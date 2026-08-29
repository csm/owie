"""The layer-19 SAE: conversion off the pickle baseline, and feature selection.

Two open items are discharged here.

**D12 — the weights ship as a `.pth` torch pickle.** It is loaded exactly once,
under ``weights_only=True``, converted to safetensors, and both files are
hashed. Everything downstream reads the safetensors copy, so the pickle is
never loaded again and the conversion is recorded rather than assumed.

**Feature selection must be frozen before outcomes are inspected**
(PREFLIGHT.md §4). Features are ranked by difference in mean activation across
the same contrast set the direction was fitted on, on **training** rows only,
and the resulting list is written with a hash before any arm runs. The Rogue
Scalpel result — that hand-picked interpretable features behave very differently
from pre-registered ones — is the reason this is a procedure rather than a
judgement call.

The SAE is layer-locked to 19 (DECISIONS.md B4). Nothing here will apply it at
another layer, because a dictionary trained on layer 19 activations does not
mean anything at layer 12.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from safetensors.torch import load_file, save_file

from directions.model import ModelHandle
from evals.schema import ContrastPair

__all__ = [
    "SAE_REPO",
    "SAE_REVISION",
    "SAE_LAYER",
    "SAE",
    "load_sae",
    "select_features",
]

SAE_REPO = "Goodfire/Llama-3.1-8B-Instruct-SAE-l19"
SAE_REVISION = "f6775a221e47b44233af4bac2c7b65189265519a"
SAE_FILE = "Llama-3.1-8B-Instruct-SAE-l19.pth"
SAE_LAYER = 19


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


@dataclass(frozen=True)
class SAE:
    """A loaded dictionary. Vectors keep the scales the SAE was trained with."""

    encoder_weight: torch.Tensor  # (n_features, d_model)
    encoder_bias: torch.Tensor  # (n_features,)
    decoder_weight: torch.Tensor  # (d_model, n_features)
    layer: int
    repo: str
    revision: str
    source_hash: str
    safetensors_hash: str

    @property
    def n_features(self) -> int:
        return int(self.encoder_weight.shape[0])

    @property
    def d_model(self) -> int:
        return int(self.encoder_weight.shape[1])

    def encode(self, hidden: torch.Tensor) -> torch.Tensor:
        """Feature activations for (..., d_model) states. ReLU, as trained."""
        weight = self.encoder_weight.to(device=hidden.device, dtype=torch.float32)
        bias = self.encoder_bias.to(device=hidden.device, dtype=torch.float32)
        return torch.relu(hidden.to(torch.float32) @ weight.T + bias)

    def feature(self, index: int) -> tuple[torch.Tensor, float, torch.Tensor]:
        """Encoder row, encoder bias, and decoder column for one feature."""
        if not 0 <= index < self.n_features:
            raise ValueError(f"feature {index} out of range (0, {self.n_features})")
        return (
            self.encoder_weight[index].clone(),
            float(self.encoder_bias[index]),
            self.decoder_weight[:, index].clone().contiguous(),
        )

    def describe(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "revision": self.revision,
            "layer": self.layer,
            "n_features": self.n_features,
            "d_model": self.d_model,
            "source_pth_hash": self.source_hash,
            "safetensors_hash": self.safetensors_hash,
            "conversion": "torch.load(weights_only=True) -> safetensors, once",
        }


def load_sae(*, cache_dir: Path, local_files_only: bool = False) -> SAE:
    """Fetch, convert once, and load. Reuses the converted copy on later calls."""
    from huggingface_hub import hf_hub_download

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    converted = cache_dir / "sae-l19.safetensors"
    metadata_path = cache_dir / "sae-l19.json"

    source = Path(
        hf_hub_download(
            SAE_REPO,
            SAE_FILE,
            revision=SAE_REVISION,
            local_files_only=local_files_only,
        )
    )
    source_hash = _sha256(source)

    if not converted.is_file():
        state = torch.load(source, weights_only=True, map_location="cpu")
        expected = {
            "encoder_linear.weight",
            "encoder_linear.bias",
            "decoder_linear.weight",
            "decoder_linear.bias",
        }
        missing = expected - set(state)
        if missing:
            raise RuntimeError(f"SAE checkpoint is missing {sorted(missing)}")
        save_file(
            {key: value.contiguous() for key, value in state.items()}, str(converted)
        )
        metadata_path.write_text(
            json.dumps(
                {
                    "repo": SAE_REPO,
                    "revision": SAE_REVISION,
                    "layer": SAE_LAYER,
                    "source_file": SAE_FILE,
                    "source_pth_hash": source_hash,
                    "safetensors_hash": _sha256(converted),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    tensors = load_file(str(converted))
    return SAE(
        encoder_weight=tensors["encoder_linear.weight"],
        encoder_bias=tensors["encoder_linear.bias"],
        decoder_weight=tensors["decoder_linear.weight"],
        layer=SAE_LAYER,
        repo=SAE_REPO,
        revision=SAE_REVISION,
        source_hash=source_hash,
        safetensors_hash=_sha256(converted),
    )


@torch.inference_mode()
def select_features(
    handle: ModelHandle,
    sae: SAE,
    pairs: Sequence[ContrastPair],
    *,
    top_k: int,
    split: str = "train",
) -> dict[str, Any]:
    """Rank features by difference in mean activation on training rows.

    Returns the frozen selection, hashed. The returned dict is written to the
    run directory *before* any SAE arm executes; the hash is what makes "we did
    not reselect after seeing the outcome" checkable rather than asserted.
    """
    from server.rendering import render_chat

    from directions.extract import span_masks

    totals = {
        polarity: torch.zeros(sae.n_features, dtype=torch.float32)
        for polarity in ("positive", "negative")
    }
    counts = {"positive": 0, "negative": 0}

    for pair in pairs:
        if pair.split != split:
            continue
        for polarity in ("positive", "negative"):
            rendered = render_chat(handle.tokenizer, pair.messages(polarity))
            masks = span_masks(
                rendered,
                pair.varied_span_positive
                if polarity == "positive"
                else pair.varied_span_negative,
                pair.member(polarity),
            )
            input_ids = torch.tensor(
                [rendered.input_ids], dtype=torch.long, device=handle.device
            )
            outputs = handle.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                output_hidden_states=True,
                use_cache=False,
            )
            hidden = outputs.hidden_states[sae.layer + 1][0]
            index = masks["tool_content"].nonzero(as_tuple=True)[0].to(hidden.device)
            selected = hidden.index_select(0, index)
            totals[polarity] += sae.encode(selected).mean(dim=0).cpu()
            counts[polarity] += 1

    if not counts["positive"] or counts["positive"] != counts["negative"]:
        raise RuntimeError(
            f"unbalanced feature-selection pass: {counts}. An unbalanced mean "
            "difference measures composition, not the concept."
        )
    difference = totals["positive"] / counts["positive"] - totals["negative"] / counts["negative"]
    order = torch.argsort(difference.abs(), descending=True)[:top_k]
    selection = {
        "sae": sae.describe(),
        "split": split,
        "rows_per_class": counts["positive"],
        "ranking": "absolute difference in mean feature activation, positive minus negative",
        "top_k": top_k,
        "features": [
            {
                "feature_index": int(index),
                "mean_activation_difference": round(float(difference[index]), 6),
                "mean_activation_positive": round(
                    float(totals["positive"][index] / counts["positive"]), 6
                ),
                "mean_activation_negative": round(
                    float(totals["negative"][index] / counts["negative"]), 6
                ),
            }
            for index in order.tolist()
        ],
    }
    payload = json.dumps(selection, sort_keys=True, separators=(",", ":"))
    selection["selection_hash"] = f"sha256:{hashlib.sha256(payload.encode()).hexdigest()}"
    return selection
