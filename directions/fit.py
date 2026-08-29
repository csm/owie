"""Difference-in-means fitting, and the bundles it writes.

One bundle per (concept, layer), fitted on the **training** split only. The
held-out families never touch the fit, so every number the sweep reports is
measured on scenario families the direction has not seen.

Two things are recorded that are easy to lose and expensive to lose:

*Polarity.* The vector is always ``mean(positive) - mean(negative)``. For C3
that means it points from the refusal-eliciting member toward the benign one,
which is the reversed polarity B3 asked for. It is stated in the manifest
notes, because an arm run against a sign-flipped direction produces a clean,
plausible, and wrong result.

*Scale.* Bundles carry unit-norm vectors, since projection requires it and the
kernel refuses a non-unit direction. The raw difference norm and the layer's
mean residual norm are kept in the manifest ``extra`` block: the additive
steering grid is expressed as a multiple of the residual norm, and without that
number recorded next to the vector the grid is uninterpretable later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from safetensors.torch import save_file

from directions.bundle import current_git_revision, hash_contrasts, write_bundle
from directions.extract import FITTING_RULE, POSITION_RULES, ExtractionResult
from directions.model import HOOK_POINT, ModelHandle
from evals.schema import ContrastPair

__all__ = [
    "FittedDirection",
    "direction_id",
    "fit_directions",
    "write_diagnostic_directions",
]

FITTING_METHOD = "difference_in_means"

_POLARITY_NOTE = {
    "c1": "positive = imperative member; vector points imperative-minus-declarative",
    "c2": "positive = directive present; vector points directive-minus-neutral",
    "c3": "positive = policy-violating request; vector points violating-minus-benign, "
    "which is the reversed polarity of a refusal direction (DECISIONS.md B3)",
}


def direction_id(concept: str, layer: int, *, pilot: bool = False) -> str:
    """Stable, collision-free bundle id. Pilot bundles are named as such."""
    suffix = "-pilot" if pilot else ""
    return f"{concept}-l{layer:02d}-dim{suffix}"


@dataclass(frozen=True)
class FittedDirection:
    concept: str
    layer: int
    vector: torch.Tensor
    raw_norm: float
    mean_residual_norm: float
    bundle_path: Path


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denominator = float(a.norm()) * float(b.norm())
    if denominator == 0.0:
        return 0.0
    return float(torch.dot(a, b) / denominator)


def fit_directions(
    handle: ModelHandle,
    result: ExtractionResult,
    pairs: Sequence[ContrastPair],
    *,
    root: Path,
    extraction_config: dict[str, Any],
    layers: Sequence[int] | None = None,
) -> list[FittedDirection]:
    """Fit and write one bundle per layer for one concept.

    ``pairs`` is the complete contrast set, including held-out rows: the
    manifest hash must identify the whole frozen set, not the training slice,
    or two bundles fitted from different splits of the same file would be
    indistinguishable by hash.
    """
    if result.split != "train":
        raise ValueError(
            f"directions are fitted on the training split only, got {result.split!r}"
        )
    differences = result.difference(FITTING_RULE)
    residual_norms = result.mean_residual_norm()
    code_revision = current_git_revision()
    contrast_hash = hash_contrasts([pair.to_dict() for pair in pairs])
    selected = list(layers) if layers is not None else list(range(handle.n_layers))

    fitted: list[FittedDirection] = []
    for layer in selected:
        raw = differences[layer]
        raw_norm = float(raw.norm())
        if raw_norm == 0.0:
            raise ValueError(f"layer {layer}: difference in means is exactly zero")
        unit = (raw / raw_norm).to(torch.float32).contiguous()

        diagnostics = {
            rule: round(
                _cosine(raw, result.difference(rule)[layer]),
                6,
            )
            for rule in POSITION_RULES
            if rule != FITTING_RULE
        }
        config = dict(extraction_config)
        config.update(
            {
                "position_rule": FITTING_RULE,
                "pooling": "mean over masked token positions, per row",
                "accumulation_dtype": "float32",
                "layer": layer,
                "train_pairs": result.pairs,
                "train_rows_per_class": result.rules[FITTING_RULE]["positive"].count,
            }
        )
        path = write_bundle(
            direction_id(result.concept, layer, pilot=handle.is_pilot),
            unit,
            {
                "model_id": handle.model_id,
                "model_revision": handle.model_revision,
                "layer": layer,
                "hook_point": HOOK_POINT,
                "token_extraction_rule": FITTING_RULE,
                "fitting_method": FITTING_METHOD,
                "normalization": "unit",
                "contrast_set_hash": contrast_hash,
                "extraction_code_git_revision": code_revision,
                "notes": _POLARITY_NOTE.get(result.concept, "positive minus negative"),
                "extra": {
                    "raw_difference_norm": round(raw_norm, 6),
                    "mean_residual_norm": round(float(residual_norms[layer]), 6),
                    "relative_difference_norm": round(
                        raw_norm / float(residual_norms[layer]), 6
                    ),
                    "cosine_to_diagnostic_rules": diagnostics,
                    "is_pilot": handle.is_pilot,
                },
            },
            [pair.to_dict() for pair in pairs],
            config,
            root=root,
        )
        fitted.append(
            FittedDirection(
                concept=result.concept,
                layer=layer,
                vector=unit,
                raw_norm=raw_norm,
                mean_residual_norm=float(residual_norms[layer]),
                bundle_path=path,
            )
        )
    return fitted


def write_diagnostic_directions(path: Path, result: ExtractionResult) -> None:
    """Store the non-fitting position rules alongside the run, not as bundles.

    These are diagnostics for D2, not directions any arm applies. Keeping them
    out of the bundle namespace means nothing can accidentally load one and
    report it as the fitted direction.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {
        f"{result.concept}.{rule}": result.difference(rule).contiguous()
        for rule in POSITION_RULES
    }
    save_file(tensors, str(path))
    path.with_suffix(".json").write_text(
        json.dumps(result.summary(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
