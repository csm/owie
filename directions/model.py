"""Loading the pinned model, and the one place that defines what a layer is.

"Layer 14" is ambiguous in a transformer, and the ambiguity is expensive: a
direction fitted against one indexing convention and applied against another is
off by one block and produces a quietly wrong effect size. This module fixes
the convention once:

    layer L  ==  the residual stream **after** decoder block L  ==
                 ``output_hidden_states[L + 1]``

``hidden_states[0]`` is the embedding output and is not a layer. The hook point
name recorded in every manifest is ``resid_post``, matching the Checkpoint 3
runtime, which registers its forward hook on ``model.model.layers[L]`` and so
sees exactly this tensor.

One measured subtlety, worth stating because it looks like a bug when first
encountered: under the pinned ``transformers``, ``output_hidden_states``
captures each block's output **before** user-registered forward hooks run. An
intervention at layer ``L`` therefore leaves ``hidden_states[L + 1]`` looking
untouched while genuinely changing ``hidden_states[L + 2]`` and the logits.
Extraction is unaffected — it runs with no hook installed — but any check that
an intervention "did something" must read one block downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from server.rendering import MODEL_ID, MODEL_REVISION

__all__ = [
    "HOOK_POINT",
    "ModelHandle",
    "load_model",
    "resolve_device",
]

HOOK_POINT = "resid_post"


def resolve_device(requested: str | None = None) -> torch.device:
    """Pick a device, preferring Metal. There is no CUDA on this machine."""
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass(frozen=True)
class ModelHandle:
    """A loaded model with the facts the rest of Phase 0 needs about it."""

    model: Any
    tokenizer: Any
    model_id: str
    model_revision: str
    device: torch.device
    dtype: torch.dtype
    n_layers: int
    d_model: int
    is_pilot: bool

    def block(self, layer: int) -> Any:
        """The decoder block whose output is ``layer``'s residual stream."""
        if not 0 <= layer < self.n_layers:
            raise ValueError(
                f"layer {layer} is out of range for a {self.n_layers}-layer model"
            )
        return self.model.model.layers[layer]

    def describe(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
            "n_layers": self.n_layers,
            "d_model": self.d_model,
            "is_pilot": self.is_pilot,
            "hook_point": HOOK_POINT,
        }


def load_model(
    *,
    model_id: str = MODEL_ID,
    revision: str = MODEL_REVISION,
    device: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    local_files_only: bool = False,
) -> ModelHandle:
    """Load a model at an immutable revision.

    ``model_id``/``revision`` default to the pinned model. Passing the approved
    3B pilot (DECISIONS.md B6) is allowed and is recorded as ``is_pilot`` on the
    handle, which propagates into every run manifest so that pilot output can
    never be mistaken for a reported result.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, local_files_only=local_files_only
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=dtype,
        local_files_only=local_files_only,
    )
    model.to(resolved)
    model.eval()
    config = model.config
    return ModelHandle(
        model=model,
        tokenizer=tokenizer,
        model_id=model_id,
        model_revision=revision,
        device=resolved,
        dtype=dtype,
        n_layers=int(config.num_hidden_layers),
        d_model=int(config.hidden_size),
        is_pilot=model_id != MODEL_ID,
    )
