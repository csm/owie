"""Activation extraction over contrast sets, streamed into mean accumulators.

One forward pass per contrast member captures every layer at once, so the whole
sweep's extraction is a single pass over the data rather than one pass per
layer. Nothing is cached to disk: PREFLIGHT.md §6 costs a full activation cache
at 10-20 GB, and it buys nothing, because the only statistic the fit needs is a
per-class mean.

Four position rules are captured from the same forward pass. Only the first is
the fitting rule; the rest are the diagnostics D2 asked to be retained, and
they are nearly free once the activations exist:

``tool_content``
    Every token the primary mask selects — the positions the intervention will
    actually act on. **This is the fitting rule** (D2, ruled 2026-08-24).
``varied_span``
    The subset of tool-content tokens overlapping the varied span alone.
    Diagnostic: how much of the difference lives in the manipulated clause
    rather than in the shared body.
``last_prompt``
    The final prompt token, which is where CAST-style extraction reads. Kept so
    the departure from that precedent is measurable rather than merely argued.
``first_generated``
    The position of the model's own first greedily generated token. Requires
    one extra decode step per member.

Accumulation is in float32 on CPU. MPS has no float64 (PREFLIGHT.md §8, risk
15) and a bf16 running sum over hundreds of rows loses precision where it
matters most — in the small difference between two large means.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from directions.model import HOOK_POINT, ModelHandle
from evals.schema import ContrastPair

__all__ = [
    "POSITION_RULES",
    "FITTING_RULE",
    "MeanAccumulator",
    "ExtractionResult",
    "extract_contrast_means",
    "span_masks",
]

POSITION_RULES = ("tool_content", "varied_span", "last_prompt", "first_generated")

# The rule the direction is fitted from. The others are diagnostics and must
# never be silently substituted for it.
FITTING_RULE = "tool_content"


class ExtractionError(RuntimeError):
    """Extraction cannot proceed on a row, and skipping it would bias the fit."""


def span_masks(
    rendered: Any, varied_span: str, member: str
) -> dict[str, torch.Tensor]:
    """Boolean token masks for every position rule except ``first_generated``.

    ``tool_content`` comes straight from the renderer's primary mask, so the
    fitting positions are by construction the same positions the Checkpoint 3
    runtime intervenes on. ``varied_span`` is derived from the renderer's
    per-character source indices rather than by searching the rendered text:
    tool content is JSON-escaped, so a substring search over rendered text can
    miss or misplace the span whenever it contains an escaped character.
    """
    n_tokens = len(rendered.input_ids)
    primary = torch.tensor(rendered.primary_mask, dtype=torch.bool)
    if int(primary.sum()) == 0:
        raise ExtractionError(
            "the primary tool-content mask selected no tokens; the pair would "
            "contribute nothing to the fit and is silently dropping data"
        )

    start = member.index(varied_span)
    end = start + len(varied_span)
    span_chars = {
        (region.start, region.end)
        for region in rendered.regions
        if region.role == "tool"
        and region.region == "content"
        and region.source_character is not None
        and start <= region.source_character < end
    }
    varied = torch.zeros(n_tokens, dtype=torch.bool)
    for index, (token_start, token_end) in enumerate(rendered.offsets):
        if token_start >= token_end:
            continue
        for char_start, char_end in span_chars:
            if token_start < char_end and char_start < token_end:
                varied[index] = True
                break
    if int(varied.sum()) == 0:
        raise ExtractionError("the varied-span mask selected no tokens")

    last_prompt = torch.zeros(n_tokens, dtype=torch.bool)
    last_prompt[-1] = True
    return {"tool_content": primary, "varied_span": varied, "last_prompt": last_prompt}


@dataclass
class MeanAccumulator:
    """Streaming per-layer mean of activation vectors, in float32 on CPU."""

    n_layers: int
    d_model: int
    total: torch.Tensor = field(init=False)
    count: int = 0

    def __post_init__(self) -> None:
        self.total = torch.zeros(self.n_layers, self.d_model, dtype=torch.float32)

    def add(self, vectors: torch.Tensor) -> None:
        """Add one row's per-layer vector, shape (n_layers, d_model)."""
        if vectors.shape != self.total.shape:
            raise ExtractionError(
                f"expected {tuple(self.total.shape)}, got {tuple(vectors.shape)}"
            )
        self.total += vectors.to(torch.float32)
        self.count += 1

    def mean(self) -> torch.Tensor:
        if self.count == 0:
            raise ExtractionError("no rows accumulated; the mean is undefined")
        return self.total / self.count


@dataclass
class ExtractionResult:
    """Per-class means for every rule, plus the scale statistics the sweep needs."""

    concept: str
    split: str
    rules: dict[str, dict[str, MeanAccumulator]]
    # Mean L2 norm of the residual stream at tool-content positions, per layer.
    # The additive-steering alpha grid is expressed as a multiple of this, so
    # that one pre-registered grid is comparable across layers of very
    # different scale (EXPERIMENT_PROTOCOL.md §5).
    residual_norm_total: torch.Tensor
    residual_norm_count: int = 0
    pairs: int = 0
    tokens_by_rule: dict[str, int] = field(default_factory=dict)

    def mean_residual_norm(self) -> torch.Tensor:
        if self.residual_norm_count == 0:
            raise ExtractionError("no residual norms accumulated")
        return self.residual_norm_total / self.residual_norm_count

    def difference(self, rule: str = FITTING_RULE) -> torch.Tensor:
        """Difference in means, positive minus negative, shape (n_layers, d)."""
        accumulators = self.rules[rule]
        return accumulators["positive"].mean() - accumulators["negative"].mean()

    def summary(self) -> dict[str, Any]:
        return {
            "concept": self.concept,
            "split": self.split,
            "pairs": self.pairs,
            "rows_by_rule": {
                rule: {
                    polarity: accumulator.count
                    for polarity, accumulator in polarities.items()
                }
                for rule, polarities in self.rules.items()
            },
            "tokens_by_rule": dict(self.tokens_by_rule),
            "mean_residual_norm_by_layer": [
                round(float(value), 4) for value in self.mean_residual_norm()
            ],
        }


def _pooled(hidden_states: Sequence[torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool masked positions for every layer. Returns (n_layers, d_model).

    ``hidden_states[0]`` is the embedding output and is dropped, so index ``L``
    of the result is the residual stream after block ``L`` (``directions.model``).
    """
    index = mask.nonzero(as_tuple=True)[0]
    stacked = torch.stack(
        [state[0].index_select(0, index.to(state.device)) for state in hidden_states[1:]]
    )
    return stacked.to(torch.float32).mean(dim=1).cpu()


@torch.inference_mode()
def extract_contrast_means(
    handle: ModelHandle,
    pairs: Iterable[ContrastPair],
    *,
    concept: str,
    split: str,
    progress: Any = None,
) -> ExtractionResult:
    """One forward pass per member; every rule and every layer from that pass.

    Raises on any row that cannot be processed. Silently skipping a row would
    unbalance the two classes, and an unbalanced difference-in-means is a
    difference in composition as much as a difference in concept.
    """
    from server.rendering import render_chat

    rules = {
        rule: {
            polarity: MeanAccumulator(handle.n_layers, handle.d_model)
            for polarity in ("positive", "negative")
        }
        for rule in POSITION_RULES
    }
    result = ExtractionResult(
        concept=concept,
        split=split,
        rules=rules,
        residual_norm_total=torch.zeros(handle.n_layers, dtype=torch.float32),
        tokens_by_rule={rule: 0 for rule in POSITION_RULES},
    )

    for pair in pairs:
        if pair.split != split:
            continue
        result.pairs += 1
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
                use_cache=True,
            )
            hidden_states = outputs.hidden_states
            for rule, mask in masks.items():
                rules[rule][polarity].add(_pooled(hidden_states, mask))
                result.tokens_by_rule[rule] += int(mask.sum())

            tool_index = masks["tool_content"].nonzero(as_tuple=True)[0]
            norms = torch.stack(
                [
                    state[0]
                    .index_select(0, tool_index.to(state.device))
                    .to(torch.float32)
                    .norm(dim=-1)
                    .mean()
                    for state in hidden_states[1:]
                ]
            ).cpu()
            result.residual_norm_total += norms
            result.residual_norm_count += 1

            # Rule (c): one greedy decode step, then read the position the
            # generated token occupies. Cheap because the prefill cache exists.
            next_token = outputs.logits[0, -1].argmax().reshape(1, 1)
            step = handle.model(
                input_ids=next_token,
                past_key_values=outputs.past_key_values,
                output_hidden_states=True,
                use_cache=False,
            )
            first_generated = torch.stack(
                [state[0, -1].to(torch.float32) for state in step.hidden_states[1:]]
            ).cpu()
            rules["first_generated"][polarity].add(first_generated)
            result.tokens_by_rule["first_generated"] += 1

            if progress is not None:
                progress(pair.pair_id, polarity)
    return result


def write_extraction_summary(path: Path, results: Sequence[ExtractionResult]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "hook_point": HOOK_POINT,
                "fitting_rule": FITTING_RULE,
                "position_rules": list(POSITION_RULES),
                "results": [result.summary() for result in results],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
