"""Serialized request state and position-aware intervention hooks."""

from __future__ import annotations

import asyncio
import contextvars
import inspect
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterator, Literal, TypeVar

import anyio
import torch

from interventions import Norm, add_vector, clamp_sae_feature, project_out

T = TypeVar("T")


@dataclass(frozen=True)
class SAEFeature:
    """One SAE feature, carried as the payload of a ``clamp_sae`` intervention.

    Both vectors are the dictionary's own, unnormalized: ``clamp_value`` is in
    the units of this feature's activation, so rescaling either vector would
    redefine what the clamp means (``interventions.clamp_sae_feature``).
    """

    feature_index: int
    encoder_row: torch.Tensor
    encoder_bias: float
    decoder_column: torch.Tensor
    clamp_value: float = 0.0
    sae_hash: str | None = None


@dataclass(frozen=True)
class InterventionConfig:
    enabled: bool = False
    direction_id: str | None = None
    layer: int | None = None
    mode: Literal["project", "add", "clamp_sae"] = "project"
    scope: Literal["tool_content", "whole_tool_block"] = "tool_content"
    direction_norm: Literal["unit", "raw"] = "unit"
    alpha: float = 0.0

    def __post_init__(self) -> None:
        if self.enabled and self.mode != "clamp_sae" and (
            not self.direction_id or self.layer is None
        ):
            raise ValueError("enabled interventions require direction_id and layer")
        if self.enabled and self.mode == "clamp_sae" and self.layer is None:
            raise ValueError("a clamp_sae intervention requires a layer")
        if self.layer is not None and (isinstance(self.layer, bool) or self.layer < 0):
            raise ValueError("layer must be a non-negative integer")
        if self.mode not in {"project", "add", "clamp_sae"}:
            raise ValueError("mode must be 'project', 'add', or 'clamp_sae'")
        if self.scope not in {"tool_content", "whole_tool_block"}:
            raise ValueError("scope must be 'tool_content' or 'whole_tool_block'")
        if self.direction_norm not in {"unit", "raw"}:
            raise ValueError("direction_norm must be 'unit' or 'raw'")


@dataclass(frozen=True)
class RequestState:
    config: InterventionConfig
    primary_mask: tuple[bool, ...]
    whole_tool_block_mask: tuple[bool, ...]

    @property
    def selected_mask(self) -> tuple[bool, ...]:
        if self.config.scope == "whole_tool_block":
            return self.whole_tool_block_mask
        return self.primary_mask


_REQUEST_STATE: contextvars.ContextVar[RequestState | None] = contextvars.ContextVar(
    "owie_request_state", default=None
)


def current_request_state() -> RequestState | None:
    return _REQUEST_STATE.get()


class SerializedGenerationRuntime:
    """One generation at a time, with request state always reset in finally."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    async def run(
        self, state: RequestState, operation: Callable[[], T | Awaitable[T]]
    ) -> T:
        async with self._lock:
            token = _REQUEST_STATE.set(state)
            try:
                if inspect.iscoroutinefunction(operation):
                    result = operation()
                else:
                    result = await anyio.to_thread.run_sync(operation)
                if inspect.isawaitable(result):
                    return await result
                return result
            finally:
                _REQUEST_STATE.reset(token)


class PositionAwareHook:
    """Apply an intervention only where absolute positions select prompt tokens."""

    def __init__(
        self,
        state: RequestState,
        direction: torch.Tensor | None,
        sae_feature: "SAEFeature | None" = None,
    ) -> None:
        if state.config.mode == "clamp_sae":
            if sae_feature is None:
                raise ValueError("a clamp_sae intervention requires an SAEFeature")
        elif direction is None:
            raise ValueError(f"mode {state.config.mode!r} requires a direction")
        self.state = state
        self.direction = direction
        self.sae_feature = sae_feature
        self._prefill_seen = False
        self._next_position = 0

    def _positions(self, seq_len: int, kwargs: dict[str, Any]) -> torch.Tensor:
        for position_field in ("cache_position", "position_ids"):
            supplied = kwargs.get(position_field)
            if supplied is None:
                continue
            positions = torch.as_tensor(supplied).reshape(-1)
            if positions.numel() == seq_len:
                self._prefill_seen = True
                self._next_position = max(
                    self._next_position, int(positions.max().item()) + 1
                )
                return positions.to(dtype=torch.long)
        if seq_len == 1 and self._prefill_seen:
            position = self._next_position
            self._next_position += 1
            return torch.tensor([position], dtype=torch.long)
        self._prefill_seen = True
        self._next_position = max(self._next_position, seq_len)
        return torch.arange(seq_len, dtype=torch.long)

    def _apply(self, hidden: torch.Tensor, kwargs: dict[str, Any]) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[0] != 1:
            raise RuntimeError(
                f"checkpoint 3 supports hidden shape (1, seq, d_model), got {tuple(hidden.shape)}"
            )
        positions = self._positions(hidden.shape[1], kwargs)
        source_mask = self.state.selected_mask
        selected = [
            bool(source_mask[position]) if 0 <= position < len(source_mask) else False
            for position in positions.tolist()
        ]
        mask = torch.tensor(selected, dtype=torch.bool, device=hidden.device).unsqueeze(0)
        if self.state.config.mode == "clamp_sae":
            feature = self.sae_feature
            assert feature is not None  # guaranteed by __init__
            return clamp_sae_feature(
                hidden,
                feature.encoder_row.to(device=hidden.device, dtype=hidden.dtype),
                feature.encoder_bias,
                feature.decoder_column.to(device=hidden.device, dtype=hidden.dtype),
                feature.clamp_value,
                mask,
            )
        assert self.direction is not None  # guaranteed by __init__
        direction = self.direction.to(device=hidden.device, dtype=hidden.dtype)
        if self.state.config.mode == "project":
            norm = (
                Norm.ASSERT_UNIT
                if self.state.config.direction_norm == "unit"
                else Norm.NORMALIZE
            )
            return project_out(hidden, direction, mask, norm=norm)
        norm = (
            Norm.ASSERT_UNIT
            if self.state.config.direction_norm == "unit"
            else Norm.AS_IS
        )
        return add_vector(hidden, direction, self.state.config.alpha, mask, norm=norm)

    def __call__(
        self,
        _module: Any,
        _args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> Any:
        if isinstance(output, tuple):
            return (self._apply(output[0], kwargs), *output[1:])
        return self._apply(output, kwargs)


def _layer_module(model: Any, layer: int) -> Any:
    candidates = [
        getattr(getattr(model, "model", None), "layers", None),
        getattr(
            getattr(getattr(model, "base_model", None), "model", None),
            "layers",
            None,
        ),
    ]
    for layers in candidates:
        if layers is not None:
            try:
                return layers[layer]
            except IndexError as exc:
                raise ValueError(f"model has no layer {layer}") from exc
    raise TypeError("cannot locate Llama decoder layers on model")


@contextmanager
def installed_intervention_hook(
    model: Any,
    state: RequestState,
    direction: torch.Tensor | None,
    sae_feature: "SAEFeature | None" = None,
) -> Iterator[PositionAwareHook]:
    if not state.config.enabled or state.config.layer is None:
        raise ValueError("cannot install a hook for a disabled intervention")
    hook = PositionAwareHook(state, direction, sae_feature)
    handle = _layer_module(model, state.config.layer).register_forward_hook(
        hook, with_kwargs=True
    )
    try:
        yield hook
    finally:
        handle.remove()
