"""Transformers generation backend for the minimal chat-completions surface."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .rendering import RenderedChat
from .runtime import RequestState, installed_intervention_hook


@dataclass(frozen=True)
class BackendCompletion:
    content: str | None
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str = "stop"
    tool_calls: tuple[dict[str, Any], ...] = ()
    direction_bundle_hash: str | None = None


@dataclass(frozen=True)
class RegisteredDirection:
    vector: torch.Tensor
    layer: int
    normalization: str
    hook_point: str = "resid_post"
    bundle_hash: str | None = None


def hash_direction_bundle(path: Path) -> str:
    """Hash every required bundle artifact in a stable filename order."""
    digest = hashlib.sha256()
    for name in (
        "contrasts.jsonl",
        "extraction_config.json",
        "manifest.json",
        "vector.safetensors",
    ):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update((path / name).read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def encode_tool_call(content: str) -> tuple[str | None, tuple[dict[str, Any], ...]]:
    """Translate the pinned model's JSON call envelope to OpenAI encoding."""
    try:
        value = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return content, ()
    if not isinstance(value, dict) or not isinstance(value.get("name"), str):
        return content, ()
    parameters = value.get("parameters")
    if not isinstance(parameters, dict):
        return content, ()
    canonical = json.dumps(parameters, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256((value["name"] + "\0" + canonical).encode()).hexdigest()[:24]
    call = {
        "id": f"call_{digest}",
        "type": "function",
        "function": {"name": value["name"], "arguments": canonical},
    }
    return None, (call,)


@contextmanager
def _seeded_rng(seed: int | None, device: torch.device):
    """Seed generation request-locally and restore the prior RNG state."""
    if seed is None:
        yield
        return
    cpu_state = torch.random.get_rng_state()
    mps_state = torch.mps.get_rng_state() if device.type == "mps" else None
    try:
        torch.manual_seed(seed)
        if device.type == "mps":
            torch.mps.manual_seed(seed)
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)


def _eos_token_ids(model: Any, tokenizer: Any) -> set[int]:
    value = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if value is None:
        value = getattr(tokenizer, "eos_token_id", None)
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    return {int(token_id) for token_id in value}


class TransformersBackend:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        directions: Mapping[str, RegisteredDirection] | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.directions = dict(directions or {})

    def complete(
        self, rendered: RenderedChat, request: Any, state: RequestState
    ) -> BackendCompletion:
        config = state.config
        hook_context: Any = nullcontext()
        direction_bundle_hash: str | None = None
        if config.enabled:
            try:
                registered = self.directions[config.direction_id or ""]
            except KeyError as exc:
                raise ValueError(f"unknown direction_id {config.direction_id!r}") from exc
            if config.layer != registered.layer:
                raise ValueError(
                    f"direction {config.direction_id!r} was fitted at layer "
                    f"{registered.layer}, not requested layer {config.layer}"
                )
            if config.direction_norm != registered.normalization:
                raise ValueError(
                    f"direction {config.direction_id!r} has manifest normalization "
                    f"{registered.normalization!r}, not {config.direction_norm!r}"
                )
            if registered.hook_point != "resid_post":
                raise ValueError(
                    f"direction {config.direction_id!r} was fitted at hook point "
                    f"{registered.hook_point!r}; this server supports only 'resid_post'"
                )
            direction = registered.vector
            direction_bundle_hash = registered.bundle_hash
            hook_context = installed_intervention_hook(self.model, state, direction)

        device = next(self.model.parameters()).device
        input_ids = torch.tensor([rendered.input_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        temperature = float(getattr(request, "temperature", 0.0) or 0.0)
        max_new_tokens = int(getattr(request, "max_tokens", None) or 256)
        generation = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0.0,
            "use_cache": bool(getattr(request, "use_cache", True)),
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0.0:
            generation["temperature"] = temperature
        seed = getattr(request, "seed", None)
        with hook_context, _seeded_rng(seed, device), torch.inference_mode():
            output = self.model.generate(
                input_ids=input_ids, attention_mask=attention_mask, **generation
            )
        new_ids = output[0, input_ids.shape[1] :]
        text = self.tokenizer.decode(
            new_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        content, tool_calls = encode_tool_call(text)
        hit_eos = bool(new_ids.numel()) and int(new_ids[-1].item()) in _eos_token_ids(
            self.model, self.tokenizer
        )
        if int(new_ids.numel()) >= max_new_tokens and not hit_eos:
            finish_reason = "length"
        elif tool_calls:
            finish_reason = "tool_calls"
        else:
            finish_reason = "stop"
        return BackendCompletion(
            content=content,
            tool_calls=tool_calls,
            prompt_tokens=input_ids.shape[1],
            completion_tokens=int(new_ids.numel()),
            finish_reason=finish_reason,
            direction_bundle_hash=direction_bundle_hash,
        )
