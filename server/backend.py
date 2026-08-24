"""Transformers generation backend for the minimal chat-completions surface."""

from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from dataclasses import dataclass
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


@dataclass(frozen=True)
class RegisteredDirection:
    vector: torch.Tensor
    layer: int
    normalization: str


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
    canonical = json.dumps(
        parameters, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    digest = hashlib.sha256((value["name"] + "\0" + canonical).encode()).hexdigest()[:24]
    call = {
        "id": f"call_{digest}",
        "type": "function",
        "function": {"name": value["name"], "arguments": canonical},
    }
    return None, (call,)


class TransformersBackend:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        directions: Mapping[str, torch.Tensor | RegisteredDirection] | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.directions = dict(directions or {})

    def complete(
        self, rendered: RenderedChat, request: Any, state: RequestState
    ) -> BackendCompletion:
        config = state.config
        hook_context: Any = nullcontext()
        if config.enabled:
            try:
                registered = self.directions[config.direction_id or ""]
            except KeyError as exc:
                raise ValueError(f"unknown direction_id {config.direction_id!r}") from exc
            if isinstance(registered, RegisteredDirection):
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
                direction = registered.vector
            else:
                direction = registered
            hook_context = installed_intervention_hook(self.model, state, direction)

        device = next(self.model.parameters()).device
        input_ids = torch.tensor([rendered.input_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        temperature = float(getattr(request, "temperature", 0.0) or 0.0)
        generation = {
            "max_new_tokens": int(getattr(request, "max_tokens", None) or 256),
            "do_sample": temperature > 0.0,
            "use_cache": True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0.0:
            generation["temperature"] = temperature
        with hook_context, torch.inference_mode():
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
        return BackendCompletion(
            content=content,
            tool_calls=tool_calls,
            prompt_tokens=input_ids.shape[1],
            completion_tokens=int(new_ids.numel()),
            finish_reason="tool_calls" if tool_calls else "stop",
        )
