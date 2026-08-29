"""Synchronous in-process client for exact-prefix experiments."""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict
from typing import Any, Mapping

from .api import ChatCompletionRequest
from .backend import TransformersBackend
from .rendering import MODEL_ID, MODEL_REVISION, render_chat
from .runtime import RequestState


class DirectChatClient:
    """Run the OpenAI-compatible request path without HTTP or concurrency."""

    def __init__(
        self,
        backend: TransformersBackend,
        tokenizer: Any,
        *,
        model_id: str = MODEL_ID,
        model_revision: str = MODEL_REVISION,
    ) -> None:
        self.backend = backend
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.model_revision = model_revision

    def complete(self, request: Mapping[str, Any]) -> dict[str, Any]:
        parsed = ChatCompletionRequest.model_validate(dict(request))
        if parsed.model != self.model_id:
            raise ValueError(f"model {parsed.model!r} is not served")
        rendered = render_chat(
            self.tokenizer,
            parsed.messages,
            tools=parsed.tools,
            add_generation_prompt=True,
        )
        config = parsed.intervention.runtime_config()
        state = RequestState(
            config,
            rendered.primary_mask,
            rendered.whole_tool_block_mask,
        )
        completion = self.backend.complete(rendered, parsed, state)
        message: dict[str, Any] = {"role": "assistant", "content": completion.content}
        if completion.tool_calls:
            message["tool_calls"] = list(completion.tool_calls)
        fingerprint = hashlib.sha256(
            rendered.text.encode()
            + b"\0"
            + (completion.content or "").encode()
            + repr(completion.tool_calls).encode()
        ).hexdigest()[:24]
        prompt_hash = f"sha256:{hashlib.sha256(rendered.text.encode()).hexdigest()}"
        return {
            "id": f"chatcmpl-{fingerprint}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": completion.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": completion.prompt_tokens,
                "completion_tokens": completion.completion_tokens,
                "total_tokens": completion.prompt_tokens + completion.completion_tokens,
            },
            "owie": {
                "model_revision": self.model_revision,
                "rendered_prompt_hash": prompt_hash,
                "prompt_token_count": len(rendered.input_ids),
                "ambiguous_token_count": len(rendered.ambiguous_tokens),
                "seed": parsed.seed,
                "intervention": {
                    "config": asdict(config),
                    "selected_token_count": sum(state.selected_mask),
                    "primary_token_count": sum(state.primary_mask),
                    "whole_tool_block_token_count": sum(
                        state.whole_tool_block_mask
                    ),
                    "direction_bundle_hash": completion.direction_bundle_hash,
                },
            },
        }
