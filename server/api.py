"""The deliberately small OpenAI-compatible HTTP surface."""

from __future__ import annotations

import hashlib
import time
from typing import Any, Literal, Protocol

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .backend import BackendCompletion
from .rendering import MODEL_ID, MODEL_REVISION, RenderedChat, render_chat
from .runtime import InterventionConfig, RequestState, SerializedGenerationRuntime


class InterventionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    direction_id: str | None = None
    layer: int | None = None
    mode: Literal["project", "add"] = "project"
    scope: Literal["tool_content", "whole_tool_block"] = "tool_content"
    direction_norm: Literal["unit", "raw"] = "unit"
    alpha: float = 0.0

    def runtime_config(self) -> InterventionConfig:
        return InterventionConfig(**self.model_dump())


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = MODEL_ID
    messages: list[dict[str, Any]] = Field(min_length=1)
    tools: list[dict[str, Any]] | None = None
    stream: bool = False
    temperature: float = 0.0
    max_tokens: int | None = Field(default=None, gt=0)
    seed: int | None = None
    intervention: InterventionRequest = Field(default_factory=InterventionRequest)

    @model_validator(mode="after")
    def supported_surface(self) -> "ChatCompletionRequest":
        if self.stream:
            raise ValueError("streaming is out of scope; stream must be false")
        return self


class Backend(Protocol):
    def complete(
        self, rendered: RenderedChat, request: ChatCompletionRequest, state: RequestState
    ) -> BackendCompletion: ...


def create_app(
    backend: Backend,
    tokenizer: Any,
    *,
    model_id: str = MODEL_ID,
    model_revision: str = MODEL_REVISION,
    runtime: SerializedGenerationRuntime | None = None,
) -> FastAPI:
    app = FastAPI(title="owie experimental shim", version="0.0.0")
    generation_runtime = runtime or SerializedGenerationRuntime()

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "local",
                    "revision": model_revision,
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
        if request.model != model_id:
            raise HTTPException(
                status_code=404, detail=f"model {request.model!r} is not served"
            )
        try:
            rendered = render_chat(
                tokenizer,
                request.messages,
                tools=request.tools,
                add_generation_prompt=True,
            )
            config = request.intervention.runtime_config()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        state = RequestState(
            config, rendered.primary_mask, rendered.whole_tool_block_mask
        )
        try:
            completion = await generation_runtime.run(
                state, lambda: backend.complete(rendered, request, state)
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        fingerprint = hashlib.sha256(
            rendered.text.encode()
            + b"\0"
            + (completion.content or "").encode()
            + repr(completion.tool_calls).encode()
        ).hexdigest()[:24]
        message: dict[str, Any] = {"role": "assistant", "content": completion.content}
        if completion.tool_calls:
            message["tool_calls"] = list(completion.tool_calls)
        return {
            "id": f"chatcmpl-{fingerprint}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
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
        }

    return app
