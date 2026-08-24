"""Checkpoint 3 provenance-aware OpenAI-compatible shim."""

from .api import ChatCompletionRequest, InterventionRequest, create_app
from .backend import (
    BackendCompletion,
    RegisteredDirection,
    TransformersBackend,
    encode_tool_call,
)
from .rendering import (
    MODEL_ID,
    MODEL_REVISION,
    AmbiguousToken,
    CharacterRegion,
    RenderedChat,
    RenderError,
    load_pinned_tokenizer,
    render_chat,
    render_characters,
)
from .runtime import (
    InterventionConfig,
    PositionAwareHook,
    RequestState,
    SerializedGenerationRuntime,
    current_request_state,
    installed_intervention_hook,
)

__all__ = [
    "MODEL_ID",
    "MODEL_REVISION",
    "AmbiguousToken",
    "BackendCompletion",
    "CharacterRegion",
    "ChatCompletionRequest",
    "InterventionConfig",
    "InterventionRequest",
    "PositionAwareHook",
    "RegisteredDirection",
    "RenderError",
    "RenderedChat",
    "RequestState",
    "SerializedGenerationRuntime",
    "TransformersBackend",
    "create_app",
    "current_request_state",
    "encode_tool_call",
    "installed_intervention_hook",
    "load_pinned_tokenizer",
    "render_characters",
    "render_chat",
]
