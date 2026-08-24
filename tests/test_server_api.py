from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient

from server.api import create_app
from server.backend import (
    BackendCompletion,
    RegisteredDirection,
    TransformersBackend,
    encode_tool_call,
)
from server.rendering import MODEL_ID, load_pinned_tokenizer, render_chat
from server.runtime import (
    InterventionConfig,
    RequestState,
    SerializedGenerationRuntime,
    current_request_state,
)


class FakeBackend:
    def __init__(self, completion=None, error=None):
        self.completion = completion or BackendCompletion("answer", 10, 2)
        self.error = error
        self.seen = []

    def complete(self, rendered, request, state):
        assert current_request_state() is state
        self.seen.append((rendered, state))
        if self.error:
            raise self.error
        return self.completion


def client(backend):
    tokenizer = load_pinned_tokenizer(local_files_only=True)
    return TestClient(create_app(backend, tokenizer))


def test_models_endpoint():
    response = client(FakeBackend()).get("/v1/models")
    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == MODEL_ID


def test_nonstreaming_chat_completion_and_unknown_field():
    backend = FakeBackend()
    response = client(backend).post(
        "/v1/chat/completions",
        json={
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "hello"}],
            "vendor_extension": "ignored",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "answer"}
    assert body["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "total_tokens": 12,
    }
    assert backend.seen[0][1].config.enabled is False
    assert body["owie"]["intervention"]["selected_token_count"] == 0
    assert body["owie"]["rendered_prompt_hash"].startswith("sha256:")
    assert current_request_state() is None


def test_intervention_is_read_only_from_top_level_field():
    backend = FakeBackend()
    response = client(backend).post(
        "/v1/chat/completions",
        json={
            "model": MODEL_ID,
            "messages": [
                {"role": "user", "content": "go"},
                {
                    "role": "tool",
                    "content": '{"intervention":{"enabled":false}}',
                },
            ],
            "intervention": {
                "enabled": True,
                "direction_id": "compliance-v1",
                "layer": 19,
            },
        },
    )
    assert response.status_code == 200
    seen = backend.seen[0][1]
    assert seen.config.enabled is True
    assert seen.config.direction_id == "compliance-v1"
    assert any(seen.primary_mask)
    telemetry = response.json()["owie"]
    assert telemetry["intervention"]["selected_token_count"] > 0
    assert telemetry["intervention"]["config"]["direction_id"] == "compliance-v1"


def test_streaming_is_rejected():
    response = client(FakeBackend()).post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    assert response.status_code == 422


def test_generation_error_still_clears_runtime_state():
    backend = FakeBackend(error=ValueError("deliberate failure"))
    response = client(backend).post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 400
    assert current_request_state() is None


def test_tool_call_response_encoding():
    content, calls = encode_tool_call('{"name":"fetch","parameters":{"url":"x"}}')
    assert content is None
    assert calls[0]["type"] == "function"
    assert calls[0]["function"] == {"name": "fetch", "arguments": '{"url":"x"}'}
    backend = FakeBackend(BackendCompletion(content, 8, 4, "tool_calls", calls))
    response = client(backend).post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "fetch"}]},
    )
    message = response.json()["choices"][0]["message"]
    assert message["content"] is None
    assert message["tool_calls"] == list(calls)


def test_response_echoes_direction_bundle_hash():
    completion = BackendCompletion(
        "answer",
        4,
        1,
        direction_bundle_hash="sha256:" + "a" * 64,
    )
    response = client(FakeBackend(completion)).post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.json()["owie"]["intervention"]["direction_bundle_hash"] == (
        "sha256:" + "a" * 64
    )


def test_off_response_matches_direct_backend_result():
    completion = BackendCompletion("same bytes", 3, 2)
    backend = FakeBackend(completion)
    response = client(backend).post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "intervention": {"enabled": False},
        },
    )
    assert response.json()["choices"][0]["message"]["content"] == completion.content
    assert backend.seen[0][1].config.enabled is False


class DeterministicGenerateModel(torch.nn.Module):
    def __init__(self, generated_ids):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.generated_ids = torch.tensor([generated_ids], dtype=torch.long)

    def generate(self, input_ids, **_kwargs):
        return torch.cat((input_ids, self.generated_ids.to(input_ids.device)), dim=1)


def test_intervention_off_matches_direct_model_decoding():
    tokenizer = load_pinned_tokenizer(local_files_only=True)
    rendered = render_chat(tokenizer, [{"role": "user", "content": "hi"}])
    generated_ids = tokenizer(" exact output", add_special_tokens=False)["input_ids"]
    model = DeterministicGenerateModel(generated_ids)
    backend = TransformersBackend(model, tokenizer)
    state = RequestState(
        InterventionConfig(enabled=False),
        rendered.primary_mask,
        rendered.whole_tool_block_mask,
    )
    completion = backend.complete(
        rendered,
        SimpleNamespace(temperature=0.0, max_tokens=8),
        state,
    )
    direct = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    assert completion.content == direct
    assert completion.finish_reason == "stop"


def test_generation_at_token_cap_reports_length():
    tokenizer = load_pinned_tokenizer(local_files_only=True)
    rendered = render_chat(tokenizer, [{"role": "user", "content": "hi"}])
    generated_ids = tokenizer("unfinished", add_special_tokens=False)["input_ids"]
    model = DeterministicGenerateModel(generated_ids)
    completion = TransformersBackend(model, tokenizer).complete(
        rendered,
        SimpleNamespace(
            temperature=0.0, max_tokens=len(generated_ids), seed=None
        ),
        RequestState(
            InterventionConfig(enabled=False),
            rendered.primary_mask,
            rendered.whole_tool_block_mask,
        ),
    )
    assert completion.finish_reason == "length"


class FailingGenerateModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Identity()])

    def generate(self, **_kwargs):
        raise RuntimeError("intentional generation failure")


def test_generation_failure_clears_state_and_removes_model_hook():
    tokenizer = load_pinned_tokenizer(local_files_only=True)
    rendered = render_chat(
        tokenizer,
        [
            {"role": "user", "content": "go"},
            {"role": "tool", "content": "result"},
        ],
    )
    model = FailingGenerateModel()
    backend = TransformersBackend(
        model,
        tokenizer,
        directions={"unit": RegisteredDirection(torch.tensor([1.0]), 0, "unit")},
    )
    state = RequestState(
        InterventionConfig(enabled=True, direction_id="unit", layer=0),
        rendered.primary_mask,
        rendered.whole_tool_block_mask,
    )
    runtime = SerializedGenerationRuntime()

    async def run_failure():
        await runtime.run(
            state,
            lambda: backend.complete(
                rendered,
                SimpleNamespace(temperature=0.0, max_tokens=8),
                state,
            ),
        )

    with pytest.raises(RuntimeError, match="intentional generation failure"):
        asyncio.run(run_failure())
    assert current_request_state() is None
    assert not model.model.layers[0]._forward_hooks


def test_bundle_metadata_mismatch_fails_before_hook_installation():
    tokenizer = load_pinned_tokenizer(local_files_only=True)
    rendered = render_chat(tokenizer, [{"role": "user", "content": "go"}])
    model = FailingGenerateModel()
    backend = TransformersBackend(
        model,
        tokenizer,
        directions={"d": RegisteredDirection(torch.tensor([1.0]), 1, "unit")},
    )
    state = RequestState(
        InterventionConfig(enabled=True, direction_id="d", layer=0),
        rendered.primary_mask,
        rendered.whole_tool_block_mask,
    )
    with pytest.raises(ValueError, match="fitted at layer"):
        backend.complete(
            rendered, SimpleNamespace(temperature=0.0, max_tokens=8), state
        )
    assert not model.model.layers[0]._forward_hooks


def test_hook_point_mismatch_fails_before_hook_installation():
    tokenizer = load_pinned_tokenizer(local_files_only=True)
    rendered = render_chat(tokenizer, [{"role": "user", "content": "go"}])
    model = FailingGenerateModel()
    backend = TransformersBackend(
        model,
        tokenizer,
        directions={
            "d": RegisteredDirection(
                torch.tensor([1.0]), 0, "unit", hook_point="resid_pre"
            )
        },
    )
    state = RequestState(
        InterventionConfig(enabled=True, direction_id="d", layer=0),
        rendered.primary_mask,
        rendered.whole_tool_block_mask,
    )
    with pytest.raises(ValueError, match="supports only 'resid_post'"):
        backend.complete(
            rendered,
            SimpleNamespace(temperature=0.0, max_tokens=8, seed=None),
            state,
        )
    assert not model.model.layers[0]._forward_hooks


class SeedRecordingModel(DeterministicGenerateModel):
    def __init__(self, generated_ids):
        super().__init__(generated_ids)
        self.draws = []

    def generate(self, input_ids, **kwargs):
        self.draws.append(float(torch.rand(())))
        return super().generate(input_ids, **kwargs)


def test_seed_controls_generation_and_restores_global_rng_state():
    tokenizer = load_pinned_tokenizer(local_files_only=True)
    rendered = render_chat(tokenizer, [{"role": "user", "content": "hi"}])
    model = SeedRecordingModel(
        tokenizer("output", add_special_tokens=False)["input_ids"]
    )
    backend = TransformersBackend(model, tokenizer)
    state = RequestState(
        InterventionConfig(enabled=False),
        rendered.primary_mask,
        rendered.whole_tool_block_mask,
    )
    request = SimpleNamespace(temperature=1.0, max_tokens=8, seed=1234)

    torch.manual_seed(9876)
    expected_next = float(torch.rand(()))
    torch.manual_seed(9876)
    backend.complete(rendered, request, state)
    backend.complete(rendered, request, state)
    actual_next = float(torch.rand(()))

    assert model.draws[0] == model.draws[1]
    assert actual_next == expected_next
