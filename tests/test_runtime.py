from __future__ import annotations

import asyncio

import pytest
import torch

from server.runtime import (
    InterventionConfig,
    PositionAwareHook,
    RequestState,
    SerializedGenerationRuntime,
    current_request_state,
    installed_intervention_hook,
)


def state(mask=(False, True, False), **config):
    return RequestState(
        InterventionConfig(enabled=True, direction_id="d", layer=0, **config),
        tuple(mask),
        tuple(True for _ in mask),
    )


def test_prefill_changes_only_masked_source_positions():
    hook = PositionAwareHook(state(), torch.tensor([1.0, 0.0]))
    hidden = torch.tensor([[[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]])
    result = hook(None, (), {}, hidden)
    assert torch.equal(result[:, 0], hidden[:, 0])
    assert torch.equal(result[:, 2], hidden[:, 2])
    assert torch.equal(result[:, 1], torch.tensor([[0.0, 5.0]]))


def test_incremental_assistant_decode_is_never_selected():
    hook = PositionAwareHook(state(), torch.tensor([1.0, 0.0]))
    hook(None, (), {}, torch.ones(1, 3, 2))
    generated = torch.tensor([[[9.0, 8.0]]])
    assert torch.equal(hook(None, (), {}, generated), generated)


def test_cache_positions_map_back_to_prompt_mask():
    hook = PositionAwareHook(state(mask=(True, False, True)), torch.tensor([1.0, 0.0]))
    hidden = torch.tensor([[[2.0, 1.0], [3.0, 1.0]]])
    result = hook(None, (), {"cache_position": torch.tensor([2, 3])}, hidden)
    assert torch.equal(result[0, 0], torch.tensor([0.0, 1.0]))
    assert torch.equal(result[0, 1], hidden[0, 1])


def test_llama_position_ids_map_back_to_prompt_mask():
    hook = PositionAwareHook(
        state(mask=(False, True, False)), torch.tensor([1.0, 0.0])
    )
    hidden = torch.tensor([[[2.0, 1.0], [3.0, 1.0]]])
    result = hook(None, (), {"position_ids": torch.tensor([[1, 3]])}, hidden)
    assert torch.equal(result[0, 0], torch.tensor([0.0, 1.0]))
    assert torch.equal(result[0, 1], hidden[0, 1])


def test_additive_mode_uses_alpha():
    hook = PositionAwareHook(
        state(mode="add", alpha=-2.0), torch.tensor([1.0, 0.0])
    )
    hidden = torch.zeros(1, 3, 2)
    result = hook(None, (), {}, hidden)
    assert torch.equal(result[0, 1], torch.tensor([-2.0, 0.0]))
    assert torch.count_nonzero(result[0, [0, 2]]) == 0


def test_tuple_layer_output_is_preserved():
    hook = PositionAwareHook(state(), torch.tensor([1.0, 0.0]))
    result = hook(None, (), {}, (torch.ones(1, 3, 2), "cache"))
    assert result[1] == "cache"


def test_generation_error_clears_request_state():
    runtime = SerializedGenerationRuntime()

    async def scenario():
        def fail():
            assert current_request_state() is not None
            raise RuntimeError("generation failed")

        with pytest.raises(RuntimeError, match="generation failed"):
            await runtime.run(state(), fail)
        assert current_request_state() is None

    asyncio.run(scenario())


def test_sequential_configs_cannot_contaminate_one_another():
    runtime = SerializedGenerationRuntime()

    async def scenario():
        seen = []
        for scope in ("tool_content", "whole_tool_block"):
            request_state = state(scope=scope)
            await runtime.run(
                request_state,
                lambda: seen.append(current_request_state().config.scope),
            )
            assert current_request_state() is None
        assert seen == ["tool_content", "whole_tool_block"]

    asyncio.run(scenario())


class FakeLayer(torch.nn.Module):
    def forward(self, hidden, **kwargs):
        return hidden


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([FakeLayer()])


def test_installed_hook_is_removed_after_raised_error():
    model = FakeModel()
    with pytest.raises(RuntimeError, match="boom"):
        with installed_intervention_hook(model, state(), torch.tensor([1.0, 0.0])):
            assert len(model.model.layers[0]._forward_hooks) == 1
            raise RuntimeError("boom")
    assert len(model.model.layers[0]._forward_hooks) == 0


def test_invalid_enabled_config_fails_loudly():
    with pytest.raises(ValueError, match="require"):
        InterventionConfig(enabled=True)
