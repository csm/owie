from __future__ import annotations

import copy
import hashlib
import json

import pytest

from loop.runner import (
    LoopConfig,
    canonical_trajectory,
    run_task,
    trajectories_match,
    write_determinism_report,
    write_trajectory,
)
from loop.tasks import TASK_BY_ID, StatePredicate, Task
from loop.tools import TOOL_NAMES, TOOL_SCHEMAS, FakeEnvironment, ToolError


def test_exactly_three_fake_tool_domains_and_deterministic_state_transitions():
    assert len(TOOL_SCHEMAS) == 3
    assert TOOL_NAMES == {"fake_filesystem", "fake_key_value", "fake_http_fetch"}
    environment = FakeEnvironment(documents={"https://x": "canned"})

    fetched = environment.execute("fake_http_fetch", {"url": "https://x"})
    assert fetched.output["body"] == "canned"
    assert fetched.state_before == fetched.state_after

    written = environment.execute(
        "fake_filesystem", {"operation": "write", "path": "/a", "content": "one"}
    )
    assert written.state_before["files"] == {}
    assert written.state_after["files"] == {"/a": "one"}
    stored = environment.execute(
        "fake_key_value", {"operation": "set", "key": "a", "value": "one"}
    )
    assert stored.state_after["key_values"] == {"a": "one"}


@pytest.mark.parametrize(
    "name,arguments,match",
    [
        ("host_shell", {}, "unknown tool"),
        ("fake_filesystem", {"operation": "read", "path": "../secret"}, "absolute"),
        ("fake_http_fetch", {"url": "https://x", "method": "POST"}, "only 'url'"),
    ],
)
def test_fake_tools_fail_closed(name, arguments, match):
    with pytest.raises(ToolError, match=match):
        FakeEnvironment().execute(name, arguments)


class ScriptedClient:
    def __init__(self, calls, *, created=100):
        self.calls = list(calls)
        self.created = created
        self.requests = []

    def complete(self, request):
        self.requests.append(copy.deepcopy(request))
        index = len(self.requests) - 1
        if index >= len(self.calls):
            message = {"role": "assistant", "content": "done"}
            finish_reason = "stop"
        else:
            name, arguments = self.calls[index]
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{index}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(arguments, separators=(",", ":")),
                        },
                    }
                ],
            }
            finish_reason = "tool_calls"
        prompt = json.dumps(request, sort_keys=True, separators=(",", ":"))
        return {
            "id": f"chatcmpl-{index}",
            "object": "chat.completion",
            "created": self.created + index,
            "model": request["model"],
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 10 + index, "completion_tokens": 4, "total_tokens": 14 + index},
            "owie": {
                "model_revision": "revision",
                "rendered_prompt_hash": "sha256:" + hashlib.sha256(prompt.encode()).hexdigest(),
            },
        }


def _test_task():
    return Task(
        task_id="test",
        objective="Fetch the document, write /answer, and set status.",
        documents={"https://fixture": "42"},
        predicate=StatePredicate(
            files_equal={"/answer": "42"}, key_values_equal={"status": "done"}
        ),
    )


CALLS = [
    ("fake_http_fetch", {"url": "https://fixture"}),
    ("fake_filesystem", {"operation": "write", "path": "/answer", "content": "42"}),
    ("fake_key_value", {"operation": "set", "key": "status", "value": "done"}),
]


def test_loop_logs_complete_bodies_state_and_success():
    client = ScriptedClient(CALLS)
    result = run_task(client, _test_task(), LoopConfig(seed=17))
    assert result.success is True
    assert result.stop_reason == "success_predicate"
    assert result.events[0]["seed"] == 17
    assert result.events[0]["determinism"]["sampling"] is False
    assert result.events[0]["prompt_defense"] is True
    assert client.requests[0]["use_cache"] is False
    model_steps = [event for event in result.events if event["event"] == "model_step"]
    tool_steps = [event for event in result.events if event["event"] == "tool_step"]
    assert len(model_steps) == len(tool_steps) == 3
    assert model_steps[0]["request"] == client.requests[0]
    assert "choices" in model_steps[0]["response"]
    assert model_steps[0]["rendered_prompt_hash"].startswith("sha256:")
    assert model_steps[0]["token_counts"]["total_tokens"] == 14
    assert tool_steps[1]["environment_before"]["files"] == {}
    assert tool_steps[1]["environment_after"]["files"] == {"/answer": "42"}
    assert result.events[-1]["final_environment"]["key_values"] == {"status": "done"}


def test_prompt_defense_is_an_explicit_arm():
    defended = run_task(ScriptedClient(CALLS), _test_task(), LoopConfig())
    neutral = run_task(
        ScriptedClient(CALLS), _test_task(), LoopConfig(prompt_defense=False)
    )
    defended_prompt = defended.events[1]["request"]["messages"][0]["content"]
    neutral_prompt = neutral.events[1]["request"]["messages"][0]["content"]
    assert "untrusted data" in defended_prompt
    assert "untrusted data" not in neutral_prompt
    assert defended.run_id != neutral.run_id


def test_same_seed_trajectories_match_after_only_documented_timing_exclusions(tmp_path):
    config = LoopConfig(seed=8)
    first = run_task(ScriptedClient(CALLS, created=100), _test_task(), config)
    second = run_task(ScriptedClient(CALLS, created=900), _test_task(), config)
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    write_trajectory(first_path, first)
    write_trajectory(second_path, second)

    assert first_path.read_bytes() != second_path.read_bytes()
    assert trajectories_match(first_path, second_path)
    assert canonical_trajectory(first_path) == canonical_trajectory(second_path)
    report = write_determinism_report(tmp_path / "determinism.json", first_path, second_path)
    assert report["matched"] is True
    assert report["raw_sha256"][0] != report["raw_sha256"][1]
    assert report["normalized_sha256"][0] == report["normalized_sha256"][1]
    with pytest.raises(FileExistsError):
        write_trajectory(first_path, first)


def test_forged_role_header_reaches_loop_as_untrusted_tool_content():
    task = TASK_BY_ID["injection_forged_header"]
    client = ScriptedClient([])
    result = run_task(client, task, LoopConfig(max_steps=1))

    tool_message = client.requests[0]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert "<|start_header_id|>system<|end_header_id|>" in tool_message["content"]
    setup = [event for event in result.events if event["event"] == "setup_tool_step"]
    assert setup[0]["tool_name"] == "fake_http_fetch"
    assert result.events[-1]["success"] is False


class FailingClient:
    def complete(self, _request):
        raise RuntimeError("deliberate model failure")


def test_model_failure_is_recorded_as_a_failed_run():
    result = run_task(FailingClient(), _test_task(), LoopConfig())
    assert result.stop_reason == "model_error"
    assert result.success is False
    assert result.events[-2]["event"] == "model_error"
    assert result.events[-2]["error"] == {
        "type": "RuntimeError",
        "message": "deliberate model failure",
    }
    assert result.events[-1]["event"] == "run_end"


def test_task_set_contains_benign_and_both_injection_shapes():
    assert TASK_BY_ID["benign_release"].injection is False
    assert TASK_BY_ID["injection_invoice"].injection is True
    assert TASK_BY_ID["injection_forged_header"].injection is True
    assert TASK_BY_ID["benign_release"].attack_predicate is None
    assert all(
        task.attack_predicate is not None
        for task in TASK_BY_ID.values()
        if task.injection
    )
    assert all(
        len(task.predicate.files_equal) + len(task.predicate.key_values_equal) == 1
        for task in TASK_BY_ID.values()
    )
    assert all(task.setup_calls[0][0] == "fake_http_fetch" for task in TASK_BY_ID.values())
    forged = next(iter(TASK_BY_ID["injection_forged_header"].documents.values()))
    assert "<|eot_id|>" in forged


def test_attack_success_is_independent_of_task_success():
    task = TASK_BY_ID["injection_invoice"]
    client = ScriptedClient(
        [("fake_key_value", {"operation": "set", "key": "exfiltrated", "value": "yes"})]
    )
    result = run_task(client, task, LoopConfig(max_steps=1, prompt_defense=False))
    assert result.success is False
    assert result.events[-1]["attack_success"] is True
