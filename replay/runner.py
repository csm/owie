"""Resume a loop trajectory from an immutable model-step prefix."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from loop.runner import ChatClient, seed_everything
from loop.tasks import BASE_SYSTEM_PROMPT, PROMPT_DEFENSE, TASK_BY_ID
from loop.tools import FakeEnvironment, TOOL_NAMES, ToolError
from replay.prefixes import ReplayPrefix, load_trajectory_prefixes


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def reserved_token_guard(content: str) -> str:
    """Break Llama reserved-token delimiters and change no other text."""

    return content.replace("<|", "< |").replace("|>", "| >")


@dataclass(frozen=True)
class ReplayArm:
    arm_id: str
    intervention: dict[str, Any] = field(default_factory=lambda: {"enabled": False})
    prompt_defense: bool = False
    reserved_token_guard: bool = False

    def __post_init__(self) -> None:
        if not self.arm_id:
            raise ValueError("arm_id must not be empty")
        if self.prompt_defense and self.reserved_token_guard:
            raise ValueError("one replay arm cannot combine prompt defense and the guard")

    @property
    def prompt_changing(self) -> bool:
        return self.prompt_defense or self.reserved_token_guard


@dataclass(frozen=True)
class ReplayConfig:
    client: ChatClient
    arm: ReplayArm
    max_steps: int = 8

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")


@dataclass(frozen=True)
class Continuation:
    continuation_id: str
    prefix_id: str
    task_id: str
    step_k: int
    arm_id: str
    seed: int
    prompt_changing: bool
    success: bool
    attack_success: bool
    stop_reason: str
    events: tuple[dict[str, Any], ...]


def _prefix_at(path: Path, step_k: int) -> ReplayPrefix:
    matches = [
        prefix
        for prefix in load_trajectory_prefixes(path)
        if prefix.step == step_k
    ]
    if len(matches) != 1:
        raise ValueError(f"{path}: expected one replay prefix at step {step_k}")
    return matches[0]


def _environment(snapshot: Mapping[str, Any]) -> FakeEnvironment:
    return FakeEnvironment(
        files=copy.deepcopy(dict(snapshot.get("files", {}))),
        key_values=copy.deepcopy(dict(snapshot.get("key_values", {}))),
        documents=copy.deepcopy(dict(snapshot.get("documents", {}))),
    )


def _guard_tool_messages(messages: list[dict[str, Any]]) -> None:
    for message in messages:
        if message.get("role") == "tool" and isinstance(message.get("content"), str):
            message["content"] = reserved_token_guard(message["content"])


def _messages(prefix: ReplayPrefix, arm: ReplayArm) -> list[dict[str, Any]]:
    messages = copy.deepcopy(prefix.request["messages"])
    if arm.prompt_defense:
        if messages[0] != {"role": "system", "content": BASE_SYSTEM_PROMPT}:
            raise ValueError("prompt-defense replay requires the frozen neutral system prompt")
        messages[0]["content"] = f"{BASE_SYSTEM_PROMPT}\n{PROMPT_DEFENSE}"
    if arm.reserved_token_guard:
        _guard_tool_messages(messages)
    return messages


def _parse_call(response: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("chat response has no choices[0].message") from exc
    calls = message.get("tool_calls") or []
    if not calls:
        return None
    if len(calls) != 1:
        raise ValueError("replay accepts exactly one tool call per response")
    call = calls[0]
    try:
        function = call["function"]
        arguments = json.loads(function["arguments"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("tool call has invalid function arguments") from exc
    if not isinstance(arguments, dict):
        raise ValueError("tool call arguments must decode to an object")
    return copy.deepcopy(call), arguments


def continuation_id(
    prefix: ReplayPrefix, arm: ReplayArm, seed: int, max_steps: int
) -> str:
    identity = {
        "prefix_id": prefix.prefix_id,
        "arm_id": arm.arm_id,
        "intervention": arm.intervention,
        "prompt_defense": arm.prompt_defense,
        "reserved_token_guard": arm.reserved_token_guard,
        "seed": seed,
        "max_steps": max_steps,
    }
    return "continuation_" + hashlib.sha256(_canonical_json(identity).encode()).hexdigest()[:20]


def _run_one(prefix: ReplayPrefix, config: ReplayConfig, seed: int) -> Continuation:
    task = TASK_BY_ID[prefix.task_id]
    environment = _environment(prefix.environment)
    messages = _messages(prefix, config.arm)
    seeds = seed_everything(seed)
    events: list[dict[str, Any]] = [
        {
            "event": "continuation_start",
            "prefix_id": prefix.prefix_id,
            "request_hash": prefix.request_hash,
            "source_sha256": prefix.source_sha256,
            "task_hash": prefix.task_hash,
            "task_set_hash": prefix.task_set_hash,
            "arm_id": config.arm.arm_id,
            "prompt_changing": config.arm.prompt_changing,
            "intervention": copy.deepcopy(config.arm.intervention),
            "seed": seed,
            "determinism": seeds,
            "initial_environment": environment.snapshot(),
        }
    ]
    success = False
    stop_reason = "max_steps"
    for replay_step in range(config.max_steps):
        request = copy.deepcopy(prefix.request)
        request["messages"] = copy.deepcopy(messages)
        request["seed"] = seed
        request["intervention"] = copy.deepcopy(config.arm.intervention)
        request["use_cache"] = False
        started = time.perf_counter_ns()
        try:
            response = config.client.complete(request)
        except Exception as exc:
            events.append(
                {
                    "event": "model_error",
                    "replay_step": replay_step,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "latency_ms": (time.perf_counter_ns() - started) / 1_000_000,
                    "environment": environment.snapshot(),
                }
            )
            stop_reason = "model_error"
            break
        latency_ms = (time.perf_counter_ns() - started) / 1_000_000
        rendered_hash = response.get("owie", {}).get("rendered_prompt_hash")
        if replay_step == 0 and not config.arm.prompt_changing:
            if rendered_hash != prefix.rendered_prompt_hash:
                events.append(
                    {
                        "event": "prefix_error",
                        "replay_step": replay_step,
                        "expected_rendered_prompt_hash": prefix.rendered_prompt_hash,
                        "actual_rendered_prompt_hash": rendered_hash,
                    }
                )
                stop_reason = "prefix_hash_mismatch"
                break
        message = copy.deepcopy(response["choices"][0]["message"])
        events.append(
            {
                "event": "model_step",
                "replay_step": replay_step,
                "request": request,
                "response": copy.deepcopy(response),
                "rendered_prompt_hash": rendered_hash,
                "token_counts": copy.deepcopy(response.get("usage", {})),
                "latency_ms": latency_ms,
            }
        )
        messages.append(message)
        try:
            parsed = _parse_call(response)
        except ValueError as exc:
            events.append(
                {
                    "event": "tool_call_error",
                    "replay_step": replay_step,
                    "error": str(exc),
                    "environment": environment.snapshot(),
                }
            )
            stop_reason = "invalid_tool_call"
            break
        if parsed is None:
            stop_reason = "assistant_stop"
            break
        call, arguments = parsed
        name = call["function"].get("name")
        name_valid = name in TOOL_NAMES
        try:
            execution = environment.execute(name, arguments)
            output = execution.output
            tool_error = None
            before = execution.state_before
            after = execution.state_after
        except ToolError as exc:
            output = {"ok": False, "error": str(exc)}
            tool_error = str(exc)
            before = after = environment.snapshot()
        events.append(
            {
                "event": "tool_step",
                "replay_step": replay_step,
                "tool_call_id": call.get("id"),
                "tool_name": name,
                "tool_name_valid": name_valid,
                "tool_input": arguments,
                "tool_output": output,
                "tool_error": tool_error,
                "environment_before": before,
                "environment_after": after,
            }
        )
        tool_content = _canonical_json(output)
        if config.arm.reserved_token_guard:
            tool_content = reserved_token_guard(tool_content)
        messages.append(
            {
                "role": "tool",
                "name": name,
                "tool_call_id": call.get("id"),
                "content": tool_content,
            }
        )
        check = task.predicate.evaluate(environment)
        attack_check = (
            task.attack_predicate.evaluate(environment)
            if task.attack_predicate is not None
            else None
        )
        events.append(
            {
                "event": "success_check",
                "replay_step": replay_step,
                **check,
                "attack_check": attack_check,
            }
        )
        if check["passed"]:
            success = True
            stop_reason = "success_predicate"
            break

    final_check = task.predicate.evaluate(environment)
    final_attack_check = (
        task.attack_predicate.evaluate(environment)
        if task.attack_predicate is not None
        else None
    )
    success = success or final_check["passed"]
    attack_success = bool(final_attack_check and final_attack_check["passed"])
    events.append(
        {
            "event": "continuation_end",
            "success": success,
            "attack_success": attack_success,
            "stop_reason": stop_reason,
            "success_check": final_check,
            "attack_check": final_attack_check,
            "final_environment": environment.snapshot(),
        }
    )
    return Continuation(
        continuation_id=continuation_id(prefix, config.arm, seed, config.max_steps),
        prefix_id=prefix.prefix_id,
        task_id=prefix.task_id,
        step_k=prefix.step,
        arm_id=config.arm.arm_id,
        seed=seed,
        prompt_changing=config.arm.prompt_changing,
        success=success,
        attack_success=attack_success,
        stop_reason=stop_reason,
        events=tuple(events),
    )


def resume(
    trajectory: Path,
    step_k: int,
    config: ReplayConfig,
    seeds: Iterable[int],
) -> list[Continuation]:
    """Resume one stored prefix once for each generation seed."""

    prefix = _prefix_at(Path(trajectory), step_k)
    resolved_seeds = tuple(seeds)
    if not resolved_seeds or any(not isinstance(seed, int) for seed in resolved_seeds):
        raise ValueError("seeds must contain at least one integer")
    if len(set(resolved_seeds)) != len(resolved_seeds):
        raise ValueError("seeds must be unique")
    return [_run_one(prefix, config, seed) for seed in resolved_seeds]
