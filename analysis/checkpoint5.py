"""Paired Checkpoint 5 summaries and task-clustered bootstrap intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from loop.tasks import TASK_BY_ID


BOOTSTRAP_RESAMPLES = 2_000
BOOTSTRAP_SEED = 20260829


def read_collection(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            rows.append(row)
    return rows


def _events(row: Mapping[str, Any], kind: str) -> list[Mapping[str, Any]]:
    return [event for event in row.get("events", []) if event.get("event") == kind]


def summarize_arm(rows: Iterable[Mapping[str, Any]], arm_id: str) -> dict[str, Any]:
    selected = [row for row in rows if row.get("arm_id") == arm_id]
    if not selected:
        raise ValueError(f"no records for arm {arm_id!r}")
    injection = [row for row in selected if TASK_BY_ID[row["task_id"]].injection]
    benign = [row for row in selected if not TASK_BY_ID[row["task_id"]].injection]
    tool_steps = [event for row in selected for event in _events(row, "tool_step")]
    call_errors = [event for row in selected for event in _events(row, "tool_call_error")]
    model_steps = [event for row in selected for event in _events(row, "model_step")]
    call_attempts = len(tool_steps) + len(call_errors)
    valid_arguments = sum(event.get("tool_error") is None for event in tool_steps)

    def rate(items: list[Mapping[str, Any]], key: str) -> float:
        return sum(bool(item[key]) for item in items) / len(items) if items else 0.0

    return {
        "arm_id": arm_id,
        "continuations": len(selected),
        "task_success": rate(selected, "success"),
        "benign_task_success": rate(benign, "success"),
        "injection_task_success": rate(injection, "success"),
        "attack_success": rate(injection, "attack_success"),
        "attack_success_count": sum(bool(row["attack_success"]) for row in injection),
        "attack_denominator": len(injection),
        "tool_call_envelope_validity": len(tool_steps) / call_attempts if call_attempts else 1.0,
        "tool_name_validity": (
            sum(bool(event["tool_name_valid"]) for event in tool_steps) / len(tool_steps)
            if tool_steps
            else 1.0
        ),
        "argument_schema_validity": (
            valid_arguments / len(tool_steps) if tool_steps else 1.0
        ),
        "model_steps": len(model_steps),
        "tool_calls": len(tool_steps),
        "prompt_tokens": sum(
            int(event.get("token_counts", {}).get("prompt_tokens", 0))
            for event in model_steps
        ),
        "completion_tokens": sum(
            int(event.get("token_counts", {}).get("completion_tokens", 0))
            for event in model_steps
        ),
        "latency_ms": sum(float(event.get("latency_ms", 0.0)) for event in model_steps),
        "stop_reasons": {
            reason: sum(row.get("stop_reason") == reason for row in selected)
            for reason in sorted({str(row.get("stop_reason")) for row in selected})
        },
    }


def paired_task_bootstrap(
    rows: Iterable[Mapping[str, Any]],
    arm_id: str,
    baseline_id: str,
    outcome: str,
    *,
    injection_only: bool,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Bootstrap tasks after averaging generation seeds within each task."""

    values: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        current_arm = row.get("arm_id")
        task_id = row.get("task_id")
        if current_arm not in {arm_id, baseline_id} or task_id not in TASK_BY_ID:
            continue
        if injection_only and not TASK_BY_ID[task_id].injection:
            continue
        values.setdefault(task_id, {}).setdefault(current_arm, []).append(
            float(bool(row[outcome]))
        )
    complete = sorted(
        task_id
        for task_id, arms in values.items()
        if arm_id in arms and baseline_id in arms
    )
    if not complete:
        raise ValueError("paired bootstrap has no complete task pairs")
    differences = np.asarray(
        [
            np.mean(values[task_id][arm_id])
            - np.mean(values[task_id][baseline_id])
            for task_id in complete
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(resamples, len(differences)))
    samples = differences[indices].mean(axis=1)
    return {
        "arm_id": arm_id,
        "baseline_id": baseline_id,
        "outcome": outcome,
        "estimate": float(differences.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "task_units": len(complete),
        "seed_pairs": sum(
            min(len(values[task_id][arm_id]), len(values[task_id][baseline_id]))
            for task_id in complete
        ),
        "resamples": resamples,
        "bootstrap_seed": seed,
        "tasks": complete,
    }


def analyze_collection(rows: list[dict[str, Any]]) -> dict[str, Any]:
    arms = sorted({str(row["arm_id"]) for row in rows})
    if "none" not in arms:
        raise ValueError("collection has no no-intervention arm")
    return {
        "arms": [summarize_arm(rows, arm) for arm in arms],
        "attack_differences": [
            paired_task_bootstrap(
                rows,
                arm,
                "none",
                "attack_success",
                injection_only=True,
            )
            for arm in arms
            if arm != "none"
        ],
        "task_success_differences": [
            paired_task_bootstrap(
                rows,
                arm,
                "none",
                "success",
                injection_only=False,
            )
            for arm in arms
            if arm != "none"
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    analysis = analyze_collection(read_collection(args.results))
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(analysis, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
