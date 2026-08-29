"""Paired Checkpoint 5 task, retain, and safety analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from loop.tasks import TASK_BY_ID
from replay.calibration import summarize_records


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


def _paired_metric_rows(
    rows: Iterable[Mapping[str, Any]],
    arm_id: str,
    baseline_id: str,
    metric: str,
    *,
    predicate: Any,
) -> tuple[list[str], dict[str, dict[str, Mapping[str, Any]]]]:
    values: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        current_arm = row.get("arm_id")
        item_id = row.get("item_id")
        if (
            current_arm not in {arm_id, baseline_id}
            or row.get("metric") != metric
            or not isinstance(item_id, str)
            or not predicate(row)
        ):
            continue
        arm_rows = values.setdefault(item_id, {})
        if current_arm in arm_rows:
            raise ValueError(
                f"duplicate {metric} record for item {item_id!r} and arm {current_arm!r}"
            )
        arm_rows[current_arm] = row
    complete = sorted(
        item_id
        for item_id, arm_rows in values.items()
        if arm_id in arm_rows and baseline_id in arm_rows
    )
    if not complete:
        raise ValueError(f"paired {metric} bootstrap has no complete item pairs")
    return complete, values


def paired_retain_bootstrap(
    rows: Iterable[Mapping[str, Any]],
    arm_id: str,
    baseline_id: str = "none",
    *,
    tool_dependent: bool | None,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Resample paired retain items and recompute token-weighted perplexity."""

    complete, values = _paired_metric_rows(
        rows,
        arm_id,
        baseline_id,
        "retain",
        predicate=lambda row: (
            tool_dependent is None
            or bool(row.get("tool_dependent")) is tool_dependent
        ),
    )
    arm_logprob = np.asarray(
        [float(values[item_id][arm_id]["logprob"]) for item_id in complete],
        dtype=np.float64,
    )
    arm_tokens = np.asarray(
        [int(values[item_id][arm_id]["tokens"]) for item_id in complete],
        dtype=np.float64,
    )
    baseline_logprob = np.asarray(
        [float(values[item_id][baseline_id]["logprob"]) for item_id in complete],
        dtype=np.float64,
    )
    baseline_tokens = np.asarray(
        [int(values[item_id][baseline_id]["tokens"]) for item_id in complete],
        dtype=np.float64,
    )
    if np.any(arm_tokens <= 0) or np.any(baseline_tokens <= 0):
        raise ValueError("retain bootstrap requires positive token counts")

    def perplexity(logprob: np.ndarray, tokens: np.ndarray) -> float:
        return float(np.exp(-logprob.sum() / tokens.sum()))

    arm_perplexity = perplexity(arm_logprob, arm_tokens)
    baseline_perplexity = perplexity(baseline_logprob, baseline_tokens)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(complete), size=(resamples, len(complete)))
    arm_samples = np.exp(
        -arm_logprob[indices].sum(axis=1) / arm_tokens[indices].sum(axis=1)
    )
    baseline_samples = np.exp(
        -baseline_logprob[indices].sum(axis=1)
        / baseline_tokens[indices].sum(axis=1)
    )
    ratio_samples = arm_samples / baseline_samples
    difference_samples = arm_samples - baseline_samples
    return {
        "arm_id": arm_id,
        "baseline_id": baseline_id,
        "metric": "retain_perplexity",
        "group": (
            "all"
            if tool_dependent is None
            else "tool_dependent" if tool_dependent else "tool_independent"
        ),
        "arm_perplexity": arm_perplexity,
        "baseline_perplexity": baseline_perplexity,
        "ratio": arm_perplexity / baseline_perplexity,
        "ratio_ci_low": float(np.quantile(ratio_samples, 0.025)),
        "ratio_ci_high": float(np.quantile(ratio_samples, 0.975)),
        "difference": arm_perplexity - baseline_perplexity,
        "difference_ci_low": float(np.quantile(difference_samples, 0.025)),
        "difference_ci_high": float(np.quantile(difference_samples, 0.975)),
        "item_pairs": len(complete),
        "resamples": resamples,
        "bootstrap_seed": seed,
        "items": complete,
    }


def paired_safety_bootstrap(
    rows: Iterable[Mapping[str, Any]],
    arm_id: str,
    baseline_id: str = "none",
    *,
    channel: str,
    outcome: str,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Resample paired safety items within one arrival channel."""

    if channel not in {"user", "tool"}:
        raise ValueError("safety channel must be 'user' or 'tool'")
    if outcome not in {"refused", "emitted_tool_call", "complied", "complied_in_text"}:
        raise ValueError(f"unsupported safety outcome {outcome!r}")
    complete, values = _paired_metric_rows(
        rows,
        arm_id,
        baseline_id,
        "safety",
        predicate=lambda row: row.get("channel") == channel,
    )
    arm_values = np.asarray(
        [float(bool(values[item_id][arm_id][outcome])) for item_id in complete],
        dtype=np.float64,
    )
    baseline_values = np.asarray(
        [float(bool(values[item_id][baseline_id][outcome])) for item_id in complete],
        dtype=np.float64,
    )
    differences = arm_values - baseline_values
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(complete), size=(resamples, len(complete)))
    samples = differences[indices].mean(axis=1)
    return {
        "arm_id": arm_id,
        "baseline_id": baseline_id,
        "metric": "safety",
        "channel": channel,
        "outcome": outcome,
        "arm_count": int(arm_values.sum()),
        "baseline_count": int(baseline_values.sum()),
        "denominator": len(complete),
        "arm_rate": float(arm_values.mean()),
        "baseline_rate": float(baseline_values.mean()),
        "difference": float(differences.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "item_pairs": len(complete),
        "resamples": resamples,
        "bootstrap_seed": seed,
        "items": complete,
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


def analyze_arm_evaluation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize arm metrics and pair every intervention with ``none``."""

    arms = sorted({str(row["arm_id"]) for row in rows})
    if "none" not in arms:
        raise ValueError("arm evaluation has no no-intervention arm")
    summaries = {
        arm: summarize_records(row for row in rows if row.get("arm_id") == arm)
        for arm in arms
    }
    comparisons = []
    for arm in arms:
        if arm == "none":
            continue
        comparisons.append(
            {
                "arm_id": arm,
                "retain": [
                    paired_retain_bootstrap(
                        rows,
                        arm,
                        tool_dependent=group,
                    )
                    for group in (None, True, False)
                ],
                "safety": [
                    paired_safety_bootstrap(
                        rows,
                        arm,
                        channel=channel,
                        outcome=outcome,
                    )
                    for channel in ("user", "tool")
                    for outcome in (
                        "refused",
                        "emitted_tool_call",
                        "complied",
                        "complied_in_text",
                    )
                ],
            }
        )
    return {"arms": summaries, "paired_comparisons": comparisons}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--arm-evaluation-results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    analysis = analyze_collection(read_collection(args.results))
    if args.arm_evaluation_results is not None:
        analysis["arm_evaluation"] = analyze_arm_evaluation(
            read_collection(args.arm_evaluation_results)
        )
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(analysis, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
