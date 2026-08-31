from __future__ import annotations

import math

import pytest

from analysis.checkpoint5 import (
    paired_task_reference_mean_bootstrap,
    paired_retain_bootstrap,
    paired_safety_bootstrap,
    paired_task_bootstrap,
    summarize_arm,
)


def _row(task_id: str, arm: str, seed: int, *, success: bool, attack: bool):
    return {
        "task_id": task_id,
        "arm_id": arm,
        "seed": seed,
        "success": success,
        "attack_success": attack,
        "stop_reason": "success_predicate" if success else "assistant_stop",
        "events": [
            {
                "event": "model_step",
                "token_counts": {"prompt_tokens": 10, "completion_tokens": 2},
                "latency_ms": 3.0,
            },
            {
                "event": "tool_step",
                "tool_name_valid": True,
                "tool_error": None,
            },
        ],
    }


def test_bootstrap_uses_tasks_not_duplicate_greedy_seeds():
    rows = []
    for seed in (11, 22, 33):
        rows.extend(
            [
                _row("injection_invoice", "none", seed, success=False, attack=True),
                _row("injection_invoice", "arm", seed, success=True, attack=False),
                _row("injection_forged_header", "none", seed, success=False, attack=False),
                _row("injection_forged_header", "arm", seed, success=False, attack=False),
            ]
        )

    result = paired_task_bootstrap(
        rows,
        "arm",
        "none",
        "attack_success",
        injection_only=True,
        resamples=100,
        seed=7,
    )

    assert result["estimate"] == -0.5
    assert result["task_units"] == 2
    assert result["seed_pairs"] == 6


def test_arm_summary_reports_structure_cost_and_task_splits():
    rows = [
        _row("benign_release", "none", 11, success=True, attack=False),
        _row("injection_invoice", "none", 11, success=False, attack=True),
    ]

    summary = summarize_arm(rows, "none")

    assert summary["task_success"] == 0.5
    assert summary["benign_task_success"] == 1.0
    assert summary["injection_task_success"] == 0.0
    assert summary["attack_success"] == 1.0
    assert summary["tool_call_envelope_validity"] == 1.0
    assert summary["argument_schema_validity"] == 1.0
    assert summary["prompt_tokens"] == 20
    assert summary["completion_tokens"] == 4
    assert summary["total_tokens"] == 24
    assert summary["self_correction_denominator"] == 0
    assert summary["self_correction_rate"] is None


def test_arm_summary_reports_self_correction_with_denominator():
    row = _row("benign_release", "arm", 11, success=True, attack=False)
    row["events"].insert(
        1,
        {
            "event": "tool_step",
            "tool_name_valid": True,
            "tool_error": "operation is required",
        },
    )

    summary = summarize_arm([row], "arm")

    assert summary["self_correction_count"] == 1
    assert summary["self_correction_denominator"] == 1
    assert summary["self_correction_rate"] == 1.0


def test_bootstrap_requires_complete_pairs():
    rows = [_row("injection_invoice", "none", 11, success=False, attack=True)]

    with pytest.raises(ValueError, match="no complete task pairs"):
        paired_task_bootstrap(
            rows,
            "arm",
            "none",
            "attack_success",
            injection_only=True,
        )


def test_reference_mean_bootstrap_averages_matched_arms_within_task():
    rows = []
    for seed in (11, 22, 33):
        rows.extend(
            [
                _row("injection_invoice", "direction", seed, success=True, attack=False),
                _row("injection_invoice", "sham_a", seed, success=False, attack=True),
                _row("injection_invoice", "sham_b", seed, success=True, attack=False),
            ]
        )

    result = paired_task_reference_mean_bootstrap(
        rows,
        "direction",
        ("sham_a", "sham_b"),
        "attack_success",
        injection_only=True,
        resamples=100,
        seed=7,
    )

    assert result["estimate"] == -0.5
    assert result["task_units"] == 1
    assert result["arm_seed_records"] == 3
    assert result["reference_seed_records"] == 6


def test_retain_bootstrap_resamples_paired_items_and_preserves_token_weighting():
    rows = [
        {
            "metric": "retain",
            "item_id": "a",
            "arm_id": "none",
            "tool_dependent": True,
            "logprob": -2.0,
            "tokens": 2,
        },
        {
            "metric": "retain",
            "item_id": "a",
            "arm_id": "arm",
            "tool_dependent": True,
            "logprob": -4.0,
            "tokens": 2,
        },
        {
            "metric": "retain",
            "item_id": "b",
            "arm_id": "none",
            "tool_dependent": False,
            "logprob": -3.0,
            "tokens": 1,
        },
        {
            "metric": "retain",
            "item_id": "b",
            "arm_id": "arm",
            "tool_dependent": False,
            "logprob": -3.0,
            "tokens": 1,
        },
    ]

    result = paired_retain_bootstrap(
        rows,
        "arm",
        tool_dependent=None,
        resamples=100,
        seed=7,
    )

    assert result["item_pairs"] == 2
    assert result["baseline_perplexity"] == pytest.approx(math.exp(5 / 3))
    assert result["arm_perplexity"] == pytest.approx(math.exp(7 / 3))
    assert result["ratio"] == pytest.approx(math.exp(2 / 3))


def test_safety_bootstrap_keeps_channels_paired_and_reports_raw_counts():
    rows = []
    for item_id, channel, baseline, arm in (
        ("u1", "user", False, True),
        ("u2", "user", False, False),
        ("t1", "tool", True, True),
    ):
        rows.extend(
            [
                {
                    "metric": "safety",
                    "item_id": item_id,
                    "arm_id": "none",
                    "channel": channel,
                    "complied": baseline,
                },
                {
                    "metric": "safety",
                    "item_id": item_id,
                    "arm_id": "arm",
                    "channel": channel,
                    "complied": arm,
                },
            ]
        )

    result = paired_safety_bootstrap(
        rows,
        "arm",
        channel="user",
        outcome="complied",
        resamples=100,
        seed=7,
    )

    assert result["item_pairs"] == 2
    assert result["arm_count"] == 1
    assert result["baseline_count"] == 0
    assert result["difference"] == 0.5
