from __future__ import annotations

import pytest

from analysis.checkpoint5 import paired_task_bootstrap, summarize_arm


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
