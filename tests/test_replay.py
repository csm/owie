from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from loop.runner import LoopConfig, run_task, write_trajectory
from loop.tasks import TASK_BY_ID, TASK_SET_HASH
from replay.prefixes import (
    build_prefix_manifest,
    load_trajectory_prefixes,
    verify_prefix_manifest,
    write_prefix_manifest,
)
from replay.calibration import CalibrationConfig, summarize_records

from test_loop import ScriptedClient


RELEASE_ARGUMENTS = {
    "operation": "write",
    "path": "/release-date.txt",
    "content": "2026-09-03",
}


def _write_release_trajectory(path: Path) -> None:
    result = run_task(
        ScriptedClient([("fake_filesystem", RELEASE_ARGUMENTS)]),
        TASK_BY_ID["benign_release"],
        LoopConfig(
            model_revision="test-revision",
            prompt_defense=False,
            intervention={"enabled": False},
        ),
    )
    write_trajectory(path, result)


def test_load_trajectory_prefix_captures_exact_request_and_state(tmp_path):
    path = tmp_path / "trajectory.jsonl"
    _write_release_trajectory(path)

    (prefix,) = load_trajectory_prefixes(path)

    assert prefix.task_id == "benign_release"
    assert prefix.task_set_hash == TASK_SET_HASH
    assert prefix.step == 0
    assert prefix.request["messages"][-1]["role"] == "tool"
    call = prefix.baseline_message["tool_calls"][0]
    assert call["function"]["name"] == "fake_filesystem"
    assert json.loads(call["function"]["arguments"]) == RELEASE_ARGUMENTS
    assert prefix.environment["files"] == {}
    assert prefix.request_hash.startswith("sha256:")
    assert prefix.source_sha256.startswith("sha256:")


def test_prefix_identity_changes_when_request_changes(tmp_path):
    path = tmp_path / "trajectory.jsonl"
    _write_release_trajectory(path)
    original = load_trajectory_prefixes(path)[0]

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    model_step = next(row for row in rows if row["event"] == "model_step")
    model_step["request"]["max_tokens"] += 1
    changed_path = tmp_path / "changed.jsonl"
    changed_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    changed = load_trajectory_prefixes(changed_path)[0]
    assert changed.request_hash != original.request_hash
    assert changed.prefix_id != original.prefix_id


def test_neutral_prefix_rejects_prompt_defense(tmp_path):
    path = tmp_path / "trajectory.jsonl"
    result = run_task(
        ScriptedClient([("fake_filesystem", RELEASE_ARGUMENTS)]),
        TASK_BY_ID["benign_release"],
        LoopConfig(model_revision="test-revision", prompt_defense=True),
    )
    write_trajectory(path, result)

    with pytest.raises(ValueError, match="prompt_defense=false"):
        load_trajectory_prefixes(path)


def test_complete_manifest_requires_every_candidate_task(tmp_path):
    path = tmp_path / "trajectory.jsonl"
    _write_release_trajectory(path)

    with pytest.raises(ValueError, match="must cover every candidate task"):
        build_prefix_manifest([path])


def test_manifest_writer_never_overwrites(tmp_path):
    path = tmp_path / "manifest.json"
    write_prefix_manifest(path, {"schema_version": 1})

    with pytest.raises(FileExistsError):
        write_prefix_manifest(path, {"schema_version": 1})


def test_prefix_manifest_verification_rejects_changed_content():
    manifest = {
        "task_set_hash": TASK_SET_HASH,
        "schema_version": 1,
    }
    import hashlib

    wire = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    manifest["manifest_hash"] = "sha256:" + hashlib.sha256(wire.encode()).hexdigest()
    verify_prefix_manifest(manifest)

    manifest["schema_version"] = 2
    with pytest.raises(ValueError, match="hash"):
        verify_prefix_manifest(manifest)


def test_calibration_config_rejects_spent_budget():
    with pytest.raises(ValueError, match="within the checkpoint budget"):
        CalibrationConfig(checkpoint_budget_hours=1.0, prior_checkpoint_hours=1.0)


def test_calibration_summary_keeps_retain_and_safety_channels_separate():
    rows = [
        {
            "metric": "retain",
            "tokens": 2,
            "logprob": -2.0,
            "tool_dependent": True,
        },
        {
            "metric": "retain",
            "tokens": 2,
            "logprob": -4.0,
            "tool_dependent": False,
        },
        {
            "metric": "safety",
            "channel": "user",
            "complied": False,
            "complied_in_text": False,
            "emitted_tool_call": False,
        },
        {
            "metric": "safety",
            "channel": "tool",
            "complied": True,
            "complied_in_text": False,
            "emitted_tool_call": True,
        },
    ]

    summary = summarize_records(rows)

    assert summary["retain"]["perplexity"] == pytest.approx(math.exp(1.5))
    assert summary["retain"]["tool_dependent_perplexity"] == pytest.approx(math.e)
    assert summary["safety"]["user_compliance_rate"] == 0.0
    assert summary["safety"]["tool_compliance_rate"] == 1.0
