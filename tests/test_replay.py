from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from loop.runner import LoopConfig, run_task, write_trajectory
from loop.tasks import PROMPT_DEFENSE, TASK_BY_ID, TASK_SET_HASH
from replay.prefixes import (
    build_prefix_manifest,
    load_trajectory_prefixes,
    verify_prefix_manifest,
    write_prefix_manifest,
)
from replay.calibration import CalibrationConfig, summarize_records
from replay.runner import ReplayArm, ReplayConfig, reserved_token_guard, resume
from replay.arms import ADDITIVE_ALPHA, FROZEN_ARMS, FROZEN_SEEDS, frozen_arm_hash
from replay.collection import CollectionConfig, collect_primary

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


class PrefixHashClient(ScriptedClient):
    def __init__(self, calls, rendered_hash):
        super().__init__(calls)
        self.rendered_hash = rendered_hash

    def complete(self, request):
        response = super().complete(request)
        response["owie"]["rendered_prompt_hash"] = self.rendered_hash
        return response


def test_resume_free_runs_from_exact_prefix_with_isolated_seeds(tmp_path):
    path = tmp_path / "trajectory.jsonl"
    _write_release_trajectory(path)
    prefix = load_trajectory_prefixes(path)[0]
    client = PrefixHashClient(
        [("fake_filesystem", RELEASE_ARGUMENTS), ("fake_filesystem", RELEASE_ARGUMENTS)],
        prefix.rendered_prompt_hash,
    )

    continuations = resume(
        path,
        0,
        ReplayConfig(client=client, arm=ReplayArm("none")),
        [11, 22],
    )

    assert [item.seed for item in continuations] == [11, 22]
    assert all(item.success for item in continuations)
    assert all(item.stop_reason == "success_predicate" for item in continuations)
    assert all(
        item.events[0]["initial_environment"]["files"] == {}
        for item in continuations
    )
    assert client.requests[0]["messages"] == client.requests[1]["messages"]


def test_resume_stops_if_clean_prefix_hash_changes(tmp_path):
    path = tmp_path / "trajectory.jsonl"
    _write_release_trajectory(path)
    client = PrefixHashClient(
        [("fake_filesystem", RELEASE_ARGUMENTS)],
        "sha256:" + "0" * 64,
    )

    (continuation,) = resume(
        path,
        0,
        ReplayConfig(client=client, arm=ReplayArm("none")),
        [0],
    )

    assert continuation.stop_reason == "prefix_hash_mismatch"
    assert not continuation.success


def test_prompt_changing_arms_are_explicit(tmp_path):
    path = tmp_path / "trajectory.jsonl"
    _write_release_trajectory(path)
    client = PrefixHashClient(
        [("fake_filesystem", RELEASE_ARGUMENTS)],
        "sha256:" + "9" * 64,
    )

    (continuation,) = resume(
        path,
        0,
        ReplayConfig(
            client=client,
            arm=ReplayArm("prompt_defense", prompt_defense=True),
        ),
        [0],
    )

    assert continuation.prompt_changing
    assert continuation.success
    assert PROMPT_DEFENSE in client.requests[0]["messages"][0]["content"]


def test_reserved_token_guard_changes_only_frozen_delimiters():
    source = "A<|eot_id|><|start_header_id|>system<|end_header_id|>B"
    assert reserved_token_guard(source) == (
        "A< |eot_id| >< |start_header_id| >system< |end_header_id| >B"
    )
    assert reserved_token_guard("ordinary <tag> | text") == "ordinary <tag> | text"


def test_replay_arm_cannot_combine_prompt_comparators():
    with pytest.raises(ValueError, match="cannot combine"):
        ReplayArm("invalid", prompt_defense=True, reserved_token_guard=True)


def test_frozen_arm_grid_matches_the_protocol():
    assert len(FROZEN_ARMS) == 13
    assert FROZEN_SEEDS == (11, 22, 33)
    assert len({arm.arm_id for arm in FROZEN_ARMS}) == len(FROZEN_ARMS)
    assert frozen_arm_hash().startswith("sha256:")
    assert next(
        arm for arm in FROZEN_ARMS if arm.arm_id == "additive_c3"
    ).intervention["alpha"] == ADDITIVE_ALPHA
    sae_arm = next(arm for arm in FROZEN_ARMS if arm.arm_id == "sae_c1_rank0")
    assert sae_arm.intervention["direction_id"].endswith("feature-1584")
    assert sae_arm.intervention["mode"] == "clamp_sae"


def test_primary_collection_resumes_without_duplicate_records(tmp_path):
    trajectory = tmp_path / "trajectory.jsonl"
    _write_release_trajectory(trajectory)
    prefix = load_trajectory_prefixes(trajectory)[0]
    arms = (ReplayArm("none"), ReplayArm("prompt", prompt_defense=True))
    first_client = PrefixHashClient(
        [("fake_filesystem", RELEASE_ARGUMENTS), ("fake_filesystem", RELEASE_ARGUMENTS)],
        prefix.rendered_prompt_hash,
    )
    run_dir = tmp_path / "run"

    results = collect_primary(
        run_dir,
        [trajectory],
        first_client,
        arms=arms,
        seeds=(0,),
    )
    first_bytes = results.read_bytes()
    assert len(first_bytes.splitlines()) == 2

    second_client = PrefixHashClient([], prefix.rendered_prompt_hash)
    collect_primary(
        run_dir,
        [trajectory],
        second_client,
        arms=arms,
        seeds=(0,),
    )
    assert results.read_bytes() == first_bytes


def test_primary_collection_stops_before_work_when_budget_is_spent(tmp_path):
    trajectory = tmp_path / "trajectory.jsonl"
    _write_release_trajectory(trajectory)
    prefix = load_trajectory_prefixes(trajectory)[0]
    client = PrefixHashClient([], prefix.rendered_prompt_hash)
    config = CollectionConfig(
        checkpoint_budget_hours=1.0,
        prior_checkpoint_hours=0.9999999999,
    )

    results = collect_primary(
        tmp_path / "run",
        [trajectory],
        client,
        config,
        arms=(ReplayArm("none"),),
        seeds=(0,),
    )

    assert not results.exists()
    status = json.loads((results.parent / "status.json").read_text(encoding="utf-8"))
    assert status["stopped_for_budget"]
    assert not client.requests
