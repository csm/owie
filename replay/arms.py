"""Frozen Checkpoint 5 arms and generation seeds."""

from __future__ import annotations

import hashlib
import json

from replay.runner import ReplayArm


FROZEN_SEEDS = (11, 22, 33)
ADDITIVE_ALPHA = 3.3473639488220215


FROZEN_ARMS: tuple[ReplayArm, ...] = (
    ReplayArm("none"),
    ReplayArm(
        "projection_c1",
        intervention={
            "enabled": True,
            "direction_id": "c1-l10-dim",
            "layer": 10,
            "mode": "project",
            "scope": "tool_content",
            "direction_norm": "unit",
            "alpha": 0.0,
        },
    ),
    ReplayArm(
        "additive_c3",
        intervention={
            "enabled": True,
            "direction_id": "c3-l10-dim",
            "layer": 10,
            "mode": "add",
            "scope": "tool_content",
            "direction_norm": "unit",
            "alpha": ADDITIVE_ALPHA,
        },
    ),
    ReplayArm("prompt_defense", prompt_defense=True),
    ReplayArm("reserved_token_guard", reserved_token_guard=True),
    *tuple(
        ReplayArm(
            f"projection_sham_{seed}",
            intervention={
                "enabled": True,
                "direction_id": f"sham-{seed}",
                "layer": 10,
                "mode": "project",
                "scope": "tool_content",
                "direction_norm": "unit",
                "alpha": 0.0,
            },
        )
        for seed in FROZEN_SEEDS
    ),
    *tuple(
        ReplayArm(
            f"additive_sham_{seed}",
            intervention={
                "enabled": True,
                "direction_id": f"sham-{seed}",
                "layer": 10,
                "mode": "add",
                "scope": "tool_content",
                "direction_norm": "unit",
                "alpha": ADDITIVE_ALPHA,
            },
        )
        for seed in FROZEN_SEEDS
    ),
    ReplayArm(
        "whole_tool_projection_c1",
        intervention={
            "enabled": True,
            "direction_id": "c1-l10-dim",
            "layer": 10,
            "mode": "project",
            "scope": "whole_tool_block",
            "direction_norm": "unit",
            "alpha": 0.0,
        },
    ),
    ReplayArm(
        "sae_c1_rank0",
        intervention={
            "enabled": True,
            "direction_id": "sae-c1-rank0-feature-1584",
            "layer": 19,
            "mode": "clamp_sae",
            "scope": "tool_content",
            "direction_norm": "raw",
            "alpha": 0.0,
            "feature_index": 1584,
            "clamp_value": 0.0,
        },
    ),
)


def frozen_arm_hash() -> str:
    records = [
        {
            "arm_id": arm.arm_id,
            "intervention": arm.intervention,
            "prompt_defense": arm.prompt_defense,
            "reserved_token_guard": arm.reserved_token_guard,
        }
        for arm in FROZEN_ARMS
    ]
    wire = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(wire.encode()).hexdigest()
