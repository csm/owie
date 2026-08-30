"""Checkpoint 5 paired-replay inputs and execution."""

from .prefixes import (
    ReplayPrefix,
    build_prefix_manifest,
    load_trajectory_prefixes,
    verify_prefix_manifest,
    write_prefix_manifest,
)
from .runner import (
    Continuation,
    ReplayArm,
    ReplayConfig,
    continuation_id,
    reserved_token_guard,
    resume,
)
from .arms import FROZEN_ARMS, FROZEN_SEEDS, frozen_arm_hash
from .collection import CollectionConfig, collect_primary, collect_secondary

__all__ = [
    "ReplayPrefix",
    "build_prefix_manifest",
    "load_trajectory_prefixes",
    "verify_prefix_manifest",
    "write_prefix_manifest",
    "Continuation",
    "ReplayArm",
    "ReplayConfig",
    "continuation_id",
    "reserved_token_guard",
    "resume",
    "FROZEN_ARMS",
    "FROZEN_SEEDS",
    "frozen_arm_hash",
    "CollectionConfig",
    "collect_primary",
    "collect_secondary",
]
