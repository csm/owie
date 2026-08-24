"""Versioned direction bundles: write, read, and verify.

Layout, fixed by the Checkpoint 1 specification::

    directions/<direction-id>/
      vector.safetensors
      contrasts.jsonl
      manifest.json
      extraction_config.json

Two invariants are enforced by this module rather than by discipline:

*Bundles are never overwritten.* ``write_bundle`` refuses to write into an
existing directory. Raw data is immutable (working discipline); a re-fit gets a
new id, so a result can always be traced to the exact bundle that produced it.

*A bundle verifies itself on read.* ``read_bundle`` recomputes the contrast-set
hash and re-checks the vector against the manifest's shape, dtype, and declared
normalization. A bundle that has drifted from its manifest raises instead of
being loaded, because a direction silently mismatched to its provenance would
corrupt every downstream effect size while looking fine.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from safetensors.torch import load_file, save_file

from directions.manifest import (
    DirectionManifest,
    ManifestError,
    utc_now_iso,
    validate_direction_id,
)

__all__ = [
    "BundleError",
    "BundleExistsError",
    "BundleIntegrityError",
    "Bundle",
    "DEFAULT_ROOT",
    "VECTOR_KEY",
    "hash_contrasts",
    "canonical_contrast_line",
    "write_bundle",
    "read_bundle",
    "current_git_revision",
]

DEFAULT_ROOT = Path(__file__).resolve().parent
VECTOR_KEY = "vector"

_MANIFEST_NAME = "manifest.json"
_VECTOR_NAME = "vector.safetensors"
_CONTRASTS_NAME = "contrasts.jsonl"
_EXTRACTION_CONFIG_NAME = "extraction_config.json"

_DTYPE_BY_NAME = {
    "float64": torch.float64,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
_NAME_BY_DTYPE = {v: k for k, v in _DTYPE_BY_NAME.items()}

# Used when checking that a vector stored as normalization="unit" really is
# unit-norm. Must match interventions.kernel._UNIT_TOL: a bundle that reads
# clean here but is then rejected by the kernel would be a maddening bug.
# See that module for why float64 is 1e-6 and not tighter.
_UNIT_TOL = {
    torch.bfloat16: 1e-2,
    torch.float16: 1e-2,
    torch.float32: 1e-5,
    torch.float64: 1e-6,
}


class BundleError(RuntimeError):
    """Base class for bundle read/write failures."""


class BundleExistsError(BundleError):
    """Refusing to overwrite an existing bundle."""


class BundleIntegrityError(BundleError):
    """Bundle contents do not match its manifest."""


def canonical_contrast_line(row: Any) -> str:
    """Canonical JSON for one contrast row.

    Sorted keys and fixed separators, so that a hash depends on the *content*
    of the contrast set and not on how some writer happened to format it.
    Non-ASCII is preserved rather than escaped — the contrast sets are natural
    language and will contain it.
    """
    return json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hash_contrasts(rows: Iterable[Any]) -> str:
    """Hash a contrast set to ``sha256:<hex>``.

    Order-sensitive by design: the split into training and held-out scenario
    families is positional in the file, so two files with the same rows in a
    different order are not the same contrast set.
    """
    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical_contrast_line(row).encode("utf-8"))
        digest.update(b"\n")
    return f"sha256:{digest.hexdigest()}"


def _read_jsonl(path: Path) -> list[Any]:
    rows: list[Any] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise BundleIntegrityError(
                    f"{path}: line {lineno} is not valid JSON: {exc}"
                ) from exc
    return rows


def current_git_revision(repo: Path | None = None) -> str:
    """Resolve HEAD, suffixed ``+dirty`` when the tree has uncommitted changes.

    A bundle fitted from a dirty tree is not reproducible from its recorded
    revision alone. That is recorded rather than forbidden, so the fact shows
    up in the manifest instead of being lost.
    """
    cwd = str(repo or Path.cwd())
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd, capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise BundleError(
            f"cannot resolve git revision in {cwd!r}: {exc}. The manifest "
            "requires the extraction-code revision; it is not optional."
        ) from exc
    return f"{head}+dirty" if dirty else head


@dataclass(frozen=True)
class Bundle:
    """A loaded, verified direction bundle."""

    manifest: DirectionManifest
    vector: torch.Tensor
    contrasts: list[Any]
    extraction_config: dict[str, Any]
    path: Path

    @property
    def direction_id(self) -> str:
        return self.manifest.direction_id


def _check_vector_against_manifest(
    vector: torch.Tensor, manifest: DirectionManifest, where: str
) -> None:
    if vector.ndim != 1:
        raise BundleIntegrityError(
            f"{where}: vector must be 1-D, got shape {tuple(vector.shape)}"
        )
    if vector.shape[0] != manifest.d_model:
        raise BundleIntegrityError(
            f"{where}: vector length {vector.shape[0]} does not match manifest "
            f"d_model {manifest.d_model}"
        )
    expected_dtype = _DTYPE_BY_NAME[manifest.dtype]
    if vector.dtype is not expected_dtype:
        raise BundleIntegrityError(
            f"{where}: vector dtype {vector.dtype} does not match manifest "
            f"dtype {manifest.dtype}"
        )
    if not torch.isfinite(vector).all():
        raise BundleIntegrityError(f"{where}: vector contains NaN or Inf")

    length = float(torch.linalg.vector_norm(vector.to(torch.float32)))
    if manifest.normalization == "unit":
        tol = _UNIT_TOL[expected_dtype]
        if abs(length - 1.0) > tol:
            raise BundleIntegrityError(
                f"{where}: manifest declares normalization='unit' but the stored "
                f"vector has norm {length!r} (tolerance {tol}). The manifest and "
                "the vector disagree; one of them is wrong."
            )
    elif length <= 0.0:
        raise BundleIntegrityError(f"{where}: vector is all zeros")


def write_bundle(
    direction_id: str,
    vector: torch.Tensor,
    manifest_fields: dict[str, Any],
    contrasts: Sequence[Any],
    extraction_config: dict[str, Any],
    *,
    root: Path | None = None,
) -> Path:
    """Write a new bundle. Refuses to overwrite an existing one.

    ``manifest_fields`` supplies provenance; ``direction_id``, ``d_model``,
    ``dtype``, ``contrast_set_hash`` and ``created_at`` are derived here if not
    given, and cross-checked if they are. The write goes to a temporary
    directory and is renamed into place, so an interrupted write cannot leave a
    half-bundle that later reads as valid.
    """
    validate_direction_id(direction_id)
    root = Path(root) if root is not None else DEFAULT_ROOT
    target = root / direction_id

    if target.exists():
        raise BundleExistsError(
            f"{target} already exists. Direction bundles are never overwritten "
            "(working discipline); fit under a new direction_id instead."
        )
    if not isinstance(vector, torch.Tensor):
        raise BundleError(f"vector must be a Tensor, got {type(vector).__name__}")
    if vector.dtype not in _NAME_BY_DTYPE:
        raise BundleError(f"unsupported vector dtype {vector.dtype}")

    vector = vector.detach().cpu().contiguous()
    contrast_hash = hash_contrasts(contrasts)

    fields = dict(manifest_fields)
    fields.setdefault("direction_id", direction_id)
    fields.setdefault("d_model", int(vector.shape[0]) if vector.ndim == 1 else -1)
    fields.setdefault("dtype", _NAME_BY_DTYPE[vector.dtype])
    fields.setdefault("contrast_set_hash", contrast_hash)
    fields.setdefault("created_at", utc_now_iso())

    if fields["direction_id"] != direction_id:
        raise ManifestError(
            f"manifest direction_id {fields['direction_id']!r} does not match "
            f"the requested id {direction_id!r}"
        )
    if fields["contrast_set_hash"] != contrast_hash:
        raise BundleIntegrityError(
            f"manifest contrast_set_hash {fields['contrast_set_hash']!r} does not "
            f"match the hash of the supplied contrasts {contrast_hash!r}"
        )

    manifest = DirectionManifest(**fields)
    _check_vector_against_manifest(vector, manifest, where="write_bundle")

    root.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{direction_id}.partial.", dir=root))
    try:
        save_file({VECTOR_KEY: vector}, str(tmp / _VECTOR_NAME))
        with (tmp / _CONTRASTS_NAME).open("w", encoding="utf-8") as handle:
            for row in contrasts:
                handle.write(canonical_contrast_line(row))
                handle.write("\n")
        (tmp / _EXTRACTION_CONFIG_NAME).write_text(
            json.dumps(extraction_config, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (tmp / _MANIFEST_NAME).write_text(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.rename(tmp, target)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return target


def read_bundle(direction_id: str, *, root: Path | None = None) -> Bundle:
    """Load and verify a bundle. Raises rather than returning a suspect one."""
    validate_direction_id(direction_id)
    root = Path(root) if root is not None else DEFAULT_ROOT
    path = root / direction_id
    if not path.is_dir():
        raise BundleError(f"no bundle at {path}")

    for name in (_MANIFEST_NAME, _VECTOR_NAME, _CONTRASTS_NAME, _EXTRACTION_CONFIG_NAME):
        if not (path / name).is_file():
            raise BundleIntegrityError(f"{path}: missing required file {name!r}")

    try:
        manifest_data = json.loads((path / _MANIFEST_NAME).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BundleIntegrityError(f"{path}: manifest.json is not valid JSON: {exc}") from exc
    manifest = DirectionManifest.from_dict(manifest_data)

    if manifest.direction_id != direction_id:
        raise BundleIntegrityError(
            f"{path}: manifest direction_id {manifest.direction_id!r} does not "
            f"match its directory name {direction_id!r}"
        )

    tensors = load_file(str(path / _VECTOR_NAME))
    if set(tensors) != {VECTOR_KEY}:
        raise BundleIntegrityError(
            f"{path}: vector.safetensors must contain exactly one tensor named "
            f"{VECTOR_KEY!r}, found {sorted(tensors)}"
        )
    vector = tensors[VECTOR_KEY]
    _check_vector_against_manifest(vector, manifest, where=str(path))

    contrasts = _read_jsonl(path / _CONTRASTS_NAME)
    recomputed = hash_contrasts(contrasts)
    if recomputed != manifest.contrast_set_hash:
        raise BundleIntegrityError(
            f"{path}: contrast-set hash mismatch. manifest says "
            f"{manifest.contrast_set_hash!r}, contrasts.jsonl hashes to "
            f"{recomputed!r}. The bundle has drifted from its provenance."
        )

    try:
        extraction_config = json.loads(
            (path / _EXTRACTION_CONFIG_NAME).read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise BundleIntegrityError(
            f"{path}: extraction_config.json is not valid JSON: {exc}"
        ) from exc

    return Bundle(
        manifest=manifest,
        vector=vector,
        contrasts=contrasts,
        extraction_config=extraction_config,
        path=path,
    )
