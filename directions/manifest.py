"""Direction-bundle manifest: schema, validation, and the placeholder guard.

A manifest is the provenance record for one fitted direction. Its job is to
make a bundle re-derivable and to make an under-specified bundle impossible to
publish, so validation here is deliberately unforgiving.
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "FORMAT_VERSION",
    "DirectionManifest",
    "ManifestError",
    "PlaceholderRevisionError",
]

FORMAT_VERSION = "1"

_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
_CODE_REV_RE = re.compile(r"^[0-9a-f]{40}(\+dirty)?$")
_CONTRAST_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

# Mutable refs that resolve to "whatever is current". Recording one of these as
# a model revision would make a bundle un-reproducible while looking complete,
# which is the specific failure this guard exists to prevent.
_PLACEHOLDER_REVISIONS = frozenset(
    {"", "main", "master", "head", "latest", "none", "null", "tbd", "todo", "unknown", "xxx"}
)

_ALLOWED_NORMALIZATION = frozenset({"unit", "raw"})
_ALLOWED_DTYPES = frozenset(
    {"float64", "float32", "float16", "bfloat16"}
)

# Names that must not be used as direction ids, because bundles live as
# subdirectories of the `directions/` package and would shadow its modules.
_RESERVED_IDS = frozenset({"__init__", "__pycache__", "bundle", "manifest"})


class ManifestError(ValueError):
    """Manifest is missing a required field, or a field is malformed."""


class PlaceholderRevisionError(ManifestError):
    """A revision field holds a mutable ref instead of an immutable commit.

    Raised loudly and separately from other manifest errors because this is the
    failure mode ``DECISIONS.md`` B2 specifically requires be impossible to
    reach by accident: an extraction run that records ``main`` instead of a
    pinned SHA produces results nobody can reproduce.
    """


def validate_direction_id(direction_id: str) -> str:
    if not isinstance(direction_id, str) or not _ID_RE.match(direction_id):
        raise ManifestError(
            f"direction_id {direction_id!r} must match {_ID_RE.pattern} "
            "(lowercase alphanumeric, dot, dash, underscore)"
        )
    if direction_id in _RESERVED_IDS or direction_id.endswith(".py"):
        raise ManifestError(
            f"direction_id {direction_id!r} is reserved: bundles are stored as "
            "subdirectories of the `directions/` package and must not shadow "
            "its modules."
        )
    return direction_id


def _require_immutable_revision(value: str, fieldname: str) -> str:
    if not isinstance(value, str):
        raise ManifestError(f"{fieldname} must be a string, got {type(value).__name__}")
    if value.strip().lower() in _PLACEHOLDER_REVISIONS:
        raise PlaceholderRevisionError(
            f"{fieldname} is {value!r}, which is a mutable ref or a placeholder. "
            "An immutable 40-character commit SHA is required. "
            "See DECISIONS.md B2 and docs/PINS.md."
        )
    if not _SHA1_RE.match(value):
        raise PlaceholderRevisionError(
            f"{fieldname} is {value!r}, which is not an immutable 40-character "
            "lowercase commit SHA."
        )
    return value


@dataclass(frozen=True)
class DirectionManifest:
    """Provenance for one fitted direction.

    Every field required by the Checkpoint 1 specification is mandatory and
    validated in ``__post_init__``; there are no optional provenance fields.
    """

    direction_id: str
    model_id: str
    model_revision: str
    layer: int
    hook_point: str
    token_extraction_rule: str
    fitting_method: str
    normalization: str
    contrast_set_hash: str
    extraction_code_git_revision: str
    dtype: str
    d_model: int
    created_at: str
    format_version: str = FORMAT_VERSION
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_direction_id(self.direction_id)

        if self.format_version != FORMAT_VERSION:
            raise ManifestError(
                f"format_version {self.format_version!r} is not supported by this "
                f"code, which writes and reads version {FORMAT_VERSION!r}"
            )

        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ManifestError("model_id must be a non-empty string")
        _require_immutable_revision(self.model_revision, "model_revision")
        _require_immutable_revision(
            self.extraction_code_git_revision.removesuffix("+dirty"),
            "extraction_code_git_revision",
        )
        if not _CODE_REV_RE.match(self.extraction_code_git_revision):
            raise ManifestError(
                "extraction_code_git_revision must be a 40-character SHA, "
                "optionally suffixed '+dirty'"
            )

        if not isinstance(self.layer, int) or isinstance(self.layer, bool) or self.layer < 0:
            raise ManifestError(f"layer must be a non-negative int, got {self.layer!r}")
        if not isinstance(self.d_model, int) or isinstance(self.d_model, bool) or self.d_model <= 0:
            raise ManifestError(f"d_model must be a positive int, got {self.d_model!r}")

        for name in ("hook_point", "token_extraction_rule", "fitting_method"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ManifestError(f"{name} must be a non-empty string")

        if self.normalization not in _ALLOWED_NORMALIZATION:
            raise ManifestError(
                f"normalization must be one of {sorted(_ALLOWED_NORMALIZATION)}, "
                f"got {self.normalization!r}. The bundle must state explicitly "
                "whether the stored vector is unit-norm or raw."
            )
        if self.dtype not in _ALLOWED_DTYPES:
            raise ManifestError(
                f"dtype must be one of {sorted(_ALLOWED_DTYPES)}, got {self.dtype!r}"
            )
        if not _CONTRAST_HASH_RE.match(self.contrast_set_hash or ""):
            raise ManifestError(
                f"contrast_set_hash must look like 'sha256:<64 hex>', "
                f"got {self.contrast_set_hash!r}"
            )

        try:
            parsed = _dt.datetime.fromisoformat(self.created_at)
        except (TypeError, ValueError) as exc:
            raise ManifestError(
                f"created_at must be an ISO-8601 timestamp, got {self.created_at!r}"
            ) from exc
        if parsed.tzinfo is None:
            raise ManifestError(
                f"created_at {self.created_at!r} has no timezone. Use UTC, e.g. "
                "'2026-08-23T12:00:00+00:00'."
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DirectionManifest":
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ManifestError(
                f"manifest has unknown field(s) {sorted(unknown)}. Unrecognised "
                "provenance is rejected rather than ignored: it usually means "
                "the bundle was written by a different format version."
            )
        try:
            return cls(**data)
        except TypeError as exc:
            raise ManifestError(f"manifest is missing required field(s): {exc}") from exc


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string with explicit offset."""
    return _dt.datetime.now(_dt.timezone.utc).isoformat()
