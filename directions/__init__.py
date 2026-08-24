"""Direction bundles: versioned, self-verifying provenance for fitted directions.

Bundle *data* lives in subdirectories of this package, one per direction id, as
the Checkpoint 1 specification requires (``directions/<direction-id>/``). Ids
are validated so they cannot shadow this package's own modules.
"""

from directions.bundle import (
    Bundle,
    BundleError,
    BundleExistsError,
    BundleIntegrityError,
    current_git_revision,
    hash_contrasts,
    read_bundle,
    write_bundle,
)
from directions.manifest import (
    FORMAT_VERSION,
    DirectionManifest,
    ManifestError,
    PlaceholderRevisionError,
    utc_now_iso,
)

__all__ = [
    "FORMAT_VERSION",
    "DirectionManifest",
    "ManifestError",
    "PlaceholderRevisionError",
    "utc_now_iso",
    "Bundle",
    "BundleError",
    "BundleExistsError",
    "BundleIntegrityError",
    "hash_contrasts",
    "read_bundle",
    "write_bundle",
    "current_git_revision",
]
