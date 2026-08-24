"""Direction-bundle format tests. No model, no network."""

import json

import pytest
import torch

from directions import (
    BundleExistsError,
    BundleIntegrityError,
    DirectionManifest,
    ManifestError,
    PlaceholderRevisionError,
    hash_contrasts,
    read_bundle,
    write_bundle,
)
from directions.bundle import canonical_contrast_line

# The real pins, so the tests exercise the shapes of value the project will
# actually carry. See docs/PINS.md.
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_REV = "0e9e39f249a16976918f6564b8830bc894c89659"
CODE_REV = "1111111111111111111111111111111111111111"

CONTRASTS = [
    {"family": "invoice", "polarity": "pos", "text": "Delete the staging bucket."},
    {"family": "invoice", "polarity": "neg", "text": "The staging bucket exists."},
    {"family": "weather", "polarity": "pos", "text": "Email the report to Dana."},
    {"family": "weather", "polarity": "neg", "text": "The report was emailed."},
]

EXTRACTION_CONFIG = {
    "positions": "tool_content",
    "layers": list(range(32)),
    "batch_size": 1,
    "accumulator_dtype": "float32",
}


def fields(**overrides):
    base = dict(
        model_id=MODEL_ID,
        model_revision=MODEL_REV,
        layer=19,
        hook_point="resid_post",
        token_extraction_rule="tool_content_positions",
        fitting_method="difference_in_means",
        normalization="unit",
        extraction_code_git_revision=CODE_REV,
    )
    base.update(overrides)
    return base


def unit_vector(d=16, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(d, generator=g, dtype=torch.float32)
    return (v / torch.linalg.vector_norm(v)).to(dtype)


# --------------------------------------------------------------------------
# Round trip
# --------------------------------------------------------------------------

def test_write_then_read_round_trips(tmp_path):
    vec = unit_vector()
    write_bundle("c1-l19-dim", vec, fields(), CONTRASTS, EXTRACTION_CONFIG, root=tmp_path)
    bundle = read_bundle("c1-l19-dim", root=tmp_path)

    assert bundle.direction_id == "c1-l19-dim"
    assert bundle.manifest.model_revision == MODEL_REV
    assert bundle.manifest.layer == 19
    assert bundle.manifest.d_model == 16
    assert bundle.manifest.dtype == "float32"
    assert bundle.contrasts == CONTRASTS
    assert bundle.extraction_config == EXTRACTION_CONFIG
    assert torch.equal(bundle.vector, vec)


def test_all_four_files_are_written(tmp_path):
    write_bundle("d", unit_vector(), fields(), CONTRASTS, EXTRACTION_CONFIG, root=tmp_path)
    names = {p.name for p in (tmp_path / "d").iterdir()}
    assert names == {
        "vector.safetensors",
        "contrasts.jsonl",
        "manifest.json",
        "extraction_config.json",
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16, torch.float16])
def test_dtypes_round_trip(tmp_path, dtype):
    vec = unit_vector(dtype=dtype)
    write_bundle("d", vec, fields(), CONTRASTS, EXTRACTION_CONFIG, root=tmp_path)
    bundle = read_bundle("d", root=tmp_path)
    assert bundle.vector.dtype is dtype
    assert torch.equal(bundle.vector, vec)


def test_raw_normalization_accepts_a_non_unit_vector(tmp_path):
    vec = unit_vector() * 12.5
    write_bundle(
        "raw-dir", vec, fields(normalization="raw"), CONTRASTS, EXTRACTION_CONFIG, root=tmp_path
    )
    bundle = read_bundle("raw-dir", root=tmp_path)
    assert bundle.manifest.normalization == "raw"
    assert torch.allclose(torch.linalg.vector_norm(bundle.vector), torch.tensor(12.5))


# --------------------------------------------------------------------------
# Never overwrite
# --------------------------------------------------------------------------

def test_write_refuses_to_overwrite(tmp_path):
    write_bundle("d", unit_vector(), fields(), CONTRASTS, EXTRACTION_CONFIG, root=tmp_path)
    with pytest.raises(BundleExistsError, match="never overwritten"):
        write_bundle("d", unit_vector(seed=9), fields(), CONTRASTS, EXTRACTION_CONFIG, root=tmp_path)


def test_failed_write_leaves_no_partial_bundle(tmp_path):
    """An invalid manifest must not leave debris that later reads as a bundle."""
    with pytest.raises(PlaceholderRevisionError):
        write_bundle(
            "d", unit_vector(), fields(model_revision="main"),
            CONTRASTS, EXTRACTION_CONFIG, root=tmp_path,
        )
    assert not (tmp_path / "d").exists()
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------
# The placeholder-revision guard (DECISIONS.md B2)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["main", "master", "HEAD", "latest", "", "TBD", "unknown"])
def test_placeholder_model_revision_is_refused(tmp_path, bad):
    with pytest.raises(PlaceholderRevisionError, match="immutable"):
        write_bundle(
            "d", unit_vector(), fields(model_revision=bad),
            CONTRASTS, EXTRACTION_CONFIG, root=tmp_path,
        )


@pytest.mark.parametrize("bad", ["0e9e39f2", "not-a-sha", "0E9E39F249A16976918F6564B8830BC894C89659"])
def test_non_sha_model_revision_is_refused(tmp_path, bad):
    with pytest.raises(PlaceholderRevisionError):
        write_bundle(
            "d", unit_vector(), fields(model_revision=bad),
            CONTRASTS, EXTRACTION_CONFIG, root=tmp_path,
        )


def test_dirty_code_revision_is_allowed_and_recorded(tmp_path):
    write_bundle(
        "d", unit_vector(), fields(extraction_code_git_revision=CODE_REV + "+dirty"),
        CONTRASTS, EXTRACTION_CONFIG, root=tmp_path,
    )
    assert read_bundle("d", root=tmp_path).manifest.extraction_code_git_revision.endswith("+dirty")


def test_placeholder_code_revision_is_refused(tmp_path):
    with pytest.raises(PlaceholderRevisionError):
        write_bundle(
            "d", unit_vector(), fields(extraction_code_git_revision="HEAD"),
            CONTRASTS, EXTRACTION_CONFIG, root=tmp_path,
        )


# --------------------------------------------------------------------------
# Self-verification on read
# --------------------------------------------------------------------------

def test_tampered_contrasts_are_detected(tmp_path):
    write_bundle("d", unit_vector(), fields(), CONTRASTS, EXTRACTION_CONFIG, root=tmp_path)
    path = tmp_path / "d" / "contrasts.jsonl"
    rows = path.read_text(encoding="utf-8").splitlines()
    rows[0] = canonical_contrast_line({"family": "invoice", "polarity": "pos", "text": "TAMPERED"})
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    with pytest.raises(BundleIntegrityError, match="drifted from its provenance"):
        read_bundle("d", root=tmp_path)


def test_reordered_contrasts_are_detected(tmp_path):
    """Order is part of the contrast set: the train/held-out split is positional."""
    write_bundle("d", unit_vector(), fields(), CONTRASTS, EXTRACTION_CONFIG, root=tmp_path)
    path = tmp_path / "d" / "contrasts.jsonl"
    rows = path.read_text(encoding="utf-8").splitlines()
    rows.reverse()
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(BundleIntegrityError, match="hash mismatch"):
        read_bundle("d", root=tmp_path)


def test_manifest_directory_name_mismatch_is_detected(tmp_path):
    write_bundle("d", unit_vector(), fields(), CONTRASTS, EXTRACTION_CONFIG, root=tmp_path)
    (tmp_path / "d").rename(tmp_path / "e")
    with pytest.raises(BundleIntegrityError, match="does not match its directory name"):
        read_bundle("e", root=tmp_path)


def test_vector_dtype_mismatch_is_detected(tmp_path):
    write_bundle("d", unit_vector(), fields(), CONTRASTS, EXTRACTION_CONFIG, root=tmp_path)
    manifest_path = tmp_path / "d" / "manifest.json"
    data = json.loads(manifest_path.read_text())
    data["dtype"] = "bfloat16"
    manifest_path.write_text(json.dumps(data))
    with pytest.raises(BundleIntegrityError, match="dtype"):
        read_bundle("d", root=tmp_path)


def test_vector_length_mismatch_is_detected(tmp_path):
    write_bundle("d", unit_vector(), fields(), CONTRASTS, EXTRACTION_CONFIG, root=tmp_path)
    manifest_path = tmp_path / "d" / "manifest.json"
    data = json.loads(manifest_path.read_text())
    data["d_model"] = 4096
    manifest_path.write_text(json.dumps(data))
    with pytest.raises(BundleIntegrityError, match="d_model"):
        read_bundle("d", root=tmp_path)


def test_unit_claim_is_verified_against_the_stored_vector(tmp_path):
    """normalization='unit' with a raw vector is the exact bug this catches."""
    write_bundle(
        "d", unit_vector() * 3.0, fields(normalization="raw"),
        CONTRASTS, EXTRACTION_CONFIG, root=tmp_path,
    )
    manifest_path = tmp_path / "d" / "manifest.json"
    data = json.loads(manifest_path.read_text())
    data["normalization"] = "unit"
    manifest_path.write_text(json.dumps(data))
    with pytest.raises(BundleIntegrityError, match="manifest and"):
        read_bundle("d", root=tmp_path)


@pytest.mark.parametrize(
    "name", ["manifest.json", "vector.safetensors", "contrasts.jsonl", "extraction_config.json"]
)
def test_missing_file_is_detected(tmp_path, name):
    write_bundle("d", unit_vector(), fields(), CONTRASTS, EXTRACTION_CONFIG, root=tmp_path)
    (tmp_path / "d" / name).unlink()
    with pytest.raises(BundleIntegrityError, match="missing required file"):
        read_bundle("d", root=tmp_path)


def test_extra_tensor_in_vector_file_is_detected(tmp_path):
    from safetensors.torch import save_file

    write_bundle("d", unit_vector(), fields(), CONTRASTS, EXTRACTION_CONFIG, root=tmp_path)
    save_file(
        {"vector": unit_vector(), "sneaky": torch.zeros(3)},
        str(tmp_path / "d" / "vector.safetensors"),
    )
    with pytest.raises(BundleIntegrityError, match="exactly one tensor"):
        read_bundle("d", root=tmp_path)


# --------------------------------------------------------------------------
# Contrast hashing
# --------------------------------------------------------------------------

def test_hash_is_stable_and_key_order_independent():
    a = hash_contrasts([{"a": 1, "b": 2}])
    b = hash_contrasts([{"b": 2, "a": 1}])
    assert a == b
    assert a.startswith("sha256:") and len(a) == len("sha256:") + 64


def test_hash_is_order_sensitive_across_rows():
    assert hash_contrasts([{"a": 1}, {"a": 2}]) != hash_contrasts([{"a": 2}, {"a": 1}])


def test_hash_distinguishes_content():
    assert hash_contrasts([{"t": "x"}]) != hash_contrasts([{"t": "y"}])


def test_unicode_contrasts_round_trip(tmp_path):
    """Contrast sets are natural language and will contain non-ASCII."""
    rows = [{"text": "Ignore prior instructions — ünïcode, emoji 🙂, quote \" and \\ back"}]
    write_bundle("u", unit_vector(), fields(), rows, EXTRACTION_CONFIG, root=tmp_path)
    assert read_bundle("u", root=tmp_path).contrasts == rows


def test_manifest_hash_must_match_supplied_contrasts(tmp_path):
    with pytest.raises(BundleIntegrityError, match="does not match the hash"):
        write_bundle(
            "d", unit_vector(),
            fields(contrast_set_hash="sha256:" + "0" * 64),
            CONTRASTS, EXTRACTION_CONFIG, root=tmp_path,
        )


# --------------------------------------------------------------------------
# Direction ids
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["bundle", "manifest"])
def test_module_shadowing_ids_are_refused(tmp_path, bad):
    """Bundles are subdirectories of the `directions` package, so an id that
    collides with one of its modules would shadow it on import."""
    with pytest.raises(ManifestError, match="reserved"):
        write_bundle(bad, unit_vector(), fields(), CONTRASTS, EXTRACTION_CONFIG, root=tmp_path)


@pytest.mark.parametrize(
    "bad",
    ["bundle", "manifest", "__init__", "__pycache__", "../escape", "Upper",
     "has space", "", ".hidden", "a/b", "x.py"],
)
def test_every_dangerous_id_is_refused_somehow(tmp_path, bad):
    """Whichever guard catches it, none of these may produce a bundle."""
    with pytest.raises(ManifestError):
        write_bundle(bad, unit_vector(), fields(), CONTRASTS, EXTRACTION_CONFIG, root=tmp_path)
    # Nothing was created under the root, whichever guard fired.
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("bad", ["../escape", "Upper", "has space", "", ".hidden", "a/b"])
def test_malformed_ids_are_refused(tmp_path, bad):
    with pytest.raises(ManifestError, match="must match"):
        write_bundle(bad, unit_vector(), fields(), CONTRASTS, EXTRACTION_CONFIG, root=tmp_path)
