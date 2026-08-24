"""Manifest schema tests."""

import pytest

from directions import DirectionManifest, ManifestError, PlaceholderRevisionError, utc_now_iso

REV = "0e9e39f249a16976918f6564b8830bc894c89659"


def manifest(**overrides):
    base = dict(
        direction_id="c1-l19-dim",
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        model_revision=REV,
        layer=19,
        hook_point="resid_post",
        token_extraction_rule="tool_content_positions",
        fitting_method="difference_in_means",
        normalization="unit",
        contrast_set_hash="sha256:" + "a" * 64,
        extraction_code_git_revision="b" * 40,
        dtype="float32",
        d_model=4096,
        created_at=utc_now_iso(),
    )
    base.update(overrides)
    return DirectionManifest(**base)


def test_every_required_field_is_present():
    m = manifest()
    required = {
        "model_id", "model_revision", "layer", "hook_point",
        "token_extraction_rule", "fitting_method", "normalization",
        "contrast_set_hash", "extraction_code_git_revision", "dtype", "created_at",
    }
    assert required <= set(m.to_dict())


def test_round_trips_through_dict():
    m = manifest()
    assert DirectionManifest.from_dict(m.to_dict()) == m


def test_unknown_field_is_rejected():
    data = manifest().to_dict()
    data["surprise"] = 1
    with pytest.raises(ManifestError, match="unknown field"):
        DirectionManifest.from_dict(data)


def test_missing_field_is_rejected():
    data = manifest().to_dict()
    del data["hook_point"]
    with pytest.raises(ManifestError, match="missing required field"):
        DirectionManifest.from_dict(data)


@pytest.mark.parametrize("bad", ["main", "HEAD", "", "latest", "abc123"])
def test_placeholder_revision_raises_its_own_error(bad):
    with pytest.raises(PlaceholderRevisionError):
        manifest(model_revision=bad)


def test_negative_layer_is_rejected():
    with pytest.raises(ManifestError, match="non-negative"):
        manifest(layer=-1)


def test_bool_is_not_accepted_as_layer():
    with pytest.raises(ManifestError, match="non-negative int"):
        manifest(layer=True)


def test_unknown_normalization_is_rejected():
    with pytest.raises(ManifestError, match="normalization"):
        manifest(normalization="l2ish")


def test_unknown_dtype_is_rejected():
    with pytest.raises(ManifestError, match="dtype"):
        manifest(dtype="int8")


@pytest.mark.parametrize("bad", ["", "sha256:zz", "a" * 64, "md5:" + "a" * 32])
def test_malformed_contrast_hash_is_rejected(bad):
    with pytest.raises(ManifestError, match="contrast_set_hash"):
        manifest(contrast_set_hash=bad)


def test_naive_timestamp_is_rejected():
    with pytest.raises(ManifestError, match="no timezone"):
        manifest(created_at="2026-08-23T12:00:00")


def test_unparseable_timestamp_is_rejected():
    with pytest.raises(ManifestError, match="ISO-8601"):
        manifest(created_at="last Tuesday")


def test_empty_hook_point_is_rejected():
    with pytest.raises(ManifestError, match="hook_point"):
        manifest(hook_point="   ")


def test_manifest_is_frozen():
    m = manifest()
    with pytest.raises(Exception):
        m.layer = 20
