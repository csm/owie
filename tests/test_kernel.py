"""Intervention-kernel tests.

Acceptance for Checkpoint 1 is that these pass **without loading the model**.
Nothing here touches Hugging Face, weights, or the network.
"""

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from interventions import (
    DirectionError,
    MaskError,
    Norm,
    ShapeError,
    add_vector,
    clamp_feature,
    project_out,
)

DTYPES = [torch.float32, torch.float64, torch.bfloat16, torch.float16]

# Tolerances by dtype. float32 is the ceiling on Apple Metal (no float64 on
# MPS, PREFLIGHT.md §1), so the float32 row is the one that matters in
# practice; float64 is kept because it is available on CPU and pins down that
# the algebra itself is right rather than merely well-conditioned.
ATOL = {
    torch.float64: 1e-12,
    torch.float32: 1e-5,
    torch.bfloat16: 5e-2,
    torch.float16: 5e-3,
}


def unit(d_model: int, dtype=torch.float32, seed: int = 0) -> torch.Tensor:
    """A unit vector normalized *in the target dtype*.

    Normalizing in float32 and casting up would leave a float64 vector whose
    norm is off by ~1e-8, which is realistic (see
    ``test_float32_normalized_vector_upcast_to_float64_is_accepted``) but makes
    a poor fixture for testing the algebra.
    """
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(d_model, generator=g, dtype=torch.float32).to(dtype)
    return v / torch.linalg.vector_norm(v)


def hidden(batch=1, seq=6, d_model=8, dtype=torch.float32, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, seq, d_model, generator=g, dtype=torch.float32).to(dtype)


def mask_from(rows, batch=1) -> torch.Tensor:
    return torch.tensor([rows] * batch, dtype=torch.bool)


# --------------------------------------------------------------------------
# No in-place mutation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", DTYPES)
def test_no_in_place_mutation(dtype):
    x = hidden(dtype=dtype)
    before = x.clone()
    m = mask_from([True, True, False, True, False, False])
    r = unit(8, dtype)

    project_out(x, r, m)
    add_vector(x, r, 2.0, m, norm=Norm.ASSERT_UNIT)
    clamp_feature(x, r, 1.5, m)

    assert torch.equal(x, before), "input tensor was mutated"


@pytest.mark.parametrize("dtype", DTYPES)
def test_returns_a_distinct_tensor(dtype):
    x = hidden(dtype=dtype)
    m = mask_from([True] * 6)
    out = project_out(x, unit(8, dtype), m)
    assert out.data_ptr() != x.data_ptr()


# --------------------------------------------------------------------------
# Exact identity outside the mask
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fn", ["project_out", "add_vector", "clamp_feature"])
def test_outside_mask_is_bitwise_identity(dtype, fn):
    x = hidden(dtype=dtype)
    r = unit(8, dtype)
    rows = [True, False, True, False, False, True]
    m = mask_from(rows)

    if fn == "project_out":
        out = project_out(x, r, m)
    elif fn == "add_vector":
        out = add_vector(x, r, 3.0, m, norm=Norm.ASSERT_UNIT)
    else:
        out = clamp_feature(x, r, 2.0, m)

    outside = ~m
    # Bitwise, not approximate. An unmasked position must not drift by one ulp.
    assert torch.equal(out[outside], x[outside])


@pytest.mark.parametrize("dtype", DTYPES)
def test_all_zero_mask_is_exact_identity(dtype):
    x = hidden(dtype=dtype)
    m = mask_from([False] * 6)
    r = unit(8, dtype)

    assert torch.equal(project_out(x, r, m), x)
    assert torch.equal(add_vector(x, r, 7.0, m, norm=Norm.ASSERT_UNIT), x)
    assert torch.equal(clamp_feature(x, r, 7.0, m), x)


@pytest.mark.parametrize("dtype", DTYPES)
def test_all_one_mask_modifies_everything(dtype):
    x = hidden(dtype=dtype)
    m = mask_from([True] * 6)
    r = unit(8, dtype)
    out = project_out(x, r, m)
    # Score in the widest of (input dtype, float32): casting a float64 result
    # down to float32 to check it would measure the cast, not the projection.
    score_dtype = torch.float64 if dtype is torch.float64 else torch.float32
    coeff = (out.to(score_dtype) * r.to(score_dtype)).sum(-1)
    assert torch.allclose(coeff, torch.zeros_like(coeff), atol=ATOL[dtype])


def test_identity_holds_when_unmasked_positions_are_non_finite():
    """Exactness must survive garbage outside the mask.

    A NaN at an unmasked position is not this function's business, and must
    pass through untouched rather than being spread by the arithmetic.
    """
    x = hidden()
    x[0, 3, :] = float("nan")
    x[0, 4, 0] = float("inf")
    m = mask_from([True, True, True, False, False, True])
    out = project_out(x, unit(8), m)
    assert torch.equal(out[0, 3].isnan(), x[0, 3].isnan())
    assert torch.isinf(out[0, 4, 0])
    assert not torch.isnan(out[0, :3]).any()


# --------------------------------------------------------------------------
# Device and dtype preservation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", DTYPES)
def test_dtype_and_device_preserved(dtype):
    x = hidden(dtype=dtype)
    m = mask_from([True] * 6)
    for out in (
        project_out(x, unit(8, dtype), m),
        add_vector(x, unit(8, dtype), 1.0, m, norm=Norm.ASSERT_UNIT),
        clamp_feature(x, unit(8, dtype), 1.0, m),
    ):
        assert out.dtype == x.dtype
        assert out.device == x.device
        assert out.shape == x.shape


def test_direction_dtype_may_differ_from_hidden():
    """A float32 bundle vector against bf16 activations is the normal case."""
    x = hidden(dtype=torch.bfloat16)
    r = unit(8, torch.float32)
    out = project_out(x, r, mask_from([True] * 6))
    assert out.dtype is torch.bfloat16


# --------------------------------------------------------------------------
# Projection: orthogonality and idempotence
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_projection_is_numerically_orthogonal(dtype):
    x = hidden(seq=16, d_model=64, dtype=dtype)
    r = unit(64, dtype)
    out = project_out(x, r, mask_from([True] * 16))
    residual = (out * r).sum(-1)
    assert residual.abs().max().item() < ATOL[dtype]


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_projection_is_idempotent(dtype):
    x = hidden(seq=16, d_model=64, dtype=dtype)
    r = unit(64, dtype)
    m = mask_from([True] * 16)
    once = project_out(x, r, m)
    twice = project_out(once, r, m)
    assert torch.allclose(once, twice, atol=ATOL[dtype], rtol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_projection_orthogonal_within_low_precision_tolerance(dtype):
    x = hidden(seq=16, d_model=64, dtype=dtype)
    r = unit(64, dtype)
    out = project_out(x, r, mask_from([True] * 16))
    residual = (out.to(torch.float32) * r.to(torch.float32)).sum(-1)
    assert residual.abs().max().item() < ATOL[dtype]


def test_projection_preserves_orthogonal_component():
    """Only the component along r is removed; the rest survives."""
    d = 32
    r = unit(d, torch.float64, seed=3)
    g = torch.Generator().manual_seed(9)
    perp = torch.randn(d, generator=g, dtype=torch.float64)
    perp = perp - (perp @ r) * r
    x = perp.reshape(1, 1, d) + 4.2 * r.reshape(1, 1, d)
    out = project_out(x, r, mask_from([True], batch=1))
    assert torch.allclose(out[0, 0], perp, atol=1e-12)


def test_normalize_and_assert_unit_agree_for_a_scaled_direction():
    x = hidden(d_model=16, dtype=torch.float64)
    m = mask_from([True] * 6)
    r = unit(16, torch.float64, seed=5)
    a = project_out(x, r, m, norm=Norm.ASSERT_UNIT)
    b = project_out(x, r * 37.0, m, norm=Norm.NORMALIZE)
    assert torch.allclose(a, b, atol=1e-12)


# --------------------------------------------------------------------------
# Normalization handling is explicit
# --------------------------------------------------------------------------

def test_assert_unit_rejects_non_unit_direction():
    x = hidden(dtype=torch.float64)
    m = mask_from([True] * 6)
    with pytest.raises(DirectionError, match="ASSERT_UNIT"):
        project_out(x, unit(8, torch.float64) * 2.0, m, norm=Norm.ASSERT_UNIT)


def test_normalize_rejects_zero_direction():
    x = hidden()
    m = mask_from([True] * 6)
    with pytest.raises(DirectionError, match="too small to normalize"):
        project_out(x, torch.zeros(8), m, norm=Norm.NORMALIZE)


def test_project_out_rejects_as_is():
    x = hidden()
    m = mask_from([True] * 6)
    with pytest.raises(DirectionError, match="not projection"):
        project_out(x, unit(8), m, norm=Norm.AS_IS)


def test_clamp_feature_rejects_as_is():
    x = hidden()
    m = mask_from([True] * 6)
    with pytest.raises(DirectionError, match="not projection"):
        clamp_feature(x, unit(8), 1.0, m, norm=Norm.AS_IS)


def test_add_vector_requires_explicit_norm():
    x = hidden()
    m = mask_from([True] * 6)
    with pytest.raises(TypeError):
        add_vector(x, unit(8), 1.0, m)  # norm is keyword-only and mandatory


def test_norm_must_be_the_enum_not_a_string():
    x = hidden()
    m = mask_from([True] * 6)
    with pytest.raises(DirectionError, match="Norm enum"):
        project_out(x, unit(8), m, norm="assert_unit")


def test_float32_normalized_vector_upcast_to_float64_is_accepted():
    """The realistic case on this hardware, and it must not be rejected.

    MPS has no float64 (PREFLIGHT.md §1), so any float64 direction was
    normalized at float32 and carries ~1e-8 of norm error before it is stored.
    A unit-norm tolerance tight enough to reject that would reject every
    correctly-fitted bundle.
    """
    g = torch.Generator().manual_seed(0)
    v32 = torch.randn(16, generator=g, dtype=torch.float32)
    v32 = v32 / torch.linalg.vector_norm(v32)
    v64 = v32.to(torch.float64)
    assert abs(float(torch.linalg.vector_norm(v64)) - 1.0) > 1e-9  # genuinely off

    x = hidden(d_model=16, dtype=torch.float64)
    out = project_out(x, v64, mask_from([True] * 6), norm=Norm.ASSERT_UNIT)
    assert (out * v64).sum(-1).abs().max().item() < 1e-7


def test_direction_with_nan_is_rejected():
    x = hidden()
    m = mask_from([True] * 6)
    bad = unit(8).clone()
    bad[0] = float("nan")
    with pytest.raises(DirectionError, match="NaN or Inf"):
        project_out(x, bad, m, norm=Norm.NORMALIZE)


# --------------------------------------------------------------------------
# Malformed masks fail loudly
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "bad_mask, match",
    [
        (torch.ones(1, 6, dtype=torch.long), "dtype torch.bool"),
        (torch.ones(1, 6, dtype=torch.float32), "dtype torch.bool"),
        (torch.ones(1, 6, dtype=torch.uint8), "dtype torch.bool"),
        (torch.ones(6, dtype=torch.bool), r"\(batch, seq\)"),
        (torch.ones(1, 6, 8, dtype=torch.bool), r"\(batch, seq\)"),
        (torch.ones(1, 5, dtype=torch.bool), "does not match"),
        (torch.ones(2, 6, dtype=torch.bool), "does not match"),
    ],
)
def test_malformed_masks_raise(bad_mask, match):
    x = hidden()
    with pytest.raises(MaskError, match=match):
        project_out(x, unit(8), bad_mask)


def test_mask_is_not_broadcast_from_a_singleton_batch():
    """A (1, seq) mask against a (4, seq) hidden is a bug, not a broadcast."""
    x = hidden(batch=4)
    with pytest.raises(MaskError, match="Broadcasting is not attempted"):
        project_out(x, unit(8), torch.ones(1, 6, dtype=torch.bool))


def test_non_tensor_mask_raises():
    x = hidden()
    with pytest.raises(MaskError, match="must be a Tensor"):
        project_out(x, unit(8), [[True] * 6])


# --------------------------------------------------------------------------
# Shape checks
# --------------------------------------------------------------------------

def test_hidden_must_be_3d():
    with pytest.raises(ShapeError, match=r"\(batch, seq, d_model\)"):
        project_out(torch.randn(6, 8), unit(8), torch.ones(1, 6, dtype=torch.bool))


def test_direction_must_be_1d():
    x = hidden()
    with pytest.raises(ShapeError, match="must be 1-D"):
        project_out(x, torch.randn(1, 8), mask_from([True] * 6))


def test_direction_length_must_match_d_model():
    x = hidden(d_model=8)
    with pytest.raises(ShapeError, match="d_model"):
        project_out(x, unit(9), mask_from([True] * 6))


def test_integer_hidden_is_rejected():
    with pytest.raises(ShapeError, match="floating-point"):
        project_out(
            torch.ones(1, 6, 8, dtype=torch.long),
            unit(8),
            torch.ones(1, 6, dtype=torch.bool),
        )


def test_non_finite_alpha_is_rejected():
    x = hidden()
    m = mask_from([True] * 6)
    with pytest.raises(ValueError, match="finite"):
        add_vector(x, unit(8), float("inf"), m, norm=Norm.ASSERT_UNIT)
    with pytest.raises(ValueError, match="finite"):
        clamp_feature(x, unit(8), float("nan"), m)


# --------------------------------------------------------------------------
# add_vector and clamp_feature semantics
# --------------------------------------------------------------------------

def test_add_vector_alpha_zero_is_identity():
    x = hidden(dtype=torch.float64)
    m = mask_from([True] * 6)
    assert torch.equal(add_vector(x, unit(8, torch.float64), 0.0, m, norm=Norm.ASSERT_UNIT), x)


def test_add_vector_is_directional():
    """Sign matters, unlike projection. This is the whole point of the arm."""
    x = hidden(dtype=torch.float64)
    m = mask_from([True] * 6)
    v = unit(8, torch.float64)
    plus = add_vector(x, v, 1.0, m, norm=Norm.ASSERT_UNIT)
    minus = add_vector(x, v, -1.0, m, norm=Norm.ASSERT_UNIT)
    assert not torch.allclose(plus, minus)
    assert torch.allclose((plus + minus) / 2, x, atol=1e-12)


def test_add_vector_as_is_uses_raw_magnitude():
    x = hidden(dtype=torch.float64)
    m = mask_from([True] * 6)
    v = unit(8, torch.float64) * 5.0
    as_is = add_vector(x, v, 1.0, m, norm=Norm.AS_IS)
    normalized = add_vector(x, v, 1.0, m, norm=Norm.NORMALIZE)
    assert not torch.allclose(as_is, normalized)
    assert torch.allclose(as_is, add_vector(x, v / 5.0, 5.0, m, norm=Norm.ASSERT_UNIT), atol=1e-12)


def test_clamp_to_zero_equals_project_out():
    """The consistency property: clamping to zero *is* ablation."""
    x = hidden(seq=10, d_model=32, dtype=torch.float64)
    r = unit(32, torch.float64, seed=7)
    m = mask_from([True, False] * 5)
    assert torch.allclose(
        clamp_feature(x, r, 0.0, m), project_out(x, r, m), atol=1e-12
    )


def test_clamp_sets_the_coefficient():
    x = hidden(seq=10, d_model=32, dtype=torch.float64)
    r = unit(32, torch.float64, seed=11)
    m = mask_from([True] * 10)
    out = clamp_feature(x, r, 3.25, m)
    coeff = (out * r).sum(-1)
    assert torch.allclose(coeff, torch.full_like(coeff, 3.25), atol=1e-12)


def test_clamp_is_idempotent():
    x = hidden(seq=10, d_model=32, dtype=torch.float64)
    r = unit(32, torch.float64, seed=13)
    m = mask_from([True] * 10)
    once = clamp_feature(x, r, -1.5, m)
    twice = clamp_feature(once, r, -1.5, m)
    assert torch.allclose(once, twice, atol=1e-12)


# --------------------------------------------------------------------------
# Batch-size-one is the supported case; larger batches must not silently differ
# --------------------------------------------------------------------------

def test_batch_one_is_the_primary_case():
    x = hidden(batch=1, seq=12, d_model=32, dtype=torch.float64)
    r = unit(32, torch.float64)
    m = mask_from([True, False] * 6, batch=1)
    out = project_out(x, r, m)
    assert out.shape == x.shape


def test_batched_rows_match_the_same_rows_run_singly():
    """Phases 0-2 run at batch one. If a batch dimension is ever used, it must
    agree row-for-row with the batch-one path rather than quietly differing."""
    x = hidden(batch=3, seq=7, d_model=16, dtype=torch.float64)
    r = unit(16, torch.float64, seed=17)
    rows = torch.tensor(
        [[True, False, True, True, False, False, True],
         [False] * 7,
         [True] * 7],
        dtype=torch.bool,
    )
    batched = project_out(x, r, rows)
    for i in range(3):
        single = project_out(x[i : i + 1], r, rows[i : i + 1])
        assert torch.equal(batched[i : i + 1], single)


# --------------------------------------------------------------------------
# Property tests
# --------------------------------------------------------------------------

@settings(max_examples=40, deadline=None)
@given(
    d_model=st.integers(min_value=1, max_value=64),
    seq=st.integers(min_value=1, max_value=16),
    seed=st.integers(min_value=0, max_value=10_000),
    mask_seed=st.integers(min_value=0, max_value=10_000),
)
def test_property_outside_mask_always_bitwise_identical(d_model, seq, seed, mask_seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, seq, d_model, generator=g, dtype=torch.float64)
    r = unit(d_model, torch.float64, seed=seed)
    mg = torch.Generator().manual_seed(mask_seed)
    m = torch.rand(1, seq, generator=mg) < 0.5
    out = project_out(x, r, m)
    assert torch.equal(out[~m], x[~m])


@settings(max_examples=40, deadline=None)
@given(
    d_model=st.integers(min_value=2, max_value=64),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_property_projection_orthogonal_and_idempotent(d_model, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 5, d_model, generator=g, dtype=torch.float64)
    r = unit(d_model, torch.float64, seed=seed + 1)
    m = torch.ones(1, 5, dtype=torch.bool)
    once = project_out(x, r, m)
    twice = project_out(once, r, m)
    assert (once * r).sum(-1).abs().max().item() < 1e-10
    assert torch.allclose(once, twice, atol=1e-12)


@settings(max_examples=40, deadline=None)
@given(
    alpha=st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False),
    d_model=st.integers(min_value=1, max_value=32),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_property_add_then_subtract_round_trips(alpha, d_model, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 4, d_model, generator=g, dtype=torch.float64)
    v = unit(d_model, torch.float64, seed=seed)
    m = torch.ones(1, 4, dtype=torch.bool)
    there = add_vector(x, v, alpha, m, norm=Norm.ASSERT_UNIT)
    back = add_vector(there, v, -alpha, m, norm=Norm.ASSERT_UNIT)
    assert torch.allclose(back, x, atol=1e-10)


@settings(max_examples=40, deadline=None)
@given(
    value=st.floats(min_value=-20, max_value=20, allow_nan=False, allow_infinity=False),
    d_model=st.integers(min_value=2, max_value=32),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_property_clamp_sets_coefficient(value, d_model, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 4, d_model, generator=g, dtype=torch.float64)
    d = unit(d_model, torch.float64, seed=seed + 2)
    m = torch.ones(1, 4, dtype=torch.bool)
    out = clamp_feature(x, d, value, m)
    coeff = (out * d).sum(-1)
    assert torch.allclose(coeff, torch.full_like(coeff, value), atol=1e-9)


# --------------------------------------------------------------------------
# The real accelerator
#
# Model-free, but run on MPS where the experiment will actually execute.
# Device preservation and exact-identity-outside-the-mask are backend
# behaviours, not just arithmetic, and MPS is not the backend these primitives
# were written against.
# --------------------------------------------------------------------------

mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS not available"
)


@mps
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_mps_preserves_device_and_dtype(dtype):
    x = hidden(seq=12, d_model=64, dtype=dtype).to("mps")
    r = unit(64, dtype).to("mps")
    m = mask_from([True, False] * 6).to("mps")
    out = project_out(x, r, m)
    assert out.device.type == "mps"
    assert out.dtype is dtype


@mps
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mps_outside_mask_is_bitwise_identity(dtype):
    x = hidden(seq=12, d_model=64, dtype=dtype).to("mps")
    r = unit(64, dtype).to("mps")
    m = mask_from([True, False] * 6).to("mps")
    out = project_out(x, r, m)
    assert torch.equal(out[~m], x[~m])


@mps
def test_mps_projection_is_orthogonal():
    x = hidden(seq=12, d_model=64, dtype=torch.float32).to("mps")
    r = unit(64, torch.float32).to("mps")
    m = torch.ones(1, 12, dtype=torch.bool, device="mps")
    out = project_out(x, r, m)
    assert (out * r).sum(-1).abs().max().item() < ATOL[torch.float32]


@mps
def test_mps_and_cpu_agree_closely():
    """Not bitwise — different backends reorder reductions — but close."""
    x = hidden(seq=12, d_model=64, dtype=torch.float32)
    r = unit(64, torch.float32)
    m = mask_from([True, False] * 6)
    on_cpu = project_out(x, r, m)
    on_mps = project_out(x.to("mps"), r.to("mps"), m.to("mps")).cpu()
    assert torch.allclose(on_cpu, on_mps, atol=1e-5)


@mps
def test_mps_device_mismatch_fails_loudly():
    """A CPU direction against MPS activations is a bug, not a silent copy."""
    x = hidden(d_model=8).to("mps")
    m = mask_from([True] * 6).to("mps")
    with pytest.raises(DirectionError, match="is on"):
        project_out(x, unit(8), m)
    with pytest.raises(MaskError, match="is on"):
        project_out(x, unit(8).to("mps"), mask_from([True] * 6))
