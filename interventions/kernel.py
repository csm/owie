"""Pure intervention primitives.

Three functions, no state, no model. Each takes a hidden-state tensor and
returns a **new** tensor; none mutates its input.

Design commitments, each traceable to a Checkpoint 1 requirement:

*No in-place mutation.* Every function returns a fresh tensor. The input is
never written to, including through views.

*Exact identity outside the mask.* Results are assembled with ``torch.where``
against the untouched input, so positions outside the mask are **bitwise**
identical to the input — not merely close. This is what makes a masked
intervention a clean experimental manipulation: an unmasked position cannot
drift, even by one ulp, and so cannot silently contribute to an effect size.

*Explicit normalization handling.* There is no default. Every call states how
its direction is to be treated via ``norm=``, and a direction that fails the
declared contract raises rather than being quietly fixed up. Projection onto a
non-unit vector is silently wrong rather than loudly wrong, which is precisely
why it is not allowed to happen by omission.

*Precision.* Low-precision inputs (bf16/fp16) are upcast to float32 for the
arithmetic and cast back on return, so the caller's dtype is preserved while
the projection is not computed at bf16's ~3 significant digits. Note that
float32 is the ceiling on Apple Metal — MPS has no float64 (PREFLIGHT.md §1) —
so tolerances in tests are set against float32, not double.
"""

from __future__ import annotations

import enum

import torch
from torch import Tensor

__all__ = [
    "Norm",
    "InterventionError",
    "ShapeError",
    "MaskError",
    "DirectionError",
    "project_out",
    "add_vector",
    "clamp_feature",
]


class InterventionError(ValueError):
    """Base class for every malformed-input failure in this module."""


class ShapeError(InterventionError):
    """Tensor rank or shape does not match the declared contract."""


class MaskError(InterventionError):
    """Span mask is malformed: wrong dtype, rank, shape, or device."""


class DirectionError(InterventionError):
    """Direction vector is malformed, or violates its declared normalization."""


class Norm(enum.Enum):
    """How a caller's direction vector is to be treated.

    There is no default and no inference. The caller declares, and a vector
    that does not honour the declaration raises.
    """

    ASSERT_UNIT = "assert_unit"
    """Vector is already unit-norm. Verify it, and raise if it is not."""

    NORMALIZE = "normalize"
    """Vector has arbitrary norm. Divide by it, and raise if it is ~zero."""

    AS_IS = "as_is"
    """Use the raw vector including its magnitude.

    Only meaningful where magnitude carries meaning, i.e. ``add_vector``.
    ``project_out`` and ``clamp_feature`` reject it, because projection onto a
    non-unit vector is not projection.
    """


# Tolerance for the ASSERT_UNIT check, keyed to the dtype the vector is stored
# in. This check exists to catch a *declaration* mismatch — a manifest claiming
# normalization="unit" over a raw difference-in-means vector, whose norm is
# nowhere near 1 — not to audit floating-point precision. It is therefore
# deliberately generous.
#
# float64 gets 1e-6 rather than something near double epsilon because on this
# hardware a float64 direction cannot have been normalized at float64: MPS has
# no float64 at all (PREFLIGHT.md §1), so the division happened at float32 and
# the stored vector carries ~1e-8 of norm error before it is ever written. A
# tighter bound would reject correctly-fitted bundles.
_UNIT_TOL: dict[torch.dtype, float] = {
    torch.bfloat16: 1e-2,
    torch.float16: 1e-2,
    torch.float32: 1e-5,
    torch.float64: 1e-6,
}


def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
    """Arithmetic dtype for a given input dtype. Upcasts low precision only."""
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _check_hidden(hidden: Tensor) -> None:
    if not isinstance(hidden, Tensor):
        raise ShapeError(f"hidden must be a Tensor, got {type(hidden).__name__}")
    if not hidden.is_floating_point():
        raise ShapeError(f"hidden must be a floating-point tensor, got dtype {hidden.dtype}")
    if hidden.ndim != 3:
        raise ShapeError(
            "hidden must have shape (batch, seq, d_model); "
            f"got {hidden.ndim}-D tensor of shape {tuple(hidden.shape)}"
        )
    if hidden.shape[-1] == 0:
        raise ShapeError("hidden has d_model == 0")


def _check_mask(span_mask: Tensor, hidden: Tensor) -> None:
    """Validate the span mask. Malformed masks fail loudly, never silently.

    A mask is the entire experimental manipulation: it decides which token
    positions are inside the intervention and which are untouched. A mask that
    is quietly coerced — an int tensor treated as truthy, a broadcastable-but-
    wrong shape stretched to fit — would change what the experiment measures
    without changing what it reports. Hence: exact dtype, exact rank, exact
    shape, same device. No coercion.
    """
    if not isinstance(span_mask, Tensor):
        raise MaskError(f"span_mask must be a Tensor, got {type(span_mask).__name__}")
    if span_mask.dtype is not torch.bool:
        raise MaskError(
            f"span_mask must have dtype torch.bool, got {span_mask.dtype}. "
            "Integer and float masks are rejected rather than coerced: a mask "
            "of 0/1 floats is ambiguous about what a 0.5 means."
        )
    if span_mask.ndim != 2:
        raise MaskError(
            "span_mask must have shape (batch, seq); "
            f"got {span_mask.ndim}-D tensor of shape {tuple(span_mask.shape)}"
        )
    expected = tuple(hidden.shape[:2])
    if tuple(span_mask.shape) != expected:
        raise MaskError(
            f"span_mask shape {tuple(span_mask.shape)} does not match hidden's "
            f"(batch, seq) = {expected}. Broadcasting is not attempted."
        )
    if span_mask.device != hidden.device:
        raise MaskError(
            f"span_mask is on {span_mask.device} but hidden is on {hidden.device}"
        )


def _prepare_direction(
    direction: Tensor,
    hidden: Tensor,
    norm: Norm,
    *,
    allow_as_is: bool,
    argname: str,
) -> Tensor:
    """Validate a direction and return it in the compute dtype.

    Returns a vector ready to use: unit-norm for ASSERT_UNIT and NORMALIZE,
    raw for AS_IS.
    """
    if not isinstance(norm, Norm):
        raise DirectionError(
            f"norm must be a Norm enum member, got {type(norm).__name__}. "
            "Normalization handling is explicit by design; there is no default."
        )
    if not isinstance(direction, Tensor):
        raise DirectionError(f"{argname} must be a Tensor, got {type(direction).__name__}")
    if not direction.is_floating_point():
        raise DirectionError(f"{argname} must be floating-point, got dtype {direction.dtype}")
    if direction.ndim != 1:
        raise ShapeError(
            f"{argname} must be 1-D of shape (d_model,); "
            f"got {direction.ndim}-D tensor of shape {tuple(direction.shape)}"
        )
    d_model = hidden.shape[-1]
    if direction.shape[0] != d_model:
        raise ShapeError(
            f"{argname} has length {direction.shape[0]} but hidden has "
            f"d_model = {d_model}"
        )
    if direction.device != hidden.device:
        raise DirectionError(
            f"{argname} is on {direction.device} but hidden is on {hidden.device}"
        )
    if norm is Norm.AS_IS and not allow_as_is:
        raise DirectionError(
            "Norm.AS_IS is not valid here: projecting onto a non-unit vector is "
            "not projection. Use Norm.ASSERT_UNIT or Norm.NORMALIZE."
        )

    compute_dtype = _compute_dtype(hidden.dtype)
    vec = direction.to(compute_dtype)

    if not torch.isfinite(vec).all():
        raise DirectionError(f"{argname} contains NaN or Inf")

    if norm is Norm.AS_IS:
        return vec

    length = torch.linalg.vector_norm(vec)
    if norm is Norm.NORMALIZE:
        # Guard against a direction that is numerically zero. Dividing by it
        # would produce NaN or Inf and poison every masked position.
        if float(length) <= torch.finfo(compute_dtype).tiny:
            raise DirectionError(
                f"{argname} has norm {float(length)!r}, too small to normalize"
            )
        return vec / length

    # ASSERT_UNIT: verify rather than fix. A caller that declares a unit vector
    # and supplies something else has a bug upstream — most likely a bundle
    # whose manifest says normalization="unit" but whose stored vector is raw.
    tol = _UNIT_TOL.get(direction.dtype, 1e-5)
    if abs(float(length) - 1.0) > tol:
        raise DirectionError(
            f"{argname} declared Norm.ASSERT_UNIT but has norm {float(length)!r} "
            f"(tolerance {tol} for stored dtype {direction.dtype}). "
            "Use Norm.NORMALIZE to normalize it explicitly."
        )
    return vec


def _check_scalar(value: float | Tensor, argname: str) -> float:
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise InterventionError(
                f"{argname} must be a scalar; got tensor of shape {tuple(value.shape)}"
            )
        value = float(value.item())
    else:
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise InterventionError(f"{argname} must be a real number: {exc}") from exc
    if value != value or value in (float("inf"), float("-inf")):
        raise InterventionError(f"{argname} must be finite, got {value!r}")
    return value


def _assemble(hidden: Tensor, modified: Tensor, span_mask: Tensor) -> Tensor:
    """Combine modified and original states, exact outside the mask.

    ``modified`` is computed for every position and then discarded outside the
    mask. That is deliberately wasteful and deliberately simple: it makes
    "outside the mask is untouched" a property of the assembly step rather than
    of the arithmetic, so it cannot be broken by a change to the arithmetic.
    At batch one the waste is irrelevant.
    """
    out = torch.where(span_mask.unsqueeze(-1), modified.to(hidden.dtype), hidden)
    # torch.where allocates; the input is never written through.
    assert out.dtype == hidden.dtype
    assert out.device == hidden.device
    return out


def project_out(
    hidden: Tensor,
    direction: Tensor,
    span_mask: Tensor,
    *,
    norm: Norm = Norm.ASSERT_UNIT,
) -> Tensor:
    r"""Remove the component along ``direction`` at masked positions.

    .. math:: x' = x - \hat r (\hat r^\top x)

    The primary primitive of the project (settled decision 5). Note that it is
    **sign-agnostic**: it removes the component along the direction regardless
    of sign, so anything sharing that direction goes with it. That is a
    property of the operator, not a defect, and it is why additive steering is
    retained as a qualitatively different comparison arm (PREFLIGHT.md §8).

    Args:
        hidden: (batch, seq, d_model) floating-point states.
        direction: (d_model,) direction to remove.
        span_mask: (batch, seq) bool. True positions are modified.
        norm: How to treat ``direction``. ``AS_IS`` is rejected.

    Returns:
        A new tensor, same dtype and device as ``hidden``, bitwise identical to
        ``hidden`` at every False position of the mask.
    """
    _check_hidden(hidden)
    _check_mask(span_mask, hidden)
    r = _prepare_direction(
        direction, hidden, norm, allow_as_is=False, argname="direction"
    )

    x = hidden.to(_compute_dtype(hidden.dtype))
    coeff = (x * r).sum(dim=-1, keepdim=True)
    modified = x - coeff * r
    return _assemble(hidden, modified, span_mask)


def add_vector(
    hidden: Tensor,
    vector: Tensor,
    alpha: float | Tensor,
    span_mask: Tensor,
    *,
    norm: Norm,
) -> Tensor:
    r"""Add ``alpha * vector`` at masked positions.

    .. math:: x' = x + \alpha v

    The comparison arm (settled decision 6), not the default. Unlike
    ``project_out`` this is directional: the sign of ``alpha`` matters, and
    steering "away from" a behaviour is a different operation from removing the
    subspace that carries it.

    ``norm`` has no default here because the meaning of ``alpha`` depends
    entirely on it: against a unit vector, ``alpha`` is a magnitude in the units
    of the residual stream; against a raw vector it is a dimensionless
    multiplier of whatever norm that vector happens to have. An alpha sweep is
    only interpretable if which of the two is in force was chosen, not
    defaulted.

    Args:
        hidden: (batch, seq, d_model) floating-point states.
        vector: (d_model,) steering vector.
        alpha: Scalar coefficient. Must be finite.
        span_mask: (batch, seq) bool. True positions are modified.
        norm: How to treat ``vector``. All three modes are valid.

    Returns:
        A new tensor, same dtype and device as ``hidden``, bitwise identical to
        ``hidden`` at every False position of the mask.
    """
    _check_hidden(hidden)
    _check_mask(span_mask, hidden)
    v = _prepare_direction(vector, hidden, norm, allow_as_is=True, argname="vector")
    a = _check_scalar(alpha, "alpha")

    x = hidden.to(_compute_dtype(hidden.dtype))
    modified = x + a * v
    return _assemble(hidden, modified, span_mask)


def clamp_feature(
    hidden: Tensor,
    decoder_direction: Tensor,
    value: float | Tensor,
    span_mask: Tensor,
    *,
    norm: Norm = Norm.ASSERT_UNIT,
) -> Tensor:
    r"""Set the coefficient along ``decoder_direction`` to ``value``.

    .. math:: x' = x - (\hat d^\top x)\hat d + v\,\hat d

    At ``value == 0`` this is exactly ``project_out``, which is the useful
    consistency property: clamping to zero *is* ablation.

    .. warning::
       This is **not** a full SAE feature clamp, and must not be reported as
       one without qualification. A real clamp reads the feature's activation
       through the SAE *encoder* (with its bias, and its nonlinearity), then
       rewrites the decoder contribution. With only the decoder direction
       available, the current coefficient is read by projection instead, which
       coincides with the true feature activation only when the decoder
       dictionary is orthonormal — which SAE dictionaries are not.

       The signature is fixed by the Checkpoint 1 specification, so this
       implements that signature faithfully rather than inventing a wider one.
       The gap is recorded as an open item (``DECISIONS.md`` D14): the
       Checkpoint 2 SAE arm needs encoder-based activation reading, and that
       needs a signature this checkpoint does not have.

    Args:
        hidden: (batch, seq, d_model) floating-point states.
        decoder_direction: (d_model,) decoder direction for one feature.
        value: Target coefficient along the direction. Must be finite.
        span_mask: (batch, seq) bool. True positions are modified.
        norm: How to treat ``decoder_direction``. ``AS_IS`` is rejected,
            because ``value`` would otherwise be in units of an arbitrary norm.

    Returns:
        A new tensor, same dtype and device as ``hidden``, bitwise identical to
        ``hidden`` at every False position of the mask.
    """
    _check_hidden(hidden)
    _check_mask(span_mask, hidden)
    d = _prepare_direction(
        decoder_direction, hidden, norm, allow_as_is=False, argname="decoder_direction"
    )
    v = _check_scalar(value, "value")

    x = hidden.to(_compute_dtype(hidden.dtype))
    coeff = (x * d).sum(dim=-1, keepdim=True)
    modified = x + (v - coeff) * d
    return _assemble(hidden, modified, span_mask)
