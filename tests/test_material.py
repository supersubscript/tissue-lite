"""Tests for the material definitions and their validation."""

import numpy as np
import pytest

from tissue_lite.material import (
    Boundary,
    ConcentrationSoftening,
    Material,
    Pressure,
    TRBSMembrane,
    compute_fibre_direction,
)
from tissue_lite.rand import make_rand_state


@pytest.mark.parametrize("nu", [-0.1, 0.5, 0.7])
def test_membrane_rejects_poisson_ratios(nu: float) -> None:
    with pytest.raises(ValueError, match="Poisson"):
        TRBSMembrane(1.0, 1.0, poisson_longitudinal=nu)


@pytest.mark.parametrize("ramp", [0.0, -1.0])
def test_pressure_rejects_ramp_time(ramp: float) -> None:
    with pytest.raises(ValueError, match="ramp_time"):
        Pressure(0.01, ramp)


def test_boundary_rejects_axes() -> None:
    with pytest.raises(ValueError, match="fixed_axes"):
        Boundary(fixed_axes=(3,))


@pytest.mark.parametrize(
    ("softening", "expected"),
    [
        (None, Material(1.0, 2.0, 0.2, 0.3, False, 0.0, 0.0)),
        (
            ConcentrationSoftening(2.0, 3.0),
            Material(1.0, 2.0, 0.2, 0.3, True, 2.0, 3.0),
        ),
    ],
)
def test_membrane_flattens_for_kernels(
    softening: ConcentrationSoftening | None, expected: Material
) -> None:
    membrane = TRBSMembrane(1.0, 2.0, 0.2, 0.3, softening)
    assert membrane.to_kernel() == expected


@pytest.mark.parametrize(
    ("microtubule", "fidelity", "expected"),
    [
        ((0.0, 0.0, 1.0), True, None),
        ((0.0, 0.0, 1.0), False, (0.0, 0.0)),
        # In plane above 1e-4, but compressed below it by F = I / 2 (T2).
        ((1.5e-4, 0.0, 1.0), True, None),
        ((1.5e-4, 0.0, 1.0), False, (1.0, 0.0)),
    ],
)
def test_normal_microtubule_fallback(
    microtubule: tuple[float, float, float],
    fidelity: bool,
    expected: tuple[float, float] | None,
) -> None:
    rand_state = make_rand_state()
    before = rand_state.copy()
    half = ((0.5, 0.0), (0.0, 0.5))
    fibre = compute_fibre_direction(
        half, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), microtubule, rand_state, fidelity
    )
    if expected is None:  # Tissue draws a random direction (T1)
        assert np.hypot(*fibre) == pytest.approx(1.0)
        assert not np.array_equal(rand_state, before)
    else:
        assert fibre == expected
        np.testing.assert_array_equal(rand_state, before)
