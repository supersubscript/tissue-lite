"""Tests for the material definitions and their validation."""

import pytest

from tissue_lite.material import (
    Boundary,
    ConcentrationSoftening,
    Material,
    Pressure,
    TRBSMembrane,
)


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
