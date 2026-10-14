"""Tests for the material definitions and their validation."""

import pytest

from tissue_lite.mechanics import (
    Boundary,
    ConcentrationSoftening,
    MaterialParameters,
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
        (None, (1.0, 2.0, 0.2, 0.3, True, False, 0, 0.0, 0.0)),
        (
            ConcentrationSoftening(41, 2.0, 3.0),
            (1.0, 2.0, 0.2, 0.3, True, True, 41, 2.0, 3.0),
        ),
    ],
)
def test_material_parameters_flatten(
    softening: ConcentrationSoftening | None, expected: tuple
) -> None:
    membrane = TRBSMembrane(1.0, 2.0, 0.2, 0.3, True, softening)
    assert MaterialParameters.from_membrane(membrane).as_tuple() == expected
