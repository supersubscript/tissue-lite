"""Tests for fibre-stiffness and microtubule feedback."""

import numpy as np
import pytest

from tissue_lite.feedback import (
    Coupling,
    FibreStiffness,
    MicrotubuleAlignment,
    MicrotubuleKernel,
    initiate_fibres,
    initiate_microtubules,
    update_fibres,
    update_microtubules,
)
from tissue_lite.material import TRBSMembrane
from tissue_lite.tissue import CellState

YM, YF = 120.0, 120.0
MEMBRANE = TRBSMembrane(YM, YF)


def _cells(anisotropy: float, velocity: float = 0.0) -> CellState:
    cells = CellState.zeros(1)
    cells.stress_anisotropy[0] = anisotropy
    cells.velocity[0] = velocity
    cells.young_longitudinal[0] = YM
    return cells


@pytest.mark.parametrize(
    ("coupling", "anisotropy", "target"),
    [
        ("linear", 0.5, YM + 0.75 * YF),
        ("hill", 2 / 7, YM + 0.75 * YF),  # a / (1 - a) = k gives f = 1/2
        ("hill", 0.0, YM + 0.5 * YF),
    ],
)
def test_fibre_initiation_targets(
    coupling: Coupling, anisotropy: float, target: float
) -> None:
    cells = _cells(anisotropy)
    initiate_fibres(FibreStiffness(rate=1.0, coupling=coupling), MEMBRANE, cells)
    assert cells.young_longitudinal[0] == pytest.approx(target)


def test_fibre_update_relaxes_towards_target() -> None:
    cells = _cells(0.5)
    fibres = FibreStiffness(rate=1.0, coupling="linear").to_kernel(MEMBRANE)
    update_fibres(cells, 0.1, fibres)
    assert cells.young_longitudinal[0] == pytest.approx(YM + 0.1 * 0.75 * YF)


def test_fibre_update_is_gated_by_velocity() -> None:
    cells = _cells(0.5, velocity=500.0)
    fibres = FibreStiffness(rate=1.0, coupling="linear").to_kernel(MEMBRANE)
    update_fibres(cells, 0.1, fibres)
    assert cells.young_longitudinal[0] == YM


def test_hill_direct_sets_modulus_without_matrix() -> None:
    cells = _cells(2 / 7, velocity=500.0)
    fibres = FibreStiffness(rate=1.0, coupling="hill_direct").to_kernel(MEMBRANE)
    update_fibres(cells, 0.1, fibres)
    assert cells.young_longitudinal[0] == pytest.approx(0.75 * YF)


def test_microtubules_rotate_towards_stress_and_stay_unit() -> None:
    cells = CellState.zeros(1)
    cells.stress_direction[0] = (0.0, 1.0, 0.0)
    cells.microtubule[0] = (-1.0, 0.0, 0.0)
    update_microtubules(cells, 0.5, MicrotubuleKernel(1.0, 0, 300.0, 0.0))
    mt = cells.microtubule[0]
    np.testing.assert_allclose(np.linalg.norm(mt), 1.0)
    assert mt[1] > 0 and mt[0] < 0


def test_microtubules_initiate_from_stress() -> None:
    cells = CellState.zeros(2)
    cells.stress_direction[:] = (0.0, 0.0, 1.0)
    initiate_microtubules(cells)
    np.testing.assert_array_equal(cells.microtubule, cells.stress_direction)


def test_stress_threshold_needs_velocity_threshold() -> None:
    with pytest.raises(ValueError, match="velocity_threshold"):
        MicrotubuleAlignment(rate=1.0, velocity_threshold=None)


@pytest.mark.parametrize(
    ("alignment", "expected"),
    [
        (MicrotubuleAlignment(rate=1.0), MicrotubuleKernel(1.0, 2, 300.0, 0.0)),
        (
            MicrotubuleAlignment(rate=1.0, stress_difference_threshold=None),
            MicrotubuleKernel(1.0, 1, 300.0, 0.0),
        ),
        (
            MicrotubuleAlignment(
                rate=1.0, velocity_threshold=None, stress_difference_threshold=None
            ),
            MicrotubuleKernel(1.0, 0, 0.0, 0.0),
        ),
    ],
)
def test_microtubule_gate_levels(
    alignment: MicrotubuleAlignment, expected: MicrotubuleKernel
) -> None:
    assert alignment.to_kernel() == expected


def test_fibre_moduli_fall_back_to_membrane() -> None:
    membrane = TRBSMembrane(1.0, 2.0)
    assert FibreStiffness(rate=1.0).get_moduli(membrane) == (1.0, 2.0)
    assert FibreStiffness(rate=1.0, young_matrix=5.0).get_moduli(membrane) == (
        5.0,
        2.0,
    )
