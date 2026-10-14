"""Tests for fibre-stiffness and microtubule feedback."""

import numpy as np
import pytest

from tissue_lite import layout
from tissue_lite.feedback import (
    COUPLING_CODES,
    Coupling,
    FibreStiffness,
    MicrotubuleAlignment,
    initiate_fibres,
    initiate_microtubules,
    update_fibres,
    update_microtubules,
)

YM, YF = 120.0, 120.0


def _cells(anisotropy: float, velocity: float = 0.0) -> np.ndarray:
    v = np.zeros((1, 35))
    v[0, layout.STRESS_ANISOTROPY] = anisotropy
    v[0, layout.VELOCITY] = velocity
    v[0, layout.YOUNG_LONGITUDINAL] = YM
    return v


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
    v = _cells(anisotropy)
    initiate_fibres(FibreStiffness(rate=1.0, coupling=coupling), YM, YF, v)
    assert v[0, layout.YOUNG_LONGITUDINAL] == pytest.approx(target)


def test_fibre_update_relaxes_towards_target() -> None:
    v = _cells(0.5)
    update_fibres(v, 0.1, 1.0, COUPLING_CODES["linear"], 0.4, 2.0, 300.0, YM, YF)
    assert v[0, layout.YOUNG_LONGITUDINAL] == pytest.approx(YM + 0.1 * 0.75 * YF)


def test_fibre_update_is_gated_by_velocity() -> None:
    v = _cells(0.5, velocity=500.0)
    update_fibres(v, 0.1, 1.0, COUPLING_CODES["linear"], 0.4, 2.0, 300.0, YM, YF)
    assert v[0, layout.YOUNG_LONGITUDINAL] == YM


def test_hill_direct_sets_modulus_without_matrix() -> None:
    v = _cells(2 / 7, velocity=500.0)
    update_fibres(v, 0.1, 1.0, COUPLING_CODES["hill_direct"], 0.4, 2.0, 300.0, YM, YF)
    assert v[0, layout.YOUNG_LONGITUDINAL] == pytest.approx(0.75 * YF)


def test_microtubules_rotate_towards_stress_and_stay_unit() -> None:
    v = np.zeros((1, 35))
    v[0, layout.STRESS_DIRECTION] = (0.0, 1.0, 0.0)
    v[0, layout.MICROTUBULE] = (-1.0, 0.0, 0.0)
    update_microtubules(v, 0.5, 1.0, 0, 300.0, 0.0)
    mt = v[0, layout.MICROTUBULE]
    np.testing.assert_allclose(np.linalg.norm(mt), 1.0)
    assert mt[1] > 0 and mt[0] < 0


def test_microtubules_initiate_from_stress() -> None:
    v = np.zeros((2, 35))
    v[:, layout.STRESS_DIRECTION] = (0.0, 0.0, 1.0)
    initiate_microtubules(v)
    np.testing.assert_array_equal(
        v[:, layout.MICROTUBULE], v[:, layout.STRESS_DIRECTION]
    )


def test_stress_threshold_needs_velocity_threshold() -> None:
    with pytest.raises(ValueError, match="velocity_threshold"):
        MicrotubuleAlignment(rate=1.0, velocity_threshold=None)


def test_fibre_moduli_fall_back_to_membrane() -> None:
    assert FibreStiffness(rate=1.0).get_moduli(1.0, 2.0) == (1.0, 2.0)
    assert FibreStiffness(rate=1.0, young_matrix=5.0).get_moduli(1.0, 2.0) == (5.0, 2.0)
