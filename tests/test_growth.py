"""Tests for strain-driven wall growth."""

from dataclasses import replace

import numpy as np
import pytest
from conftest import CASES_DIR, case_state, read_case_tissue

from tissue_lite.model import ModelState


def test_growth_lengthens_rest_lengths_under_tension() -> None:
    state = case_state(CASES_DIR / "aniso_growth")
    tissue = state.tissue
    state.positions *= 1.05
    spokes = state.mesh.spokes.copy()
    walls = tissue.wall_lengths.copy()
    state.compute_forces()
    state.update(1e-3)
    assert (tissue.wall_lengths >= walls).all() or (state.mesh.spokes >= spokes).all()
    assert (tissue.cells.growth_count >= 0).all()


def test_no_growth_without_strain() -> None:
    state = case_state(CASES_DIR / "aniso_growth")
    tissue = state.tissue
    walls = tissue.wall_lengths.copy()
    state.update(0.0)
    np.testing.assert_array_equal(tissue.wall_lengths, walls)


@pytest.mark.parametrize(("velocity_threshold", "grows"), [(None, True), (0.0, False)])
def test_velocity_threshold_gates_growth(
    velocity_threshold: float | None, grows: bool
) -> None:
    model = case_state(CASES_DIR / "aniso_growth").model
    assert model.growth is not None
    gated = replace(
        model, growth=replace(model.growth, velocity_threshold=velocity_threshold)
    )
    state = ModelState(model=gated, tissue=read_case_tissue(CASES_DIR / "aniso_growth"))
    state.positions *= 1.05
    spokes = state.mesh.spokes.copy()
    state.compute_forces()
    state.update(1e-3)
    assert (not np.array_equal(state.mesh.spokes, spokes)) == grows
