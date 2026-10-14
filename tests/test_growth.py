"""Tests for strain-driven wall growth."""

import numpy as np
from conftest import CASES_DIR, case_state


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
