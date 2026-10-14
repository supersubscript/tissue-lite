"""Tests for strain-driven wall growth."""

import numpy as np
from conftest import CASES_DIR, case_state

from tissue_lite import layout


def test_growth_lengthens_rest_lengths_under_tension() -> None:
    state = case_state(CASES_DIR / "aniso_growth")
    tissue = state.tissue
    tissue.vertices *= 1.05
    state.triangulation.centres[:] *= 1.05
    spokes = state.triangulation.spokes.copy()
    walls = tissue.wall_lengths.copy()
    state.compute_derivatives(
        tissue.cell_variables, state.triangulation.centres, tissue.vertices
    )
    state.update(1e-3)
    assert (tissue.wall_lengths >= walls).all() or (
        state.triangulation.spokes >= spokes
    ).all()
    assert (tissue.cell_variables[:, layout.GROWTH_COUNT] >= 0).all()


def test_no_growth_without_strain() -> None:
    state = case_state(CASES_DIR / "aniso_growth")
    tissue = state.tissue
    walls = tissue.wall_lengths.copy()
    state.update(0.0)
    np.testing.assert_array_equal(tissue.wall_lengths, walls)
