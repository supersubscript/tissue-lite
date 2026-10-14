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


def test_velocity_threshold_gates_growth_per_cell() -> None:
    case = CASES_DIR / "aniso_growth"
    model = case_state(case).model
    assert model.growth is not None
    state = ModelState(
        model=replace(model, track_velocity=True), tissue=read_case_tissue(case)
    )
    state.positions *= 1.05
    state.compute_forces()
    velocity = state.tissue.cells.velocity.copy()
    threshold = float(np.median(velocity))
    gated = replace(
        model,
        track_velocity=True,
        growth=replace(model.growth, velocity_threshold=threshold),
    )
    state = ModelState(model=gated, tissue=read_case_tissue(case))
    state.positions *= 1.05
    spokes = state.mesh.spokes.copy()
    state.compute_forces()
    state.update(1e-3)
    changed = (state.mesh.spokes != spokes).any(axis=1)
    cells = state.tissue.topology.triangle_cells
    grew = np.bincount(cells, weights=changed, minlength=len(velocity)) > 0
    assert grew.any()
    assert not grew[velocity >= threshold].any()


def test_growth_gate_needs_tracked_velocity() -> None:
    model = case_state(CASES_DIR / "aniso_growth").model
    assert model.growth is not None
    with pytest.raises(ValueError, match="track_velocity"):
        replace(model, growth=replace(model.growth, velocity_threshold=1.0))
