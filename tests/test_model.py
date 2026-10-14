"""Tests for the model definition, TOML IO and the per-call evaluation."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from conftest import CASES_DIR, MESHES_DIR, MODELS, case_state, read_case_tissue

from tissue_lite.legacy.init_file import read_init
from tissue_lite.legacy.model_file import read_model_file
from tissue_lite.material import Boundary, Pressure, TRBSMembrane
from tissue_lite.model import MeristemModel, ModelState


@pytest.mark.parametrize("path", MODELS, ids=lambda p: p.parent.name)
def test_toml_round_trip(path: Path, tmp_path: Path) -> None:
    model = read_model_file(path).model
    toml = tmp_path / "model.toml"
    model.to_toml(toml)
    assert MeristemModel.from_toml(toml) == model


def _rest_state(model: MeristemModel) -> ModelState:
    tissue = read_init(MESHES_DIR / "patch19.init")
    ends = tissue.vertices[tissue.topology.wall_vertices]
    tissue.wall_lengths = np.linalg.norm(ends[:, 0] - ends[:, 1], axis=1)
    return ModelState(model=model, tissue=tissue)


def test_no_forces_at_rest_without_pressure() -> None:
    model = MeristemModel(
        membrane=TRBSMembrane(120.0, 0.0),
        pressure=Pressure(0.0, 0.001),
        boundary=Boundary(fixed_axes=()),
    )
    state = _rest_state(model)
    np.testing.assert_allclose(state.compute_forces(), 0.0, atol=1e-10)


def test_internal_forces_balance() -> None:
    model = MeristemModel(
        membrane=TRBSMembrane(120.0, 0.0),
        pressure=Pressure(0.0, 0.001),
        boundary=Boundary(fixed_axes=()),
    )
    state = _rest_state(model)
    derivatives = state.compute_forces_at(state.tissue.cells, state.positions * 1.05)
    assert np.abs(derivatives).max() > 1e-3
    np.testing.assert_allclose(derivatives.sum(0), 0.0, atol=1e-9)


@pytest.mark.parametrize("axes", [(0, 1, 2), (2,)])
def test_boundary_vertices_do_not_move_along_fixed_axes(axes: tuple[int, ...]) -> None:
    model = MeristemModel(
        membrane=TRBSMembrane(120.0, 0.0),
        pressure=Pressure(0.01, 0.001),
        boundary=Boundary(fixed_axes=axes),
    )
    state = _rest_state(model)
    state.update(1.0)
    vd = state.compute_forces()[: state.tissue.topology.num_vertices]
    edge = state.tissue.topology.boundary
    np.testing.assert_array_equal(vd[edge][:, list(axes)], 0.0)
    assert np.abs(vd[~edge]).max() > 0


def test_pressure_ramps_to_full() -> None:
    model = MeristemModel(
        membrane=TRBSMembrane(120.0, 0.0), pressure=Pressure(0.01, ramp_time=0.5)
    )
    state = _rest_state(model)
    assert state.pressure_factor == 0.0
    state.update(0.25)
    assert state.pressure_factor == pytest.approx(0.5)
    state.update(1.0)
    assert state.pressure_factor == 1.0


def test_softening_scales_forces() -> None:
    case = CASES_DIR / "auxin_softening_hill"
    model = read_model_file(case / "model.model").model
    tissue = read_case_tissue(case)

    def max_force(concentration: float) -> float:
        t = tissue.copy()
        t.cells.concentration[:] = concentration
        t.vertices *= 1.02
        state = ModelState(model=model, tissue=t)
        return float(np.abs(state.compute_forces()).max())

    # Hill repression 1 - c^2 / (4 + c^2): 1 at c = 0, 0.5 at c = K = 2.
    assert max_force(2.0) == pytest.approx(0.5 * max_force(0.0), rel=1e-6)


def test_cell_stress_after_update() -> None:
    state = case_state(CASES_DIR / "sweep_120_120")
    state.compute_forces()
    state.update(1e-3)
    cells = state.tissue.cells
    n = state.tissue.topology.num_cells
    assert cells.stress_tensor.shape == (n, 3, 3)
    np.testing.assert_allclose(
        cells.stress_tensor, np.swapaxes(cells.stress_tensor, 1, 2), atol=1e-12
    )
    np.testing.assert_allclose(np.linalg.norm(cells.normal, axis=1), 1.0, atol=1e-9)
    np.testing.assert_allclose(
        np.linalg.norm(cells.stress_direction, axis=1), 1.0, atol=1e-9
    )
    assert (cells.rest_area > 0).all()


def test_membrane_fidelity_keeps_regular_forces() -> None:
    state = case_state(CASES_DIR / "sweep_120_120")
    model = replace(state.model, membrane=replace(state.model.membrane, fidelity=False))
    fixed = ModelState(
        model=model, tissue=read_case_tissue(CASES_DIR / "sweep_120_120")
    )
    # No sliver or vanishing fibre in the case, so the forces are unchanged.
    np.testing.assert_array_equal(fixed.compute_forces(), state.compute_forces())
    fixed.update(1e-3)
    membrane = model.membrane
    cells = fixed.tissue.cells
    np.testing.assert_allclose(  # Y_T, not the anisotropic energy (T7)
        cells.young_transverse,
        2 * membrane.young_matrix + membrane.young_fibre - cells.young_longitudinal,
    )


def test_pressure_fidelity_sets_nodal_share() -> None:
    def total_force(fidelity: bool) -> np.ndarray:
        model = MeristemModel(
            membrane=TRBSMembrane(0.0, 0.0),
            pressure=Pressure(1.0, 1e-3, fidelity=fidelity),
            boundary=Boundary(fixed_axes=()),
        )
        state = _rest_state(model)
        state.update(1.0)
        return state.compute_forces().sum(0)

    # P A / 3 per node adds up to P A per triangle; Tissue's P A / 2 to 1.5 P A (P1).
    np.testing.assert_allclose(total_force(False), total_force(True) * 2 / 3)


@pytest.mark.parametrize(("fidelity", "balanced"), [(True, False), (False, True)])
def test_sliver_fix_breaks_force_balance(fidelity: bool, balanced: bool) -> None:
    model = MeristemModel(
        membrane=TRBSMembrane(120.0, 0.0, fidelity=fidelity),
        pressure=Pressure(0.0, 0.001),
        boundary=Boundary(fixed_axes=()),
    )
    state = _rest_state(model)
    # The last triangle, so that the fix adds forces accumulated before it.
    spokes, wall = state.mesh.spokes, state.mesh.wall_lengths[state.mesh.wall[-1]]
    spokes[-1, 1] = spokes[-1, 0] + wall * (1 - 1e-9)  # rest angles near 0° and 180°
    derivatives = state.compute_forces_at(state.tissue.cells, state.positions * 1.05)
    scale = np.abs(derivatives).max()
    assert (np.abs(derivatives.sum(0)).max() < 1e-9 * scale) == balanced
