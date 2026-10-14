"""Tests for the model definition, TOML IO and the per-call evaluation."""

from pathlib import Path

import numpy as np
import pytest
from conftest import CASES_DIR, MESHES_DIR, MODELS, case_state

from tissue_lite import layout
from tissue_lite.growth import StrainGrowth
from tissue_lite.legacy import read_model_file
from tissue_lite.mechanics import Boundary, Pressure, TRBSMembrane
from tissue_lite.model import MeristemModel, ModelState
from tissue_lite.tissue import Tissue


@pytest.mark.parametrize("path", MODELS, ids=lambda p: p.parent.name)
def test_toml_round_trip(path: Path, tmp_path: Path) -> None:
    model = read_model_file(path)
    toml = tmp_path / "model.toml"
    model.to_toml(toml)
    assert MeristemModel.from_toml(toml) == model


def test_growth_needs_double_rest_lengths() -> None:
    with pytest.raises(ValueError, match="double_rest_lengths"):
        MeristemModel(
            membrane=TRBSMembrane(50.0, 0.0),
            pressure=Pressure(0.01, 0.001),
            growth=StrainGrowth(rate=1.0, threshold=0.0),
        )


def _rest_state(model: MeristemModel) -> ModelState:
    tissue = Tissue.read_init(MESHES_DIR / "patch19.init")
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
    vd, cd = state.compute_derivatives(
        state.tissue.cell_variables, state.triangulation.centres, state.tissue.vertices
    )
    np.testing.assert_allclose(vd, 0.0, atol=1e-10)
    np.testing.assert_allclose(cd, 0.0, atol=1e-10)


def test_internal_forces_balance() -> None:
    model = MeristemModel(
        membrane=TRBSMembrane(120.0, 0.0),
        pressure=Pressure(0.0, 0.001),
        boundary=Boundary(fixed_axes=()),
    )
    state = _rest_state(model)
    stretched = state.tissue.vertices * 1.05
    vd, cd = state.compute_derivatives(
        state.tissue.cell_variables, state.triangulation.centres * 1.05, stretched
    )
    assert np.abs(vd).max() > 1e-3
    np.testing.assert_allclose(vd.sum(0) + cd.sum(0), 0.0, atol=1e-9)


@pytest.mark.parametrize("axes", [(0, 1, 2), (2,)])
def test_boundary_vertices_do_not_move_along_fixed_axes(axes: tuple[int, ...]) -> None:
    model = MeristemModel(
        membrane=TRBSMembrane(120.0, 0.0),
        pressure=Pressure(0.01, 0.001),
        boundary=Boundary(fixed_axes=axes),
    )
    state = _rest_state(model)
    state.update(1.0)
    vd, _ = state.compute_derivatives(
        state.tissue.cell_variables, state.triangulation.centres, state.tissue.vertices
    )
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
    model = read_model_file(case / "model.model")
    tissue = Tissue.read_init(case / "input.init")

    def max_force(concentration: float) -> float:
        t = tissue.copy()
        t.cell_variables[:, 41] = concentration
        t.vertices *= 1.02
        state = ModelState(model=model, tissue=t)
        vd, _ = state.compute_derivatives(
            t.cell_variables, state.triangulation.centres, t.vertices
        )
        return float(np.abs(vd).max())

    # Hill repression 1 - c^2 / (4 + c^2): 1 at c = 0, 0.5 at c = K = 2.
    assert max_force(2.0) == pytest.approx(0.5 * max_force(0.0), rel=1e-6)


def test_cell_mechanics_after_update() -> None:
    state = case_state(CASES_DIR / "sweep_120_120")
    state.compute_derivatives(
        state.tissue.cell_variables, state.triangulation.centres, state.tissue.vertices
    )
    state.update(1e-3)
    mech = state.get_cell_mechanics()
    n = state.tissue.topology.num_cells
    assert mech.stress_tensor.shape == (n, 3, 3)
    np.testing.assert_allclose(
        mech.stress_tensor, np.swapaxes(mech.stress_tensor, 1, 2), atol=1e-12
    )
    np.testing.assert_allclose(
        mech.stress_max, state.tissue.cell_variables[:, layout.STRESS_MAX]
    )
    np.testing.assert_allclose(np.linalg.norm(mech.normal, axis=1), 1.0, atol=1e-9)
