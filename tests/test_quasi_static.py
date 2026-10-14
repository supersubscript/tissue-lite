"""Tests for the quasi-static solver."""

import numpy as np
import pytest
from conftest import CASES_DIR, case_state, read_case_model, read_case_tissue

from tissue_lite.solvers.quasi_static import QuasiStatic
from tissue_lite.solvers.simulate import integrate, simulate


def test_step_reaches_equilibrium_on_a_fixed_boundary() -> None:
    state = case_state(CASES_DIR / "aniso")
    run = QuasiStatic(force_tolerance=1e-8).start(state)
    result = run.step(0.0, 1.0, state.compute_forces())
    assert result.t == 1.0 and result.accepted
    assert state.pressure_factor == 1.0
    assert np.abs(state.compute_forces().ravel()[run.free]).max() <= 1e-8
    assert run.iterations > 0


def test_forces_end_as_a_rigid_drift_on_a_z_fixed_boundary() -> None:
    state = case_state(CASES_DIR / "sweep_pre")
    run = QuasiStatic(force_tolerance=1e-8).start(state)
    run.step(0.0, 1.0, state.compute_forces())
    forces = state.compute_forces()
    centred = state.positions - state.positions.mean(axis=0)
    rigid = np.stack(
        [np.tile([1.0, 0.0, 0.0], len(forces)), np.tile([0.0, 1.0, 0.0], len(forces))]
        + [np.cross([0.0, 0.0, 1.0], centred).ravel()],
        axis=1,
    )[run.free]
    drift, *_ = np.linalg.lstsq(rigid, forces.ravel()[run.free], rcond=None)
    assert np.abs(forces.ravel()[run.free]).max() > 1e-4  # the drift itself
    assert np.abs(forces.ravel()[run.free] - rigid @ drift).max() <= 1e-8


def test_fixed_boundary_components_stay() -> None:
    case = CASES_DIR / "auxin_softening"
    tissue = read_case_tissue(case)
    before = tissue.vertices.copy()
    result = simulate(
        read_case_model(case).model, tissue, t_end=2.0, n_frames=2, solver=QuasiStatic()
    )
    edge = tissue.topology.boundary
    np.testing.assert_array_equal(result.final.vertices[edge, 0], before[edge, 0])
    assert np.abs(result.final.vertices[~edge] - before[~edge]).max() > 0


def test_frames_follow_the_schedule() -> None:
    case = CASES_DIR / "sweep_pre"
    result = simulate(
        read_case_model(case).model,
        read_case_tissue(case),
        t_end=3.0,
        n_frames=4,
        solver=QuasiStatic(),
    )
    np.testing.assert_allclose(result.times, [0.0, 1.0, 2.0, 3.0, 3.0])
    assert result.accepted_steps == 4  # three feedback steps and the S6 sliver to t_end


def test_converged_run_stops_iterating() -> None:
    state = case_state(CASES_DIR / "sweep_pre")
    run = QuasiStatic().start(state)
    integrate(state, 0.0, 2.0, 0, QuasiStatic())
    before = run.evaluations
    run.step(2.0, 3.0, state.compute_forces())
    assert run.evaluations - before == 1  # one residual check, no Jacobian


@pytest.mark.parametrize(
    ("step", "tolerance", "iterations"),
    [(0.0, 1e-6, 50), (1.0, 0.0, 50), (1.0, 1e-6, 0)],
)
def test_rejects_bad_settings(step: float, tolerance: float, iterations: int) -> None:
    with pytest.raises(ValueError):
        QuasiStatic(step=step, force_tolerance=tolerance, max_iterations=iterations)
