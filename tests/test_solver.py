"""Regression of whole runs against Tissue e74e457 (see `reference/generate.py`)."""

from pathlib import Path

import numpy as np
import pytest
from conftest import CASES, CASES_DIR, CaseRunner, assert_tissues_close

from tissue_lite.legacy import read_model_file, read_solver_file
from tissue_lite.solver import CashKarp, prerun_and_simulate, simulate
from tissue_lite.tissue import Tissue


@pytest.mark.parametrize("case", CASES)
def test_final_state_matches_tissue(case: Path, run_case: CaseRunner) -> None:
    result, _ = run_case(case)
    assert_tissues_close(result.final, Tissue.read_init(case / "final.init"))


@pytest.mark.parametrize("case", CASES)
def test_frame_count_and_times(case: Path, run_case: CaseRunner) -> None:
    settings = read_solver_file(case / "solver.rk5")
    result, _ = run_case(case)
    expected = np.linspace(settings.t_start, settings.t_end, settings.n_frames)
    if settings.n_frames > 2:
        # Tissue prints t_end twice: just before it and after the last step.
        expected = np.append(expected, settings.t_end)
    assert len(result.frames) == len(list((case / "vtk").glob("VTK_cells*.vtu")))
    np.testing.assert_allclose(result.times, expected, atol=1e-8)


def test_tight_tolerance_rejects_steps(run_case: CaseRunner) -> None:
    result, _ = run_case(CASES_DIR / "sweep_rejections")
    assert result.rejected_steps > 0


def test_prerun_handoff_matches_chained_tissue_runs() -> None:
    pre, main = CASES_DIR / "sweep_pre", CASES_DIR / "sweep_120_0"
    pre_settings = read_solver_file(pre / "solver.rk5")
    settings = read_solver_file(main / "solver.rk5")
    _, result = prerun_and_simulate(
        read_model_file(main / "model.model"),
        read_model_file(pre / "model.model"),
        Tissue.read_init(pre / "input.init"),
        t_end=settings.t_end,
        n_frames=settings.n_frames,
        solver=settings.solver,
        prerun_t_end=pre_settings.t_end,
        prerun_frames=pre_settings.n_frames,
        prerun_solver=pre_settings.solver,
    )
    assert_tissues_close(result.final, Tissue.read_init(main / "final.init"))


def test_input_is_not_modified() -> None:
    case = CASES_DIR / "sweep_pre"
    tissue = Tissue.read_init(case / "input.init")
    before = tissue.copy()
    simulate(read_model_file(case / "model.model"), tissue, t_end=0.05, n_frames=2)
    np.testing.assert_array_equal(tissue.vertices, before.vertices)
    np.testing.assert_array_equal(tissue.cell_variables, before.cell_variables)


@pytest.mark.parametrize(
    ("max_step", "tolerance"), [(0.0, 1e-4), (5e-3, 0.0), (-1.0, 1e-4)]
)
def test_cash_karp_rejects_non_positive_settings(
    max_step: float, tolerance: float
) -> None:
    with pytest.raises(ValueError, match="positive"):
        CashKarp(max_step=max_step, tolerance=tolerance)


def test_simulate_rejects_empty_time_span() -> None:
    case = CASES_DIR / "sweep_pre"
    with pytest.raises(ValueError, match="t_end"):
        simulate(
            read_model_file(case / "model.model"),
            Tissue.read_init(case / "input.init"),
            t_end=0.0,
        )
