"""Shared paths and helpers for the regression cases in `tests/data/cases/`."""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from _pytest.mark import ParameterSet  # public pytest.param return type

from tissue_lite.legacy.init_file import read_init
from tissue_lite.legacy.model_file import LegacyModel, read_model_file
from tissue_lite.legacy.slots import SlotLayout
from tissue_lite.legacy.solver_file import read_solver_file
from tissue_lite.model import ModelState
from tissue_lite.output import VtuWriter
from tissue_lite.solvers.simulate import SimulationResult, simulate
from tissue_lite.tissue import Tissue

DATA = Path(__file__).parent / "data"
CASES_DIR = DATA / "cases"
MESHES_DIR = DATA / "meshes"

# Tissue and tissue-lite agree bit for bit on the reference machine; the tolerances
# allow for libm differences (pow, acos, sqrt rounding) on other platforms.
RTOL = 1e-9
ATOL = 1e-12


def _case_param(path: Path) -> ParameterSet:
    """Returns the case as a pytest param, marked slow if its provenance says so."""
    tags = json.loads((path / "provenance.json").read_text())["tags"]
    marks = [pytest.mark.slow] if "slow" in tags else []
    return pytest.param(path, id=path.name, marks=marks)


CASES = [_case_param(p) for p in sorted(CASES_DIR.iterdir()) if p.is_dir()]
MODELS = sorted(CASES_DIR.glob("*/model.model"))


def read_case_model(case: Path) -> LegacyModel:
    """Reads a reference case's model file."""
    return read_model_file(case / "model.model")


def read_case_tissue(case: Path, name: str = "input.init") -> Tissue:
    """Reads one of a reference case's init files with the case's slot layout."""
    return read_init(case / name, read_case_model(case).layout)


def case_state(case: Path) -> ModelState:
    """Returns the initiated model state of a reference case."""
    return ModelState(model=read_case_model(case).model, tissue=read_case_tissue(case))


type CaseRunner = Callable[[Path], tuple[SimulationResult, Path]]


@pytest.fixture(scope="session")
def run_case(tmp_path_factory: pytest.TempPathFactory) -> CaseRunner:
    """Runs each reference case once per session; returns (result, VTU directory)."""
    runs: dict[Path, tuple[SimulationResult, Path]] = {}

    def run(case: Path) -> tuple[SimulationResult, Path]:
        if case not in runs:
            legacy = read_case_model(case)
            settings = read_solver_file(case / "solver.rk5")
            vtk = tmp_path_factory.mktemp(case.name)
            result = simulate(
                legacy.model,
                read_case_tissue(case),
                t_end=settings.t_end,
                n_frames=settings.n_frames,
                solver=settings.solver,
                t_start=settings.t_start,
                writer=VtuWriter(vtk, settings.n_frames, legacy.layout),
            )
            runs[case] = (result, vtk)
        return runs[case]

    return run


def assert_tissues_close(
    actual: Tissue, expected: Tissue, layout: SlotLayout = SlotLayout()
) -> None:
    """Asserts equal topology, and state within tolerance in Tissue's numbering."""
    np.testing.assert_array_equal(
        actual.topology.wall_cells, expected.topology.wall_cells
    )
    np.testing.assert_array_equal(
        actual.topology.wall_vertices, expected.topology.wall_vertices
    )
    for name in ("vertices", "wall_lengths", "wall_variables"):
        np.testing.assert_allclose(
            getattr(actual, name),
            getattr(expected, name),
            rtol=RTOL,
            atol=ATOL,
            err_msg=name,
        )
    np.testing.assert_allclose(
        layout.to_cell_variables(actual),
        layout.to_cell_variables(expected),
        rtol=RTOL,
        atol=ATOL,
        err_msg="cell variables",
    )
