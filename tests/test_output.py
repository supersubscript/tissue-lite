"""Tests for the VTU/PVD writer against Tissue's own files."""

from pathlib import Path

import numpy as np
import pytest
from conftest import CASES, CASES_DIR

from tissue_lite.legacy import read_model_file, read_solver_file
from tissue_lite.output import format_pvd
from tissue_lite.solver import simulate
from tissue_lite.tissue import Tissue


def _tokens_close(actual: str, expected: str) -> None:
    """Compares files token by token; numbers within %g rounding."""
    a, e = actual.split(), expected.split()
    assert len(a) == len(e)
    numeric_a, numeric_e = [], []
    for x, y in zip(a, e, strict=True):
        try:
            numeric_e.append(float(y))
        except ValueError:
            assert x == y
            continue
        numeric_a.append(float(x))
    np.testing.assert_allclose(numeric_a, numeric_e, rtol=1e-5, atol=1e-12)


@pytest.mark.parametrize("case", CASES)
def test_vtu_frames_match_tissue(case: Path, tmp_path: Path) -> None:
    settings = read_solver_file(case / "solver.rk5")
    simulate(
        read_model_file(case / "model.model"),
        Tissue.read_init(case / "input.init"),
        t_end=settings.t_end,
        n_frames=settings.n_frames,
        solver=settings.solver,
        output_dir=tmp_path,
    )
    expected = sorted(p.name for p in (case / "vtk").iterdir())
    assert sorted(p.name for p in tmp_path.iterdir()) == expected
    for name in expected:
        _tokens_close((tmp_path / name).read_text(), (case / "vtk" / name).read_text())


@pytest.mark.parametrize("n_frames", [2, 3])
def test_pvd_matches_tissue(n_frames: int) -> None:
    case = next(
        p
        for p in sorted(CASES_DIR.iterdir())
        if read_solver_file(p / "solver.rk5").n_frames == n_frames
    )
    assert format_pvd(n_frames) == (case / "vtk" / "tissue.pvd").read_text()
