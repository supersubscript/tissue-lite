"""Tests for the VTU/PVD writer against Tissue's own files."""

import re
from pathlib import Path

import numpy as np
import pytest
from conftest import CASES, CASES_DIR, CaseRunner

from tissue_lite.legacy.solver_file import read_solver_file
from tissue_lite.output import format_pvd

_NUMBER = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)(e[-+]?\d+)?|[-+]?(inf|nan)")


def _tokens_close(actual: str, expected: str) -> None:
    """Compares files token by token; numbers within %g rounding."""
    a, e = actual.split(), expected.split()
    assert len(a) == len(e)
    numeric_a, numeric_e = [], []
    for x, y in zip(a, e, strict=True):
        if _NUMBER.fullmatch(y):
            numeric_a.append(float(x))
            numeric_e.append(float(y))
        else:
            assert x == y
    np.testing.assert_allclose(numeric_a, numeric_e, rtol=1e-5, atol=1e-12)


@pytest.mark.parametrize("case", CASES)
def test_vtu_frames_match_tissue(case: Path, run_case: CaseRunner) -> None:
    _, vtk = run_case(case)
    expected = sorted(p.name for p in (case / "vtk").iterdir())
    assert sorted(p.name for p in vtk.iterdir()) == expected
    for name in expected:
        _tokens_close((vtk / name).read_text(), (case / "vtk" / name).read_text())


@pytest.mark.parametrize("n_frames", [2, 3])
def test_pvd_matches_tissue(n_frames: int) -> None:
    case = next(
        p
        for p in sorted(CASES_DIR.iterdir())
        if read_solver_file(p / "solver.rk5").n_frames == n_frames
    )
    assert format_pvd(n_frames) == (case / "vtk" / "tissue.pvd").read_text()
