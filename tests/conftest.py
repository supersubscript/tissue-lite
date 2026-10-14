"""Shared paths and helpers for the regression cases in `tests/data/cases/`."""

import json
from pathlib import Path

import numpy as np
import pytest

from tissue_lite.tissue import Tissue

DATA = Path(__file__).parent / "data"
CASES_DIR = DATA / "cases"
MESHES_DIR = DATA / "meshes"

# Tissue and tissue-lite agree bit for bit on the reference machine; the tolerances
# allow for libm differences (pow, acos, sqrt rounding) on other platforms.
RTOL = 1e-9
ATOL = 1e-12


def _case_param(path: Path) -> object:
    tags = json.loads((path / "provenance.json").read_text())["tags"]
    marks = [pytest.mark.slow] if "slow" in tags else []
    return pytest.param(path, id=path.name, marks=marks)


CASES = [_case_param(p) for p in sorted(CASES_DIR.iterdir()) if p.is_dir()]


def assert_tissues_close(actual: Tissue, expected: Tissue) -> None:
    """Asserts equal topology and positions, lengths and variables within tolerance."""
    np.testing.assert_array_equal(
        actual.topology.wall_cells, expected.topology.wall_cells
    )
    np.testing.assert_array_equal(
        actual.topology.wall_vertices, expected.topology.wall_vertices
    )
    for name in ("vertices", "wall_lengths", "wall_variables", "cell_variables"):
        np.testing.assert_allclose(
            getattr(actual, name),
            getattr(expected, name),
            rtol=RTOL,
            atol=ATOL,
            err_msg=name,
        )
