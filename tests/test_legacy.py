"""Tests for the legacy `.model` and `.rk5` readers."""

import logging
from pathlib import Path

import pytest
from conftest import CASES_DIR, MODELS

from tissue_lite.legacy.model_file import parse_reactions, read_model_file
from tissue_lite.legacy.solver_file import read_solver_file


def _model_text(case: str) -> str:
    return (CASES_DIR / case / "model.model").read_text()


@pytest.mark.parametrize("path", MODELS, ids=lambda p: p.parent.name)
def test_reads_every_reference_model(path: Path) -> None:
    model = read_model_file(path).model
    assert model.membrane.young_matrix > 0 or model.membrane.young_fibre > 0


@pytest.mark.parametrize(
    ("case", "second_axes", "has_growth", "softened"),
    [
        ("sweep_120_0", False, False, False),
        ("aniso_growth", True, True, False),
        ("auxin_softening", False, False, True),
    ],
)
def test_maps_reactions_to_model(
    case: str, second_axes: bool, has_growth: bool, softened: bool
) -> None:
    legacy = read_model_file(CASES_DIR / case / "model.model")
    assert legacy.layout.second_axes == second_axes
    assert (legacy.model.growth is not None) == has_growth
    assert (legacy.model.membrane.softening is not None) == softened


def test_softening_parameters() -> None:
    legacy = read_model_file(CASES_DIR / "auxin_softening" / "model.model")
    assert legacy.model.membrane.softening is not None
    assert legacy.layout.concentration_slot == 41


def test_warns_on_fibre_moduli_mismatch(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        model = read_model_file(CASES_DIR / "iso_growth" / "model.model").model
    assert "differ from the TRBS" in caplog.text
    assert model.fibres is not None
    assert model.fibres.young_matrix == 100


def test_parse_reactions_skips_comments() -> None:
    reactions = parse_reactions(_model_text("sweep_pre"))
    names = [r.name for r in reactions]
    assert names[0] == "FiberModel"
    assert reactions[0].parameters[0] == 0.01


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("Calculate::VertexVelocity", "Calculate::Unknown", "Unsupported reactions"),
        ("Pressure3D::CenterTriangulation::Linear", "Unknown", "Unsupported"),
    ],
)
def test_rejects_unsupported_reactions(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    path = tmp_path / "model.model"
    path.write_text(_model_text("sweep_120_0").replace(old, new, 1))
    with pytest.raises(ValueError, match=message):
        read_model_file(path)


def test_rejects_other_solvers(tmp_path: Path) -> None:
    path = tmp_path / "solver.rk5"
    path.write_text("RK4\n0 1\n2 2\n0.01\n")
    with pytest.raises(ValueError, match="RK5Adaptive"):
        read_solver_file(path)


def test_reads_solver_file() -> None:
    settings = read_solver_file(CASES_DIR / "sweep_rejections" / "solver.rk5")
    assert (settings.t_start, settings.t_end, settings.n_frames) == (0.0, 0.2, 2)
    assert settings.solver.tolerance == 1e-8
