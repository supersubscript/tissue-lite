"""Tests for the Tissue-compatible command line."""

from pathlib import Path

from click.testing import CliRunner
from conftest import CASES_DIR, assert_tissues_close

from tissue_lite.cli import main
from tissue_lite.legacy import read_model_file
from tissue_lite.tissue import Tissue


def _args(case: Path) -> list[str]:
    return [str(case / n) for n in ("model.model", "input.init", "solver.rk5")]


def test_tissue_style_options(tmp_path: Path) -> None:
    case = CASES_DIR / "sweep_pre"
    out = tmp_path / "final.init"
    vtk = tmp_path / "vtk"
    result = CliRunner().invoke(
        main, [*_args(case), "-init_output", str(out), "-vtk_output", str(vtk)]
    )
    assert result.exit_code == 0, result.output
    assert out.read_text() == (case / "final.init").read_text()
    assert sorted(p.name for p in vtk.iterdir()) == sorted(
        p.name for p in (case / "vtk").iterdir()
    )


def test_toml_model(tmp_path: Path) -> None:
    case = CASES_DIR / "sweep_pre"
    toml = tmp_path / "model.toml"
    read_model_file(case / "model.model").to_toml(toml)
    out = tmp_path / "final.init"
    result = CliRunner().invoke(
        main,
        [
            str(toml),
            str(case / "input.init"),
            str(case / "solver.rk5"),
            "--init-output",
            str(out),
            "--vtk-output",
            str(tmp_path / "vtk"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert_tissues_close(Tissue.read_init(out), Tissue.read_init(case / "final.init"))


def test_missing_file_fails() -> None:
    result = CliRunner().invoke(main, ["missing.model", "a.init", "b.rk5"])
    assert result.exit_code != 0
