"""Generates the regression reference data under `tests/data/cases/` from Tissue.

Usage: uv run python reference/generate.py [--simulator BIN] [--models DIR] [CASE ...]

Runs Tissue's `simulator` (built by `reference/build_reference.sh`) on each case and
stores its inputs (model, solver, init), Tissue's `-init_output` state, its VTU
frames and a `provenance.json` per case. Model files are copied from
the `--models` directory, so the tests do not need it.
"""

import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import click
import numpy as np
from cut_patch import RAW_LAYOUT, resize_cell_variables
from loguru import logger

from tissue_lite.legacy.init_file import read_init, write_init
from tissue_lite.tissue import Tissue

ROOT = Path(__file__).resolve().parents[1]
CASES_DIR = ROOT / "tests" / "data" / "cases"
MESHES_DIR = ROOT / "tests" / "data" / "meshes"
TISSUE_COMMIT = "e74e457869410e634a474a68f1071ceb14df753e"
TISSUE_FLAGS = "-O3 -DNDEBUG -Wall -pedantic -ansi -std=c++11"
CONCENTRATION_INDEX = 41


@dataclass(frozen=True)
class Case:
    """One reference run.

    Attributes:
        name: Directory name under `tests/data/cases/`.
        model: Model file, relative to `--models`.
        init: A mesh name in `tests/data/meshes/` or `case:<name>` for another
            case's final state (as `prerunAndSim.sh` chains runs).
        n_vars: Cell variables to pad or truncate the init to.
        concentration: Sets cell variable 41 to a ramp over the cells (MF=10).
        edits: (old, new) replacements applied once each to the model text.
        description: What the case covers.
        tags: "slow" marks cases skipped by default in the tests.
    """

    name: str
    model: str
    init: str
    t_end: float
    n_frames: int
    n_vars: int
    tolerance: float = 1e-4
    concentration: bool = False
    edits: tuple[tuple[str, str], ...] = ()
    description: str = ""
    tags: tuple[str, ...] = ()


ZFIX = "results/zfix_rerun"
CASES = (
    Case(
        "sweep_pre",
        f"{ZFIX}/mod_120_0_04/model_pre.model",
        "patch19",
        1.0,
        2,
        35,
        description="Sweep prerun: pressure ramp, no feedback (prerunAndSim step 1).",
    ),
    Case(
        "sweep_120_0",
        f"{ZFIX}/mod_120_0_04/model.model",
        "case:sweep_pre",
        2.0,
        3,
        35,
        description="Sweep main run Ym=120, Yf=0, from the prerun state.",
    ),
    Case(
        "sweep_120_120",
        f"{ZFIX}/mod_120_120_04/model.model",
        "case:sweep_pre",
        2.0,
        3,
        35,
        description="Sweep main run Ym=120, Yf=120 (fibre feedback active).",
    ),
    Case(
        "sweep_rejections",
        f"{ZFIX}/mod_120_120_04/model.model",
        "case:sweep_pre",
        0.2,
        2,
        35,
        tolerance=1e-8,
        description="Tight tolerance, forcing rejected Cash-Karp steps.",
    ),
    Case(
        "iso",
        "models/iso/model.model",
        "patch19",
        1.0,
        3,
        39,
        description="Isotropic wall, fibre init flag 1, MT velocity gate.",
    ),
    Case(
        "aniso",
        "models/aniso/model.model",
        "patch19",
        1.0,
        3,
        39,
        description="Anisotropic wall, hill_direct fibre coupling, all axes fixed.",
    ),
    Case(
        "aniso_growth",
        "models/aniso-growth/model.model",
        "patch19",
        1.0,
        3,
        39,
        description="Strain-driven growth with double rest lengths.",
    ),
    Case(
        "iso_growth",
        "models/iso-growth/model.model",
        "patch19",
        1.0,
        3,
        39,
        description="Growth; FiberModel moduli differ from TRBS.",
    ),
    Case(
        "auxin_softening",
        "results/celly/test_auxin_decay/model.model",
        "patch19",
        1.0,
        3,
        43,
        concentration=True,
        description="MF=10 as used (p9=p10=0, so K=0, n=0: constant factor 1/2).",
    ),
    Case(
        "auxin_softening_hill",
        "results/celly/test_auxin_decay/model.model",
        "patch19",
        1.0,
        3,
        43,
        concentration=True,
        edits=(
            ("0    # MT feedback flag 0 none   1 Teta", "2    # K (Hill)"),
            ("0    # unused", "2    # n (Hill)"),
        ),
        description="MF=10 with K=2, n=2 (p9, p10) and a ramp in variable 41.",
    ),
    Case(
        "patch100",
        f"{ZFIX}/mod_120_120_04/model.model",
        "patch100",
        0.5,
        2,
        35,
        description="100-cell end-to-end run.",
        tags=("slow",),
    ),
)


def compute_sha256(path: Path) -> str:
    """Returns the file's SHA-256 hex digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_solver_text(case: Case) -> str:
    """Returns the RK5Adaptive solver file for the case (print flag 2, VTU)."""
    return (
        f"RK5Adaptive\n0 {case.t_end:g}\n2 {case.n_frames}\n5.0e-3 {case.tolerance:g}\n"
    )


def make_input_tissue(case: Case) -> Tissue:
    """Loads the case's input state and adapts it to the model's variable count."""
    if case.init.startswith("case:"):
        tissue = read_init(CASES_DIR / case.init[5:] / "final.init")
    else:
        tissue = read_init(MESHES_DIR / f"{case.init}.init")
    values = resize_cell_variables(RAW_LAYOUT.to_cell_variables(tissue), case.n_vars)
    if case.concentration:
        values[:, CONCENTRATION_INDEX] = np.linspace(0.0, 1.0, len(values))
    tissue.cells = RAW_LAYOUT.from_cell_variables(values)
    return tissue


def run_case(case: Case, simulator: Path, models: Path) -> None:
    """Runs Tissue on one case and stores inputs, outputs and provenance."""
    out = CASES_DIR / case.name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    text = (models / case.model).read_text()
    for old, new in case.edits:
        if old not in text:
            raise ValueError(f"{case.name}: {old!r} not in {case.model}")
        text = text.replace(old, new, 1)
    (out / "model.model").write_text(text)
    (out / "solver.rk5").write_text(make_solver_text(case))
    write_init(make_input_tissue(case), out / "input.init")
    command = [
        str(simulator),
        "model.model",
        "input.init",
        "solver.rk5",
        "-init_output",
        "final.init",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for name in ("model.model", "solver.rk5", "input.init"):
            shutil.copyfile(out / name, work / name)
        subprocess.run(command, cwd=work, capture_output=True, text=True, check=True)
        shutil.copyfile(work / "final.init", out / "final.init")
        shutil.copytree(work / "vtk", out / "vtk")
    provenance = {
        "case": case.name,
        "description": case.description,
        "tags": list(case.tags),
        "tissue_commit": TISSUE_COMMIT,
        "tissue_url": "https://gitlab.com/slcu/teamHJ/tissue",
        "compiler": get_compiler_version(),
        "compiler_flags": TISSUE_FLAGS,
        "platform": platform.platform(),
        "command": " ".join(["simulator", *command[1:]]),
        "source_model": case.model,
        "model_edits": [list(e) for e in case.edits],
        "source_init": case.init,
        "n_vars": case.n_vars,
        "concentration_ramp_index": CONCENTRATION_INDEX if case.concentration else None,
        "sha256": {
            name: compute_sha256(out / name)
            for name in ("model.model", "solver.rk5", "input.init", "final.init")
        },
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    logger.info("Wrote {}", out)


def get_compiler_version() -> str:
    """Returns the first line of `g++ --version`."""
    proc = subprocess.run(["g++", "--version"], capture_output=True, text=True)
    return proc.stdout.splitlines()[0] if proc.returncode == 0 else "unknown"


@click.command()
@click.option(
    "--simulator",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=ROOT / "reference" / "build" / "bin" / "simulator",
    help="Tissue simulator built at the reference commit.",
)
@click.option(
    "--models",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory holding the model files.",
)
@click.argument("names", nargs=-1)
def main(simulator: Path, models: Path, names: tuple[str, ...]) -> None:
    """Regenerates the named cases (default: all)."""
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{message}")
    known = {c.name for c in CASES}
    unknown = set(names) - known
    if unknown:
        raise click.BadParameter(f"Unknown cases: {sorted(unknown)}")
    for case in CASES:
        if not names or case.name in names:
            run_case(case, simulator.resolve(), models)


if __name__ == "__main__":
    main()
