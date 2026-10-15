"""Command line: Tissue's `simulator` arguments on supported models.

`tissue-lite MODEL INIT SOLVER [-init_output FILE] [-vtk_output DIR]` takes the same
arguments as Tissue. MODEL is a legacy `.model` file or a TOML model; SOLVER is an
RK5Adaptive file. `--solver cash-karp` reproduces Tissue's results (the
`TISSUE_BIN=tissue-lite` drop-in); the default, quasi-static, uses only the file's
time span and frame count.
"""

import logging
from pathlib import Path

import click
import numba

from tissue_lite.legacy.init_file import read_init, write_init
from tissue_lite.legacy.model_file import read_model_file
from tissue_lite.legacy.slots import SlotLayout
from tissue_lite.legacy.solver_file import read_solver_file
from tissue_lite.model import MeristemModel
from tissue_lite.output import VtuWriter
from tissue_lite.solvers.implicit import Implicit
from tissue_lite.solvers.quasi_static import QuasiStatic
from tissue_lite.solvers.simulate import simulate
from tissue_lite.solvers.stepper import Stepper

logger = logging.getLogger(__name__)

_FILE = click.Path(exists=True, dir_okay=False, path_type=Path)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("model_file", type=_FILE)
@click.argument("init_file", type=_FILE)
@click.argument("solver_file", type=_FILE)
@click.option(
    "-init_output",
    "--init-output",
    "init_output",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Writes the final state as a Tissue init file.",
)
@click.option(
    "-vtk_output",
    "--vtk-output",
    "vtk_output",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("vtk"),
    show_default=True,
    help="Directory for the VTU frames and tissue.pvd.",
)
@click.option(
    "--second-axes",
    is_flag=True,
    help="TOML models only: store the second stress and strain axes in the growth "
    "models' slots (legacy model files carry their own layout).",
)
@click.option(
    "--concentration-slot",
    type=int,
    help="TOML models only: cell-variable column holding the softening concentration.",
)
@click.option(
    "--solver",
    "solver_name",
    type=click.Choice(["cash-karp", "quasi-static", "implicit"]),
    default="quasi-static",
    show_default=True,
    help="cash-karp follows SOLVER_FILE as Tissue does; quasi-static and implicit "
    "(Radau) take only its time span and frame count and feed back every --step "
    "(KNOWN_ISSUES N1–N6).",
)
@click.option(
    "--step",
    type=float,
    default=0.1,
    show_default=True,
    help="Feedback step of the quasi-static and implicit solvers.",
)
@click.option(
    "--threads",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Threads for the force kernel; results do not depend on it.",
)
@click.option("-v", "--verbose", is_flag=True, help="Logs debug messages.")
def main(
    model_file: Path,
    init_file: Path,
    solver_file: Path,
    init_output: Path | None,
    vtk_output: Path,
    second_axes: bool,
    concentration_slot: int | None,
    solver_name: str,
    step: float,
    threads: int,
    verbose: bool,
) -> None:
    """Simulates MODEL_FILE from INIT_FILE with the settings in SOLVER_FILE."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO, format="%(message)s"
    )
    numba.set_num_threads(threads)
    if model_file.suffix == ".toml":
        model = MeristemModel.from_toml(model_file)
        layout = SlotLayout(
            second_axes=second_axes, concentration_slot=concentration_slot
        )
        if model.membrane.softening is not None and concentration_slot is None:
            raise click.UsageError("A softening model needs --concentration-slot")
    else:
        legacy = read_model_file(model_file)
        model, layout = legacy.model, legacy.layout
    settings = read_solver_file(solver_file)
    solvers: dict[str, Stepper] = {
        "cash-karp": settings.solver,
        "quasi-static": QuasiStatic(step=step),
        "implicit": Implicit(step=step),
    }
    result = simulate(
        model,
        read_init(init_file, layout),
        t_end=settings.t_end,
        n_frames=settings.n_frames,
        solver=solvers[solver_name],
        t_start=settings.t_start,
        writer=VtuWriter(vtk_output, settings.n_frames, layout),
    )
    if init_output is not None:
        write_init(result.final, init_output, layout)
        logger.info("Wrote %s", init_output)
