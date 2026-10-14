"""Command line: a drop-in replacement for Tissue's `simulator` on supported models.

`tissue-lite MODEL INIT SOLVER [-init_output FILE] [-vtk_output DIR]` takes the same
arguments as Tissue, so `TISSUE_BIN=tissue-lite` works with existing run scripts.
MODEL is a legacy `.model` file or a TOML model; SOLVER is an RK5Adaptive file.
"""

import logging
from pathlib import Path

import click

from tissue_lite.legacy import read_model_file, read_solver_file
from tissue_lite.model import MeristemModel
from tissue_lite.solver import simulate
from tissue_lite.tissue import Tissue

logger = logging.getLogger(__name__)

_VTU_PRINT_FLAG = 2

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
@click.option("-v", "--verbose", is_flag=True, help="Logs debug messages.")
def main(
    model_file: Path,
    init_file: Path,
    solver_file: Path,
    init_output: Path | None,
    vtk_output: Path,
    verbose: bool,
) -> None:
    """Simulates MODEL_FILE from INIT_FILE with the settings in SOLVER_FILE."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO, format="%(message)s"
    )
    if model_file.suffix == ".toml":
        model = MeristemModel.from_toml(model_file)
    else:
        model = read_model_file(model_file)
    settings = read_solver_file(solver_file)
    if settings.print_flag != _VTU_PRINT_FLAG:
        logger.warning(
            "Print flag %d is not supported; writing VTU (flag 2) instead",
            settings.print_flag,
        )
    result = simulate(
        model,
        Tissue.read_init(init_file),
        t_end=settings.t_end,
        n_frames=settings.n_frames,
        solver=settings.solver,
        t_start=settings.t_start,
        output_dir=vtk_output,
    )
    if init_output is not None and result.final is not None:
        result.final.write_init(init_output)
        logger.info("Wrote %s", init_output)
