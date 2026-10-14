"""Command line: a drop-in replacement for Tissue's `simulator` on supported models.

`tissue-lite MODEL INIT SOLVER [-init_output FILE] [-vtk_output DIR]` takes the same
arguments as Tissue, so `TISSUE_BIN=tissue-lite` works with existing run scripts.
MODEL is a legacy `.model` file or a TOML model; SOLVER is an RK5Adaptive file.
"""

import logging
from pathlib import Path

import click

from tissue_lite.legacy.init_file import read_init, write_init
from tissue_lite.legacy.model_file import read_model_file
from tissue_lite.legacy.slots import SlotLayout
from tissue_lite.legacy.solver_file import read_solver_file
from tissue_lite.model import MeristemModel
from tissue_lite.solvers.simulate import simulate

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
@click.option("-v", "--verbose", is_flag=True, help="Logs debug messages.")
def main(
    model_file: Path,
    init_file: Path,
    solver_file: Path,
    init_output: Path | None,
    vtk_output: Path,
    second_axes: bool,
    concentration_slot: int | None,
    verbose: bool,
) -> None:
    """Simulates MODEL_FILE from INIT_FILE with the settings in SOLVER_FILE."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO, format="%(message)s"
    )
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
    result = simulate(
        model,
        read_init(init_file, layout),
        t_end=settings.t_end,
        n_frames=settings.n_frames,
        solver=settings.solver,
        t_start=settings.t_start,
        output_dir=vtk_output,
        slot_layout=layout,
    )
    if init_output is not None:
        write_init(result.final, init_output, layout)
        logger.info("Wrote %s", init_output)
