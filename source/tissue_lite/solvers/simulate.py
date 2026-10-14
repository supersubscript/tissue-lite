"""Runs a model with any stepper, printing frames on Tissue's schedule.

The schedule is Tissue's own (`RK5Adaptive::simulate`): with more than two frames
it prints t_end twice, just before the last step and after it, and the VTU tests
pin that (see KNOWN_ISSUES).
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tissue_lite.arrays import FloatArray
from tissue_lite.legacy.slots import SlotLayout
from tissue_lite.model import MeristemModel, ModelState
from tissue_lite.output import VtuWriter
from tissue_lite.solvers.cash_karp import CashKarp
from tissue_lite.solvers.stepper import Stepper
from tissue_lite.tissue import Tissue

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Frame:
    """One printed time point; `tissue.cells` carries the stress and strain."""

    time: float
    tissue: Tissue


@dataclass(frozen=True)
class SimulationResult:
    """Printed frames and step statistics of one run; `final` is the end state."""

    frames: list[Frame]
    accepted_steps: int
    rejected_steps: int
    final: Tissue

    @property
    def times(self) -> FloatArray:
        """Times of the printed frames."""
        return np.array([f.time for f in self.frames])


def simulate(
    model: MeristemModel,
    tissue: Tissue,
    t_end: float,
    n_frames: int = 100,
    solver: Stepper = CashKarp(),
    t_start: float = 0.0,
    output_dir: Path | None = None,
    slot_layout: SlotLayout = SlotLayout(),
) -> SimulationResult:
    """Runs the model from t_start to t_end, printing n_frames frames as Tissue does.

    The input tissue is not modified. With output_dir, frames are also written as
    Tissue-compatible VTU files plus `tissue.pvd`, numbered by `slot_layout`.

    Raises:
        ValueError: If the time span is empty.
        RuntimeError: If the step size underflows.
    """
    writer = (
        None if output_dir is None else VtuWriter(output_dir, n_frames, slot_layout)
    )
    state = ModelState(model=model, tissue=tissue.copy())
    return integrate(state, t_start, t_end, n_frames, solver, writer)


def integrate(
    state: ModelState,
    t_start: float,
    t_end: float,
    n_frames: int,
    solver: Stepper,
    writer: VtuWriter | None = None,
) -> SimulationResult:
    """Port of the frame loop of `RK5Adaptive::simulate` on an initiated model state."""
    if not (t_end - t_start > 0.0):
        raise ValueError("t_end must exceed t_start")
    run = solver.start(state)
    tissue = state.tissue
    frames: list[Frame] = []
    accepted = rejected = 0
    tiny = run.time_slack
    print_time = t_end + tiny
    print_delta = t_end + 2.0 * tiny
    do_print = n_frames > 0
    if n_frames >= 2:
        print_time = t_start - tiny
    if n_frames > 2:
        print_delta = (t_end - t_start) / float(n_frames - 1)
    t = t_start

    def emit() -> None:
        frame = Frame(time=t, tissue=tissue.copy())
        if writer is not None:
            writer.write(len(frames), frame.tissue)
        frames.append(frame)

    while True:
        vertex_forces, centre_forces = state.compute_forces(
            tissue.cells, state.triangulation.centres, tissue.vertices
        )
        if do_print and t >= print_time:
            print_time += print_delta
            emit()
        t_bound = t_end if t_end < print_time else print_time
        result = run.step(t, t_bound, vertex_forces, centre_forces)
        t = result.t
        if result.accepted:
            accepted += 1
        else:
            rejected += 1
        if t >= t_end:
            if do_print:
                state.compute_forces(
                    tissue.cells, state.triangulation.centres, tissue.vertices
                )
                emit()
            logger.info("Simulation done: %d ok, %d bad steps", accepted, rejected)
            return SimulationResult(frames, accepted, rejected, tissue.copy())


def prerun_and_simulate(
    model: MeristemModel,
    prerun_model: MeristemModel,
    tissue: Tissue,
    t_end: float,
    n_frames: int = 100,
    solver: Stepper = CashKarp(),
    prerun_t_end: float = 1.0,
    prerun_frames: int = 2,
    prerun_solver: Stepper = CashKarp(),
    output_dir: Path | None = None,
    slot_layout: SlotLayout = SlotLayout(),
) -> tuple[SimulationResult, SimulationResult]:
    """Reproduces flower-experiments' `prerunAndSim.sh`: prerun, then main run.

    The main run starts from the prerun's final state as Tissue re-reads it from
    `-init_output`: cell variables and positions kept, centre triangulation rebuilt.
    """
    prerun = simulate(
        prerun_model,
        tissue,
        prerun_t_end,
        prerun_frames,
        prerun_solver,
        output_dir=output_dir,
        slot_layout=slot_layout,
    )
    start = prerun.final.without_triangulation()
    main = simulate(
        model,
        start,
        t_end,
        n_frames,
        solver,
        output_dir=output_dir,
        slot_layout=slot_layout,
    )
    return prerun, main
