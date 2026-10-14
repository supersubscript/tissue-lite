"""Runs a model with any stepper, printing frames on Tissue's schedule.

The schedule is Tissue's own (`RK5Adaptive::simulate`): with more than two frames
it prints t_end twice, just before the last step and after it, and the VTU tests
pin that (KNOWN_ISSUES S6). Steppers without `fidelity` print it once.
"""

import logging
import math
from dataclasses import dataclass

import numpy as np

from tissue_lite.arrays import FloatArray
from tissue_lite.model import MeristemModel, ModelState
from tissue_lite.output import VtuWriter
from tissue_lite.solvers.quasi_static import QuasiStatic
from tissue_lite.solvers.stepper import Stepper
from tissue_lite.tissue import Tissue

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Frame:
    """One printed time point; `tissue` is a copy, with stress and strain in `cells`."""

    time: float
    tissue: Tissue


@dataclass(frozen=True)
class SimulationResult:
    """Printed frames and step statistics of one run; `final` is the end state.

    `frames` holds `n_frames` frames, one more for `n_frames > 2` with a fidelity
    stepper (KNOWN_ISSUES S6), fewer after a steady stop. A step counts as rejected
    when its trial step had to shrink.
    """

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
    solver: Stepper = QuasiStatic(),
    t_start: float = 0.0,
    writer: VtuWriter | None = None,
    steady_tolerance: float | None = None,
) -> SimulationResult:
    """Runs the model from t_start to t_end, printing n_frames frames as Tissue does.

    The input tissue is not modified. A `writer` also receives every frame.
    `n_frames` of 0 prints nothing, 1 the end state, 2 the start and end, more evenly
    spaced frames, with t_end twice for a fidelity stepper (KNOWN_ISSUES S6). With
    `steady_tolerance`, the run ends early, printing its state, after the first step
    whose largest position change per unit time is below it (S7). Each call restarts
    `rand()` from seed 1 (T1).

    Raises:
        ValueError: If the time span is empty.
        RuntimeError: If the step size underflows.
    """
    state = ModelState(model=model, tissue=tissue.copy())
    return integrate(state, t_start, t_end, n_frames, solver, writer, steady_tolerance)


def integrate(
    state: ModelState,
    t_start: float,
    t_end: float,
    n_frames: int,
    solver: Stepper,
    writer: VtuWriter | None = None,
    steady_tolerance: float | None = None,
) -> SimulationResult:
    """Port of the frame loop of `RK5Adaptive::simulate` on an initiated model state.

    Advances `state` and its tissue in place; `simulate` runs it on a copy and
    documents the arguments.

    Raises:
        ValueError: If `t_end` does not exceed `t_start`.
        RuntimeError: If the step size underflows.
    """
    if not (t_end - t_start > 0.0):
        raise ValueError("t_end must exceed t_start")
    run = solver.start(state)
    tissue = state.tissue
    frames: list[Frame] = []
    accepted = rejected = 0
    tiny = solver.time_slack
    print_time = t_end + tiny
    print_delta = t_end + 2.0 * tiny
    do_print = n_frames > 0
    if n_frames >= 2:
        print_time = t_start - tiny
    if n_frames > 2:
        print_delta = (t_end - t_start) / float(n_frames - 1)
    # Without fidelity, a step ending within the frame slack of t_end ends the run (S6).
    t_last = t_end if solver.fidelity else t_end - tiny
    t = t_start

    def emit() -> None:
        frame = Frame(time=t, tissue=tissue.copy())
        if writer is not None:
            writer.write(len(frames), frame.tissue)
        frames.append(frame)

    while True:
        derivatives = state.compute_forces()
        if do_print and t >= print_time:
            print_time += print_delta
            emit()
            if not solver.fidelity and len(frames) == n_frames - 1:
                print_time = math.inf  # the end state is the last frame (S6)
        t_bound = t_end if t_end < print_time else print_time
        previous = state.positions.copy() if steady_tolerance is not None else None
        t_previous = t
        result = run.step(t, t_bound, derivatives)
        t = result.t
        if result.accepted:
            accepted += 1
        else:
            rejected += 1
        steady = (
            previous is not None
            and np.abs(state.positions - previous).max() / (t - t_previous)
            < steady_tolerance
        )
        if t >= t_last or steady:
            if do_print:
                state.compute_forces()
                emit()
            logger.info(
                "Simulation done at t = %g: %d ok, %d bad steps", t, accepted, rejected
            )
            return SimulationResult(frames, accepted, rejected, tissue.copy())


def prerun_and_simulate(
    model: MeristemModel,
    prerun_model: MeristemModel,
    tissue: Tissue,
    t_end: float,
    n_frames: int = 100,
    solver: Stepper = QuasiStatic(),
    prerun_t_end: float = 1.0,
    prerun_frames: int = 2,
    prerun_solver: Stepper = QuasiStatic(),
    writer: VtuWriter | None = None,
) -> tuple[SimulationResult, SimulationResult]:
    """Reproduces flower-experiments' `prerunAndSim.sh`: prerun, then main run.

    The main run starts from the prerun's final state as Tissue re-reads it from
    `-init_output`: cell variables and positions kept, centre triangulation rebuilt,
    `rand()` restarted. Both runs write their frames through the same `writer`; the main
    run's frame indices restart at 0, so its files overwrite the prerun's.
    """
    prerun = simulate(
        prerun_model, tissue, prerun_t_end, prerun_frames, prerun_solver, writer=writer
    )
    main = simulate(model, prerun.final, t_end, n_frames, solver, writer=writer)
    return prerun, main
