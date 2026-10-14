"""Tissue's adaptive Cash–Karp Runge–Kutta solver (`RK5Adaptive`), quirks included.

Only vertex positions and cell centres have non-zero derivatives, so only they are
integrated; every other variable is carried through each step unchanged, exactly as
Tissue's `y + h * 0` does. Kept as in Tissue: the step is capped at `max_step` after
every step, per-step updates receive the trial step rather than the accepted one,
and the error scale is `|y| + |h dy| + 1e-9 tolerance`.
"""

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tissue_lite.mechanics import CellMechanics
from tissue_lite.model import MeristemModel, ModelState
from tissue_lite.output import VtuWriter
from tissue_lite.tissue import FloatArray, Tissue

logger = logging.getLogger(__name__)

_SAFETY = 0.9
_PGROW = -0.2
_PSHRINK = -0.25
_ERRCON = 1.89e-4

_B21 = 0.2
_B31, _B32 = 3.0 / 40.0, 9.0 / 40.0
_B41, _B42, _B43 = 0.3, -0.9, 1.2
_B51, _B52, _B53, _B54 = -11.0 / 54.0, 2.5, -70.0 / 27.0, 35.0 / 27.0
_B61, _B62, _B63, _B64, _B65 = (
    1631.0 / 55296.0,
    175.0 / 512.0,
    575.0 / 13824.0,
    44275.0 / 110592.0,
    253.0 / 4096.0,
)
_C1, _C3, _C4, _C6 = 37.0 / 378.0, 250.0 / 621.0, 125.0 / 594.0, 512.0 / 1771.0
_DC5 = -277.00 / 14336.0
_DC1 = _C1 - 2825.0 / 27648.0
_DC3 = _C3 - 18575.0 / 48384.0
_DC4 = _C4 - 13525.0 / 55296.0
_DC6 = _C6 - 0.25


@dataclass(frozen=True)
class CashKarp:
    """Tissue's RK5Adaptive settings: initial and maximal step, relative tolerance."""

    max_step: float = 5e-3
    tolerance: float = 1e-4

    def __post_init__(self) -> None:
        if self.max_step <= 0 or self.tolerance <= 0:
            raise ValueError("max_step and tolerance must be positive")


@dataclass(frozen=True)
class Frame:
    """One printed time point."""

    time: float
    tissue: Tissue
    mechanics: CellMechanics | None


@dataclass
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
    solver: CashKarp = CashKarp(),
    t_start: float = 0.0,
    output_dir: Path | None = None,
) -> SimulationResult:
    """Runs the model from t_start to t_end, printing n_frames frames as Tissue does.

    The input tissue is not modified. With output_dir, frames are also written as
    Tissue-compatible VTU files plus `tissue.pvd`.

    Raises:
        ValueError: If the time span is empty.
        RuntimeError: If the step size underflows.
    """
    writer = None if output_dir is None else VtuWriter(output_dir, n_frames)
    state = ModelState(model=model, tissue=tissue.copy())
    return integrate(state, t_start, t_end, n_frames, solver, writer)


def integrate(
    state: ModelState,
    t_start: float,
    t_end: float,
    n_frames: int,
    solver: CashKarp,
    writer: VtuWriter | None = None,
) -> SimulationResult:
    """Port of `RK5Adaptive::simulate` on an initiated model state."""
    if not (t_end - t_start > 0.0):
        raise ValueError("t_end must exceed t_start")
    tissue = state.tissue
    frames: list[Frame] = []
    accepted = rejected = 0
    eps = solver.tolerance
    h1 = solver.max_step
    tiny = 1e-9 * eps
    h = h1
    print_time = t_end + tiny
    print_delta = t_end + 2.0 * tiny
    do_print = n_frames > 0
    if n_frames >= 2:
        print_time = t_start - tiny
    if n_frames > 2:
        print_delta = (t_end - t_start) / float(n_frames - 1)
    t = t_start

    def emit() -> None:
        mech = state.cell_mechanics() if state.mechanics_valid else None
        frame = Frame(time=t, tissue=tissue.copy(), mechanics=mech)
        if writer is not None:
            writer.write(len(frames), frame.tissue)
        frames.append(frame)

    while True:
        centres = state.triangulation.centres
        vd, cd = state.derivatives(tissue.cell_variables, centres, tissue.vertices)
        scale_v = np.abs(tissue.vertices) + np.abs(vd * h) + tiny
        scale_c = np.abs(centres) + np.abs(cd * h) + tiny
        if do_print and t >= print_time:
            print_time += print_delta
            emit()
        t_min = t_end if t_end < print_time else print_time
        if t + h > t_min:
            h = t_min - t
        h_did, h_next = _step(state, t, h, vd, cd, scale_v, scale_c, eps)
        t += h_did
        if h_did == h:
            accepted += 1
        else:
            rejected += 1
        state.update(h)
        if t >= t_end:
            if do_print:
                state.derivatives(
                    tissue.cell_variables, state.triangulation.centres, tissue.vertices
                )
                emit()
            logger.info("Simulation done: %d ok, %d bad steps", accepted, rejected)
            return SimulationResult(frames, accepted, rejected, tissue.copy())
        h = h_next
        if h > h1:
            h = h1


def _step(
    state: ModelState,
    t: float,
    h_try: float,
    vd: FloatArray,
    cd: FloatArray,
    scale_v: FloatArray,
    scale_c: FloatArray,
    eps: float,
) -> tuple[float, float]:
    """Port of `rkqs`: retries until the error is within tolerance; updates state."""
    h = h_try
    tissue = state.tissue
    triangulation = state.triangulation
    while True:
        x_out, c_out, x_err, c_err = _cash_karp(
            state, tissue.vertices, triangulation.centres, vd, cd, h
        )
        err_max = max(
            float(np.max(np.abs(c_err / scale_c))),
            float(np.max(np.abs(x_err / scale_v))),
            0.0,
        )
        err_max /= eps
        if err_max <= 1.0:
            break
        h_temp = _SAFETY * h * math.pow(err_max, _PSHRINK)
        h = h_temp if h_temp > 0.1 * h else 0.1 * h  # h > 0 (integrate checks)
        if t + h == t:
            raise RuntimeError("Step size underflow")
    h_next = _SAFETY * h * math.pow(err_max, _PGROW) if err_max > _ERRCON else 5.0 * h
    tissue.vertices[:] = x_out
    triangulation.centres[:] = c_out
    return h, h_next


def _cash_karp(
    state: ModelState,
    x: FloatArray,
    c: FloatArray,
    k1x: FloatArray,
    k1c: FloatArray,
    h: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Port of `rkck`: one Cash–Karp step with its embedded error estimate.

    Stage derivatives see a copy of the cell variables, so their side effects are
    discarded as in Tissue.
    """
    cell_vars = state.tissue.cell_variables

    def stage(xs: FloatArray, cs: FloatArray) -> tuple[FloatArray, FloatArray]:
        return state.derivatives(cell_vars.copy(), cs, xs)

    k2x, k2c = stage(x + _B21 * h * k1x, c + _B21 * h * k1c)
    k3x, k3c = stage(
        x + h * (_B31 * k1x + _B32 * k2x), c + h * (_B31 * k1c + _B32 * k2c)
    )
    k4x, k4c = stage(
        x + h * (_B41 * k1x + _B42 * k2x + _B43 * k3x),
        c + h * (_B41 * k1c + _B42 * k2c + _B43 * k3c),
    )
    k5x, k5c = stage(
        x + h * (_B51 * k1x + _B52 * k2x + _B53 * k3x + _B54 * k4x),
        c + h * (_B51 * k1c + _B52 * k2c + _B53 * k3c + _B54 * k4c),
    )
    k6x, k6c = stage(
        x + h * (_B61 * k1x + _B62 * k2x + _B63 * k3x + _B64 * k4x + _B65 * k5x),
        c + h * (_B61 * k1c + _B62 * k2c + _B63 * k3c + _B64 * k4c + _B65 * k5c),
    )
    x_out = x + h * (_C1 * k1x + _C3 * k3x + _C4 * k4x + _C6 * k6x)
    c_out = c + h * (_C1 * k1c + _C3 * k3c + _C4 * k4c + _C6 * k6c)
    x_err = h * (_DC1 * k1x + _DC3 * k3x + _DC4 * k4x + _DC5 * k5x + _DC6 * k6x)
    c_err = h * (_DC1 * k1c + _DC3 * k3c + _DC4 * k4c + _DC5 * k5c + _DC6 * k6c)
    return x_out, c_out, x_err, c_err


def prerun_and_simulate(
    model: MeristemModel,
    prerun_model: MeristemModel,
    tissue: Tissue,
    t_end: float,
    n_frames: int = 100,
    solver: CashKarp = CashKarp(),
    prerun_t_end: float = 1.0,
    prerun_frames: int = 2,
    prerun_solver: CashKarp = CashKarp(),
    output_dir: Path | None = None,
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
    )
    start = prerun.final.without_triangulation()
    main = simulate(model, start, t_end, n_frames, solver, output_dir=output_dir)
    return prerun, main
