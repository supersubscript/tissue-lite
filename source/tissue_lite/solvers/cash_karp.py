"""Tissue's adaptive Cash–Karp Runge–Kutta solver (`RK5Adaptive`), quirks included.

Only vertex positions and cell centres have non-zero derivatives, so only they are
integrated; every other variable is carried through each step unchanged, exactly as
Tissue's `y + h * 0` does. Kept as in Tissue: the step is capped at `max_step` after
every step, per-step updates receive the trial step rather than the accepted one,
and the error scale is `|y| + |h dy| + 1e-9 tolerance`.
"""

import math
from dataclasses import dataclass

import numpy as np

from tissue_lite.arrays import FloatArray
from tissue_lite.model import ModelState
from tissue_lite.solvers.stepper import StepResult

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

    def start(self, state: ModelState) -> CashKarpRun:
        return CashKarpRun(self, state)


class CashKarpRun:
    """Port of the step part of `RK5Adaptive::simulate` on one model state."""

    def __init__(self, config: CashKarp, state: ModelState) -> None:
        self.state = state
        self.max_step = config.max_step
        self.tolerance = config.tolerance
        self.time_slack = 1e-9 * config.tolerance
        self.h = config.max_step

    def step(
        self,
        t: float,
        t_bound: float,
        vertex_forces: FloatArray,
        centre_forces: FloatArray,
    ) -> StepResult:
        """Takes one accepted step; `update` gets the trial step, as in Tissue."""
        state = self.state
        tissue = state.tissue
        centres = state.triangulation.centres
        h = self.h
        scale_v = np.abs(tissue.vertices) + np.abs(vertex_forces * h) + self.time_slack
        scale_c = np.abs(centres) + np.abs(centre_forces * h) + self.time_slack
        if t + h > t_bound:
            h = t_bound - t
        h_did, h_next = _retry_step(
            state, t, h, vertex_forces, centre_forces, scale_v, scale_c, self.tolerance
        )
        state.update(h)
        self.h = min(h_next, self.max_step)
        return StepResult(t=t + h_did, accepted=h_did == h)


def _retry_step(
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
        x_out, c_out, x_err, c_err = _cash_karp_step(
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
        h = h_temp if h_temp > 0.1 * h else 0.1 * h  # h > 0 (simulate checks)
        if t + h == t:
            raise RuntimeError("Step size underflow")
    h_next = _SAFETY * h * math.pow(err_max, _PGROW) if err_max > _ERRCON else 5.0 * h
    tissue.vertices[:] = x_out
    triangulation.centres[:] = c_out
    return h, h_next


def _cash_karp_step(
    state: ModelState,
    x: FloatArray,
    c: FloatArray,
    k1x: FloatArray,
    k1c: FloatArray,
    h: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Port of `rkck`: one Cash–Karp step with its embedded error estimate.

    Stage derivatives see a copy of the cell state, so their side effects are
    discarded as in Tissue.
    """
    cells = state.tissue.cells

    def stage(xs: FloatArray, cs: FloatArray) -> tuple[FloatArray, FloatArray]:
        return state.compute_forces(cells.copy(), cs, xs)

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
