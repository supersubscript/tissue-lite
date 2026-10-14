"""Tissue's adaptive Cash–Karp Runge–Kutta solver (`RK5Adaptive`), quirks included.

Only vertex positions and cell centres have non-zero derivatives, so only they are
integrated; every other variable is carried through each step unchanged, exactly as
Tissue's `y + h * 0` does (KNOWN_ISSUES S3 is therefore not ported). Kept as in
Tissue: the step is capped at `max_step` after every step (S1), per-step updates
receive the trial step rather than the accepted one (S2), and the error scale is
`|y| + |h dy| + 1e-9 tolerance`. `CashKarp(fidelity=False)` fixes S2, S5 and S6.
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
    """Tissue's RK5Adaptive settings: initial and maximal step, relative tolerance.

    `max_step` is both the initial step and the cap applied after every step
    (KNOWN_ISSUES S1), so error control can only shrink steps. With `fidelity` False,
    the per-step updates get the accepted step (S2) and the diagnostics they read are
    re-evaluated at the accepted state (S5), at one extra force evaluation per step
    (which may consume `rand_state`, T1), and `simulate` prints t_end once (S6).

    Raises:
        ValueError: If `max_step` or `tolerance` is not positive.
    """

    max_step: float = 5e-3
    tolerance: float = 1e-4
    fidelity: bool = True

    def __post_init__(self) -> None:
        if self.max_step <= 0 or self.tolerance <= 0:
            raise ValueError("max_step and tolerance must be positive")

    @property
    def time_slack(self) -> float:
        return 1e-9 * self.tolerance

    def start(self, state: ModelState) -> CashKarpRun:
        """Binds the settings to a model state; the run keeps h between steps."""
        return CashKarpRun(self, state)


class CashKarpRun:
    """Port of the step part of `RK5Adaptive::simulate` on one model state."""

    def __init__(self, config: CashKarp, state: ModelState) -> None:
        self.state = state
        self.max_step = config.max_step
        self.tolerance = config.tolerance
        self.fidelity = config.fidelity
        self.h = config.max_step

    def step(self, t: float, t_bound: float, derivatives: FloatArray) -> StepResult:
        """Takes one accepted step from t, no further than t_bound.

        `derivatives` are those of the current positions. The step shrinks until the
        scaled error passes; the per-step updates then get the step `CashKarp`
        describes (S2), and `accepted` is False if it had to shrink.

        Raises:
            RuntimeError: If the step size underflows.
        """
        state = self.state
        h = self.h
        scale = (
            np.abs(state.positions) + np.abs(derivatives * h) + 1e-9 * self.tolerance
        )
        if t + h > t_bound:
            h = t_bound - t
        h_did, h_next = self._find_accepted_step(t, h, derivatives, scale)
        if self.fidelity:
            state.update(h)
        else:
            # volume_change is the only diagnostic carrying state between calls.
            volume_change = state.tissue.cells.volume_change.copy()
            state.compute_forces()
            state.tissue.cells.volume_change[:] = volume_change
            state.update(h_did)
        self.h = min(h_next, self.max_step)
        return StepResult(t=t + h_did, accepted=h_did == h)

    def _find_accepted_step(
        self, t: float, h_try: float, derivatives: FloatArray, scale: FloatArray
    ) -> tuple[float, float]:
        """Port of `rkqs`: shrinks h until the scaled error passes, writes the new
        positions and returns (step taken, next step)."""
        state = self.state
        h = h_try
        while True:
            positions, error = _compute_cash_karp_step(state, derivatives, h)
            err_max = max(float(np.max(np.abs(error / scale))), 0.0) / self.tolerance
            if err_max <= 1.0:
                break
            h_temp = _SAFETY * h * math.pow(err_max, _PSHRINK)
            h = h_temp if h_temp > 0.1 * h else 0.1 * h  # h > 0 (simulate checks)
            if t + h == t:
                raise RuntimeError("Step size underflow")
        h_next = (
            _SAFETY * h * math.pow(err_max, _PGROW) if err_max > _ERRCON else 5.0 * h
        )
        state.positions[:] = positions
        return h, h_next


def _compute_cash_karp_step(
    state: ModelState, k1: FloatArray, h: float
) -> tuple[FloatArray, FloatArray]:
    """Port of `rkck`: one Cash–Karp step of length h from the state's positions
    with derivatives k1; returns (new positions, error estimate).

    Stage derivatives see a copy of the cell state, so their side effects are
    discarded as in Tissue; their diagnostics are therefore skipped.
    """
    y = state.positions
    cells = state.tissue.cells

    def stage(trial: FloatArray) -> FloatArray:
        return state.compute_forces_at(cells.copy(), trial, record_diagnostics=False)

    k2 = stage(y + _B21 * h * k1)
    k3 = stage(y + h * (_B31 * k1 + _B32 * k2))
    k4 = stage(y + h * (_B41 * k1 + _B42 * k2 + _B43 * k3))
    k5 = stage(y + h * (_B51 * k1 + _B52 * k2 + _B53 * k3 + _B54 * k4))
    k6 = stage(y + h * (_B61 * k1 + _B62 * k2 + _B63 * k3 + _B64 * k4 + _B65 * k5))
    y_out = y + h * (_C1 * k1 + _C3 * k3 + _C4 * k4 + _C6 * k6)
    error = h * (_DC1 * k1 + _DC3 * k3 + _DC4 * k4 + _DC5 * k5 + _DC6 * k6)
    return y_out, error
