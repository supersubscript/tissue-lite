"""Implicit stiff integration of the overdamped dynamics between feedback updates.

Each step applies the model's per-step updates with the step length, as the
quasi-static solver does, then integrates d(positions)/dt = forces over a window of
that length with one of scipy's stiff solvers (Radau or BDF, sparse finite-difference
Jacobian). Unlike Tissue's Cash–Karp, the window length is independent of the
stiffness (KNOWN_ISSUES N1).
"""

from dataclasses import dataclass
from typing import Literal

from scipy.integrate import solve_ivp

from tissue_lite.arrays import FloatArray
from tissue_lite.model import ModelState
from tissue_lite.solvers.sparsity import build_jacobian_pattern
from tissue_lite.solvers.stepper import StepResult


@dataclass(frozen=True)
class Implicit:
    """scipy's `solve_ivp`, stiff method, over feedback windows of length `step`."""

    step: float = 1.0
    method: Literal["Radau", "BDF"] = "Radau"
    rtol: float = 1e-6
    atol: float = 1e-9

    def __post_init__(self) -> None:
        if self.step <= 0 or self.rtol <= 0 or self.atol <= 0:
            raise ValueError("step, rtol and atol must be positive")

    @property
    def time_slack(self) -> float:
        return 1e-9 * self.step

    def start(self, state: ModelState) -> ImplicitRun:
        """Binds the settings to a model state and builds its Jacobian pattern."""
        return ImplicitRun(self, state)


class ImplicitRun:
    """One implicit run; counts the integrator's force evaluations."""

    def __init__(self, config: Implicit, state: ModelState) -> None:
        self.config = config
        self.state = state
        self.pattern = build_jacobian_pattern(state)
        self.evaluations = 0
        self.internal_steps = 0

    def step(self, t: float, t_bound: float, derivatives: FloatArray) -> StepResult:
        """Applies the feedback update, then integrates one window.

        Raises:
            RuntimeError: If the integrator fails.
        """
        state = self.state
        h = min(self.config.step, t_bound - t)
        state.update(h)
        trial = state.positions.copy()

        def fun(_t: float, y: FloatArray) -> FloatArray:
            trial.ravel()[:] = y
            return state.compute_trial_forces(trial).ravel()

        solution = solve_ivp(
            fun,
            (t, t + h),
            state.positions.ravel(),
            method=self.config.method,
            rtol=self.config.rtol,
            atol=self.config.atol,
            jac_sparsity=self.pattern,
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        self.evaluations += int(solution.nfev)
        self.internal_steps += len(solution.t) - 1
        state.positions.ravel()[:] = solution.y[:, -1]
        return StepResult(t=t + h, accepted=True)
