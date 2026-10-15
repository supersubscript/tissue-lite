"""Quasi-static solver: mechanical equilibrium after each feedback update.

Each step applies the model's per-step updates with the step length (fibres,
microtubules, growth, pressure ramp), then moves the free positions to where the
forces vanish. A boundary that leaves rigid motions free (`Boundary.fixed_axes`
shorter than all three axes) lets the pressure on a non-planar edge drive a rigid
drift that no position removes; equilibrium is then up to that drift, which is
discarded (KNOWN_ISSUES N2).

The equilibrium comes from pseudo-transient continuation (Kelley & Keyes, SIAM J.
Numer. Anal. 35, 1998): implicit Euler steps of pseudo-time τ on the overdamped
dynamics, which follow the physical path through snap-through inflations at small
τ and become Newton steps as τ grows. Each pseudo-step takes two simplified Newton
corrections with the Jacobian J of its start, (I/τ - J) δ = residual, and is
accepted when the second is smaller than the first (Deuflhard's natural
monotonicity test, Newton Methods for Nonlinear Problems, 2004, §3.3); τ shrinks
on rejection and grows on acceptance. The free rigid modes border the system so
that no step moves the tissue rigidly. J is a coloured finite difference
(`solvers.sparsity`).
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from tissue_lite.arrays import BoolArray, FloatArray
from tissue_lite.model import ModelState
from tissue_lite.solvers.sparsity import (
    build_jacobian_pattern,
    colour_columns,
    compute_jacobian,
    restrict_pattern,
)
from tissue_lite.solvers.stepper import StepResult

_THETA_MAX = 0.5  # contraction a pseudo-step must show to be accepted
_THETA_FAST = 0.25  # below this τ doubles, otherwise it grows by 1.5
_TAU_SHRINK = 4.0  # on a rejected step
_TAU_MAX = 1e12


@dataclass(frozen=True)
class QuasiStatic:
    """Equilibrium after every feedback update of length `step`.

    `force_tolerance` bounds the largest free force component (a velocity in
    Tissue's units) at equilibrium, after removing the rigid drift the boundary
    leaves free. `max_iterations` bounds the pseudo-transient Newton iterations per
    step; the pseudo-time starts at `step`.

    Raises:
        ValueError: If a setting is not positive.
    """

    step: float = 0.1
    force_tolerance: float = 1e-6
    max_iterations: int = 5000  # stiff domes need ~1000 on their first step

    def __post_init__(self) -> None:
        if min(self.step, self.force_tolerance, self.max_iterations) <= 0:
            raise ValueError("Every QuasiStatic setting must be positive")

    @property
    def time_slack(self) -> float:
        return 1e-9 * self.step

    def start(self, state: ModelState) -> QuasiStaticRun:
        """Binds the settings to a model state and colours its force Jacobian."""
        return QuasiStaticRun(self, state)


class QuasiStaticRun:
    """One quasi-static run; counts Newton iterations and force evaluations.

    `tau` is the pseudo-time step, carried over between steps: it ends large after
    a converged solve, so a step from a nearby equilibrium starts as Newton.
    """

    def __init__(self, config: QuasiStatic, state: ModelState) -> None:
        self.config = config
        self.state = state
        num_centres = state.positions.shape[0] - state.fixed.shape[0]
        self.free: BoolArray = np.concatenate(
            [~state.fixed.ravel(), np.ones(3 * num_centres, dtype=bool)]
        )
        self.pattern = restrict_pattern(build_jacobian_pattern(state), self.free)
        self.colours = colour_columns(self.pattern)
        self.tau = config.step
        self.iterations = 0
        self.evaluations = 0
        self._trial = state.positions.copy()

    def step(self, t: float, t_bound: float, derivatives: FloatArray) -> StepResult:
        """Applies the feedback update, then moves the positions to equilibrium.

        `derivatives` are not used: the forces are re-evaluated as a pure function
        of the positions (`ModelState.compute_trial_forces`).

        Raises:
            RuntimeError: If no equilibrium is found within `max_iterations`.
        """
        h = min(self.config.step, t_bound - t)
        self.state.update(h)
        self._solve_equilibrium()
        return StepResult(t=t + h, accepted=True)

    def _solve_equilibrium(self) -> None:
        """Pseudo-transient Newton on the free positions until the drift-free
        forces are within `force_tolerance`."""
        config = self.config
        positions = self.state.positions.ravel()
        y = positions[self.free]
        f = self._residual(y)
        identity = sp.identity(len(y), format="csc")
        for _ in range(config.max_iterations):
            modes = self._rigid_modes(y)
            if np.abs(_remove_drift(f, modes)).max() <= config.force_tolerance:
                positions[self.free] = y
                return
            jacobian = compute_jacobian(
                self._residual, y, f, self.pattern, self.colours
            )
            while True:
                system = identity / self.tau - jacobian
                first, drift = _solve_bordered(system, modes, f)
                f_first = self._residual(y + first)
                euler_residual = first / self.tau - f_first + modes @ drift
                second, _ = _solve_bordered(system, modes, -euler_residual)
                theta = float(np.linalg.norm(second) / np.linalg.norm(first))
                if np.isfinite(theta) and theta < _THETA_MAX:
                    break
                self.tau = min(self.tau / _TAU_SHRINK, config.step)
                if self.tau < config.time_slack:
                    raise RuntimeError("Pseudo-time step underflow")
            self.tau = min(self.tau * (2.0 if theta < _THETA_FAST else 1.5), _TAU_MAX)
            y = y + first + second
            f = self._residual(y)
            self.iterations += 1
        raise RuntimeError(f"No equilibrium within {config.max_iterations} iterations")

    def _residual(self, y: FloatArray) -> FloatArray:
        """The free forces at the free positions y; a pure function, counted."""
        self._trial.ravel()[self.free] = y
        self.evaluations += 1
        return self.state.compute_trial_forces(self._trial).ravel()[self.free]

    def _rigid_modes(self, y: FloatArray) -> FloatArray:
        """Orthonormal basis (n_free, m) of the rigid motions the boundary leaves free.

        Translation along an axis the boundary does not fix, and rotation about an
        axis (through the centroid) when it fixes no other axis; m is 0 for a fully
        fixed boundary (KNOWN_ISSUES N2).
        """
        self._trial.ravel()[self.free] = y
        positions = self._trial
        fixed_axes = self.state.fixed.any(axis=0)
        modes = []
        for axis in np.flatnonzero(~fixed_axes):
            mode = np.zeros_like(positions)
            mode[:, axis] = 1.0
            modes.append(mode)
        centred = positions - positions.mean(axis=0)
        # The boundary alone decides: the fibres stay fixed in space, so with fibre
        # stiffness a torque that a relative rotation could balance is discarded too.
        for axis in range(3):
            if not fixed_axes[np.arange(3) != axis].any():
                modes.append(np.cross(np.eye(3)[axis], centred))
        if not modes:
            return np.zeros((len(y), 0))
        basis, _ = np.linalg.qr(np.stack([m.ravel()[self.free] for m in modes], axis=1))
        return basis


def _remove_drift(f: FloatArray, modes: FloatArray) -> FloatArray:
    """Returns f without its component along the orthonormal rigid modes."""
    return f - modes @ (modes.T @ f)


def _solve_bordered(
    system: sp.csc_array, modes: FloatArray, rhs: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Solves `system` δ - modes ν = rhs with modesᵀ δ = 0; returns (δ, ν).

    δ moves nothing rigidly and ν is the rigid drift left in the forces.
    """
    n, m = modes.shape
    if m == 0:
        return spla.splu(sp.csc_array(system)).solve(rhs), np.zeros(0)
    bordered = sp.bmat(
        [[system, sp.csc_array(-modes)], [sp.csc_array(modes.T), None]], format="csc"
    )
    solution = spla.splu(bordered).solve(np.concatenate([rhs, np.zeros(m)]))
    return solution[:n], solution[n:]
