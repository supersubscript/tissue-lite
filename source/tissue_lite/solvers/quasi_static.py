"""Quasi-static solver: mechanical equilibrium after each feedback update.

Each step does two things:

1. Feedback: apply the model's per-step updates (fibres, microtubules, growth,
   pressure ramp) for one step length.
2. Mechanics: move the free node positions until the forces on them vanish.

How the equilibrium is found (pseudo-transient continuation; Kelley & Keyes,
SIAM J. Numer. Anal. 35, 1998):

- The tissue is overdamped, d(positions)/dt = forces. Instead of following this
  in real time, the solver takes implicit Euler steps of an artificial
  "pseudo-time" step τ. Each step solves (I/τ - J) Δx = forces, where J is the
  Jacobian of the forces with respect to the positions.
- Small τ makes a step a short, safe move along the physical path, which matters
  in snap-through inflations. Large τ makes I/τ negligible, and the step becomes a
  plain Newton step, which converges fast near the equilibrium.
- τ adapts to how well the forces behave. Each pseudo-step makes a first
  correction Δx and then a second one with the same matrix, which corrects what
  the first missed. If the second is less than half the first, the step is
  accepted and τ grows. Otherwise it is rejected and τ shrinks (Deuflhard's
  natural monotonicity test, Newton Methods for Nonlinear Problems, 2004, §3.3).
- J comes from finite differences, with columns grouped by colour so that one
  force evaluation fills many columns (`solvers.sparsity`).

Rigid drift: a boundary that fixes fewer than all three axes
(`Boundary.fixed_axes`) leaves some rigid motions free, and the pressure on a
non-planar edge pushes the tissue along them. No position balances that push. The
solver constrains every Δx to be free of rigid motion, reports the leftover push
as `drift`, and ignores it in the convergence test (KNOWN_ISSUES N2).
"""

from collections.abc import Callable
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

_MAX_CONTRACTION = 0.5  # largest |second correction| / |first| that is accepted
_FAST_CONTRACTION = 0.25  # below this τ doubles, otherwise it grows by 1.5
_TAU_SHRINK = 4.0  # τ is divided by this on a rejected step
_TAU_MAX = 1e12

# Solves (I/τ - J) Δx = rhs without rigid motion; returns (Δx, drift).
type RigidFreeSolver = Callable[[FloatArray], tuple[FloatArray, FloatArray]]


@dataclass(frozen=True)
class QuasiStatic:
    """Equilibrium after every feedback update of length `step`.

    `force_tolerance` bounds the largest free force component (a velocity in
    Tissue's units) at equilibrium, after removing the rigid drift the boundary
    leaves free. `max_iterations` bounds the accepted pseudo-steps per step; the
    pseudo-time step τ starts at `step`.

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
    def fidelity(self) -> bool:
        return False

    @property
    def time_slack(self) -> float:
        return 1e-9 * self.step

    def start(self, state: ModelState) -> QuasiStaticRun:
        """Binds the settings to a model state and colours its force Jacobian."""
        return QuasiStaticRun(self, state)


class QuasiStaticRun:
    """One quasi-static run; counts accepted pseudo-steps and force evaluations.

    `free` masks the flattened position components that may move: all cell
    centres and the vertex components the boundary does not fix. `tau` is the
    pseudo-time step, carried over between steps: it ends large after a converged
    solve, so the next step, starting near an equilibrium, begins as Newton.
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
        """Takes pseudo-steps until the free forces, less the rigid drift, are
        within `force_tolerance`, then writes the free positions back."""
        config = self.config
        positions = self.state.positions.ravel()
        free_positions = positions[self.free]
        forces = self._compute_forces(free_positions)
        for _ in range(config.max_iterations):
            rigid_modes = self._compute_rigid_modes(free_positions)
            balanced_forces = _remove_rigid_part(forces, rigid_modes)
            if np.abs(balanced_forces).max() <= config.force_tolerance:
                positions[self.free] = free_positions
                return
            jacobian = compute_jacobian(
                self._compute_forces, free_positions, forces, self.pattern, self.colours
            )
            first, second = self._take_pseudo_step(
                free_positions, forces, jacobian, rigid_modes
            )
            free_positions = free_positions + first + second
            forces = self._compute_forces(free_positions)
            self.iterations += 1
        raise RuntimeError(f"No equilibrium within {config.max_iterations} iterations")

    def _take_pseudo_step(
        self,
        free_positions: FloatArray,
        forces: FloatArray,
        jacobian: sp.csc_array,
        rigid_modes: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        """Returns the two corrections of one accepted implicit Euler step of size τ.

        The implicit Euler equation for a displacement Δx is
        Δx/τ = forces(x + Δx), linearised as (I/τ - J) Δx = forces(x). The first
        correction solves the linearised equation; the second, with the same
        matrix, removes the error left in the full equation. Their size ratio
        measures how linear the forces are over the step. The step is accepted
        when the ratio is below `_MAX_CONTRACTION`; otherwise τ shrinks and the
        step is retried. τ then grows for the next step.

        Raises:
            RuntimeError: If τ shrinks below `time_slack`.
        """
        identity = sp.identity(len(free_positions), format="csc")
        while True:
            solve = _factorise_rigid_free_system(
                identity / self.tau - jacobian, rigid_modes
            )
            first, drift = solve(forces)
            forces_after_first = self._compute_forces(free_positions + first)
            # Residual of the implicit Euler equation after the first correction.
            euler_residual = first / self.tau - forces_after_first + rigid_modes @ drift
            second, _ = solve(-euler_residual)
            contraction = float(np.linalg.norm(second) / np.linalg.norm(first))
            if np.isfinite(contraction) and contraction < _MAX_CONTRACTION:
                break
            self.tau = min(self.tau / _TAU_SHRINK, self.config.step)
            if self.tau < self.config.time_slack:
                raise RuntimeError("Pseudo-time step underflow")
        growth = 2.0 if contraction < _FAST_CONTRACTION else 1.5
        self.tau = min(self.tau * growth, _TAU_MAX)
        return first, second

    def _compute_forces(self, free_positions: FloatArray) -> FloatArray:
        """Returns the free force components at the given free positions.

        A pure function of its argument (the real state is not touched); each
        call is counted in `evaluations`.
        """
        self._trial.ravel()[self.free] = free_positions
        self.evaluations += 1
        return self.state.compute_trial_forces(self._trial).ravel()[self.free]

    def _compute_rigid_modes(self, free_positions: FloatArray) -> FloatArray:
        """Returns an orthonormal basis (n_free, m) of the rigid motions the
        boundary leaves free.

        Translation along each axis the boundary does not fix, and rotation about
        an axis (through the centroid) when the boundary fixes neither of the
        other two. m is 0 for a fully fixed boundary (KNOWN_ISSUES N2).
        """
        self._trial.ravel()[self.free] = free_positions
        positions = self._trial
        fixed_axes = self.state.fixed.any(axis=0)
        modes = []
        for axis in np.flatnonzero(~fixed_axes):
            translation = np.zeros_like(positions)
            translation[:, axis] = 1.0
            modes.append(translation)
        centred = positions - positions.mean(axis=0)
        # The boundary alone decides: the fibres stay fixed in space, so with fibre
        # stiffness a torque that a relative rotation could balance is discarded too.
        for axis in range(3):
            if not fixed_axes[np.arange(3) != axis].any():
                modes.append(np.cross(np.eye(3)[axis], centred))
        if not modes:
            return np.zeros((len(free_positions), 0))
        free_modes = np.stack([mode.ravel()[self.free] for mode in modes], axis=1)
        basis, _ = np.linalg.qr(free_modes)
        return basis


def _remove_rigid_part(forces: FloatArray, rigid_modes: FloatArray) -> FloatArray:
    """Returns the forces without their projection on the orthonormal rigid modes."""
    return forces - rigid_modes @ (rigid_modes.T @ forces)


def _factorise_rigid_free_system(
    system: sp.csc_array, rigid_modes: FloatArray
) -> RigidFreeSolver:
    """Factorises `system` once and returns a solver for any right-hand side.

    Without rigid modes the solver returns system⁻¹ rhs and an empty drift.

    With m rigid modes R (n × m), the displacement Δx must not move the tissue
    rigidly (Rᵀ Δx = 0), so in general no Δx satisfies system Δx = rhs exactly:
    the part of rhs along R cannot be balanced. The solver therefore also
    returns drift ν (length m), the rigid push left over, from the saddle-point
    system

        [ system  -R ] [Δx]   [rhs]
        [ Rᵀ       0 ] [ν ] = [ 0 ].

    Both pseudo-step corrections solve with the same matrix, so it is factorised
    only once.
    """
    n, m = rigid_modes.shape
    if m == 0:
        lu = spla.splu(sp.csc_array(system))
        return lambda rhs: (lu.solve(rhs), np.zeros(0))
    saddle_point = sp.bmat(
        [
            [system, sp.csc_array(-rigid_modes)],
            [sp.csc_array(rigid_modes.T), None],
        ],
        format="csc",
    )
    lu = spla.splu(saddle_point)

    def solve(rhs: FloatArray) -> tuple[FloatArray, FloatArray]:
        solution = lu.solve(np.concatenate([rhs, np.zeros(m)]))
        return solution[:n], solution[n:]

    return solve
