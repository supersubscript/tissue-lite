"""The interface between `simulate` and a solver.

A `Stepper` is a frozen configuration; `start` binds it to a model state and
returns a `StepRun` that advances the state one step at a time and applies the
model's per-step updates itself.
"""

from dataclasses import dataclass
from typing import Protocol

from tissue_lite.arrays import FloatArray
from tissue_lite.model import ModelState


@dataclass(frozen=True)
class StepResult:
    """Where a step ended; `accepted` is False when it had to be retried smaller."""

    t: float
    accepted: bool


class StepRun(Protocol):
    """One solver bound to one model state."""

    time_slack: float
    """Tolerance added to the frame schedule, as Tissue's `1e-9 * tolerance`."""

    def step(
        self,
        t: float,
        t_bound: float,
        vertex_forces: FloatArray,
        centre_forces: FloatArray,
    ) -> StepResult:
        """Advances the state from t to at most t_bound and applies `update`.

        The forces are those of the current state, already evaluated by the driver
        so that their diagnostics precede any frame written at t.
        """
        ...


class Stepper(Protocol):
    """A solver configuration."""

    def start(self, state: ModelState) -> StepRun: ...
