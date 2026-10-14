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

    def step(self, t: float, t_bound: float, derivatives: FloatArray) -> StepResult:
        """Advances the state from t to at most t_bound and applies `update`.

        `derivatives` are those of the current state, already evaluated by the
        driver so that their diagnostics precede any frame written at t.
        """
        ...


class Stepper(Protocol):
    """A solver configuration."""

    @property
    def time_slack(self) -> float:
        """Added to the frame schedule; Tissue's `1e-9 * tolerance`."""
        ...

    def start(self, state: ModelState) -> StepRun: ...
