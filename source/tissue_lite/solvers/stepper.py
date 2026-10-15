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
    """One solver bound to one model state.

    `step` takes the derivatives of the current state, which the driver evaluates first
    so that their diagnostics precede any frame written at t.
    """

    def step(self, t: float, t_bound: float, derivatives: FloatArray) -> StepResult:
        """Advances from t to at most t_bound, then applies the model's `update`."""
        ...


class Stepper(Protocol):
    """A solver configuration."""

    @property
    def fidelity(self) -> bool:
        """Whether runs print t_end twice, as Tissue does (KNOWN_ISSUES S6).

        TODO: remove `fidelity` (here and on the steppers) long-term. It only reproduces
        a Tissue artefact; the non-fidelity schedule should become the only one.
        """
        ...

    @property
    def time_slack(self) -> float:
        """Added to the frame schedule; Tissue's `1e-9 * tolerance`."""
        ...

    def start(self, state: ModelState) -> StepRun: ...
