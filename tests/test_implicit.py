"""Tests for the implicit stiff integrator."""

from typing import Literal

import numpy as np
import pytest
from conftest import CASES_DIR, case_state

from tissue_lite.solvers.cash_karp import CashKarp
from tissue_lite.solvers.implicit import Implicit
from tissue_lite.solvers.simulate import integrate


@pytest.mark.parametrize("method", ["Radau", "BDF"])
def test_window_matches_cash_karp(method: Literal["Radau", "BDF"]) -> None:
    reference = case_state(CASES_DIR / "iso")
    reference.pressure_factor = 1.0
    implicit = case_state(CASES_DIR / "iso")
    implicit.pressure_factor = 1.0
    before = reference.tissue.vertices.copy()
    integrate(reference, 0.0, 0.5, 0, CashKarp())
    integrate(implicit, 0.0, 0.5, 0, Implicit(step=0.5, method=method))
    moved = np.abs(reference.tissue.vertices - before).max()
    assert moved > 1e-2
    assert (
        np.abs(implicit.tissue.vertices - reference.tissue.vertices).max()
        < 1e-3 * moved
    )


def test_long_window_approaches_equilibrium() -> None:
    state = case_state(CASES_DIR / "iso")
    run = Implicit(step=50.0).start(state)
    run.step(0.0, 50.0, state.compute_forces())
    assert run.internal_steps > 1
    assert np.abs(state.compute_forces()).max() < 1e-6


def test_rejects_bad_settings() -> None:
    with pytest.raises(ValueError):
        Implicit(step=0.0)
