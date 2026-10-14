"""Mechanical feedback on wall material: fibre stiffness and microtubule alignment.

Ports `FiberModel::General` and `UpdateMTDirectionEquilibrium` from Tissue e74e457.
Both are explicit updates applied once per accepted solver step.
"""

from dataclasses import dataclass
from typing import Literal

import numba
import numpy as np

from tissue_lite import layout
from tissue_lite.tissue import FloatArray

type Coupling = Literal["linear", "hill", "hill_direct"]
type Initialisation = Literal["none", "isotropic", "anisotropic"]

COUPLING_CODES: dict[Coupling, int] = {"linear": 0, "hill": 1, "hill_direct": 2}


@dataclass(frozen=True)
class FibreStiffness:
    """Relaxes each cell's longitudinal modulus towards a stress-anisotropy target.

    Target: `Ym + 0.5 (1 + f(a)) Yf`, with `f(a) = a` (linear) or the Hill function
    `a^n / ((1 - a)^n k^n + a^n)`. "hill_direct" sets `0.5 (1 + f(a)) Yf` at once,
    without Ym or the velocity gate, as Tissue's flag 2 does. Ym and Yf default to
    the membrane's; Tissue lets them differ (see KNOWN_ISSUES).
    """

    rate: float
    coupling: Coupling = "hill"
    k_hill: float = 0.4
    n_hill: float = 2.0
    velocity_threshold: float = 300.0
    initialise: Initialisation = "anisotropic"
    young_matrix: float | None = None
    young_fibre: float | None = None

    def moduli(self, young_matrix: float, young_fibre: float) -> tuple[float, float]:
        """Returns (Ym, Yf), falling back to the given membrane moduli."""
        return (
            young_matrix if self.young_matrix is None else self.young_matrix,
            young_fibre if self.young_fibre is None else self.young_fibre,
        )


@dataclass(frozen=True)
class MicrotubuleAlignment:
    """Rotates each cell's microtubule axis towards its principal stress direction.

    Without thresholds every cell updates. With `velocity_threshold`, only cells whose
    mean vertex speed is below it update; `stress_difference_threshold` (requires the
    velocity gate) also requires `|(s_max - s_vm) / s_max|` to exceed it.
    """

    rate: float
    velocity_threshold: float | None = 300.0
    stress_difference_threshold: float | None = 0.0

    def __post_init__(self) -> None:
        if self.stress_difference_threshold is not None and (
            self.velocity_threshold is None
        ):
            raise ValueError("stress_difference_threshold needs velocity_threshold")

    def gate(self) -> tuple[int, float, float]:
        """Returns (gate level, velocity threshold, stress threshold) for the kernel."""
        if self.velocity_threshold is None:
            return 0, 0.0, 0.0
        if self.stress_difference_threshold is None:
            return 1, self.velocity_threshold, 0.0
        return 2, self.velocity_threshold, self.stress_difference_threshold


@numba.njit(cache=True)
def _hill_fraction(a: float, k: float, n: float) -> float:
    """Returns Tissue's `a^n / ((1 - a)^n k^n + a^n)`."""
    return a**n / ((1 - a) ** n * k**n + a**n)


def initiate_fibres(
    fibres: FibreStiffness,
    young_matrix: float,
    young_fibre: float,
    cell_vars: FloatArray,
) -> None:
    """Sets the initial longitudinal modulus from the stored stress anisotropy."""
    if fibres.initialise == "isotropic":
        cell_vars[:, layout.STRESS_ANISOTROPY] = 0
        cell_vars[:, layout.YOUNG_LONGITUDINAL] = young_matrix + 0.5 * young_fibre
    elif fibres.initialise == "anisotropic":
        for c in range(len(cell_vars)):
            a = float(cell_vars[c, layout.STRESS_ANISOTROPY])
            f = (
                a
                if fibres.coupling == "linear"
                else _hill_fraction(a, fibres.k_hill, fibres.n_hill)
            )
            cell_vars[c, layout.YOUNG_LONGITUDINAL] = (
                young_matrix + 0.5 * (1 + f) * young_fibre
            )


@numba.njit(cache=True)
def update_fibres(
    cell_vars: FloatArray,
    h: float,
    rate: float,
    coupling: int,
    k: float,
    n: float,
    velocity_threshold: float,
    young_matrix: float,
    young_fibre: float,
) -> None:
    """Applies one explicit fibre-stiffness update of length h."""
    if rate == 0.0:
        return
    for c in range(cell_vars.shape[0]):
        a = cell_vars[c, layout.STRESS_ANISOTROPY]
        young_l = cell_vars[c, layout.YOUNG_LONGITUDINAL]
        gate = (
            cell_vars[c, layout.VELOCITY] < velocity_threshold
            and young_l < young_matrix + young_fibre
        )
        if coupling == 0 and gate:
            delta = young_matrix + 0.5 * (1 + a) * young_fibre - young_l
            cell_vars[c, layout.YOUNG_LONGITUDINAL] += rate * h * delta
        hill = _hill_fraction(a, k, n)
        if coupling == 1 and gate:
            delta = young_matrix + 0.5 * (1 + hill) * young_fibre - young_l
            cell_vars[c, layout.YOUNG_LONGITUDINAL] += rate * h * delta
        if coupling == 2:
            cell_vars[c, layout.YOUNG_LONGITUDINAL] = 0.5 * (1 + hill) * young_fibre


def initiate_microtubules(cell_vars: FloatArray) -> None:
    """Copies the stored principal stress direction into the microtubule axis."""
    cell_vars[:, layout.MICROTUBULE] = cell_vars[:, layout.STRESS_DIRECTION]


@numba.njit(cache=True)
def update_microtubules(
    cell_vars: FloatArray,
    h: float,
    rate: float,
    gate_level: int,
    velocity_threshold: float,
    stress_difference_threshold: float,
) -> None:
    """Applies one explicit microtubule update of length h.

    gate_level: 0 no gate, 1 velocity, 2 velocity and stress difference.
    """
    if rate == 0.0:
        return
    target = layout.STRESS_START
    mt = layout.MICROTUBULE_START
    for c in range(cell_vars.shape[0]):
        if gate_level >= 1 and not (cell_vars[c, layout.VELOCITY] < velocity_threshold):
            continue
        if gate_level == 2:
            s_max = cell_vars[c, layout.STRESS_MAX]
            diff = abs((s_max - cell_vars[c, layout.VON_MISES_STRESS]) / s_max)
            if not diff > stress_difference_threshold:
                continue
        dot = 0.0
        for d in range(3):
            dot += cell_vars[c, mt + d] * cell_vars[c, target + d]
        if dot < 0:
            for d in range(3):
                cell_vars[c, mt + d] *= -1
        for d in range(3):
            cell_vars[c, mt + d] += (
                rate * h * (cell_vars[c, target + d] - cell_vars[c, mt + d])
            )
        norm = 0.0
        for d in range(3):
            norm += cell_vars[c, mt + d] * cell_vars[c, mt + d]
        norm = 1.0 / np.sqrt(norm)
        for d in range(3):
            cell_vars[c, mt + d] *= norm
