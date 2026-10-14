"""Mechanical feedback on wall material: fibre stiffness and microtubule alignment.

Ports `FiberModel::General` and `UpdateMTDirectionEquilibrium` from Tissue e74e457.
Both are explicit Euler updates applied once per accepted solver step, outside its
error control and with the trial step length (KNOWN_ISSUES S4, S2).
"""

from dataclasses import dataclass
from typing import Literal, NamedTuple

import numba

from tissue_lite.arrays import FloatArray
from tissue_lite.material import TRBSMembrane
from tissue_lite.tissue import CellState
from tissue_lite.triangles import dot, norm

type Coupling = Literal["linear", "hill", "hill_direct"]
type Initialisation = Literal["none", "isotropic", "anisotropic"]

COUPLING_CODES: dict[Coupling, int] = {"linear": 0, "hill": 1, "hill_direct": 2}


class FibreParameters(NamedTuple):
    """`FibreStiffness` as the kernel takes it.

    `coupling` is the `COUPLING_CODES` value of the coupling name.
    """

    rate: float
    coupling: int
    k_hill: float
    n_hill: float
    velocity_threshold: float
    young_matrix: float
    young_fibre: float


class MicrotubuleParameters(NamedTuple):
    """`MicrotubuleAlignment` as the kernel takes it.

    Gate levels: 0 none, 1 velocity, 2 velocity and stress difference.
    """

    rate: float
    gate_level: int
    velocity_threshold: float
    stress_difference_threshold: float


@dataclass(frozen=True)
class FibreStiffness:
    """Relaxes each cell's longitudinal modulus towards a stress-anisotropy target.

    Target: `Ym + 0.5 (1 + f(a)) Yf`, with `f(a) = a` (linear) or the Hill function
    `a^n / ((1 - a)^n k^n + a^n)` of the stress anisotropy `a`. "linear" and "hill"
    update only cells slower than `velocity_threshold` whose modulus is below `Ym + Yf`.
    "hill_direct" sets `0.5 (1 + f(a)) Yf` at once, without Ym or the velocity gate, as
    Tissue's flag 2 does (KNOWN_ISSUES F1). Ym and Yf default to the membrane's; Tissue
    lets them differ (F2). A `rate` of 0 disables the update but not `initialise` (F4).
    """

    rate: float
    coupling: Coupling = "hill"
    k_hill: float = 0.4
    n_hill: float = 2.0
    velocity_threshold: float = 300.0
    initialise: Initialisation = "anisotropic"
    young_matrix: float | None = None
    young_fibre: float | None = None

    def get_moduli(self, membrane: TRBSMembrane) -> tuple[float, float]:
        """Returns (Ym, Yf), falling back to the membrane's moduli."""
        return (
            membrane.young_matrix if self.young_matrix is None else self.young_matrix,
            membrane.young_fibre if self.young_fibre is None else self.young_fibre,
        )

    def to_kernel(self, membrane: TRBSMembrane) -> FibreParameters:
        """Flattens the definition for `update_fibres`."""
        young_matrix, young_fibre = self.get_moduli(membrane)
        return FibreParameters(
            rate=self.rate,
            coupling=COUPLING_CODES[self.coupling],
            k_hill=self.k_hill,
            n_hill=self.n_hill,
            velocity_threshold=self.velocity_threshold,
            young_matrix=young_matrix,
            young_fibre=young_fibre,
        )


@dataclass(frozen=True)
class MicrotubuleAlignment:
    """Rotates each cell's microtubule axis towards its principal stress direction.

    Without thresholds every cell updates. With `velocity_threshold`, only cells whose
    mean vertex speed is below it update; `stress_difference_threshold` (requires the
    velocity gate) also requires `|(s_max - s_vm) / s_max|` to exceed it.

    Raises:
        ValueError: If `stress_difference_threshold` lacks a `velocity_threshold`.
    """

    rate: float
    velocity_threshold: float | None = 300.0
    stress_difference_threshold: float | None = 0.0

    def __post_init__(self) -> None:
        if self.stress_difference_threshold is not None and (
            self.velocity_threshold is None
        ):
            raise ValueError("stress_difference_threshold needs velocity_threshold")

    def to_kernel(self) -> MicrotubuleParameters:
        """Flattens the definition for `update_microtubules`."""
        if self.velocity_threshold is None:
            return MicrotubuleParameters(self.rate, 0, 0.0, 0.0)
        if self.stress_difference_threshold is None:
            return MicrotubuleParameters(self.rate, 1, self.velocity_threshold, 0.0)
        return MicrotubuleParameters(
            self.rate, 2, self.velocity_threshold, self.stress_difference_threshold
        )


@numba.njit(cache=True)
def _hill_fraction(a: float | FloatArray, k: float, n: float) -> float | FloatArray:
    """Returns Tissue's `a^n / ((1 - a)^n k^n + a^n)`."""
    return a**n / ((1 - a) ** n * k**n + a**n)


def initiate_fibres(
    fibres: FibreStiffness, membrane: TRBSMembrane, cells: CellState
) -> None:
    """Sets the initial longitudinal modulus from the stored stress anisotropy.

    Writes `cells.young_longitudinal` unless `fibres.initialise` is "none", even when
    the rate is 0 (KNOWN_ISSUES F4). "isotropic" also zeroes `cells.stress_anisotropy`.
    "hill_direct" includes Ym here, unlike the update (F1).
    """
    young_matrix, young_fibre = fibres.get_moduli(membrane)
    if fibres.initialise == "isotropic":
        cells.stress_anisotropy[:] = 0
        cells.young_longitudinal[:] = young_matrix + 0.5 * young_fibre
    elif fibres.initialise == "anisotropic":
        a = cells.stress_anisotropy
        f = (
            a
            if fibres.coupling == "linear"
            # The Numba kernel keeps libm pow; numpy's power can differ in the last bit.
            else _hill_fraction(a, fibres.k_hill, fibres.n_hill)
        )
        cells.young_longitudinal[:] = young_matrix + 0.5 * (1 + f) * young_fibre


@numba.njit(cache=True)
def update_fibres(cells: CellState, h: float, fibres: FibreParameters) -> None:
    """Applies one explicit Euler fibre-stiffness update of length h.

    Reads `cells.stress_anisotropy` and `velocity`, writes `cells.young_longitudinal`.
    `h` is the solver's trial step (KNOWN_ISSUES S2) and the velocity is that of the
    previous accepted state (S5). Does nothing when the rate is 0.
    """
    rate, coupling, k, n, velocity_threshold, young_matrix, young_fibre = fibres
    if rate == 0.0:
        return
    for c in range(cells.young_longitudinal.shape[0]):
        a = cells.stress_anisotropy[c]
        young_l = cells.young_longitudinal[c]
        gate = (
            cells.velocity[c] < velocity_threshold
            and young_l < young_matrix + young_fibre
        )
        if coupling == 0 and gate:
            delta = young_matrix + 0.5 * (1 + a) * young_fibre - young_l
            cells.young_longitudinal[c] += rate * h * delta
        hill = _hill_fraction(a, k, n)
        if coupling == 1 and gate:
            delta = young_matrix + 0.5 * (1 + hill) * young_fibre - young_l
            cells.young_longitudinal[c] += rate * h * delta
        if coupling == 2:
            cells.young_longitudinal[c] = 0.5 * (1 + hill) * young_fibre


def initiate_microtubules(cells: CellState) -> None:
    """Copies the stored principal stress direction into the microtubule axis."""
    cells.microtubule[:] = cells.stress_direction


@numba.njit(cache=True)
def update_microtubules(
    cells: CellState, h: float, microtubules: MicrotubuleParameters
) -> None:
    """Applies one explicit Euler microtubule update of length h.

    Flips an axis that points against `cells.stress_direction`, moves it by
    `rate * h * (target - axis)` and renormalises it in place. Cells failing a gate are
    skipped; `h` and the velocity are as in `update_fibres` (KNOWN_ISSUES S2, S5).

    Raises:
        ZeroDivisionError: If the stress-difference gate meets `stress_max == 0`.
    """
    rate, gate_level, velocity_threshold, stress_difference_threshold = microtubules
    if rate == 0.0:
        return
    mt = cells.microtubule
    target = cells.stress_direction
    for c in range(mt.shape[0]):
        if gate_level >= 1 and not (cells.velocity[c] < velocity_threshold):
            continue
        if gate_level == 2:
            s_max = cells.stress_max[c]
            diff = abs((s_max - cells.von_mises_stress[c]) / s_max)
            if not diff > stress_difference_threshold:
                continue
        axis = (mt[c, 0], mt[c, 1], mt[c, 2])
        if dot(axis, (target[c, 0], target[c, 1], target[c, 2])) < 0:
            for d in range(3):
                mt[c, d] *= -1
        for d in range(3):
            mt[c, d] += rate * h * (target[c, d] - mt[c, d])
        scale = 1.0 / norm((mt[c, 0], mt[c, 1], mt[c, 2]))
        for d in range(3):
            mt[c, d] *= scale
