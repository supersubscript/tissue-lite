"""Wall material, turgor pressure and boundary definitions, plus the material law.

Ports the parameters of `VertexFromTRBScenterTriangulationMT` (plane stress, MF 1
and 10), `Pressure3D::CenterTriangulation::Linear` (area flag 1) and
`VertexNoUpdateBoundary` from Tissue e74e457. The kernels keep Tissue's
evaluation order so that results are bit-identical; do not simplify them
algebraically.
"""

from dataclasses import dataclass
from typing import NamedTuple

import numba
import numpy as np

from tissue_lite.arrays import IntArray
from tissue_lite.rand import glibc_rand
from tissue_lite.tissue import CellState
from tissue_lite.triangles import Matrix2, Vector, dot

_PI = 3.14159265
RANDOM_THRESHOLD = 0.0001


@dataclass(frozen=True)
class ConcentrationSoftening:
    """Hill repression of both moduli by `CellState.concentration` (Tissue's MF=10).

    Both moduli are scaled by `1 - c^n / (k^n + c^n)`, e.g. for auxin-driven wall
    loosening.
    """

    k_hill: float
    n_hill: float


class Material(NamedTuple):
    """The membrane scalars as the Numba kernels take them."""

    young_matrix: float
    young_fibre: float
    poisson_longitudinal: float
    poisson_transverse: float
    double_rest_lengths: bool
    soften: bool
    k_hill: float
    n_hill: float


@dataclass(frozen=True)
class TRBSMembrane:
    """Anisotropic triangular biquadratic springs on the centre triangulation.

    The longitudinal modulus is read from the cell state (set by `FibreStiffness`);
    the transverse one is `2 young_matrix + young_fibre - longitudinal`.
    """

    young_matrix: float
    young_fibre: float
    poisson_longitudinal: float = 0.2
    poisson_transverse: float = 0.2
    double_rest_lengths: bool = False
    softening: ConcentrationSoftening | None = None

    def __post_init__(self) -> None:
        for nu in (self.poisson_longitudinal, self.poisson_transverse):
            if not 0 <= nu < 0.5:
                raise ValueError("Poisson ratios must be in [0, 0.5)")

    def to_kernel(self) -> Material:
        """Flattens the definition for the kernels."""
        s = self.softening
        return Material(
            young_matrix=self.young_matrix,
            young_fibre=self.young_fibre,
            poisson_longitudinal=self.poisson_longitudinal,
            poisson_transverse=self.poisson_transverse,
            double_rest_lengths=self.double_rest_lengths,
            soften=s is not None,
            k_hill=0.0 if s is None else s.k_hill,
            n_hill=0.0 if s is None else s.n_hill,
        )


@dataclass(frozen=True)
class Pressure:
    """Turgor pressure on each triangle, proportional to its area, ramped up linearly.

    The pressure factor starts at 0 and grows by `h / ramp_time` per accepted step.
    """

    pressure: float
    ramp_time: float

    def __post_init__(self) -> None:
        if self.ramp_time <= 0:
            raise ValueError("ramp_time must be positive")


@dataclass(frozen=True)
class Boundary:
    """Vertices on the tissue edge do not move along `fixed_axes`."""

    fixed_axes: tuple[int, ...] = (0, 1, 2)

    def __post_init__(self) -> None:
        if not set(self.fixed_axes) <= {0, 1, 2}:
            raise ValueError("fixed_axes must be a subset of (0, 1, 2)")


@numba.njit(cache=True)
def compute_moduli(cells: CellState, c: int, material: Material) -> tuple[float, float]:
    """Returns (young_L, young_T) of cell c, softened when the material says so."""
    young_l = cells.young_longitudinal[c]
    young_t = 2 * material.young_matrix + material.young_fibre - young_l
    if material.soften:
        a = cells.concentration[c]
        k = material.k_hill
        n = material.n_hill
        frac = 1 - a**n / (k**n + a**n)
        young_l *= frac
        young_t *= frac
    return young_l, young_t


@numba.njit(cache=True)
def compute_lame_parameters(young: float, poisson: float) -> tuple[float, float]:
    """Returns the plane-stress (λ, μ) of a modulus, as Tissue forms them."""
    return young * poisson / (1 - poisson * poisson), young / (2 * (1 + poisson))


@numba.njit(cache=True)
def random_direction(rand_state: IntArray) -> tuple[float, float]:
    """Tissue's fallback when the projected microtubule direction vanishes."""
    angle = ((glibc_rand(rand_state) % 360) * 2 * _PI) / 360
    return np.cos(angle), np.sin(angle)


@numba.njit(cache=True)
def pull_back_direction(
    gradient: Matrix2, x_axis: Vector, y_axis: Vector, direction: Vector
) -> tuple[float, float]:
    """Projects a direction onto the triangle plane and maps it to the rest plane
    with adj(F): F⁻¹ without the 1/det F (Tissue, KNOWN_ISSUES T2). Unnormalised."""
    local_x = dot(x_axis, direction)
    local_y = dot(y_axis, direction)
    return (
        gradient[1][1] * local_x - gradient[0][1] * local_y,
        -gradient[1][0] * local_x + gradient[0][0] * local_y,
    )


@numba.njit(cache=True)
def compute_fibre_direction(
    gradient: Matrix2,
    x_axis: Vector,
    y_axis: Vector,
    microtubule: Vector,
    rand_state: IntArray,
) -> tuple[float, float]:
    """Unit microtubule axis in the rest plane; a random one below
    `RANDOM_THRESHOLD`, consuming `rand_state` (KNOWN_ISSUES T1)."""
    a0, a1 = pull_back_direction(gradient, x_axis, y_axis, microtubule)
    measure = np.sqrt(a0 * a0 + a1 * a1)
    if measure < RANDOM_THRESHOLD:
        return random_direction(rand_state)
    return a0 / measure, a1 / measure


@numba.njit(cache=True)
def compute_green_strain(gradient: Matrix2) -> Matrix2:
    """Returns E = (FᵀF - I) / 2 in Tissue's expression order."""
    (f00, f01), (f10, f11) = gradient
    return (
        (0.5 * (f00 * f00 + f10 * f10 - 1), 0.5 * (f00 * f01 + f10 * f11)),
        (0.5 * (f01 * f00 + f11 * f10), 0.5 * (f01 * f01 + f11 * f11 - 1)),
    )


@numba.njit(cache=True)
def compute_fibre_strain(strain: Matrix2, fibre: tuple[float, float]) -> float:
    """Returns aᵀ E a for the in-plane unit fibre direction a."""
    (e00, e01), (e10, e11) = strain
    a0, a1 = fibre
    return a0 * a0 * e00 + a0 * a1 * (e01 + e10) + a1 * a1 * e11


@numba.njit(cache=True)
def compute_anisotropic_stress(
    strain: Matrix2,
    fibre: tuple[float, float],
    delta_lambda: float,
    delta_mu: float,
) -> Matrix2:
    """Fibre part of the second Piola-Kirchhoff stress, with Δλ = λ_L - λ_T and
    Δμ = μ_L - μ_T: (Δλ/2) (tr E · aaᵀ + aᵀEa · I) + Δμ (E aaᵀ + aaᵀ E)."""
    (e00, e01), (e10, e11) = strain
    a0, a1 = fibre
    aa00 = a0 * a0
    aa01 = a0 * a1
    aa10 = a1 * a0
    aa11 = a1 * a1
    trace = e00 + e11
    fibre_strain = compute_fibre_strain(strain, fibre)
    eaa00 = e00 * aa00 + e01 * aa10
    eaa01 = e00 * aa01 + e01 * aa11
    eaa10 = e10 * aa00 + e11 * aa10
    eaa11 = e10 * aa01 + e11 * aa11
    aae00 = aa00 * e00 + aa01 * e10
    aae01 = aa00 * e01 + aa01 * e11
    aae10 = aa10 * e00 + aa11 * e10
    aae11 = aa10 * e01 + aa11 * e11
    half_lambda = delta_lambda / 2
    return (
        (
            half_lambda * (trace * aa00 + fibre_strain) + delta_mu * (eaa00 + aae00),
            half_lambda * (trace * aa01) + delta_mu * (eaa01 + aae01),
        ),
        (
            half_lambda * (trace * aa10) + delta_mu * (eaa10 + aae10),
            half_lambda * (trace * aa11 + fibre_strain) + delta_mu * (eaa11 + aae11),
        ),
    )
