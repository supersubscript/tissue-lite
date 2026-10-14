"""Wall material, turgor pressure and boundary definitions, plus the material law.

Ports the parameters of `VertexFromTRBScenterTriangulationMT` (plane stress, MF 1
and 10), `Pressure3D::CenterTriangulation::Linear` (area flag 1) and
`VertexNoUpdateBoundary` from Tissue e74e457.
"""

from dataclasses import dataclass
from typing import NamedTuple

import numba
import numpy as np

from tissue_lite.arrays import IntArray
from tissue_lite.rand import glibc_rand
from tissue_lite.triangles import Matrix2, Vector, multiply, outer, pull_back_direction

_PI = 3.14159265
_RANDOM_THRESHOLD = 0.0001


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
def compute_moduli(
    young_longitudinal: float, concentration: float, material: Material
) -> tuple[float, float]:
    """Returns (young_L, young_T) of a cell, softened when the material says so."""
    young_l = young_longitudinal
    young_t = 2 * material.young_matrix + material.young_fibre - young_l
    if material.soften:
        a = concentration
        k = material.k_hill
        n = material.n_hill
        frac = 1 - a**n / (k**n + a**n)
        young_l *= frac
        young_t *= frac
    return young_l, young_t


class Lame(NamedTuple):
    """Plane-stress Lamé parameters of the longitudinal and transverse moduli."""

    lambda_l: float
    mu_l: float
    lambda_t: float
    mu_t: float


@numba.njit(cache=True)
def compute_lame_parameters(young_l: float, young_t: float, material: Material) -> Lame:
    """Returns (λ, μ) of both moduli, each as Tissue forms them from E and ν."""
    nu_l = material.poisson_longitudinal
    nu_t = material.poisson_transverse
    return Lame(
        young_l * nu_l / (1 - nu_l * nu_l),
        young_l / (2 * (1 + nu_l)),
        young_t * nu_t / (1 - nu_t * nu_t),
        young_t / (2 * (1 + nu_t)),
    )


@numba.njit(cache=True)
def _sample_random_direction(rand_state: IntArray) -> tuple[float, float]:
    """Tissue's fallback when the projected microtubule direction vanishes."""
    angle = ((glibc_rand(rand_state) % 360) * 2 * _PI) / 360
    return np.cos(angle), np.sin(angle)


@numba.njit(cache=True)
def compute_fibre_direction(
    gradient: Matrix2,
    x_axis: Vector,
    y_axis: Vector,
    microtubule: Vector,
    rand_state: IntArray,
) -> tuple[float, float]:
    """Unit microtubule axis in the rest plane; random when it vanishes (T1).

    The random fallback consumes `rand_state` (KNOWN_ISSUES T1, T2).
    """
    a0, a1 = pull_back_direction(gradient, x_axis, y_axis, microtubule)
    measure = np.sqrt(a0 * a0 + a1 * a1)
    if measure < _RANDOM_THRESHOLD:
        return _sample_random_direction(rand_state)
    return a0 / measure, a1 / measure


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
    """Fibre part of the second Piola-Kirchhoff stress.

    (Δλ/2) (tr E · aaᵀ + aᵀEa · I) + Δμ (E aaᵀ + aaᵀ E), with Δλ = λ_L - λ_T and
    Δμ = μ_L - μ_T.
    """
    (e00, _), (_, e11) = strain
    aa = outer(fibre)
    (eaa00, eaa01), (eaa10, eaa11) = multiply(strain, aa)
    (aae00, aae01), (aae10, aae11) = multiply(aa, strain)
    trace = e00 + e11
    fibre_strain = compute_fibre_strain(strain, fibre)
    half_lambda = delta_lambda / 2
    return (
        (
            half_lambda * (trace * aa[0][0] + fibre_strain)
            + delta_mu * (eaa00 + aae00),
            half_lambda * (trace * aa[0][1]) + delta_mu * (eaa01 + aae01),
        ),
        (
            half_lambda * (trace * aa[1][0]) + delta_mu * (eaa10 + aae10),
            half_lambda * (trace * aa[1][1] + fibre_strain)
            + delta_mu * (eaa11 + aae11),
        ),
    )
