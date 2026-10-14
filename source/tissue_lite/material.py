"""Wall material, turgor pressure and boundary definitions, plus the moduli kernel.

Ports the parameters of `VertexFromTRBScenterTriangulationMT` (plane stress, MF 1
and 10), `Pressure3D::CenterTriangulation::Linear` (area flag 1) and
`VertexNoUpdateBoundary` from Tissue e74e457.
"""

from dataclasses import dataclass
from typing import NamedTuple

import numba

from tissue_lite.tissue import CellState


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
