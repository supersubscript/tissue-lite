"""Membrane mechanics: TRBS-MT forces, pressure, boundary, and per-cell stress/strain.

Ports `VertexFromTRBScenterTriangulationMT` (plane stress, MF 1 and 10),
`Pressure3D::CenterTriangulation::Linear` (area flag 1) and `VertexNoUpdateBoundary`
from Tissue e74e457. Every expression keeps Tissue's evaluation order so that results
are bit-identical; do not simplify them algebraically.
"""

from dataclasses import astuple, dataclass

import numba
import numpy as np
from numpy.typing import NDArray

from tissue_lite import layout
from tissue_lite.rand import glibc_rand
from tissue_lite.tissue import FloatArray, IntArray, Tissue

_PI = 3.14159265
_SLIVER = 0.0001
_RANDOM_THRESHOLD = 0.0001
_EIGEN_TOLERANCE = 0.000001
_C45 = 0.70710678118


@dataclass(frozen=True)
class ConcentrationSoftening:
    """Hill repression of both moduli by a cell concentration (Tissue's MF=10).

    Both moduli are scaled by `1 - c^n / (k^n + c^n)`, e.g. for auxin-driven wall
    loosening.
    """

    concentration_index: int
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


@dataclass(frozen=True)
class CellMechanics:
    """Per-cell stress and strain, area-weighted over each cell's triangles.

    Tensors are in the global frame. Directions follow Tissue's selection rules
    (largest absolute principal value; for stress, the in-plane one).
    """

    stress_tensor: FloatArray  # (C, 3, 3) Cauchy stress
    strain_tensor: FloatArray  # (C, 3, 3) Almansi strain
    normal: FloatArray  # (C, 3)
    stress_direction: FloatArray
    stress_max: FloatArray
    stress_direction_second: FloatArray
    stress_second: FloatArray
    stress_anisotropy: FloatArray
    von_mises_stress: FloatArray
    strain_direction: FloatArray
    strain_max: FloatArray
    strain_perpendicular: FloatArray  # n x strain_direction, unnormalised
    strain_second: FloatArray
    strain_anisotropy: FloatArray
    rest_area: FloatArray
    energy_isotropic: FloatArray  # per rest area
    energy_anisotropic: FloatArray  # per rest area
    young_transverse: FloatArray


@dataclass(frozen=True)
class Triangles:
    """Index arrays of the centre triangles, in Tissue's cell and wall order."""

    v2: IntArray
    v3: IntArray
    wall: IntArray
    spoke_start: IntArray  # rest length of edge centre-v2
    spoke_end: IntArray  # rest length of edge centre-v3

    @classmethod
    def from_tissue(cls, tissue: Tissue, double: bool) -> Triangles:
        """Builds the triangle index arrays for single or double rest lengths."""
        topo = tissue.topology
        flat = np.arange(len(topo.cell_vertices))
        nxt = topo.triangle_next
        return cls(
            v2=topo.cell_vertices,
            v3=topo.cell_vertices[nxt],
            wall=topo.cell_walls,
            spoke_start=2 * flat + 1 if double else flat,
            spoke_end=2 * nxt if double else nxt,
        )


type MaterialArgs = tuple[float, float, float, float, bool, bool, int, float, float]


@dataclass(frozen=True)
class MaterialParameters:
    """Scalar parameters of the TRBS kernels, flattened for Numba."""

    young_matrix: float
    young_fibre: float
    poisson_longitudinal: float
    poisson_transverse: float
    double: bool
    soften: bool
    concentration_index: int
    k_hill: float
    n_hill: float

    @classmethod
    def from_membrane(cls, membrane: TRBSMembrane) -> MaterialParameters:
        """Flattens a membrane definition."""
        s = membrane.softening
        return cls(
            young_matrix=membrane.young_matrix,
            young_fibre=membrane.young_fibre,
            poisson_longitudinal=membrane.poisson_longitudinal,
            poisson_transverse=membrane.poisson_transverse,
            double=membrane.double_rest_lengths,
            soften=s is not None,
            concentration_index=0 if s is None else s.concentration_index,
            k_hill=0.0 if s is None else s.k_hill,
            n_hill=0.0 if s is None else s.n_hill,
        )

    def as_tuple(self) -> MaterialArgs:
        """Returns the fields in kernel-argument order."""
        return astuple(self)


@numba.njit(cache=True)
def _moduli(
    cell_vars: FloatArray,
    c: int,
    ym: float,
    yf: float,
    soften: bool,
    conc_index: int,
    k: float,
    n: float,
) -> tuple[float, float]:
    """Returns (young_L, young_T) of cell c."""
    young_l = cell_vars[c, layout.YOUNG_LONGITUDINAL]
    young_t = 2 * ym + yf - young_l
    if soften:
        a = cell_vars[c, conc_index]
        frac = 1 - a**n / (k**n + a**n)
        young_l *= frac
        young_t *= frac
    return young_l, young_t


@numba.njit(cache=True)
def _random_direction(rand_state: NDArray[np.int64]) -> tuple[float, float]:
    """Tissue's fallback when the projected microtubule direction vanishes."""
    angle = ((glibc_rand(rand_state) % 360) * 2 * _PI) / 360
    return np.cos(angle), np.sin(angle)


@numba.njit(cache=True)
def compute_trbs_forces(
    offsets: IntArray,
    tri_v2: IntArray,
    tri_v3: IntArray,
    tri_wall: IntArray,
    spoke_start: IntArray,
    spoke_end: IntArray,
    wall_vertices: IntArray,
    wall_lengths: FloatArray,
    spokes: FloatArray,
    extensions: FloatArray,
    centres: FloatArray,
    cell_vars: FloatArray,
    x: FloatArray,
    ym: float,
    yf: float,
    nu_l: float,
    nu_t: float,
    double: bool,
    soften: bool,
    conc_index: int,
    k_hill: float,
    n_hill: float,
    rand_state: NDArray[np.int64],
    vd: FloatArray,
    cd: FloatArray,
) -> None:
    """Adds TRBS-MT forces to vertex (vd) and centre (cd) derivatives.

    Writes the transverse modulus to cell variable 20 unless softening is on, as
    Tissue does.
    """
    for c in range(offsets.shape[0] - 1):
        young_l, young_t = _moduli(
            cell_vars, c, ym, yf, soften, conc_index, k_hill, n_hill
        )
        if not soften:
            cell_vars[c, layout.YOUNG_TRANSVERSE] = young_t
        lam_l = young_l * nu_l / (1 - nu_l * nu_l)
        mio_l = young_l / (2 * (1 + nu_l))
        lam_t = young_t * nu_t / (1 - nu_t * nu_t)
        mio_t = young_t / (2 * (1 + nu_t))
        a_g0 = cell_vars[c, layout.MICROTUBULE_START]
        a_g1 = cell_vars[c, layout.MICROTUBULE_START + 1]
        a_g2 = cell_vars[c, layout.MICROTUBULE_START + 2]
        for t in range(offsets[c], offsets[c + 1]):
            i2 = tri_v2[t]
            i3 = tri_v3[t]
            w = tri_wall[t]
            r0 = spokes[spoke_start[t]]
            r1 = wall_lengths[w] + extensions[t] if double else wall_lengths[w]
            r2 = spokes[spoke_end[t]]
            p00, p01, p02 = centres[c, 0], centres[c, 1], centres[c, 2]
            p10, p11, p12 = x[i2, 0], x[i2, 1], x[i2, 2]
            p20, p21, p22 = x[i3, 0], x[i3, 1], x[i3, 2]
            wa = wall_vertices[w, 0]
            wb = wall_vertices[w, 1]
            l0 = np.sqrt(
                (p00 - p10) * (p00 - p10)
                + (p01 - p11) * (p01 - p11)
                + (p02 - p12) * (p02 - p12)
            )
            dist = 0.0
            for d in range(3):
                dist += (x[wa, d] - x[wb, d]) * (x[wa, d] - x[wb, d])
            l1 = np.sqrt(dist)
            l2 = np.sqrt(
                (p00 - p20) * (p00 - p20)
                + (p01 - p21) * (p01 - p21)
                + (p02 - p22) * (p02 - p22)
            )
            rest_area = (
                np.sqrt(
                    (r0 + r1 + r2) * (-r0 + r1 + r2) * (r0 - r1 + r2) * (r0 + r1 - r2)
                )
                * 0.25
            )
            cos0 = (r0 * r0 + r2 * r2 - r1 * r1) / (r0 * r2 * 2)
            cos1 = (r0 * r0 + r1 * r1 - r2 * r2) / (r0 * r1 * 2)
            cos2 = (r1 * r1 + r2 * r2 - r0 * r0) / (r1 * r2 * 2)
            tau = 1.0 / (rest_area * 16)
            cot0 = cos0 / np.sqrt(1 - cos0 * cos0)
            cot1 = cos1 / np.sqrt(1 - cos1 * cos1)
            cot2 = cos2 / np.sqrt(1 - cos2 * cos2)
            lm = lam_t + 2 * mio_t
            ts0 = (2 * cot2 * cot2 * lm + 2 * mio_t) * tau
            ts1 = (2 * cot0 * cot0 * lm + 2 * mio_t) * tau
            ts2 = (2 * cot1 * cot1 * lm + 2 * mio_t) * tau
            as0 = (2 * cot1 * cot2 * lm - 2 * mio_t) * tau
            as1 = (2 * cot0 * cot2 * lm - 2 * mio_t) * tau
            as2 = (2 * cot0 * cot1 * lm - 2 * mio_t) * tau
            dl0 = l0 * l0 - r0 * r0
            dl1 = l1 * l1 - r1 * r1
            dl2 = l2 * l2 - r2 * r2
            ca = (l0 * l0 + l1 * l1 - l2 * l2) / (l0 * l1 * 2)
            qa = ca * l0
            qc = np.sqrt(1 - ca * ca) * l0
            qb = l1
            ra = (r0 * r0 + r1 * r1 - r2 * r2) / (r0 * r1 * 2)
            pa = ra * r0
            pc = np.sqrt(1 - ra * ra) * r0
            pb = r1
            sv00 = 0.0
            sv01 = 1 / pc
            sv10 = -1 / pb
            sv11 = (pa - pb) / (pb * pc)
            sv20 = 1 / pb
            sv21 = -pa / (pb * pc)
            temp_a = np.sqrt(
                (p20 - p10) * (p20 - p10)
                + (p21 - p11) * (p21 - p11)
                + (p22 - p12) * (p22 - p12)
            )
            xc0 = (p20 - p10) / temp_a
            xc1 = (p21 - p11) / temp_a
            xc2 = (p22 - p12) / temp_a
            bc0 = (p00 - p10) / l0
            bc1 = (p01 - p11) / l0
            bc2 = (p02 - p12) / l0
            zc0 = xc1 * bc2 - xc2 * bc1
            zc1 = xc2 * bc0 - xc0 * bc2
            zc2 = xc0 * bc1 - xc1 * bc0
            zn = np.sqrt(zc0 * zc0 + zc1 * zc1 + zc2 * zc2)
            zc0 = zc0 / zn
            zc1 = zc1 / zn
            zc2 = zc2 / zn
            yc0 = zc1 * xc2 - zc2 * xc1
            yc1 = zc2 * xc0 - zc0 * xc2
            yc2 = zc0 * xc1 - zc1 * xc0
            al0 = xc0 * a_g0 + xc1 * a_g1 + xc2 * a_g2
            al1 = yc0 * a_g0 + yc1 * a_g1 + yc2 * a_g2
            d_lam = lam_l - lam_t
            d_mio = mio_l - mio_t
            f00 = qb * sv20
            f10 = 0.0
            f01 = qa * sv01 + qb * sv21
            f11 = qc * sv01
            ar0 = f11 * al0 - f01 * al1
            ar1 = -f10 * al0 + f00 * al1
            measure = np.sqrt(ar0 * ar0 + ar1 * ar1)
            if measure < _RANDOM_THRESHOLD:
                ar0, ar1 = _random_direction(rand_state)
            else:
                ar0 = ar0 / measure
                ar1 = ar1 / measure
            aa00 = ar0 * ar0
            aa01 = ar0 * ar1
            aa10 = ar1 * ar0
            aa11 = ar1 * ar1
            e00 = 0.5 * (f00 * f00 + f10 * f10 - 1)
            e01 = 0.5 * (f00 * f01 + f10 * f11)
            e10 = 0.5 * (f01 * f00 + f11 * f10)
            e11 = 0.5 * (f01 * f01 + f11 * f11 - 1)
            tr_e = e00 + e11
            at_ea = ar0 * ar0 * e00 + ar0 * ar1 * (e01 + e10) + ar1 * ar1 * e11
            eaa00 = e00 * aa00 + e01 * aa10
            eaa01 = e00 * aa01 + e01 * aa11
            eaa10 = e10 * aa00 + e11 * aa10
            eaa11 = e10 * aa01 + e11 * aa11
            aae00 = aa00 * e00 + aa01 * e10
            aae01 = aa00 * e01 + aa01 * e11
            aae10 = aa10 * e00 + aa11 * e10
            aae11 = aa10 * e01 + aa11 * e11
            ds00 = (d_lam / 2) * (tr_e * aa00 + at_ea) + d_mio * (eaa00 + aae00)
            ds01 = (d_lam / 2) * (tr_e * aa01) + d_mio * (eaa01 + aae01)
            ds10 = (d_lam / 2) * (tr_e * aa10) + d_mio * (eaa10 + aae10)
            ds11 = (d_lam / 2) * (tr_e * aa11 + at_ea) + d_mio * (eaa11 + aae11)
            tpk00 = rest_area * (f00 * ds00 + f01 * ds10)
            tpk10 = rest_area * (f10 * ds00 + f11 * ds10)
            tpk01 = rest_area * (f00 * ds01 + f01 * ds11)
            tpk11 = rest_area * (f10 * ds01 + f11 * ds11)
            lc00 = tpk00 * sv00 + tpk01 * sv01
            lc01 = tpk10 * sv00 + tpk11 * sv01
            lc10 = tpk00 * sv10 + tpk01 * sv11
            lc11 = tpk10 * sv10 + tpk11 * sv11
            lc20 = tpk00 * sv20 + tpk01 * sv21
            lc21 = tpk10 * sv20 + tpk11 * sv21
            g0a = ts0 * dl0 + as1 * dl1 + as0 * dl2
            g0b = ts2 * dl2 + as2 * dl1 + as0 * dl0
            g1a = ts0 * dl0 + as0 * dl2 + as1 * dl1
            g1b = ts1 * dl1 + as2 * dl2 + as1 * dl0
            g2a = ts2 * dl2 + as0 * dl0 + as2 * dl1
            g2b = ts1 * dl1 + as1 * dl0 + as2 * dl2
            fx = (
                g0a * (p10 - p00) + g0b * (p20 - p00) + -(xc0 * lc00 + yc0 * lc01),
                g0a * (p11 - p01) + g0b * (p21 - p01) + -(xc1 * lc00 + yc1 * lc01),
                g0a * (p12 - p02) + g0b * (p22 - p02) + -(xc2 * lc00 + yc2 * lc01),
                g1a * (p00 - p10) + g1b * (p20 - p10) + -(xc0 * lc10 + yc0 * lc11),
                g1a * (p01 - p11) + g1b * (p21 - p11) + -(xc1 * lc10 + yc1 * lc11),
                g1a * (p02 - p12) + g1b * (p22 - p12) + -(xc2 * lc10 + yc2 * lc11),
                g2a * (p00 - p20) + g2b * (p10 - p20) + -(xc0 * lc20 + yc0 * lc21),
                g2a * (p01 - p21) + g2b * (p11 - p21) + -(xc1 * lc20 + yc1 * lc21),
                g2a * (p02 - p22) + g2b * (p12 - p22) + -(xc2 * lc20 + yc2 * lc21),
            )
            if (
                abs(1 - cos0) < _SLIVER
                or abs(1 - cos1) < _SLIVER
                or abs(1 - cos2) < _SLIVER
            ):
                # Tissue discards the forces and doubles the accumulated ones.
                s0 = cd[c, 0] + vd[i2, 0] + vd[i3, 0]
                s1 = cd[c, 1] + vd[i2, 1] + vd[i3, 1]
                s2 = cd[c, 2] + vd[i2, 2] + vd[i3, 2]
                cd[c, 0] += s0
                cd[c, 1] += s1
                cd[c, 2] += s2
                vd[i2, 0] += s0
                vd[i2, 1] += s1
                vd[i2, 2] += s2
                vd[i3, 0] += s0
                vd[i3, 1] += s1
                vd[i3, 2] += s2
            else:
                cd[c, 0] += fx[0]
                cd[c, 1] += fx[1]
                cd[c, 2] += fx[2]
                vd[i2, 0] += fx[3]
                vd[i2, 1] += fx[4]
                vd[i2, 2] += fx[5]
                vd[i3, 0] += fx[6]
                vd[i3, 1] += fx[7]
                vd[i3, 2] += fx[8]


@numba.njit(cache=True)
def compute_pressure_forces(
    offsets: IntArray,
    tri_v2: IntArray,
    tri_v3: IntArray,
    tri_wall: IntArray,
    wall_vertices: IntArray,
    centres: FloatArray,
    x: FloatArray,
    factor: float,
    pressure: float,
    vd: FloatArray,
    cd: FloatArray,
) -> None:
    """Adds the area-proportional pressure force along each triangle's normal."""
    for c in range(offsets.shape[0] - 1):
        for t in range(offsets[c], offsets[c + 1]):
            i2 = tri_v2[t]
            i3 = tri_v3[t]
            w = tri_wall[t]
            p00, p01, p02 = centres[c, 0], centres[c, 1], centres[c, 2]
            p10, p11, p12 = x[i2, 0], x[i2, 1], x[i2, 2]
            p20, p21, p22 = x[i3, 0], x[i3, 1], x[i3, 2]
            l0 = np.sqrt(
                (p00 - p10) * (p00 - p10)
                + (p01 - p11) * (p01 - p11)
                + (p02 - p12) * (p02 - p12)
            )
            wa = wall_vertices[w, 0]
            wb = wall_vertices[w, 1]
            dist = 0.0
            for d in range(3):
                dist += (x[wa, d] - x[wb, d]) * (x[wa, d] - x[wb, d])
            l1 = np.sqrt(dist)
            l2 = np.sqrt(
                (p00 - p20) * (p00 - p20)
                + (p01 - p21) * (p01 - p21)
                + (p02 - p22) * (p02 - p22)
            )
            area = (
                np.sqrt(
                    (l0 + l1 + l2) * (-l0 + l1 + l2) * (l0 - l1 + l2) * (l0 + l1 - l2)
                )
                * 0.25
            )
            temp_a = np.sqrt(
                (p20 - p10) * (p20 - p10)
                + (p21 - p11) * (p21 - p11)
                + (p22 - p12) * (p22 - p12)
            )
            xc0 = (p20 - p10) / temp_a
            xc1 = (p21 - p11) / temp_a
            xc2 = (p22 - p12) / temp_a
            bc0 = (p00 - p10) / l0
            bc1 = (p01 - p11) / l0
            bc2 = (p02 - p12) / l0
            n0 = xc1 * bc2 - xc2 * bc1
            n1 = xc2 * bc0 - xc0 * bc2
            n2 = xc0 * bc1 - xc1 * bc0
            nn = np.sqrt(n0 * n0 + n1 * n1 + n2 * n2)
            n0 = n0 / nn
            n1 = n1 / nn
            n2 = n2 / nn
            coeff = factor * pressure * (area / 2)
            cd[c, 0] += coeff * n0
            cd[c, 1] += coeff * n1
            cd[c, 2] += coeff * n2
            vd[i2, 0] += coeff * n0
            vd[i2, 1] += coeff * n1
            vd[i2, 2] += coeff * n2
            vd[i3, 0] += coeff * n0
            vd[i3, 1] += coeff * n1
            vd[i3, 2] += coeff * n2


@numba.njit(cache=True)
def _pick_pivot(s: FloatArray) -> tuple[float, int, int]:
    """Returns the largest off-diagonal magnitude and its (I, J), I > J."""
    pivot, big, small = abs(s[1, 0]), 1, 0
    if abs(s[2, 0]) > pivot:
        pivot, big, small = abs(s[2, 0]), 2, 0
    if abs(s[2, 1]) > pivot:
        pivot, big, small = abs(s[2, 1]), 2, 1
    return pivot, big, small


@numba.njit(cache=True)
def _rotate(
    s: FloatArray, v: FloatArray, big: int, small: int, tmp: FloatArray
) -> None:
    """Applies Tissue's Jacobi rotation in the (small, big) plane.

    Skipping Tissue's exact products with 0 and 1 leaves the result bit-identical.
    """
    if abs(s[big, big] - s[small, small]) < _EIGEN_TOLERANCE:
        si = _C45
        co = _C45
    else:
        t = (2 * s[big, small]) / (s[small, small] - s[big, big])
        t = 1 / np.sqrt(1 + t * t)
        si = _C45 * np.sqrt(1 - t) if t < 1 else _C45 * np.sqrt(t - 1)
        co = _C45 * np.sqrt(1 + t)
    for r in range(3):
        a_lo = s[r, small]
        a_hi = s[r, big]
        tmp[r, small] = a_lo * co + a_hi * si
        tmp[r, big] = a_lo * -si + a_hi * co
        for q in range(3):
            if q != small and q != big:
                tmp[r, q] = s[r, q]
    for q in range(3):
        b_lo = tmp[small, q]
        b_hi = tmp[big, q]
        for r in range(3):
            if r == small:
                s[r, q] = co * b_lo + si * b_hi
            elif r == big:
                s[r, q] = -si * b_lo + co * b_hi
            else:
                s[r, q] = tmp[r, q]
    for r in range(3):
        v_lo = v[r, small]
        v_hi = v[r, big]
        v[r, small] = v_lo * co + v_hi * si
        v[r, big] = v_lo * -si + v_hi * co


@numba.njit(cache=True)
def _jacobi(s: FloatArray, v: FloatArray, pivot_at_top: bool, tmp: FloatArray) -> None:
    """Diagonalises s in place; columns of v become unit eigenvectors.

    `pivot_at_top` reproduces the stress loop, which tests the previous pivot and so
    always applies one extra rotation.
    """
    for r in range(3):
        for q in range(3):
            v[r, q] = 1.0 if r == q else 0.0
    if pivot_at_top:
        pivot = 1.0
        while pivot > _EIGEN_TOLERANCE:
            pivot, big, small = _pick_pivot(s)
            _rotate(s, v, big, small, tmp)
    else:
        pivot, big, small = _pick_pivot(s)
        while pivot > _EIGEN_TOLERANCE:
            _rotate(s, v, big, small, tmp)
            pivot, big, small = _pick_pivot(s)
    for q in range(3):
        norm = np.sqrt(v[0, q] * v[0, q] + v[1, q] * v[1, q] + v[2, q] * v[2, q])
        if norm > 0:
            for r in range(3):
                v[r, q] /= norm


@numba.njit(cache=True)
def _rotate_tensor(rot: FloatArray, local: FloatArray, tmp: FloatArray) -> None:
    """Replaces local by rot @ local @ rot.T, summing in Tissue's order."""
    for r in range(3):
        for q in range(3):
            acc = 0.0
            for w in range(3):
                acc += rot[r, w] * local[w, q]
            tmp[r, q] = acc
    for r in range(3):
        for q in range(3):
            acc = 0.0
            for w in range(3):
                acc += tmp[r, w] * rot[q, w]
            local[r, q] = acc


@numba.njit(cache=True)
def update_trbs(
    offsets: IntArray,
    tri_v2: IntArray,
    tri_v3: IntArray,
    tri_wall: IntArray,
    spoke_start: IntArray,
    spoke_end: IntArray,
    wall_vertices: IntArray,
    wall_lengths: FloatArray,
    spokes: FloatArray,
    extensions: FloatArray,
    centres: FloatArray,
    cell_vars: FloatArray,
    x: FloatArray,
    ym: float,
    yf: float,
    nu_l: float,
    nu_t: float,
    double: bool,
    soften: bool,
    conc_index: int,
    k_hill: float,
    n_hill: float,
    second_axes: bool,
    rand_state: NDArray[np.int64],
    stress_out: FloatArray,
    strain_out: FloatArray,
    energy_iso_out: FloatArray,
) -> None:
    """Writes per-cell stress, strain, normal and energies, as Tissue's update does.

    `second_axes` selects the growth layout (second stress axis at 31-34 over the
    normal, perpendicular strain at 26-29).
    """
    rot = np.empty((3, 3))
    tmp = np.empty((3, 3))
    strain_t = np.empty((3, 3))
    stress_t = np.empty((3, 3))
    strain_sum = np.empty((3, 3))
    stress_sum = np.empty((3, 3))
    v_stress = np.empty((3, 3))
    v_strain = np.empty((3, 3))
    normal = np.empty(3)
    for c in range(offsets.shape[0] - 1):
        young_l, young_t = _moduli(
            cell_vars, c, ym, yf, soften, conc_index, k_hill, n_hill
        )
        lam_l = young_l * nu_l / (1 - nu_l * nu_l)
        mio_l = young_l / (2 * (1 + nu_l))
        lam_t = young_t * nu_t / (1 - nu_t * nu_t)
        mio_t = young_t / (2 * (1 + nu_t))
        strain_sum[:] = 0.0
        stress_sum[:] = 0.0
        normal[:] = 0.0
        total_rest = 0.0
        total_area = 0.0
        energy_iso = 0.0
        energy_aniso = 0.0
        a_g0 = cell_vars[c, layout.MICROTUBULE_START]
        a_g1 = cell_vars[c, layout.MICROTUBULE_START + 1]
        a_g2 = cell_vars[c, layout.MICROTUBULE_START + 2]
        for t in range(offsets[c], offsets[c + 1]):
            i2 = tri_v2[t]
            i3 = tri_v3[t]
            w = tri_wall[t]
            r0 = spokes[spoke_start[t]]
            r1 = wall_lengths[w] + extensions[t] if double else wall_lengths[w]
            r2 = spokes[spoke_end[t]]
            p00, p01, p02 = centres[c, 0], centres[c, 1], centres[c, 2]
            p10, p11, p12 = x[i2, 0], x[i2, 1], x[i2, 2]
            p20, p21, p22 = x[i3, 0], x[i3, 1], x[i3, 2]
            wa = wall_vertices[w, 0]
            wb = wall_vertices[w, 1]
            l0 = np.sqrt(
                (p00 - p10) * (p00 - p10)
                + (p01 - p11) * (p01 - p11)
                + (p02 - p12) * (p02 - p12)
            )
            dist = 0.0
            for d in range(3):
                dist += (x[wa, d] - x[wb, d]) * (x[wa, d] - x[wb, d])
            l1 = np.sqrt(dist)
            l2 = np.sqrt(
                (p00 - p20) * (p00 - p20)
                + (p01 - p21) * (p01 - p21)
                + (p02 - p22) * (p02 - p22)
            )
            rest_area = (
                np.sqrt(
                    (r0 + r1 + r2) * (-r0 + r1 + r2) * (r0 - r1 + r2) * (r0 + r1 - r2)
                )
                * 0.25
            )
            area = (
                np.sqrt(
                    (l0 + l1 + l2) * (-l0 + l1 + l2) * (l0 - l1 + l2) * (l0 + l1 - l2)
                )
                * 0.25
            )
            ca = (l0 * l0 + l1 * l1 - l2 * l2) / (l0 * l1 * 2)
            qa = ca * l0
            qc = np.sqrt(1 - ca * ca) * l0
            qb = l1
            ra = (r0 * r0 + r1 * r1 - r2 * r2) / (r0 * r1 * 2)
            pa = ra * r0
            pc = np.sqrt(1 - ra * ra) * r0
            pb = r1
            sv01 = 1 / pc
            sv21 = -pa / (pb * pc)
            sv20 = 1 / pb
            temp_a = np.sqrt(
                (p20 - p10) * (p20 - p10)
                + (p21 - p11) * (p21 - p11)
                + (p22 - p12) * (p22 - p12)
            )
            xc0 = (p20 - p10) / temp_a
            xc1 = (p21 - p11) / temp_a
            xc2 = (p22 - p12) / temp_a
            bc0 = (p00 - p10) / l0
            bc1 = (p01 - p11) / l0
            bc2 = (p02 - p12) / l0
            zc0 = xc1 * bc2 - xc2 * bc1
            zc1 = xc2 * bc0 - xc0 * bc2
            zc2 = xc0 * bc1 - xc1 * bc0
            zn = np.sqrt(zc0 * zc0 + zc1 * zc1 + zc2 * zc2)
            zc0 = zc0 / zn
            zc1 = zc1 / zn
            zc2 = zc2 / zn
            yc0 = zc1 * xc2 - zc2 * xc1
            yc1 = zc2 * xc0 - zc0 * xc2
            yc2 = zc0 * xc1 - zc1 * xc0
            rot[0, 0], rot[0, 1], rot[0, 2] = xc0, yc0, zc0
            rot[1, 0], rot[1, 1], rot[1, 2] = xc1, yc1, zc1
            rot[2, 0], rot[2, 1], rot[2, 2] = xc2, yc2, zc2
            al0 = xc0 * a_g0 + xc1 * a_g1 + xc2 * a_g2
            al1 = yc0 * a_g0 + yc1 * a_g1 + yc2 * a_g2
            d_lam = lam_l - lam_t
            d_mio = mio_l - mio_t
            f00 = qb * sv20
            f10 = 0.0
            f01 = qa * sv01 + qb * sv21
            f11 = qc * sv01
            ar0 = f11 * al0 - f01 * al1
            ar1 = -f10 * al0 + f00 * al1
            measure = np.sqrt(ar0 * ar0 + ar1 * ar1)
            if measure < _RANDOM_THRESHOLD:
                ar0, ar1 = _random_direction(rand_state)
            else:
                ar0 = ar0 / measure
                ar1 = ar1 / measure
            aa00 = ar0 * ar0
            aa01 = ar0 * ar1
            aa10 = ar1 * ar0
            aa11 = ar1 * ar1
            b00 = f00 * f00 + f01 * f01
            b01 = f00 * f10 + f01 * f11
            b10 = f10 * f00 + f11 * f01
            b11 = f10 * f10 + f11 * f11
            e00 = 0.5 * (f00 * f00 + f10 * f10 - 1)
            e01 = 0.5 * (f00 * f01 + f10 * f11)
            e10 = 0.5 * (f01 * f00 + f11 * f10)
            e11 = 0.5 * (f01 * f01 + f11 * f11 - 1)
            ee00 = e00 * e00 + e01 * e10
            ee01 = e00 * e01 + e01 * e11
            ee10 = e10 * e00 + e11 * e10
            ee11 = e10 * e01 + e11 * e11
            tr_e = e00 + e11
            i2_inv = ee00 + ee11
            i5 = ar0 * ar0 * ee00 + ar0 * ar1 * (ee01 + ee10) + ar1 * ar1 * ee11
            det_b = b00 * b11 - b10 * b01
            alm00 = 0.5 * (1 - (b11 / det_b))
            alm01 = 0.5 * b01 / det_b
            alm10 = 0.5 * b10 / det_b
            alm11 = 0.5 * (1 - (b00 / det_b))
            at_ea = ar0 * ar0 * e00 + ar0 * ar1 * (e01 + e10) + ar1 * ar1 * e11
            eaa00 = e00 * aa00 + e01 * aa10
            eaa01 = e00 * aa01 + e01 * aa11
            eaa10 = e10 * aa00 + e11 * aa10
            eaa11 = e10 * aa01 + e11 * aa11
            aae00 = aa00 * e00 + aa01 * e10
            aae01 = aa00 * e01 + aa01 * e11
            aae10 = aa10 * e00 + aa11 * e10
            aae11 = aa10 * e01 + aa11 * e11
            bb00 = b00 * b00 + b01 * b10
            bb01 = b00 * b01 + b01 * b11
            bb10 = b10 * b00 + b11 * b10
            bb11 = b10 * b01 + b11 * b11
            area_factor = rest_area / area
            g = lam_t * tr_e - mio_t
            sg00 = area_factor * (g * b00 + mio_t * bb00)
            sg01 = area_factor * (g * b01 + mio_t * bb01)
            sg10 = area_factor * (g * b10 + mio_t * bb10)
            sg11 = area_factor * (g * b11 + mio_t * bb11)
            ds00 = (d_lam / 2) * (tr_e * aa00 + at_ea) + d_mio * (eaa00 + aae00)
            ds01 = (d_lam / 2) * (tr_e * aa01) + d_mio * (eaa01 + aae01)
            ds10 = (d_lam / 2) * (tr_e * aa10) + d_mio * (eaa10 + aae10)
            ds11 = (d_lam / 2) * (tr_e * aa11 + at_ea) + d_mio * (eaa11 + aae11)
            dsf00 = ds00 * f00 + ds01 * f01
            dsf10 = ds10 * f00 + ds11 * f01
            dsf01 = ds00 * f10 + ds01 * f11
            dsf11 = ds10 * f10 + ds11 * f11
            dsg00 = area_factor * (f00 * dsf00 + f01 * dsf10)
            dsg10 = area_factor * (f10 * dsf00 + f11 * dsf10)
            dsg01 = area_factor * (f00 * dsf01 + f01 * dsf11)
            dsg11 = area_factor * (f10 * dsf01 + f11 * dsf11)
            strain_t[:] = 0.0
            stress_t[:] = 0.0
            strain_t[0, 0] = alm00
            strain_t[1, 0] = alm10
            strain_t[0, 1] = alm01
            strain_t[1, 1] = alm11
            stress_t[0, 0] = sg00 + dsg00
            stress_t[1, 0] = sg10 + dsg10
            stress_t[0, 1] = sg01 + dsg01
            stress_t[1, 1] = sg11 + dsg11
            _rotate_tensor(rot, strain_t, tmp)
            _rotate_tensor(rot, stress_t, tmp)
            for r in range(3):
                for q in range(3):
                    strain_sum[r, q] += area * strain_t[r, q]
            for r in range(3):
                for q in range(3):
                    stress_sum[r, q] += area * stress_t[r, q]
            normal[0] += area * zc0
            normal[1] += area * zc1
            normal[2] += area * zc2
            total_rest = total_rest + rest_area
            total_area = total_area + area
            energy_iso += ((lam_t / 2) * tr_e * tr_e + mio_t * i2_inv) * rest_area
            energy_aniso += ((d_lam / 2) * at_ea * tr_e + d_mio * i5) * rest_area
        for r in range(3):
            for q in range(3):
                strain_sum[r, q] = strain_sum[r, q] / total_area
        for r in range(3):
            for q in range(3):
                stress_sum[r, q] = stress_sum[r, q] / total_area
        for r in range(3):
            normal[r] /= total_area
        nn = np.sqrt(
            normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]
        )
        if nn > 0:
            for r in range(3):
                normal[r] /= nn
        stress_out[c] = stress_sum
        strain_out[c] = strain_sum
        cell_vars[c, layout.NORMAL_START] = normal[0]
        cell_vars[c, layout.NORMAL_START + 1] = normal[1]
        cell_vars[c, layout.NORMAL_START + 2] = normal[2]

        _jacobi(stress_sum, v_stress, True, tmp)
        i_s = 0
        s1 = stress_sum[0, 0]
        if abs(stress_sum[1, 1]) > abs(s1):
            s1 = stress_sum[1, 1]
            i_s = 1
        if abs(stress_sum[2, 2]) > abs(s1):
            s1 = stress_sum[2, 2]
            i_s = 2
        if i_s == 0:
            i_s2, i_s3 = 1, 2
        elif i_s == 1:
            i_s2, i_s3 = 2, 0
        else:
            i_s2, i_s3 = 0, 1
        if abs(stress_sum[i_s3, i_s3]) > abs(stress_sum[i_s2, i_s2]):
            i_s2, i_s3 = i_s3, i_s2
        s2 = stress_sum[i_s2, i_s2]
        s3 = stress_sum[i_s3, i_s3]
        if (
            abs(
                normal[0] * v_stress[0, i_s]
                + normal[1] * v_stress[1, i_s]
                + normal[2] * v_stress[2, i_s]
            )
            > 0.7
        ):
            i_s = i_s2
            i_s2 = i_s3
            s1 = s2
            s2 = s3
        cell_vars[c, layout.VON_MISES_STRESS] = np.sqrt(s1 * s1 + s2 * s2 - s1 * s2)
        if abs(s1) < 0.000001:
            cell_vars[c, layout.STRESS_ANISOTROPY] = 0
        else:
            cell_vars[c, layout.STRESS_ANISOTROPY] = 1 - abs(s2 / s1)
        k0 = layout.STRESS_START
        cell_vars[c, k0] = v_stress[0, i_s]
        cell_vars[c, k0 + 1] = v_stress[1, i_s]
        cell_vars[c, k0 + 2] = v_stress[2, i_s]
        cell_vars[c, k0 + 3] = s1
        if second_axes:
            k1 = layout.STRESS_SECOND_START
            cell_vars[c, k1] = v_stress[0, i_s2]
            cell_vars[c, k1 + 1] = v_stress[1, i_s2]
            cell_vars[c, k1 + 2] = v_stress[2, i_s2]
            cell_vars[c, k1 + 3] = s2

        _jacobi(strain_sum, v_strain, False, tmp)
        i_e = 0
        e1 = strain_sum[0, 0]
        if abs(strain_sum[1, 1]) > abs(e1):
            e1 = strain_sum[1, 1]
            i_e = 1
        if abs(strain_sum[2, 2]) > abs(e1):
            e1 = strain_sum[2, 2]
            i_e = 2
        if i_e == 0:
            i_e2, i_e3 = 1, 2
        elif i_e == 1:
            i_e2, i_e3 = 0, 2
        else:
            i_e2, i_e3 = 0, 1
        if abs(strain_sum[i_e3, i_e3]) > abs(strain_sum[i_e2, i_e2]):
            i_e2, i_e3 = i_e3, i_e2
        e2 = strain_sum[i_e2, i_e2]
        perp0 = normal[1] * v_strain[2, i_e] - normal[2] * v_strain[1, i_e]
        perp1 = normal[2] * v_strain[0, i_e] - normal[0] * v_strain[2, i_e]
        perp2 = normal[0] * v_strain[1, i_e] - normal[1] * v_strain[0, i_e]
        pn = np.sqrt(perp0 * perp0 + perp1 * perp1 + perp2 * perp2)
        if abs(pn) < 0.0001:
            perp0 = v_strain[0, i_e]
            perp1 = v_strain[1, i_e]
            perp2 = v_strain[2, i_e]
        if abs(e1) < 0.000001:
            cell_vars[c, layout.STRAIN_ANISOTROPY] = 0
        else:
            cell_vars[c, layout.STRAIN_ANISOTROPY] = 1 - abs(e2 / e1)
        k0 = layout.STRAIN_START
        cell_vars[c, k0] = v_strain[0, i_e]
        cell_vars[c, k0 + 1] = v_strain[1, i_e]
        cell_vars[c, k0 + 2] = v_strain[2, i_e]
        cell_vars[c, k0 + 3] = e1
        if second_axes:
            k1 = layout.STRAIN_PERPENDICULAR_START
            cell_vars[c, k1] = perp0
            cell_vars[c, k1 + 1] = perp1
            cell_vars[c, k1 + 2] = perp2
            cell_vars[c, k1 + 3] = e2
        mt_norm = np.sqrt(a_g0 * a_g0 + a_g1 * a_g1 + a_g2 * a_g2)
        if mt_norm > 0:
            for d in range(3):
                mt = layout.MICROTUBULE_START + d
                cell_vars[c, mt] = cell_vars[c, mt] / mt_norm
        cell_vars[c, layout.REST_AREA] = total_rest
        energy_iso_out[c] = energy_iso / total_rest
        cell_vars[c, layout.YOUNG_TRANSVERSE] = energy_aniso / total_rest
