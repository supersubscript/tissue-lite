"""Forces on vertices and cell centres: TRBS-MT membrane and turgor pressure.

Ports the derivative parts of `VertexFromTRBScenterTriangulationMT` and
`Pressure3D::CenterTriangulation::Linear` from Tissue e74e457. Every expression
keeps Tissue's evaluation order so that results are bit-identical; do not simplify
them algebraically.
"""

import numba
import numpy as np

from tissue_lite.arrays import FloatArray, IntArray
from tissue_lite.material import Material, compute_moduli
from tissue_lite.rand import glibc_rand
from tissue_lite.tissue import CellState
from tissue_lite.triangles import RestLengths, TriangleMesh

_PI = 3.14159265
_SLIVER = 0.0001
RANDOM_THRESHOLD = 0.0001


@numba.njit(cache=True)
def random_direction(rand_state: IntArray) -> tuple[float, float]:
    """Tissue's fallback when the projected microtubule direction vanishes."""
    angle = ((glibc_rand(rand_state) % 360) * 2 * _PI) / 360
    return np.cos(angle), np.sin(angle)


@numba.njit(cache=True)
def compute_trbs_forces(
    mesh: TriangleMesh,
    rest: RestLengths,
    centres: FloatArray,
    vertices: FloatArray,
    cells: CellState,
    material: Material,
    rand_state: IntArray,
    vertex_forces: FloatArray,
    centre_forces: FloatArray,
) -> None:
    """Adds the TRBS-MT forces to `vertex_forces` and `centre_forces`.

    Also writes each cell's transverse modulus into `cells.young_transverse`.
    """
    offsets, tri_v2, tri_v3, tri_wall, spoke_start, spoke_end, wall_vertices = mesh
    spokes, wall_lengths, extensions = rest
    x = vertices
    vd = vertex_forces
    cd = centre_forces
    nu_l = material.poisson_longitudinal
    nu_t = material.poisson_transverse
    double = material.double_rest_lengths
    for c in range(offsets.shape[0] - 1):
        young_l, young_t = compute_moduli(cells, c, material)
        cells.young_transverse[c] = young_t
        lam_l = young_l * nu_l / (1 - nu_l * nu_l)
        mio_l = young_l / (2 * (1 + nu_l))
        lam_t = young_t * nu_t / (1 - nu_t * nu_t)
        mio_t = young_t / (2 * (1 + nu_t))
        a_g0 = cells.microtubule[c, 0]
        a_g1 = cells.microtubule[c, 1]
        a_g2 = cells.microtubule[c, 2]
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
            if measure < RANDOM_THRESHOLD:
                ar0, ar1 = random_direction(rand_state)
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
    mesh: TriangleMesh,
    centres: FloatArray,
    vertices: FloatArray,
    pressure: float,
    vertex_forces: FloatArray,
    centre_forces: FloatArray,
) -> None:
    """Adds `pressure * area / 2` along each triangle's normal to all three nodes."""
    offsets, tri_v2, tri_v3, tri_wall, _, _, wall_vertices = mesh
    x = vertices
    vd = vertex_forces
    cd = centre_forces
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
            coeff = pressure * (area / 2)
            cd[c, 0] += coeff * n0
            cd[c, 1] += coeff * n1
            cd[c, 2] += coeff * n2
            vd[i2, 0] += coeff * n0
            vd[i2, 1] += coeff * n1
            vd[i2, 2] += coeff * n2
            vd[i3, 0] += coeff * n0
            vd[i3, 1] += coeff * n1
            vd[i3, 2] += coeff * n2
