"""Forces on vertices and cell centres: TRBS-MT membrane and turgor pressure.

Ports the derivative parts of `VertexFromTRBScenterTriangulationMT` and
`Pressure3D::CenterTriangulation::Linear` from Tissue e74e457. Every expression
keeps Tissue's evaluation order so that results are bit-identical; do not simplify
them algebraically.
"""

import numba
import numpy as np

from tissue_lite.arrays import FloatArray, IntArray
from tissue_lite.material import (
    Material,
    compute_anisotropic_stress,
    compute_fibre_direction,
    compute_green_strain,
    compute_lame_parameters,
    compute_moduli,
)
from tissue_lite.tissue import CellState
from tissue_lite.triangles import (
    RestLengths,
    TriangleMesh,
    compute_deformation_gradient,
    compute_plane_coordinates,
    compute_shape_vectors,
    compute_triangle_area,
    compute_triangle_geometry,
    get_rest_lengths,
)

_SLIVER = 0.0001


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
    offsets = mesh.offsets
    vd = vertex_forces
    cd = centre_forces
    for c in range(offsets.shape[0] - 1):
        young_l, young_t = compute_moduli(cells, c, material)
        cells.young_transverse[c] = young_t
        lam_l, mio_l = compute_lame_parameters(young_l, material.poisson_longitudinal)
        lam_t, mio_t = compute_lame_parameters(young_t, material.poisson_transverse)
        d_lam = lam_l - lam_t
        d_mio = mio_l - mio_t
        microtubule = (
            cells.microtubule[c, 0],
            cells.microtubule[c, 1],
            cells.microtubule[c, 2],
        )
        for t in range(offsets[c], offsets[c + 1]):
            i2 = mesh.v2[t]
            i3 = mesh.v3[t]
            tri = compute_triangle_geometry(mesh, centres, vertices, c, t)
            p00, p01, p02 = tri.centre
            p10, p11, p12 = tri.vertex_a
            p20, p21, p22 = tri.vertex_b
            l0, l1, l2 = tri.lengths
            xc0, xc1, xc2 = tri.x_axis
            yc0, yc1, yc2 = tri.y_axis
            rest_lengths = get_rest_lengths(mesh, rest, t, material.double_rest_lengths)
            r0, r1, r2 = rest_lengths
            rest_area = compute_triangle_area(rest_lengths)
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
            shape = compute_shape_vectors(compute_plane_coordinates(rest_lengths))
            (sv00, sv01), (sv10, sv11), (sv20, sv21) = shape
            gradient = compute_deformation_gradient(
                compute_plane_coordinates(tri.lengths), shape
            )
            (f00, f01), (f10, f11) = gradient
            fibre = compute_fibre_direction(
                gradient, tri.x_axis, tri.y_axis, microtubule, rand_state
            )
            strain = compute_green_strain(gradient)
            (ds00, ds01), (ds10, ds11) = compute_anisotropic_stress(
                strain, fibre, d_lam, d_mio
            )
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
    offsets = mesh.offsets
    for c in range(offsets.shape[0] - 1):
        for t in range(offsets[c], offsets[c + 1]):
            tri = compute_triangle_geometry(mesh, centres, vertices, c, t)
            coeff = pressure * (compute_triangle_area(tri.lengths) / 2)
            for d in range(3):
                centre_forces[c, d] += coeff * tri.normal[d]
                vertex_forces[mesh.v2[t], d] += coeff * tri.normal[d]
                vertex_forces[mesh.v3[t], d] += coeff * tri.normal[d]
