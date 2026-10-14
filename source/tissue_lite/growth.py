"""Strain-driven growth of the centre-triangulation rest lengths.

Ports `CenterTriangulation::WallGrowth::StrainTRBS` (double rest lengths) from Tissue
e74e457, including its hard-coded inputs: the growth axis is the microtubule axis
(cell variables 0-2) and the two compared values are cell variables 3 and 7 (the
maximal stress). Its velocity threshold is read but never used by Tissue.
"""

from dataclasses import dataclass

import numba
import numpy as np

from tissue_lite import layout
from tissue_lite.tissue import FloatArray, IntArray


@dataclass(frozen=True)
class StrainGrowth:
    """Extends rest lengths along and across the growth axis above a threshold."""

    rate: float
    threshold: float


@numba.njit(cache=True)
def update_growth(
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
    h: float,
    rate: float,
    threshold: float,
) -> None:
    """Applies one explicit growth update of length h to the double rest lengths."""
    comp = np.empty((3, 2))
    for c in range(offsets.shape[0] - 1):
        start = offsets[c]
        n_walls = offsets[c + 1] - start
        cell_vars[c, layout.GROWTH_COUNT] = 0
        g1 = cell_vars[c, layout.GROWTH_VALUE_1]
        g2 = cell_vars[c, layout.STRESS_MAX]
        gg0 = cell_vars[c, layout.MICROTUBULE_START]
        gg1 = cell_vars[c, layout.MICROTUBULE_START + 1]
        gg2 = cell_vars[c, layout.MICROTUBULE_START + 2]
        for k in range(n_walls):
            t = start + k
            i2 = tri_v2[t]
            i3 = tri_v3[t]
            w = tri_wall[t]
            rl0 = spokes[spoke_start[t]]
            rl2 = spokes[spoke_end[t]]
            rl1 = wall_lengths[w] + extensions[t]
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
            angle = np.arccos((l0 * l0 + l1 * l1 - l2 * l2) / (l0 * l1 * 2))
            qa = np.cos(angle) * l0
            qc = np.sin(angle) * l0
            qb = l1
            rest_angle = np.arccos(
                (rl0 * rl0 + rl1 * rl1 - rl2 * rl2) / (rl0 * rl1 * 2)
            )
            pa = np.cos(rest_angle) * rl0
            pc = np.sin(rest_angle) * rl0
            pb = rl1
            sv01 = 1 / pc
            sv20 = 1 / pb
            sv21 = -pa / (pb * pc)
            f00 = qb * sv20
            f10 = 0.0
            f01 = qa * sv01 + qb * sv21
            f11 = qc * sv01
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
            gl0 = xc0 * gg0 + xc1 * gg1 + xc2 * gg2
            gl1 = yc0 * gg0 + yc1 * gg1 + yc2 * gg2
            gr0 = f11 * gl0 - f01 * gl1
            gr1 = -f10 * gl0 + f00 * gl1
            gnorm = np.sqrt(gr0 * gr0 + gr1 * gr1)
            gr0 /= gnorm
            gr1 /= gnorm
            edges = ((-pa, -pc), (pb, 0.0), (pa - pb, pc))
            rests = (rl0, rl1, rl2)
            for j in range(3):
                cos_t = abs((gr0 * edges[j][0] + gr1 * edges[j][1]) / rests[j])
                sin_t = np.sqrt(abs(1 - cos_t * cos_t))
                comp[j, 0] = rests[j] * cos_t
                comp[j, 1] = rests[j] * sin_t
            if g1 > threshold and g2 < threshold:
                cell_vars[c, layout.GROWTH_COUNT] += 1
                f1 = rate * h * (g1 - threshold)
                for j in range(3):
                    comp[j, 0] += comp[j, 0] * f1
            if g1 < threshold and g2 > threshold:
                cell_vars[c, layout.GROWTH_COUNT] += 1
                f2 = rate * h * (g2 - threshold)
                for j in range(3):
                    comp[j, 1] += comp[j, 1] * f2
            if g1 > threshold and g2 > threshold:
                cell_vars[c, layout.GROWTH_COUNT] += 2
                f1 = rate * h * (g1 - threshold)
                f2 = rate * h * (g2 - threshold)
                for j in range(3):
                    comp[j, 0] += comp[j, 0] * f1
                    comp[j, 1] += comp[j, 1] * f2
            internal = np.sqrt(comp[0, 0] * comp[0, 0] + comp[0, 1] * comp[0, 1])
            external = np.sqrt(comp[1, 0] * comp[1, 0] + comp[1, 1] * comp[1, 1])
            internal_next = np.sqrt(comp[2, 0] * comp[2, 0] + comp[2, 1] * comp[2, 1])
            extensions[t] = external - wall_lengths[w]
            spokes[spoke_start[t]] = internal
            spokes[spoke_end[t]] = internal_next
