"""Strain-driven growth of the centre-triangulation rest lengths.

Ports `CenterTriangulation::WallGrowth::StrainTRBS` (double rest lengths) from Tissue
e74e457, including its hard-coded inputs: the growth axis is the microtubule axis
and the two compared values are `growth_signal` and `stress_max`. Its velocity
threshold is read but never used by Tissue.
"""

from dataclasses import dataclass

import numba
import numpy as np

from tissue_lite.arrays import FloatArray
from tissue_lite.material import pull_back_direction
from tissue_lite.tissue import CellState
from tissue_lite.triangles import (
    RestLengths,
    TriangleMesh,
    compute_deformation_gradient,
    compute_shape_vectors,
    compute_triangle_geometry,
    get_rest_lengths,
)


@dataclass(frozen=True)
class StrainGrowth:
    """Extends rest lengths along and across the growth axis above a threshold."""

    rate: float
    threshold: float


@numba.njit(cache=True)
def update_growth(
    mesh: TriangleMesh,
    rest: RestLengths,
    centres: FloatArray,
    vertices: FloatArray,
    cells: CellState,
    h: float,
    rate: float,
    threshold: float,
) -> None:
    """Applies one explicit growth update of length h to the double rest lengths."""
    offsets = mesh.offsets
    spokes = rest.spokes
    wall_lengths = rest.wall_lengths
    extensions = rest.wall_extensions
    comp = np.empty((3, 2))
    for c in range(offsets.shape[0] - 1):
        start = offsets[c]
        n_walls = offsets[c + 1] - start
        cells.growth_count[c] = 0
        g1 = cells.growth_signal[c]
        g2 = cells.stress_max[c]
        growth_axis = (
            cells.microtubule[c, 0],
            cells.microtubule[c, 1],
            cells.microtubule[c, 2],
        )
        for k in range(n_walls):
            t = start + k
            w = mesh.wall[t]
            tri = compute_triangle_geometry(mesh, centres, vertices, c, t)
            l0, l1, l2 = tri.lengths
            rl0, rl1, rl2 = get_rest_lengths(mesh, rest, t, True)
            # Tissue's growth uses acos/cos/sin here, not the cos/sqrt(1 - cos²) of the
            # membrane, so the deformation gradient differs from it in the last bits.
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
            gradient = compute_deformation_gradient(
                (qa, qb, qc), compute_shape_vectors((pa, pb, pc))
            )
            gr0, gr1 = pull_back_direction(
                gradient, tri.x_axis, tri.y_axis, growth_axis
            )
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
                cells.growth_count[c] += 1
                f1 = rate * h * (g1 - threshold)
                for j in range(3):
                    comp[j, 0] += comp[j, 0] * f1
            if g1 < threshold and g2 > threshold:
                cells.growth_count[c] += 1
                f2 = rate * h * (g2 - threshold)
                for j in range(3):
                    comp[j, 1] += comp[j, 1] * f2
            if g1 > threshold and g2 > threshold:
                cells.growth_count[c] += 2
                f1 = rate * h * (g1 - threshold)
                f2 = rate * h * (g2 - threshold)
                for j in range(3):
                    comp[j, 0] += comp[j, 0] * f1
                    comp[j, 1] += comp[j, 1] * f2
            internal = np.sqrt(comp[0, 0] * comp[0, 0] + comp[0, 1] * comp[0, 1])
            external = np.sqrt(comp[1, 0] * comp[1, 0] + comp[1, 1] * comp[1, 1])
            internal_next = np.sqrt(comp[2, 0] * comp[2, 0] + comp[2, 1] * comp[2, 1])
            extensions[t] = external - wall_lengths[w]
            spokes[mesh.spoke_start[t]] = internal
            spokes[mesh.spoke_end[t]] = internal_next
