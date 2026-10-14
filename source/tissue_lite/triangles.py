"""Index arrays of the centre triangles and their rest lengths, for the kernels.

Each cell wall k of cell c gives one triangle (centre, vertex k, vertex k + 1) in
Tissue's cell and wall order.
"""

from typing import NamedTuple

import numpy as np

from tissue_lite.arrays import FloatArray, IntArray
from tissue_lite.tissue import CentreTriangulation, Tissue
from tissue_lite.topology import Topology


class TriangleMesh(NamedTuple):
    """Triangle t of cell c (`offsets[c] <= t < offsets[c + 1]`) has vertices v2[t]
    and v3[t] and wall[t]; its spoke rest lengths are `spokes[spoke_start[t]]`
    (centre-v2) and `spokes[spoke_end[t]]` (centre-v3)."""

    offsets: IntArray
    v2: IntArray
    v3: IntArray
    wall: IntArray
    spoke_start: IntArray
    spoke_end: IntArray
    wall_vertices: IntArray


class RestLengths(NamedTuple):
    """Rest lengths referenced from the tissue, so growth updates are seen at once."""

    spokes: FloatArray
    wall_lengths: FloatArray
    wall_extensions: FloatArray  # empty with single rest lengths


def build_triangle_mesh(topology: Topology, double_rest_lengths: bool) -> TriangleMesh:
    """Builds the triangle index arrays for single or double rest lengths."""
    flat = np.arange(len(topology.cell_vertices))
    nxt = topology.triangle_next
    return TriangleMesh(
        offsets=topology.offsets,
        v2=topology.cell_vertices,
        v3=topology.cell_vertices[nxt],
        wall=topology.cell_walls,
        spoke_start=2 * flat + 1 if double_rest_lengths else flat,
        spoke_end=2 * nxt if double_rest_lengths else nxt,
        wall_vertices=topology.wall_vertices,
    )


def build_rest_lengths(
    tissue: Tissue, triangulation: CentreTriangulation
) -> RestLengths:
    """Bundles the tissue's rest-length arrays without copying them."""
    extensions = triangulation.wall_extensions
    return RestLengths(
        spokes=triangulation.spokes,
        wall_lengths=tissue.wall_lengths,
        wall_extensions=np.empty(0) if extensions is None else extensions,
    )
