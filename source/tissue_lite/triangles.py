"""Centre triangles: index arrays, rest lengths and the per-triangle kinematics.

Each cell wall k of cell c gives one triangle (centre, vertex k, vertex k + 1) in
Tissue's cell and wall order. The kernels here keep Tissue's evaluation order so
that results are bit-identical; do not simplify them algebraically.
"""

from typing import NamedTuple

import numba
import numpy as np

from tissue_lite.arrays import FloatArray, IntArray
from tissue_lite.tissue import CentreTriangulation, Tissue
from tissue_lite.topology import Topology

type Vector = tuple[float, float, float]
type Matrix2 = tuple[tuple[float, float], tuple[float, float]]
type ShapeVectors = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]


class TriangleMesh(NamedTuple):
    """Triangle t of cell c (`offsets[c] <= t < offsets[c + 1]`) has vertices v2[t]
    and v3[t] (the cell's vertices k and k + 1) and wall[t]; its spoke rest lengths
    are `spokes[spoke_start[t]]` (centre-v2) and `spokes[spoke_end[t]]` (centre-v3)."""

    offsets: IntArray
    v2: IntArray
    v3: IntArray
    wall: IntArray
    spoke_start: IntArray
    spoke_end: IntArray


class RestLengths(NamedTuple):
    """Rest lengths referenced from the tissue, so growth updates are seen at once."""

    spokes: FloatArray
    wall_lengths: FloatArray
    wall_extensions: FloatArray  # empty with single rest lengths


class TriangleGeometry(NamedTuple):
    """One centre triangle in its current configuration.

    `lengths` are centre-vertex a, vertex a-vertex b (the wall) and centre-vertex b.
    The local frame is Tissue's: `x_axis` along the wall from a to b, `normal` the
    unit x × (centre - a), `y_axis` = normal × x.
    """

    centre: Vector
    vertex_a: Vector
    vertex_b: Vector
    lengths: Vector
    x_axis: Vector
    y_axis: Vector
    normal: Vector


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


@numba.njit(cache=True)
def subtract(a: Vector, b: Vector) -> Vector:
    """Returns a - b."""
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


@numba.njit(cache=True)
def dot(a: Vector, b: Vector) -> float:
    """Returns a · b, summed left to right."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@numba.njit(cache=True)
def norm(v: Vector) -> float:
    """Returns |v|."""
    return np.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


@numba.njit(cache=True)
def divide(v: Vector, divisor: float) -> Vector:
    """Returns v / divisor, dividing each component (not multiplying by 1/divisor)."""
    return v[0] / divisor, v[1] / divisor, v[2] / divisor


@numba.njit(cache=True)
def cross(a: Vector, b: Vector) -> Vector:
    """Returns a × b."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


@numba.njit(cache=True)
def compute_triangle_geometry(
    mesh: TriangleMesh,
    centres: FloatArray,
    vertices: FloatArray,
    c: int,
    t: int,
) -> TriangleGeometry:
    """Positions, edge lengths and local frame of triangle t of cell c."""
    a = mesh.v2[t]
    b = mesh.v3[t]
    centre = (centres[c, 0], centres[c, 1], centres[c, 2])
    vertex_a = (vertices[a, 0], vertices[a, 1], vertices[a, 2])
    vertex_b = (vertices[b, 0], vertices[b, 1], vertices[b, 2])
    centre_from_a = subtract(centre, vertex_a)
    b_from_a = subtract(vertex_b, vertex_a)
    length_a = norm(centre_from_a)
    length_wall = norm(b_from_a)  # Tissue's wall length: the same squares, summed
    length_b = norm(subtract(centre, vertex_b))
    x_axis = divide(b_from_a, length_wall)
    towards_centre = divide(centre_from_a, length_a)
    normal = cross(x_axis, towards_centre)
    normal = divide(normal, norm(normal))
    y_axis = cross(normal, x_axis)
    return TriangleGeometry(
        centre,
        vertex_a,
        vertex_b,
        (length_a, length_wall, length_b),
        x_axis,
        y_axis,
        normal,
    )


@numba.njit(cache=True)
def get_rest_lengths(
    mesh: TriangleMesh, rest: RestLengths, t: int, double_rest_lengths: bool
) -> Vector:
    """Rest lengths of triangle t in the order of `TriangleGeometry.lengths`."""
    wall = rest.wall_lengths[mesh.wall[t]]
    if double_rest_lengths:
        wall = wall + rest.wall_extensions[t]
    return rest.spokes[mesh.spoke_start[t]], wall, rest.spokes[mesh.spoke_end[t]]


@numba.njit(cache=True)
def compute_triangle_area(lengths: Vector) -> float:
    """Heron's formula in Tissue's factor order; NaN for impossible lengths."""
    a, b, c = lengths
    return np.sqrt((a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c)) * 0.25


@numba.njit(cache=True)
def compute_plane_coordinates(lengths: Vector) -> Vector:
    """Lays the triangle in its plane: vertex a at the origin, vertex b at (b, 0),
    the centre at (a, c). Returns (a, b, c)."""
    to_a, wall, to_b = lengths
    cos_a = (to_a * to_a + wall * wall - to_b * to_b) / (to_a * wall * 2)
    return cos_a * to_a, wall, np.sqrt(1 - cos_a * cos_a) * to_a


@numba.njit(cache=True)
def compute_shape_vectors(rest_plane: Vector) -> ShapeVectors:
    """Gradients of the barycentric coordinates of (centre, vertex a, vertex b) in
    the rest plane (Tissue's `ShapeVectorRest`)."""
    a, b, c = rest_plane
    return (0.0, 1 / c), (-1 / b, (a - b) / (b * c)), (1 / b, -a / (b * c))


@numba.njit(cache=True)
def compute_deformation_gradient(plane: Vector, shape: ShapeVectors) -> Matrix2:
    """F from the rest plane to the current plane; F[1][0] is 0 by construction."""
    a, b, c = plane
    return (b * shape[2][0], a * shape[0][1] + b * shape[2][1]), (0.0, c * shape[0][1])
