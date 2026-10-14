"""Centre triangles: index arrays, rest lengths and the per-triangle kinematics.

Each cell wall k of cell c gives one triangle (centre, vertex k, vertex k + 1) in
Tissue's cell and wall order.
"""

from typing import NamedTuple

import numba
import numpy as np

from tissue_lite.arrays import FloatArray, IntArray
from tissue_lite.tissue import Tissue, compute_row_norms

type Vector = tuple[float, float, float]
type Matrix2 = tuple[tuple[float, float], tuple[float, float]]
type ShapeVectors = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]


class TriangleMesh(NamedTuple):
    """Index arrays and rest lengths of the centre triangles.

    Triangle t of cell c (`offsets[c] <= t < offsets[c + 1]`) joins the centre to
    the cell's vertices k and k + 1 across `wall[t]`. Tissue stores one spoke rest
    length per vertex unless the membrane uses double rest lengths; here every
    triangle owns both spokes, and without growth the shared values stay equal.
    `wall_lengths` is the tissue's own array.
    """

    offsets: IntArray  # (C + 1,)
    vertex_a: IntArray  # (T,)
    vertex_b: IntArray  # (T,)
    wall: IntArray  # (T,)
    spokes: FloatArray  # (T, 2) rest lengths centre-a, centre-b
    wall_extensions: FloatArray  # (T,) growth added to the wall's rest length
    wall_lengths: FloatArray  # (W,)


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


def build_triangle_mesh(tissue: Tissue, centres: FloatArray) -> TriangleMesh:
    """Triangulates every cell from its centre, as `CenterTriangulation::Initiate`.

    Spoke rest lengths are the current centre-vertex distances; the walls keep the
    tissue's rest lengths.
    """
    topology = tissue.topology
    vertex_a = topology.cell_vertices
    spokes_a = compute_row_norms(
        centres[topology.triangle_cells] - tissue.vertices[vertex_a]
    )
    return TriangleMesh(
        offsets=topology.offsets,
        vertex_a=vertex_a,
        vertex_b=vertex_a[topology.triangle_next],
        wall=topology.cell_walls,
        spokes=np.column_stack([spokes_a, spokes_a[topology.triangle_next]]),
        wall_extensions=np.zeros(len(vertex_a)),
        wall_lengths=tissue.wall_lengths,
    )


@numba.njit(cache=True)
def subtract(a: Vector, b: Vector) -> Vector:
    """Returns a - b."""
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


@numba.njit(cache=True)
def combine(weight_a: float, a: Vector, weight_b: float, b: Vector) -> Vector:
    """Returns weight_a * a + weight_b * b."""
    return (
        weight_a * a[0] + weight_b * b[0],
        weight_a * a[1] + weight_b * b[1],
        weight_a * a[2] + weight_b * b[2],
    )


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
def multiply(m: Matrix2, n: Matrix2) -> Matrix2:
    """Returns the 2×2 product m n, each entry summed left to right."""
    return (
        (m[0][0] * n[0][0] + m[0][1] * n[1][0], m[0][0] * n[0][1] + m[0][1] * n[1][1]),
        (m[1][0] * n[0][0] + m[1][1] * n[1][0], m[1][0] * n[0][1] + m[1][1] * n[1][1]),
    )


@numba.njit(cache=True)
def transpose(m: Matrix2) -> Matrix2:
    """Returns mᵀ."""
    return (m[0][0], m[1][0]), (m[0][1], m[1][1])


@numba.njit(cache=True)
def outer(a: tuple[float, float]) -> Matrix2:
    """Returns a aᵀ."""
    return (a[0] * a[0], a[0] * a[1]), (a[1] * a[0], a[1] * a[1])


@numba.njit(cache=True)
def compute_triangle_geometry(
    mesh: TriangleMesh,
    centres: FloatArray,
    vertices: FloatArray,
    c: int,
    t: int,
) -> TriangleGeometry:
    """Positions, edge lengths and local frame of triangle t of cell c."""
    a = mesh.vertex_a[t]
    b = mesh.vertex_b[t]
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
def get_rest_lengths(mesh: TriangleMesh, t: int) -> Vector:
    """Rest lengths of triangle t in the order of `TriangleGeometry.lengths`."""
    wall = mesh.wall_lengths[mesh.wall[t]] + mesh.wall_extensions[t]
    return mesh.spokes[t, 0], wall, mesh.spokes[t, 1]


@numba.njit(cache=True)
def compute_triangle_area(lengths: Vector) -> float:
    """Heron's formula in Tissue's factor order; NaN for impossible lengths."""
    a, b, c = lengths
    return np.sqrt((a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c)) * 0.25


@numba.njit(cache=True)
def compute_cosine(a: float, b: float, opposite: float) -> float:
    """Law of cosines: the cosine of the angle between the sides a and b."""
    return (a * a + b * b - opposite * opposite) / (a * b * 2)


@numba.njit(cache=True)
def compute_plane_coordinates(lengths: Vector) -> Vector:
    """Lays the triangle flat: vertex a at the origin, b at (wall, 0), the centre at
    (centre_x, centre_y). Returns (centre_x, wall, centre_y)."""
    to_a, wall, to_b = lengths
    cos_a = compute_cosine(to_a, wall, to_b)
    return cos_a * to_a, wall, np.sqrt(1 - cos_a * cos_a) * to_a


@numba.njit(cache=True)
def compute_shape_vectors(rest_plane: Vector) -> ShapeVectors:
    """Barycentric gradients of (centre, vertex a, vertex b) in the rest plane
    (Tissue's `ShapeVectorRest`)."""
    centre_x, wall, centre_y = rest_plane
    return (
        (0.0, 1 / centre_y),
        (-1 / wall, (centre_x - wall) / (wall * centre_y)),
        (1 / wall, -centre_x / (wall * centre_y)),
    )


@numba.njit(cache=True)
def compute_deformation_gradient(plane: Vector, shape: ShapeVectors) -> Matrix2:
    """F from the rest plane to the current plane; F[1][0] is 0 by construction."""
    centre_x, wall, centre_y = plane
    return (
        (wall * shape[2][0], centre_x * shape[0][1] + wall * shape[2][1]),
        (0.0, centre_y * shape[0][1]),
    )


@numba.njit(cache=True)
def compute_deformation(
    lengths: Vector, rest_lengths: Vector
) -> tuple[float, ShapeVectors, Matrix2]:
    """Returns the rest area, the rest shape vectors and the deformation gradient."""
    rest_area = compute_triangle_area(rest_lengths)
    shape = compute_shape_vectors(compute_plane_coordinates(rest_lengths))
    gradient = compute_deformation_gradient(compute_plane_coordinates(lengths), shape)
    return rest_area, shape, gradient


@numba.njit(cache=True)
def compute_green_strain(gradient: Matrix2) -> Matrix2:
    """Returns E = (FᵀF - I) / 2."""
    (c00, c01), (c10, c11) = multiply(transpose(gradient), gradient)
    return (0.5 * (c00 - 1), 0.5 * c01), (0.5 * c10, 0.5 * (c11 - 1))


@numba.njit(cache=True)
def pull_back_direction(
    gradient: Matrix2, x_axis: Vector, y_axis: Vector, direction: Vector
) -> tuple[float, float]:
    """Projects a direction onto the triangle and maps it to the rest plane with
    adj(F): F⁻¹ without the 1/det F (KNOWN_ISSUES T2). Unnormalised."""
    local_x = dot(x_axis, direction)
    local_y = dot(y_axis, direction)
    return (
        gradient[1][1] * local_x - gradient[0][1] * local_y,
        -gradient[1][0] * local_x + gradient[0][0] * local_y,
    )
