"""Strain-driven growth of the centre-triangulation rest lengths.

Ports `CenterTriangulation::WallGrowth::StrainTRBS` (double rest lengths) from Tissue
e74e457, including its hard-coded inputs: the growth axis is the microtubule axis
and the two compared values are `growth_signal` and `stress_max` (KNOWN_ISSUES
G1-G3). Its velocity threshold is read but never used by Tissue; `StrainGrowth`
can apply it (G1).
"""

from dataclasses import dataclass

import numba
import numpy as np

from tissue_lite.arrays import FloatArray
from tissue_lite.tissue import CellState
from tissue_lite.triangles import (
    TriangleGeometry,
    TriangleMesh,
    Vector,
    compute_cosine,
    compute_deformation_gradient,
    compute_shape_vectors,
    compute_triangle_geometry,
    get_rest_lengths,
    pull_back_direction,
)

type EdgeComponents = tuple[
    tuple[float, float], tuple[float, float], tuple[float, float]
]  # (along, across) of the edges centre-a, a-b, b-centre


@dataclass(frozen=True)
class StrainGrowth:
    """Extends rest lengths along and across the growth axis above a threshold.

    `threshold` is compared with `growth_signal` (a strain) and `stress_max` (a stress,
    KNOWN_ISSUES G2). Without `velocity_threshold` growth runs every step, as in
    Tissue (G1); with it, only in cells slower than it.
    """

    rate: float
    threshold: float
    velocity_threshold: float | None = None


@numba.njit(cache=True)
def update_growth(
    mesh: TriangleMesh,
    centres: FloatArray,
    vertices: FloatArray,
    cells: CellState,
    h: float,
    rate: float,
    threshold: float,
    velocity_threshold: float,
) -> None:
    """Applies one explicit growth update of length h to the rest lengths in `mesh`.

    Each triangle's three rest edges are split into components along and across
    the growth axis, `cells.microtubule`. The along components are scaled by
    `1 + rate * h * (growth_signal - threshold)` where the signal exceeds `threshold`,
    the across ones likewise with `stress_max` (KNOWN_ISSUES G2). Writes `mesh.spokes`
    and `mesh.wall_extensions` in place, and the branch count into `cells.growth_count`
    (G3). Cells not slower than `velocity_threshold` do not grow (inf: none, as
    Tissue, G1). `h` is the solver's trial step (S2).

    Raises:
        ZeroDivisionError: If a cell's microtubule axis is zero or normal to a triangle.
    """
    for c in range(mesh.offsets.shape[0] - 1):
        cells.growth_count[c] = 0
        if not cells.velocity[c] < velocity_threshold:
            continue
        signal = cells.growth_signal[c]
        stress = cells.stress_max[c]
        growth_axis = (
            cells.microtubule[c, 0],
            cells.microtubule[c, 1],
            cells.microtubule[c, 2],
        )
        for t in range(mesh.offsets[c], mesh.offsets[c + 1]):
            triangle = compute_triangle_geometry(mesh, centres, vertices, c, t)
            rest_lengths = get_rest_lengths(mesh, t)
            rest_plane = _compute_plane_by_angle(rest_lengths)
            direction = _compute_growth_direction(
                triangle,
                _compute_plane_by_angle(triangle.lengths),
                rest_plane,
                growth_axis,
            )
            components = _split_rest_edges(rest_plane, rest_lengths, direction)
            components, count = _grow_components(
                components, signal, stress, rate * h, threshold
            )
            cells.growth_count[c] += count
            spoke_a, wall, spoke_b = _compute_edge_lengths(components)
            mesh.wall_extensions[t] = wall - mesh.wall_lengths[mesh.wall[t]]
            mesh.spokes[t, 0] = spoke_a
            mesh.spokes[t, 1] = spoke_b


@numba.njit(cache=True)
def _compute_plane_by_angle(lengths: Vector) -> Vector:
    """`compute_plane_coordinates` as the growth reaction writes it: through acos,
    cos and sin, which differ from cos and sqrt(1 - cos²) in the last bits."""
    to_a, wall, to_b = lengths
    angle = np.arccos(compute_cosine(to_a, wall, to_b))
    return np.cos(angle) * to_a, wall, np.sin(angle) * to_a


@numba.njit(cache=True)
def _compute_growth_direction(
    triangle: TriangleGeometry, plane: Vector, rest_plane: Vector, growth_axis: Vector
) -> tuple[float, float]:
    """Unit growth axis in the rest plane; divides by zero when it is out of plane
    (no random fallback here, unlike the membrane)."""
    gradient = compute_deformation_gradient(plane, compute_shape_vectors(rest_plane))
    g0, g1 = pull_back_direction(
        gradient, triangle.x_axis, triangle.y_axis, growth_axis
    )
    length = np.sqrt(g0 * g0 + g1 * g1)
    return g0 / length, g1 / length


@numba.njit(cache=True)
def _split_edge(
    edge: tuple[float, float], length: float, direction: tuple[float, float]
) -> tuple[float, float]:
    """Returns a rest edge's lengths along and across the growth direction."""
    cos_t = abs((direction[0] * edge[0] + direction[1] * edge[1]) / length)
    sin_t = np.sqrt(abs(1 - cos_t * cos_t))
    return length * cos_t, length * sin_t


@numba.njit(cache=True)
def _split_rest_edges(
    rest_plane: Vector, rest_lengths: Vector, direction: tuple[float, float]
) -> EdgeComponents:
    """Splits the edges centre-a, a-b and b-centre of the rest triangle."""
    centre_x, wall, centre_y = rest_plane
    return (
        _split_edge((-centre_x, -centre_y), rest_lengths[0], direction),
        _split_edge((wall, 0.0), rest_lengths[1], direction),
        _split_edge((centre_x - wall, centre_y), rest_lengths[2], direction),
    )


@numba.njit(cache=True)
def _grow_components(
    components: EdgeComponents,
    signal: float,
    stress: float,
    gain: float,
    threshold: float,
) -> tuple[EdgeComponents, int]:
    """Grows the along components by the growth signal and the across ones by the
    maximal stress, each where above `threshold`; returns Tissue's branch count."""
    along = 0.0
    across = 0.0
    count = 0
    if signal > threshold and stress < threshold:
        count += 1
        along = gain * (signal - threshold)
    if signal < threshold and stress > threshold:
        count += 1
        across = gain * (stress - threshold)
    if signal > threshold and stress > threshold:
        count += 2
        along = gain * (signal - threshold)
        across = gain * (stress - threshold)
    grown = (
        _grow_edge(components[0], along, across),
        _grow_edge(components[1], along, across),
        _grow_edge(components[2], along, across),
    )
    return grown, count


@numba.njit(cache=True)
def _grow_edge(
    edge: tuple[float, float], along: float, across: float
) -> tuple[float, float]:
    """Returns the edge components each extended by their factor (0 leaves them)."""
    return edge[0] + edge[0] * along, edge[1] + edge[1] * across


@numba.njit(cache=True)
def _compute_edge_lengths(components: EdgeComponents) -> Vector:
    """Returns the rest lengths of the edges centre-a, a-b and b-centre."""
    return (
        _compute_length(components[0]),
        _compute_length(components[1]),
        _compute_length(components[2]),
    )


@numba.njit(cache=True)
def _compute_length(edge: tuple[float, float]) -> float:
    """Returns the length of an (along, across) pair."""
    return np.sqrt(edge[0] * edge[0] + edge[1] * edge[1])
