"""Strain-driven growth of the centre-triangulation rest lengths.

Ports `CenterTriangulation::WallGrowth::StrainTRBS` (double rest lengths) from Tissue
e74e457, including its hard-coded inputs: the growth axis is the microtubule axis
and the two compared values are `growth_signal` and `stress_max` (KNOWN_ISSUES
G1-G3). Its velocity threshold is read but never used by Tissue. Every expression
keeps Tissue's evaluation order so that results are bit-identical.
"""

from dataclasses import dataclass

import numba
import numpy as np

from tissue_lite.arrays import FloatArray
from tissue_lite.material import pull_back_direction
from tissue_lite.tissue import CellState
from tissue_lite.triangles import (
    RestLengths,
    TriangleGeometry,
    TriangleMesh,
    Vector,
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
    """Applies one explicit growth update of length h to the double rest lengths.

    Each triangle's three rest edges are split into components along and across
    the growth axis; the component whose signal exceeds `threshold` grows by
    `rate * h * (signal - threshold)`. `cells.growth_count` records the branches
    taken.
    """
    for c in range(mesh.offsets.shape[0] - 1):
        cells.growth_count[c] = 0
        signal = cells.growth_signal[c]
        stress = cells.stress_max[c]
        growth_axis = (
            cells.microtubule[c, 0],
            cells.microtubule[c, 1],
            cells.microtubule[c, 2],
        )
        for t in range(mesh.offsets[c], mesh.offsets[c + 1]):
            tri = compute_triangle_geometry(mesh, centres, vertices, c, t)
            rest_lengths = get_rest_lengths(mesh, rest, t, True)
            rest_plane = _compute_plane_by_angle(rest_lengths)
            direction = _compute_growth_direction(
                tri, _compute_plane_by_angle(tri.lengths), rest_plane, growth_axis
            )
            components = _split_rest_edges(rest_plane, rest_lengths, direction)
            components, count = _grow_components(
                components, signal, stress, rate * h, threshold
            )
            cells.growth_count[c] += count
            spoke_a, wall, spoke_b = _compute_edge_lengths(components)
            rest.wall_extensions[t] = wall - rest.wall_lengths[mesh.wall[t]]
            rest.spokes[mesh.spoke_start[t]] = spoke_a
            rest.spokes[mesh.spoke_end[t]] = spoke_b


type EdgeComponents = tuple[
    tuple[float, float], tuple[float, float], tuple[float, float]
]  # (along, across) of the edges centre-a, a-b, b-centre


@numba.njit(cache=True)
def _compute_plane_by_angle(lengths: Vector) -> Vector:
    """`compute_plane_coordinates` as the growth reaction writes it: through acos,
    cos and sin, which differ from cos and sqrt(1 - cos²) in the last bits."""
    to_a, wall, to_b = lengths
    angle = np.arccos((to_a * to_a + wall * wall - to_b * to_b) / (to_a * wall * 2))
    return np.cos(angle) * to_a, wall, np.sin(angle) * to_a


@numba.njit(cache=True)
def _compute_growth_direction(
    tri: TriangleGeometry, plane: Vector, rest_plane: Vector, growth_axis: Vector
) -> tuple[float, float]:
    """Unit growth axis in the rest plane; divides by zero when it is out of plane
    (no random fallback here, unlike the membrane)."""
    gradient = compute_deformation_gradient(plane, compute_shape_vectors(rest_plane))
    g0, g1 = pull_back_direction(gradient, tri.x_axis, tri.y_axis, growth_axis)
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
    a, b, c = rest_plane
    return (
        _split_edge((-a, -c), rest_lengths[0], direction),
        _split_edge((b, 0.0), rest_lengths[1], direction),
        _split_edge((a - b, c), rest_lengths[2], direction),
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
        np.sqrt(
            components[0][0] * components[0][0] + components[0][1] * components[0][1]
        ),
        np.sqrt(
            components[1][0] * components[1][0] + components[1][1] * components[1][1]
        ),
        np.sqrt(
            components[2][0] * components[2][0] + components[2][1] * components[2][1]
        ),
    )
