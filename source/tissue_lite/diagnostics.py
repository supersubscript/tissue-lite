"""Diagnostics Tissue writes into the cell state while evaluating derivatives.

Ports `Calculate::AngleVectors`, `TemplateVolumeChange` and
`Calculate::VertexVelocity` from Tissue e74e457. They run inside the force
evaluation, so they describe the state at the start of each step.
"""

import numba

from tissue_lite.arrays import FloatArray, IntArray
from tissue_lite.tissue import CellState
from tissue_lite.triangles import dot, norm, subtract


@numba.njit(cache=True)
def _write_angle_cosines(
    first: FloatArray, second: FloatArray, store: FloatArray
) -> None:
    """Stores |cos| of the angle between the axes, normalising both in place."""
    for c in range(first.shape[0]):
        for axis in (first, second):
            length = norm((axis[c, 0], axis[c, 1], axis[c, 2]))
            if length > 0:
                for d in range(3):
                    axis[c, d] = axis[c, d] / length
        store[c] = abs(
            dot(
                (first[c, 0], first[c, 1], first[c, 2]),
                (second[c, 0], second[c, 1], second[c, 2]),
            )
        )


def update_angle_diagnostics(cells: CellState) -> None:
    """Writes the microtubule–stress and stress–strain angles, in Tissue's order.

    Normalises the microtubule, stress and strain axes as a side effect (the stress
    axis twice, which can change its last bit).
    """
    _write_angle_cosines(
        cells.microtubule, cells.stress_direction, cells.angle_mt_stress
    )
    _write_angle_cosines(
        cells.stress_direction, cells.strain_direction, cells.angle_stress_strain
    )


@numba.njit(cache=True)
def update_volume_change(
    volume_change: FloatArray,
    vertices: FloatArray,
    rest_positions: FloatArray,
    vertex_forces: FloatArray,
) -> None:
    """Stores (total displacement, its change since the last call, total speed)."""
    change = 0.0
    total = 0.0
    for v in range(vertices.shape[0]):
        vertex = (vertices[v, 0], vertices[v, 1], vertices[v, 2])
        rest = (rest_positions[v, 0], rest_positions[v, 1], rest_positions[v, 2])
        change += norm(subtract(vertex, rest))
        total += norm((vertex_forces[v, 0], vertex_forces[v, 1], vertex_forces[v, 2]))
    volume_change[1] = change - volume_change[0]
    volume_change[0] = change
    volume_change[2] = total


@numba.njit(cache=True)
def update_cell_velocity(
    velocity: FloatArray,
    offsets: IntArray,
    cell_vertices: IntArray,
    vertex_forces: FloatArray,
) -> None:
    """Stores each cell's mean vertex speed (the forces are the velocities)."""
    for c in range(len(offsets) - 1):
        total = 0.0
        for k in range(offsets[c], offsets[c + 1]):
            v = cell_vertices[k]
            total += norm(
                (vertex_forces[v, 0], vertex_forces[v, 1], vertex_forces[v, 2])
            )
        velocity[c] = total / (offsets[c + 1] - offsets[c])
