"""Diagnostics Tissue writes into the cell state while evaluating derivatives.

Ports `Calculate::AngleVectors`, `TemplateVolumeChange` and
`Calculate::VertexVelocity` from Tissue e74e457. They run inside the force
evaluation, so they describe the state at the start of each step.
"""

import numba
import numpy as np

from tissue_lite.arrays import FloatArray
from tissue_lite.tissue import CellState
from tissue_lite.triangles import TriangleMesh


@numba.njit(cache=True)
def _angle_diagnostic(first: FloatArray, second: FloatArray, store: FloatArray) -> None:
    """Stores |cos| of the angle between the axes, normalising both in place."""
    for c in range(first.shape[0]):
        for axis in (first, second):
            x, y, z = axis[c, 0], axis[c, 1], axis[c, 2]
            norm = np.sqrt(x * x + y * y + z * z)
            if norm > 0:
                for d in range(3):
                    axis[c, d] = axis[c, d] / norm
        dot = 0.0
        for d in range(3):
            dot += first[c, d] * second[c, d]
        store[c] = abs(dot)


def update_angle_diagnostics(cells: CellState) -> None:
    """Writes the microtubule–stress and stress–strain angles, in Tissue's order.

    Normalises the microtubule, stress and strain axes as a side effect (the stress
    axis twice, which can change its last bit).
    """
    _angle_diagnostic(cells.microtubule, cells.stress_direction, cells.angle_mt_stress)
    _angle_diagnostic(
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
        dx = vertices[v, 0] - rest_positions[v, 0]
        dy = vertices[v, 1] - rest_positions[v, 1]
        dz = vertices[v, 2] - rest_positions[v, 2]
        change += np.sqrt(dx * dx + dy * dy + dz * dz)
        f = vertex_forces[v]
        total += np.sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2])
    volume_change[1] = change - volume_change[0]
    volume_change[0] = change
    volume_change[2] = total


@numba.njit(cache=True)
def update_cell_velocity(
    cells: CellState, mesh: TriangleMesh, vertex_forces: FloatArray
) -> None:
    """Stores each cell's mean vertex speed (the forces are the velocities)."""
    offsets = mesh.offsets
    cell_vertices = mesh.v2
    for c in range(len(offsets) - 1):
        total = 0.0
        for k in range(offsets[c], offsets[c + 1]):
            f = vertex_forces[cell_vertices[k]]
            total += np.sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2])
        cells.velocity[c] = total / (offsets[c + 1] - offsets[c])
