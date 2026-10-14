"""Cuts a connected patch of cells around the apex from a Tissue `.init` mesh.

Usage: uv run python reference/cut_patch.py MESH.init N_CELLS OUT.init [N_VARS]

Cells are taken breadth-first from the cell whose centroid is highest (largest z),
walls and vertices keep their relative order, and walls leaving the patch face the
background. With N_VARS the cell variables are padded with zeros (or truncated).
"""

import sys
from collections import deque
from pathlib import Path

import numpy as np

from tissue_lite.tissue import (
    BACKGROUND,
    FloatArray,
    Tissue,
    build_topology,
    compute_cell_centroids,
)


def cut_patch(tissue: Tissue, n_cells: int) -> Tissue:
    """Returns the n_cells cells nearest (breadth-first) to the highest cell."""
    topo = tissue.topology
    start = int(np.argmax(compute_cell_centroids(tissue)[:, 2]))
    neighbours: list[list[int]] = [[] for _ in range(topo.num_cells)]
    for c1, c2 in topo.wall_cells.tolist():
        if BACKGROUND not in (c1, c2):
            neighbours[c1].append(c2)
            neighbours[c2].append(c1)
    chosen = [start]
    seen = {start}
    queue = deque([start])
    while queue and len(chosen) < n_cells:
        for nb in sorted(neighbours[queue.popleft()]):
            if nb not in seen and len(chosen) < n_cells:
                seen.add(nb)
                chosen.append(nb)
                queue.append(nb)
    keep = np.array(sorted(chosen))
    cell_map = np.full(topo.num_cells + 1, BACKGROUND)
    cell_map[keep] = np.arange(len(keep))
    wall_cells = cell_map[topo.wall_cells]  # BACKGROUND (-1) indexes the sentinel
    walls = np.nonzero((wall_cells != BACKGROUND).any(axis=1))[0]
    wall_vertices = topo.wall_vertices[walls]
    verts = np.unique(wall_vertices)
    vert_map = np.full(topo.num_vertices, -1)
    vert_map[verts] = np.arange(len(verts))
    new_wall_vertices = vert_map[wall_vertices]
    new_wall_cells = wall_cells[walls]
    return Tissue(
        topology=build_topology(
            num_cells=len(keep),
            num_vertices=len(verts),
            wall_cells=new_wall_cells,
            wall_vertices=new_wall_vertices,
        ),
        vertices=tissue.vertices[verts].copy(),
        wall_lengths=tissue.wall_lengths[walls].copy(),
        wall_variables=tissue.wall_variables[walls].copy(),
        cell_variables=tissue.cell_variables[keep].copy(),
    )


def resize_cell_variables(cell_vars: FloatArray, n_vars: int) -> FloatArray:
    """Returns the cell variables padded with zeros or truncated to n_vars columns."""
    resized = np.zeros((len(cell_vars), n_vars))
    resized[:, : min(n_vars, cell_vars.shape[1])] = cell_vars[:, :n_vars]
    return resized


def main() -> None:
    """Cuts the patch given on the command line (see the module docstring)."""
    mesh, n_cells, out = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
    patch = cut_patch(Tissue.read_init(mesh), n_cells)
    if len(sys.argv) > 4:
        patch.cell_variables = resize_cell_variables(
            patch.cell_variables, int(sys.argv[4])
        )
    patch.write_init(out)


if __name__ == "__main__":
    main()
