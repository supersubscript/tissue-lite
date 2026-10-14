"""Tissue's `.init` file format: topology, positions, wall and cell variables.

The cell variables are numbered as the model's `SlotLayout` says, so an init file
must be read with the layout of the model it belongs to.
"""

from pathlib import Path

import numpy as np

from tissue_lite.arrays import FloatArray
from tissue_lite.legacy.slots import SlotLayout
from tissue_lite.tissue import Tissue
from tissue_lite.topology import build_topology


def read_init(path: Path, layout: SlotLayout = SlotLayout()) -> Tissue:
    """Reads a Tissue `.init` file (without a stored centre triangulation)."""
    return parse_init(path.read_text(), layout)


def write_init(tissue: Tissue, path: Path, layout: SlotLayout = SlotLayout()) -> None:
    """Writes the state as Tissue's `-init_output` does (precision 20)."""
    path.write_text(format_init(tissue, layout))


def parse_init(text: str, layout: SlotLayout = SlotLayout()) -> Tissue:
    """Parses the contents of a Tissue `.init` file.

    Raises:
        ValueError: If the counts in the file are inconsistent.
    """
    tokens = text.split()
    pos = 0

    def take(n: int) -> list[str]:
        nonlocal pos
        if pos + n > len(tokens):
            raise ValueError("Init file ended early")
        out = tokens[pos : pos + n]
        pos += n
        return out

    num_cells, num_walls, num_vertices = (int(x) for x in take(3))
    walls = np.array(take(5 * num_walls), dtype=np.int64).reshape(num_walls, 5)
    if not np.array_equal(walls[:, 0], np.arange(num_walls)):
        raise ValueError("Walls must be listed in index order")
    count, dimension = (int(x) for x in take(2))
    if count != num_vertices or dimension != 3:
        raise ValueError("Expected the vertex block to hold 3D positions")
    vertices = np.array(take(3 * num_vertices), dtype=np.float64).reshape(-1, 3)
    count, num_length, num_wall_vars = (int(x) for x in take(3))
    if count != num_walls or num_length != 1:
        raise ValueError("Expected one length per wall")
    wall_data = np.array(take(num_walls * (1 + num_wall_vars)), dtype=np.float64)
    wall_data = wall_data.reshape(num_walls, 1 + num_wall_vars)
    count, num_cell_vars = (int(x) for x in take(2))
    if count != num_cells:
        raise ValueError("Cell block does not match the cell count")
    cell_variables = np.array(take(num_cells * num_cell_vars), dtype=np.float64)
    topology = build_topology(
        num_cells=num_cells,
        num_vertices=num_vertices,
        wall_cells=walls[:, 1:3].copy(),
        wall_vertices=walls[:, 3:5].copy(),
    )
    return Tissue(
        topology=topology,
        vertices=vertices,
        wall_lengths=wall_data[:, 0].copy(),
        wall_variables=wall_data[:, 1:].copy(),
        cells=layout.from_cell_variables(cell_variables.reshape(num_cells, -1)),
    )


def format_init(tissue: Tissue, layout: SlotLayout = SlotLayout()) -> str:
    """Formats the state in Tissue's `.init` layout with 20 significant digits."""
    topo = tissue.topology

    def row(values: FloatArray) -> str:
        return "".join(f"{x:.20g} " for x in values)

    cell_variables = layout.to_cell_variables(tissue)
    lines = [f"{topo.num_cells} {topo.num_walls} {topo.num_vertices}"]
    lines += [
        f"{i} {c1} {c2} {v1} {v2}"
        for i, ((c1, c2), (v1, v2)) in enumerate(
            zip(topo.wall_cells.tolist(), topo.wall_vertices.tolist(), strict=True)
        )
    ]
    lines += ["", f"{topo.num_vertices} 3"]
    lines += [row(x) for x in tissue.vertices]
    lines += ["", f"{topo.num_walls} 1 {tissue.wall_variables.shape[1]}"]
    wall_data = np.column_stack([tissue.wall_lengths, tissue.wall_variables])
    lines += [row(x) for x in wall_data]
    lines += ["", f"{topo.num_cells} {cell_variables.shape[1]}"]
    if cell_variables.shape[1]:
        lines += [row(x) for x in cell_variables]
        lines += [""]
    return "\n".join(lines) + "\n"
