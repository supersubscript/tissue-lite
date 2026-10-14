"""Tissue topology and state, and Tissue's `.init` file format.

Topology is built exactly as Tissue's `readInit` builds it, including its cell sort
(`Tissue::sortCellWallAndCellVertex`), because the resulting vertex order fixes the
triangle order, the triangle orientation (hence the pressure direction) and the
floating-point summation order.
"""

import logging
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

BACKGROUND = -1

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class Topology:
    """Cell–wall–vertex connectivity; each cell's walls and vertices in cyclic order.

    Cell c owns entries `offsets[c]:offsets[c + 1]` of `cell_vertices` and `cell_walls`;
    wall k of a cell joins its vertices k and k + 1.
    """

    wall_cells: IntArray  # (W, 2); BACKGROUND for the outside
    wall_vertices: IntArray  # (W, 2)
    offsets: IntArray  # (C + 1,)
    cell_vertices: IntArray  # (sum N,)
    cell_walls: IntArray  # (sum N,)
    boundary: NDArray[np.bool_]  # (V,) vertex on a wall facing the background

    @property
    def num_cells(self) -> int:
        return len(self.offsets) - 1

    @property
    def num_vertices(self) -> int:
        return len(self.boundary)

    @property
    def num_walls(self) -> int:
        return len(self.wall_cells)

    @property
    def cell_sizes(self) -> IntArray:
        """Number of vertices (and walls) of each cell."""
        return np.diff(self.offsets)

    @property
    def triangle_cells(self) -> IntArray:
        """Cell of each centre triangle (one per cell wall)."""
        return np.repeat(np.arange(self.num_cells), self.cell_sizes)

    @property
    def triangle_next(self) -> IntArray:
        """Flat index of the next vertex of each triangle's cell (cyclic)."""
        index = np.arange(len(self.cell_vertices))
        start = self.offsets[self.triangle_cells]
        sizes = self.cell_sizes[self.triangle_cells]
        return start + (index - start + 1) % sizes


@dataclass
class CentreTriangulation:
    """Tissue's centre triangulation: one centre per cell, spokes to its vertices.

    With single rest lengths, `spokes[offsets[c] + k]` is the rest length of spoke k
    (centre to vertex k). With double rest lengths spoke k has two copies, as Tissue
    stores them: `spokes[2 * (offsets[c] + k)]`, used by triangle k - 1, and
    `spokes[2 * (offsets[c] + k) + 1]`, used by triangle k. `wall_extensions` is each
    cell's growth added to the rest length of its wall k.
    """

    centres: FloatArray  # (C, 3)
    spokes: FloatArray  # (sum N,) or (2 sum N,)
    wall_extensions: FloatArray | None = None  # (sum N,), double rest lengths only


@dataclass
class Tissue:
    """A cell tissue: topology plus every state variable Tissue keeps.

    `cell_variables` keeps Tissue's cell-variable layout (see `layout`), because the
    legacy models address cell state by index.
    """

    topology: Topology
    vertices: FloatArray  # (V, 3)
    wall_lengths: FloatArray  # (W,) rest lengths
    wall_variables: FloatArray  # (W, n_wall)
    cell_variables: FloatArray  # (C, n_cell)
    triangulation: CentreTriangulation | None = field(default=None)

    @classmethod
    def read_init(cls, path: Path) -> Tissue:
        """Reads a Tissue `.init` file (without a stored centre triangulation)."""
        return parse_init(path.read_text())

    def write_init(self, path: Path) -> None:
        """Writes the state as Tissue's `-init_output` does (precision 20)."""
        path.write_text(format_init(self))

    def copy(self) -> Tissue:
        """Returns a deep copy of the state; the topology is shared."""
        return replace(
            self,
            vertices=self.vertices.copy(),
            wall_lengths=self.wall_lengths.copy(),
            wall_variables=self.wall_variables.copy(),
            cell_variables=self.cell_variables.copy(),
            triangulation=deepcopy(self.triangulation),
        )

    def without_triangulation(self) -> Tissue:
        """Returns a copy as Tissue re-reads it from an `-init_output` file."""
        tissue = self.copy()
        tissue.triangulation = None
        return tissue


def parse_init(text: str) -> Tissue:
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
        cell_variables=cell_variables.reshape(num_cells, num_cell_vars),
    )


def format_init(tissue: Tissue) -> str:
    """Formats the state in Tissue's `.init` layout with 20 significant digits."""
    topo = tissue.topology

    def row(values: FloatArray) -> str:
        return "".join(f"{x:.20g} " for x in values)

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
    lines += ["", f"{topo.num_cells} {tissue.cell_variables.shape[1]}"]
    if tissue.cell_variables.shape[1]:
        lines += [row(x) for x in tissue.cell_variables]
        lines += [""]
    return "\n".join(lines) + "\n"


def build_topology(
    num_cells: int,
    num_vertices: int,
    wall_cells: IntArray,
    wall_vertices: IntArray,
) -> Topology:
    """Builds the cyclic cell connectivity exactly as Tissue's `readInit` does."""
    cell_walls: list[list[int]] = [[] for _ in range(num_cells)]
    cell_verts: list[list[int]] = [[] for _ in range(num_cells)]
    for w, ((c1, c2), (v1, v2)) in enumerate(
        zip(wall_cells.tolist(), wall_vertices.tolist(), strict=True)
    ):
        for c in (c1, c2):
            if c != BACKGROUND:
                cell_walls[c].append(w)
        for c in (c1, c2):
            if c == BACKGROUND:
                continue
            for v in (v1, v2):
                if v not in cell_verts[c]:
                    cell_verts[c].append(v)
    for c in range(num_cells):
        if len(cell_walls[c]) != len(cell_verts[c]):
            raise ValueError(f"Cell {c} has unequal numbers of walls and vertices")

    _sort_all_cells(cell_walls, cell_verts, wall_cells, wall_vertices)

    boundary = np.zeros(num_vertices, dtype=bool)
    on_background = (wall_cells == BACKGROUND).any(axis=1)
    boundary[wall_vertices[on_background].ravel()] = True
    sizes = [len(w) for w in cell_walls]
    return Topology(
        wall_cells=wall_cells,
        wall_vertices=wall_vertices,
        offsets=np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64),
        cell_vertices=np.array([v for vs in cell_verts for v in vs], dtype=np.int64),
        cell_walls=np.array([w for ws in cell_walls for w in ws], dtype=np.int64),
        boundary=boundary,
    )


def _sort_all_cells(
    cell_walls: list[list[int]],
    cell_verts: list[list[int]],
    wall_cells: IntArray,
    wall_vertices: IntArray,
) -> None:
    """Sorts every cell in Tissue's recursive depth-first order, in place."""
    num_cells = len(cell_walls)
    cell_sort = np.zeros((len(wall_cells), 2), dtype=np.int64)
    wc = wall_cells.tolist()
    wv = wall_vertices.tolist()
    done = [False] * num_cells
    for start in range(num_cells):
        if done[start]:
            continue
        # Iterative Tissue::sortCellRecursive (recursion would exceed Python's limit).
        _sort_cell(start, cell_walls, cell_verts, wc, wv, cell_sort)
        done[start] = True
        stack = [(start, 0)]
        while stack:
            cell, k = stack.pop()
            if k >= len(cell_walls[cell]):
                continue
            stack.append((cell, k + 1))
            w = cell_walls[cell][k]
            neighbour = wc[w][1] if wc[w][0] == cell else wc[w][0]
            if neighbour == BACKGROUND or done[neighbour]:
                continue
            _sort_cell(neighbour, cell_walls, cell_verts, wc, wv, cell_sort)
            done[neighbour] = True
            stack.append((neighbour, 0))


def _sort_cell(
    cell: int,
    cell_walls: list[list[int]],
    cell_verts: list[list[int]],
    wc: list[list[int]],
    wv: list[list[int]],
    cell_sort: IntArray,
) -> None:
    """Port of `Cell::sortWallAndVertex`; keeps neighbour orientations consistent."""
    walls = cell_walls[cell]
    n = len(walls)
    sorted_walls = [-1] * n
    sorted_verts = [-1] * n

    def side(w: int) -> int:
        if wc[w][0] == cell:
            return 0
        if wc[w][1] == cell:
            return 1
        raise ValueError(f"Wall {w} does not belong to cell {cell}")

    found = next((i for i, w in enumerate(walls) if cell_sort[w, 0] != 0), None)
    if found is None:
        wall_index = wall_start = vertex_index = 0
        w = walls[0]
        sorted_walls[0] = w
        sorted_verts[0], sorted_verts[1] = wv[w]
        vertex_index = 1
        s = side(w)
        cell_sort[w, s] = 1
        if wc[w][1 - s] != BACKGROUND:
            cell_sort[w, 1 - s] = -1
    else:
        wall_index = wall_start = vertex_index = found
        w = walls[found]
        sorted_walls[found] = w
        flag = cell_sort[w, side(w)]
        if flag == 0:
            raise ValueError(f"Wall {w} marked as sorted but is not")
        first, second = wv[w] if flag == 1 else wv[w][::-1]
        sorted_verts[vertex_index] = first
        vertex_index = (vertex_index + 1) % n
        sorted_verts[vertex_index] = second

    while wall_index < wall_start + n - 1:
        for w in walls:
            current = sorted_verts[vertex_index]
            if w == sorted_walls[wall_index % n] or current not in wv[w]:
                continue
            wall_index += 1
            sorted_walls[wall_index % n] = w
            v1, v2 = wv[w]
            vertex_index = (vertex_index + 1) % n
            if wall_index < wall_start + n - 1:
                sorted_verts[vertex_index] = v2 if current == v1 else v1
            s = side(w)
            flag = cell_sort[w, s]
            if flag == 1 and v2 != sorted_verts[vertex_index]:
                raise ValueError(f"Inconsistent orientation of wall {w}")
            if flag == -1 and v1 != sorted_verts[vertex_index]:
                raise ValueError(f"Inconsistent orientation of wall {w}")
            if flag == 0:
                if v2 == sorted_verts[vertex_index]:
                    cell_sort[w, s] = 1
                    if wc[w][1 - s] != BACKGROUND:
                        cell_sort[w, 1 - s] = -1
                elif v1 == sorted_verts[vertex_index]:
                    cell_sort[w, s] = -1
                    if wc[w][1 - s] != BACKGROUND:
                        cell_sort[w, 1 - s] = 1
                else:
                    raise ValueError(f"Vertex not in sorted wall {w}")
            break
    if -1 in sorted_walls or -1 in sorted_verts:
        raise ValueError(f"Cell {cell} could not be sorted")
    cell_walls[cell] = sorted_walls
    cell_verts[cell] = sorted_verts


def cell_centroids(tissue: Tissue) -> FloatArray:
    """Returns Tissue's `positionFromVertex`: wall midpoints weighted by wall length."""
    topo = tissue.topology
    ends = tissue.vertices[topo.wall_vertices[topo.cell_walls]]  # (sum N, 2, 3)
    diff = ends[:, 0] - ends[:, 1]
    # Explicit sums and bincount keep Tissue's sequential summation order.
    length = np.sqrt(
        diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1] + diff[:, 2] * diff[:, 2]
    )
    weighted = 0.5 * (ends[:, 0] + ends[:, 1]) * length[:, None]
    cells, n = topo.triangle_cells, topo.num_cells
    total = np.bincount(cells, weights=length, minlength=n)
    sums = [np.bincount(cells, weights=weighted[:, d], minlength=n) for d in range(3)]
    return np.column_stack(sums) / total[:, None]
