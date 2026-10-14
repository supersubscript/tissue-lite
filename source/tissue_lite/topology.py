"""Cell–wall–vertex connectivity, built exactly as Tissue's `readInit` builds it.

The cell sort (`Tissue::sortCellWallAndCellVertex`) is reproduced because the
resulting vertex order fixes the triangle order, the triangle orientation (hence
the pressure direction) and the floating-point summation order.
"""

from dataclasses import dataclass

import numpy as np

from tissue_lite.arrays import BoolArray, IntArray

BACKGROUND = -1


@dataclass(frozen=True)
class Topology:
    """Each cell's walls and vertices in cyclic order.

    Cell c owns entries `offsets[c]:offsets[c + 1]` of `cell_vertices` and `cell_walls`;
    wall k of a cell joins its vertices k and k + 1 (the last wraps to the first).
    `wall_cells` names each wall's two cells, with `BACKGROUND` (-1) for the outside.
    Treated as read-only: copies of a `Tissue` share it.
    """

    wall_cells: IntArray  # (W, 2); BACKGROUND for the outside
    wall_vertices: IntArray  # (W, 2)
    offsets: IntArray  # (C + 1,)
    cell_vertices: IntArray  # (sum N,)
    cell_walls: IntArray  # (sum N,)
    boundary: BoolArray  # (V,) vertex on a wall facing the background

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


def build_topology(
    num_cells: int,
    num_vertices: int,
    wall_cells: IntArray,
    wall_vertices: IntArray,
) -> Topology:
    """Builds the cyclic cell connectivity exactly as Tissue's `readInit` does.

    `wall_cells` and `wall_vertices` are (W, 2) int arrays, `BACKGROUND` (-1) marking
    the outside of a boundary wall; they are stored without copying.

    Raises:
        ValueError: If a cell's walls do not close into one consistently oriented loop.
    """
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
