"""Tissue-compatible VTU/PVD output (Tissue's print flag 2).

Files, array names, point layout and number formatting (`%g`) follow Tissue's
`VTUostream::write_cells2` and `write_walls3`, so existing readers such as
`flower.output` load them unchanged. Cell variables are numbered by a `SlotLayout`.
"""

import logging
from pathlib import Path

import numpy as np

from tissue_lite.arrays import FloatArray
from tissue_lite.legacy.slots import SlotLayout
from tissue_lite.tissue import Tissue, compute_cell_centroids

logger = logging.getLogger(__name__)

_CELL_SHRINK = 0.1  # Tissue's WALL_RELATIVE_THICKNESS
_POLYGON = 7
_TRIANGLE = 5
_QUAD = 7  # Tissue writes VTK_POLYGON (7), not VTK_QUAD (9), for walls


def _format_g(value: float) -> str:
    """Formats a float as a default C++ ostream does."""
    return f"{value:g}"


def _row(values: FloatArray) -> str:
    """Formats values space-separated with a trailing space, as Tissue does."""
    return "".join(f"{_format_g(x)} " for x in values)


def _piece_header(n_points: int, n_cells: int) -> str:
    """Returns Tissue's <Piece> line, counts right-aligned to width 8."""
    return f'<Piece  NumberOfPoints="{n_points:>8}" NumberOfCells="{n_cells:>8}">\n'


def _shrunk_points(tissue: Tissue) -> FloatArray:
    """Each cell's vertices moved 10% of the way towards its centroid."""
    topo = tissue.topology
    centres = compute_cell_centroids(tissue)[topo.triangle_cells]
    p = tissue.vertices[topo.cell_vertices]
    return p + _CELL_SHRINK * (centres - p)


def format_cells(tissue: Tissue, layout: SlotLayout) -> str:
    """Returns the cell VTU file contents, with cell variables numbered by `layout`.

    Each cell's vertices are moved 10% towards its centroid (`WALL_RELATIVE_THICKNESS`)
    and values are written with `%g`, which keeps six significant digits.
    """
    topo = tissue.topology
    sizes = topo.cell_sizes
    points = _shrunk_points(tissue)
    v = layout.to_cell_variables(tissue)
    out = ['<VTKFile type="UnstructuredGrid" version="0.1">\n<UnstructuredGrid>\n']
    out.append(_piece_header(len(points), topo.num_cells))
    out.append(
        '<Points>\n<DataArray type="Float64" NumberOfComponents="3" format="ascii">\n'
    )
    out += [f"{_format_g(a)} {_format_g(b)} {_format_g(c)}\n" for a, b, c in points]
    out.append("</DataArray>\n</Points>\n")
    out.append('<Cells>\n<DataArray type="Int32" Name="connectivity" format="ascii">\n')
    count = 0
    for n in sizes:
        out.append("".join(f"{count + i} " for i in range(n)) + "\n")
        count += n
    out.append('</DataArray>\n<DataArray type="Int32" Name="offsets" format="ascii">\n')
    out.append("".join(f"{o} " for o in np.cumsum(sizes)) + "\n")
    out.append('</DataArray>\n<DataArray type="UInt8" Name="types" format="ascii">\n')
    cell_type = _TRIANGLE if (sizes == 3).all() else _POLYGON
    out.append(f"{cell_type} " * topo.num_cells + "\n")
    out.append("</DataArray>\n</Cells>\n")
    out.append('<CellData Scalars="cell variable 4" Vectors="cell vector">\n')
    out.append(
        '<DataArray type="Float64" NumberOfComponents="3" Name="cell vector" '
        'format="ascii">\n'
    )
    out += [f"{_format_g(a)} {_format_g(b)} {_format_g(c)}\n" for a, b, c in v[:, :3]]
    out.append("</DataArray>\n")
    out.append('<DataArray type="Float64" Name="cell vector length" format="ascii">\n')
    out.append(_row(v[:, 3]) + "\n</DataArray>\n")
    for i in range(4, v.shape[1]):
        out.append(
            f'<DataArray type="Float64" Name="cell variable {i}" format="ascii">\n'
        )
        out.append(_row(v[:, i]) + "\n</DataArray>\n")
    out.append("</CellData>\n</Piece>\n</UnstructuredGrid>\n</VTKFile>\n")
    return "".join(out)


def format_walls(tissue: Tissue) -> str:
    """Returns the wall VTU file contents (one quad per cell wall).

    Wall variables are written in pairs of columns, one per side of the wall; each quad
    takes its cell's side (the last column serves both sides when the count is odd).
    Values use `%g`.
    """
    topo = tissue.topology
    n_cell_walls = len(topo.cell_walls)
    out = ['<VTKFile type="UnstructuredGrid" version="0.1">\n<UnstructuredGrid>\n']
    out.append(_piece_header(topo.num_vertices + n_cell_walls, n_cell_walls))
    out.append(
        '<Points>\n<DataArray type="Float64" NumberOfComponents="3" format="ascii">\n'
    )
    out += [
        f"{_format_g(a)} {_format_g(b)} {_format_g(c)} \n"
        for a, b, c in tissue.vertices
    ]
    out += [
        f"{_format_g(a)} {_format_g(b)} {_format_g(c)}\n"
        for a, b, c in _shrunk_points(tissue)
    ]
    out.append("</DataArray>\n</Points>\n")
    out.append('<Cells>\n<DataArray type="Int32" Name="connectivity" format="ascii">\n')
    parts = []
    for c in range(topo.num_cells):
        start, stop = topo.offsets[c], topo.offsets[c + 1]
        verts = topo.cell_vertices[start:stop]
        temp = topo.num_vertices + start
        last = stop - start - 1
        for i in range(last):
            parts.append(f"{verts[i]} {verts[i + 1]} {temp + i + 1} {temp + i} ")
        parts.append(f"{verts[last]} {verts[0]} {temp} {temp + last} ")
    out.append("".join(parts) + "\n")
    out.append('</DataArray>\n<DataArray type="Int32" Name="offsets" format="ascii">\n')
    out.append("".join(f"{4 * (i + 1)} " for i in range(n_cell_walls)) + "\n")
    out.append('</DataArray>\n<DataArray type="UInt8" Name="types" format="ascii">\n')
    out.append(f"{_QUAD} " * n_cell_walls + "\n")
    out.append("</DataArray>\n</Cells>\n")
    out.append('<CellData Scalars="wall variable 0">\n')
    out.append('<DataArray type="Float64" Name="wall length" format="ascii">\n')
    lengths = tissue.wall_lengths[topo.cell_walls]
    out.append(_row(lengths) + "\n</DataArray>\n")
    is_second = topo.wall_cells[topo.cell_walls, 0] != topo.triangle_cells
    n_vars = tissue.wall_variables.shape[1]
    for i in range(0, n_vars, 2):
        column = i + is_second.astype(np.int64)
        values = tissue.wall_variables[topo.cell_walls, np.minimum(column, n_vars - 1)]
        out.append(
            f'<DataArray type="Float64" Name="wall variable {i // 2}" format="ascii">\n'
        )
        out.append(_row(values) + "\n</DataArray>\n")
    out.append("</CellData>\n</Piece>\n</UnstructuredGrid>\n</VTKFile>\n")
    return "".join(out)


def format_pvd(n_frames: int) -> str:
    """Returns Tissue's `tissue.pvd`, listing `n_frames` frames up front.

    Each frame has a cells and a walls dataset. For `n_frames > 2` a run with a
    fidelity stepper writes one more frame than is listed (KNOWN_ISSUES S6).
    """
    lines = ['<?xml version="1.0"?> ', '<VTKFile type="Collection" version="0.1" > ']
    lines.append("<Collection> ")
    for i in range(n_frames):
        for part, stem in enumerate(("VTK_cells", "VTK_walls")):
            lines.append(
                f'<DataSet timestep="{i}" part="{part}" file="{stem}{i:06d}.vtu"/>'
            )
    lines += ["</Collection> ", "</VTKFile> "]
    return "\n".join(lines) + "\n"


class VtuWriter:
    """Writes frames as `VTK_cells%06d.vtu` / `VTK_walls%06d.vtu` plus `tissue.pvd`.

    The PVD lists `n_frames` frames; `layout` numbers the cell variables. Existing files
    of the same name in `directory` are overwritten.
    """

    def __init__(self, directory: Path, n_frames: int, layout: SlotLayout) -> None:
        self.directory = directory
        self.n_frames = n_frames
        self.layout = layout

    def write(self, index: int, tissue: Tissue) -> None:
        """Writes frame `index`, creating the directory if needed.

        Index 0 also writes `tissue.pvd`.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        if index == 0:
            (self.directory / "tissue.pvd").write_text(format_pvd(self.n_frames))
        (self.directory / f"VTK_cells{index:06d}.vtu").write_text(
            format_cells(tissue, self.layout)
        )
        (self.directory / f"VTK_walls{index:06d}.vtu").write_text(format_walls(tissue))
        logger.debug("Wrote frame %d to %s", index, self.directory)
