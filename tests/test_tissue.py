"""Tests for init-file IO, topology and the cell sort."""

from pathlib import Path

import numpy as np
import pytest
from conftest import CASES_DIR, MESHES_DIR

from tissue_lite.legacy.init_file import format_init, parse_init, read_init
from tissue_lite.tissue import CellState, compute_cell_centroids
from tissue_lite.topology import BACKGROUND

MESHES = sorted(MESHES_DIR.glob("*.init"))
TISSUE_WRITTEN = sorted(CASES_DIR.glob("*/final.init"))


@pytest.mark.parametrize("path", TISSUE_WRITTEN, ids=lambda p: p.parent.name)
def test_format_init_reproduces_tissue_output(path: Path) -> None:
    assert format_init(parse_init(path.read_text())) == path.read_text()


@pytest.mark.parametrize("path", MESHES, ids=lambda p: p.stem)
def test_topology_is_consistent(path: Path) -> None:
    topo = read_init(path).topology
    assert topo.offsets[-1] == len(topo.cell_vertices) == len(topo.cell_walls)
    for c in range(topo.num_cells):
        walls = topo.cell_walls[topo.offsets[c] : topo.offsets[c + 1]]
        assert ((topo.wall_cells[walls] == c).any(axis=1)).all()
    on_edge = (topo.wall_cells == BACKGROUND).any(axis=1)
    edge_vertices = np.unique(topo.wall_vertices[on_edge])
    np.testing.assert_array_equal(np.nonzero(topo.boundary)[0], edge_vertices)


@pytest.mark.parametrize("path", MESHES, ids=lambda p: p.stem)
def test_cells_are_sorted_into_closed_loops(path: Path) -> None:
    topo = read_init(path).topology
    for c in range(topo.num_cells):
        start, stop = topo.offsets[c], topo.offsets[c + 1]
        verts = topo.cell_vertices[start:stop]
        for k, w in enumerate(topo.cell_walls[start:stop]):
            pair = {int(verts[k]), int(verts[(k + 1) % len(verts)])}
            assert pair == set(topo.wall_vertices[w].tolist())


def test_cell_centroids_lie_within_the_mesh() -> None:
    tissue = read_init(MESHES_DIR / "patch19.init")
    centroids = compute_cell_centroids(tissue)
    assert centroids.shape == (19, 3)
    assert (centroids >= tissue.vertices.min(0)).all()
    assert (centroids <= tissue.vertices.max(0)).all()


def test_copy_is_independent() -> None:
    tissue = read_init(MESHES_DIR / "patch19.init")
    copy = tissue.copy()
    copy.vertices[0, 0] += 1.0
    copy.cells.microtubule[0, 0] += 1.0
    assert tissue.vertices[0, 0] != copy.vertices[0, 0]
    assert tissue.cells.microtubule[0, 0] != copy.cells.microtubule[0, 0]


def test_zero_cell_state_has_every_field() -> None:
    cells = CellState.zeros(4, num_extra=2)
    assert cells.num_cells == 4
    assert cells.stress_tensor.shape == (4, 3, 3)
    assert cells.extra.shape == (4, 2)
    assert cells.volume_change.shape == (3,)


def test_parse_init_rejects_truncated_file() -> None:
    text = (MESHES_DIR / "patch19.init").read_text()
    with pytest.raises(ValueError, match="ended early"):
        parse_init(text[: len(text) // 2])
