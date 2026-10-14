"""Tests for the slot layout between `CellState` and Tissue's cell variables."""

import numpy as np
import pytest

from tissue_lite.legacy.slots import SlotLayout
from tissue_lite.tissue import CellState, Tissue
from tissue_lite.topology import build_topology


def _make_triangle_tissue(cells: CellState) -> Tissue:
    """One three-wall cell around the given state."""
    topology = build_topology(
        num_cells=1,
        num_vertices=3,
        wall_cells=np.array([[0, -1], [0, -1], [0, -1]]),
        wall_vertices=np.array([[0, 1], [1, 2], [2, 0]]),
    )
    return Tissue(
        topology=topology,
        vertices=np.zeros((3, 3)),
        wall_lengths=np.ones(3),
        wall_variables=np.zeros((3, 0)),
        cells=cells,
    )


@pytest.mark.parametrize(
    "layout",
    [SlotLayout(), SlotLayout(second_axes=True), SlotLayout(concentration_slot=41)],
    ids=["sweep", "growth", "softening"],
)
def test_slot_layout_round_trips_every_column(layout: SlotLayout) -> None:
    values = np.arange(5 * 42, dtype=float).reshape(5, 42)
    cells = layout.from_cell_variables(values)
    tissue = _make_triangle_tissue(cells)
    np.testing.assert_array_equal(layout.to_cell_variables(tissue), values)


def test_slot_layout_rejects_short_rows() -> None:
    with pytest.raises(ValueError, match="35"):
        SlotLayout().from_cell_variables(np.zeros((2, 30)))


def test_second_axes_overwrite_the_normal() -> None:
    cells = CellState.zeros(1)
    cells.normal[0] = (1.0, 2.0, 3.0)
    cells.stress_direction_second[0] = (4.0, 5.0, 6.0)
    cells.stress_second[0] = 7.0
    sweep = SlotLayout().to_cell_variables(_make_triangle_tissue(cells))
    growth = SlotLayout(second_axes=True).to_cell_variables(
        _make_triangle_tissue(cells)
    )
    np.testing.assert_array_equal(sweep[0, 32:35], (1.0, 2.0, 3.0))
    np.testing.assert_array_equal(growth[0, 31:35], (4.0, 5.0, 6.0, 7.0))
