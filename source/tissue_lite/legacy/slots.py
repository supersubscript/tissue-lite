"""Where Tissue keeps each `CellState` field in its numbered cell variables.

The numbering is fixed by the retained model files; `SlotLayout` converts between
it and `CellState` for the `.init` and VTU formats, and the model reader validates
model files against the slot constants (KNOWN_ISSUES L1).
"""

from dataclasses import dataclass

import numpy as np

from tissue_lite.arrays import FloatArray
from tissue_lite.tissue import CellState, Tissue

MICROTUBULE = slice(0, 3)
GROWTH_SIGNAL = 3  # "cell vector length"; read by StrainTRBS growth, never written
STRESS_DIRECTION = slice(4, 7)
STRESS_MAX = 7
STRAIN_DIRECTION = slice(8, 11)
STRAIN_MAX = 11
YOUNG_LONGITUDINAL = 16
STRAIN_ANISOTROPY = 17
STRESS_ANISOTROPY = 18
REST_AREA = 19  # Tissue's "area ratio index"
YOUNG_TRANSVERSE = 20  # also the iso and aniso energy index
VON_MISES_STRESS = 21  # Tissue's "MT stress index"
VELOCITY = 22
ANGLE_MT_STRESS = 23
ANGLE_STRESS_STRAIN = 24
VOLUME_CHANGE = 25  # written in cells 1, 2 and 3
STRAIN_PERPENDICULAR = slice(26, 29)  # second axes only
STRAIN_SECOND = 29  # second axes only
GROWTH_COUNT = 30
STRESS_DIRECTION_SECOND = slice(31, 34)  # second axes only (overwrites the normal)
STRESS_SECOND = 34  # second axes only
NORMAL = slice(32, 35)
STRESS_TENSOR_START = 26  # TRBS level-0 index; written only by dead branches
NUM_SLOTS = 35

_UNUSED = [12, 13, 14, 15, VOLUME_CHANGE]
_SECOND_AXES = [26, 27, 28, 29, 31]
_VOLUME_CELLS = slice(1, 4)

# Field → slot, in the order Tissue writes them (the second axes overwrite the normal).
_FIELD_SLOTS: dict[str, int | slice] = {
    "microtubule": MICROTUBULE,
    "growth_signal": GROWTH_SIGNAL,
    "stress_direction": STRESS_DIRECTION,
    "stress_max": STRESS_MAX,
    "strain_direction": STRAIN_DIRECTION,
    "strain_max": STRAIN_MAX,
    "young_longitudinal": YOUNG_LONGITUDINAL,
    "strain_anisotropy": STRAIN_ANISOTROPY,
    "stress_anisotropy": STRESS_ANISOTROPY,
    "rest_area": REST_AREA,
    "young_transverse": YOUNG_TRANSVERSE,
    "von_mises_stress": VON_MISES_STRESS,
    "velocity": VELOCITY,
    "angle_mt_stress": ANGLE_MT_STRESS,
    "angle_stress_strain": ANGLE_STRESS_STRAIN,
    "growth_count": GROWTH_COUNT,
    "normal": NORMAL,
}
_SECOND_AXES_SLOTS: dict[str, int | slice] = {
    "strain_perpendicular": STRAIN_PERPENDICULAR,
    "strain_second": STRAIN_SECOND,
    "stress_direction_second": STRESS_DIRECTION_SECOND,
    "stress_second": STRESS_SECOND,
}


@dataclass(frozen=True)
class SlotLayout:
    """Tissue's cell-variable numbering for one model.

    `second_axes` is the growth models' storage: the perpendicular strain axis in
    slots 26-29 and the second stress axis in 31-34, over the normal (KNOWN_ISSUES T9).
    `concentration_slot` is the column holding the softening concentration, if any.
    Columns no field maps to (12-15, 25, the second-axis slots when unused, and
    everything from 35 on except the concentration) pass through `CellState.extra`.
    """

    second_axes: bool = False
    concentration_slot: int | None = None

    def field_slots(self) -> dict[str, int | slice]:
        """Returns each stored `CellState` field's slot, in Tissue's write order."""
        slots = dict(_FIELD_SLOTS)
        if self.second_axes:
            slots |= _SECOND_AXES_SLOTS
        if self.concentration_slot is not None:
            slots["concentration"] = self.concentration_slot
        return slots

    def extra_columns(self, num_columns: int) -> list[int]:
        """Returns the columns `CellState.extra` carries, in order."""
        columns = _UNUSED + ([] if self.second_axes else _SECOND_AXES)
        columns += [
            c for c in range(NUM_SLOTS, num_columns) if c != self.concentration_slot
        ]
        return columns

    def from_cell_variables(self, values: FloatArray) -> CellState:
        """Builds the named state from Tissue's (C, n) cell variables, copying them.

        `volume_change` is read from slot 25 of cells 1-3 (KNOWN_ISSUES L2).

        Raises:
            ValueError: If there are fewer than 35 columns.
        """
        num_cells, num_columns = values.shape
        if num_columns < NUM_SLOTS:
            raise ValueError(f"Tissue models need at least {NUM_SLOTS} cell variables")
        extra_columns = self.extra_columns(num_columns)
        return CellState.zeros(num_cells, len(extra_columns))._replace(
            **{
                name: values[:, slot].copy()
                for name, slot in self.field_slots().items()
            },
            volume_change=_extract_volume_change(values),
            extra=values[:, extra_columns].copy(),
        )

    def to_cell_variables(self, tissue: Tissue) -> FloatArray:
        """Returns Tissue's (C, n) cell variables, writing slots in Tissue's order.

        With `second_axes` the second stress axis overwrites the normal (KNOWN_ISSUES
        T9). `volume_change` goes into slot 25 of cells 1-3 (L2). A state built without
        legacy columns (`CellState.zeros`) leaves them 0.
        """
        cells = tissue.cells
        extra_base = len(_UNUSED) + (0 if self.second_axes else len(_SECOND_AXES))
        num_columns = NUM_SLOTS + max(0, cells.extra.shape[1] - extra_base)
        if self.concentration_slot is not None:
            num_columns = max(num_columns + 1, self.concentration_slot + 1)
        cell_variables = np.zeros((cells.num_cells, num_columns))
        columns = self.extra_columns(num_columns)
        width = min(len(columns), cells.extra.shape[1])
        cell_variables[:, columns[:width]] = cells.extra[:, :width]
        for name, slot in self.field_slots().items():
            cell_variables[:, slot] = getattr(cells, name)
        cell_variables[_VOLUME_CELLS, VOLUME_CHANGE] = cells.volume_change[
            : cells.num_cells - 1
        ]
        return cell_variables


def _extract_volume_change(values: FloatArray) -> FloatArray:
    """Returns the volume-change diagnostic from slot 25 of cells 1-3."""
    volume_change = np.zeros(3)
    stored = values[_VOLUME_CELLS, VOLUME_CHANGE]
    volume_change[: len(stored)] = stored
    return volume_change
