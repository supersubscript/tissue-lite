"""Where Tissue keeps each `CellState` field in its numbered cell variables.

The numbering is fixed by the retained model files; `SlotLayout` converts between
it and `CellState` for the `.init` and VTU formats, and the model reader validates
model files against the slot constants.
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


@dataclass(frozen=True)
class SlotLayout:
    """Tissue's cell-variable numbering for one model.

    `second_axes` is the growth models' storage: the perpendicular strain axis in
    slots 26-29 and the second stress axis in 31-34, over the normal. Columns no
    field maps to (12-15, 25, the second-axis slots when unused, and everything
    from 35 on except the concentration) pass through `CellState.extra`.
    """

    second_axes: bool = False
    concentration_slot: int | None = None

    def extra_columns(self, num_columns: int) -> list[int]:
        """Returns the columns `CellState.extra` carries, in order."""
        columns = _UNUSED + ([] if self.second_axes else _SECOND_AXES)
        columns += [
            c for c in range(NUM_SLOTS, num_columns) if c != self.concentration_slot
        ]
        return columns

    def from_cell_variables(self, values: FloatArray) -> CellState:
        """Builds the named state from Tissue's (C, n) cell variables.

        Raises:
            ValueError: If there are fewer than 35 columns.
        """
        num_cells, num_columns = values.shape
        if num_columns < NUM_SLOTS:
            raise ValueError(f"Tissue models need at least {NUM_SLOTS} cell variables")
        none3 = np.zeros((num_cells, 3))
        none1 = np.zeros(num_cells)
        v = values

        def column(index: int | slice, used: bool = True) -> FloatArray:
            if not used:
                return none1.copy() if isinstance(index, int) else none3.copy()
            return v[:, index].copy()

        return CellState(
            microtubule=column(MICROTUBULE),
            growth_signal=column(GROWTH_SIGNAL),
            young_longitudinal=column(YOUNG_LONGITUDINAL),
            young_transverse=column(YOUNG_TRANSVERSE),
            stress_direction=column(STRESS_DIRECTION),
            stress_max=column(STRESS_MAX),
            stress_direction_second=column(STRESS_DIRECTION_SECOND, self.second_axes),
            stress_second=column(STRESS_SECOND, self.second_axes),
            stress_anisotropy=column(STRESS_ANISOTROPY),
            von_mises_stress=column(VON_MISES_STRESS),
            strain_direction=column(STRAIN_DIRECTION),
            strain_max=column(STRAIN_MAX),
            strain_perpendicular=column(STRAIN_PERPENDICULAR, self.second_axes),
            strain_second=column(STRAIN_SECOND, self.second_axes),
            strain_anisotropy=column(STRAIN_ANISOTROPY),
            normal=column(NORMAL),
            rest_area=column(REST_AREA),
            energy_isotropic=none1.copy(),
            energy_anisotropic=column(YOUNG_TRANSVERSE),
            stress_tensor=np.zeros((num_cells, 3, 3)),
            strain_tensor=np.zeros((num_cells, 3, 3)),
            velocity=column(VELOCITY),
            angle_mt_stress=column(ANGLE_MT_STRESS),
            angle_stress_strain=column(ANGLE_STRESS_STRAIN),
            growth_count=column(GROWTH_COUNT),
            concentration=none1.copy()
            if self.concentration_slot is None
            else column(self.concentration_slot),
            volume_change=_read_volume_change(v),
            extra=v[:, self.extra_columns(num_columns)].copy(),
        )

    def to_cell_variables(self, tissue: Tissue) -> FloatArray:
        """Returns Tissue's (C, n) cell variables, writing slots in Tissue's order.

        Slot 20 holds the transverse modulus, or the anisotropic energy when the
        model softens walls (Tissue then never writes the modulus).
        """
        cells = tissue.cells
        extra_base = len(_UNUSED) + (0 if self.second_axes else len(_SECOND_AXES))
        num_columns = NUM_SLOTS + max(0, cells.extra.shape[1] - extra_base)
        if self.concentration_slot is not None:
            num_columns = max(num_columns + 1, self.concentration_slot + 1)
        v = np.zeros((cells.num_cells, num_columns))
        # A state built without legacy columns (`CellState.zeros`) leaves them 0.
        columns = self.extra_columns(num_columns)
        width = min(len(columns), cells.extra.shape[1])
        v[:, columns[:width]] = cells.extra[:, :width]
        v[:, MICROTUBULE] = cells.microtubule
        v[:, GROWTH_SIGNAL] = cells.growth_signal
        v[:, STRESS_DIRECTION] = cells.stress_direction
        v[:, STRESS_MAX] = cells.stress_max
        v[:, STRAIN_DIRECTION] = cells.strain_direction
        v[:, STRAIN_MAX] = cells.strain_max
        v[:, YOUNG_LONGITUDINAL] = cells.young_longitudinal
        v[:, STRAIN_ANISOTROPY] = cells.strain_anisotropy
        v[:, STRESS_ANISOTROPY] = cells.stress_anisotropy
        v[:, REST_AREA] = cells.rest_area
        v[:, YOUNG_TRANSVERSE] = (
            cells.young_transverse
            if self.concentration_slot is None
            else cells.energy_anisotropic
        )
        v[:, VON_MISES_STRESS] = cells.von_mises_stress
        v[:, VELOCITY] = cells.velocity
        v[:, ANGLE_MT_STRESS] = cells.angle_mt_stress
        v[:, ANGLE_STRESS_STRAIN] = cells.angle_stress_strain
        v[_VOLUME_CELLS, VOLUME_CHANGE] = cells.volume_change[: cells.num_cells - 1]
        v[:, GROWTH_COUNT] = cells.growth_count
        v[:, NORMAL] = cells.normal
        if self.second_axes:
            v[:, STRAIN_PERPENDICULAR] = cells.strain_perpendicular
            v[:, STRAIN_SECOND] = cells.strain_second
            v[:, STRESS_DIRECTION_SECOND] = cells.stress_direction_second
            v[:, STRESS_SECOND] = cells.stress_second
        if self.concentration_slot is not None:
            v[:, self.concentration_slot] = cells.concentration
        return v


def _read_volume_change(values: FloatArray) -> FloatArray:
    """Returns the volume-change diagnostic from slot 25 of cells 1-3."""
    volume_change = np.zeros(3)
    stored = values[_VOLUME_CELLS, VOLUME_CHANGE]
    volume_change[: len(stored)] = stored
    return volume_change
