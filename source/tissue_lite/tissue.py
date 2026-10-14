"""Tissue state: topology, vertex positions, wall rest lengths and named cell fields.

Tissue keeps all per-cell quantities in one numbered array; here each has its own
array in `CellState`. The numbering survives only in `tissue_lite.legacy.slots`.
"""

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import NamedTuple

import numpy as np

from tissue_lite.arrays import FloatArray
from tissue_lite.topology import Topology


class CellState(NamedTuple):
    """Per-cell arrays, (C,) or (C, 3) unless noted, updated in place by the kernels.

    A NamedTuple rather than a dataclass so that Numba kernels take it directly.
    `volume_change` is a tissue-level diagnostic that Tissue stores in cells 1-3;
    `extra` holds the legacy cell-variable columns no field maps to, so that files
    round-trip unchanged.
    """

    microtubule: FloatArray  # unit axis
    growth_signal: FloatArray  # Tissue's slot 3 ("cell vector length"), read by growth
    young_longitudinal: FloatArray
    young_transverse: FloatArray
    stress_direction: FloatArray
    stress_max: FloatArray
    stress_direction_second: FloatArray
    stress_second: FloatArray
    stress_anisotropy: FloatArray
    von_mises_stress: FloatArray
    strain_direction: FloatArray
    strain_max: FloatArray
    strain_perpendicular: FloatArray  # normal x strain_direction, unnormalised
    strain_second: FloatArray
    strain_anisotropy: FloatArray
    normal: FloatArray
    rest_area: FloatArray
    energy_isotropic: FloatArray  # per rest area
    energy_anisotropic: FloatArray  # per rest area
    stress_tensor: FloatArray  # (C, 3, 3) Cauchy, global frame
    strain_tensor: FloatArray  # (C, 3, 3) Almansi, global frame
    velocity: FloatArray  # mean vertex speed
    angle_mt_stress: FloatArray  # |cos| between microtubule and stress axes
    angle_stress_strain: FloatArray
    growth_count: FloatArray
    concentration: FloatArray  # input to ConcentrationSoftening; zeros when unused
    volume_change: FloatArray  # (3,): total displacement, its change, total speed
    extra: FloatArray  # (C, k)

    @classmethod
    def zeros(cls, num_cells: int, num_extra: int = 0) -> CellState:
        """Returns a state of the given size with every field zero."""
        return cls(
            **{
                name: np.zeros((num_cells, *_FIELD_SHAPES.get(name, ())))
                for name in cls._fields
                if name not in ("volume_change", "extra")
            },
            volume_change=np.zeros(3),
            extra=np.zeros((num_cells, num_extra)),
        )

    @property
    def num_cells(self) -> int:
        return len(self.growth_signal)

    def copy(self) -> CellState:
        return CellState(*(a.copy() for a in self))


_FIELD_SHAPES: dict[str, tuple[int, ...]] = {
    "microtubule": (3,),
    "stress_direction": (3,),
    "stress_direction_second": (3,),
    "strain_direction": (3,),
    "strain_perpendicular": (3,),
    "normal": (3,),
    "stress_tensor": (3, 3),
    "strain_tensor": (3, 3),
}


@dataclass
class CentreTriangulation:
    """Tissue's centre triangulation: one centre per cell, spokes to its vertices.

    `spokes[t]` holds the rest lengths of triangle t's two spokes (centre to vertex
    k, centre to vertex k + 1). Tissue stores one spoke per vertex unless the
    membrane uses double rest lengths; here every triangle owns both, and without
    growth the shared values stay equal. `wall_extensions[t]` is each triangle's
    growth added to its wall's rest length (0 without growth).
    """

    centres: FloatArray  # (C, 3)
    spokes: FloatArray  # (T, 2)
    wall_extensions: FloatArray  # (T,)


@dataclass
class Tissue:
    """A cell tissue: topology plus every state variable the simulation carries."""

    topology: Topology
    vertices: FloatArray  # (V, 3)
    wall_lengths: FloatArray  # (W,) rest lengths
    wall_variables: FloatArray  # (W, n), carried through unchanged
    cells: CellState
    triangulation: CentreTriangulation | None = None

    def copy(self) -> Tissue:
        """Returns a deep copy of the state; the topology is shared."""
        return replace(
            self,
            vertices=self.vertices.copy(),
            wall_lengths=self.wall_lengths.copy(),
            wall_variables=self.wall_variables.copy(),
            cells=self.cells.copy(),
            triangulation=deepcopy(self.triangulation),
        )

    def without_triangulation(self) -> Tissue:
        """Returns a copy as Tissue re-reads it from an `-init_output` file."""
        tissue = self.copy()
        tissue.triangulation = None
        return tissue


def compute_cell_centroids(tissue: Tissue) -> FloatArray:
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


def initiate_triangulation(tissue: Tissue) -> CentreTriangulation:
    """Creates the centre triangulation from scratch, as Tissue does without one stored.

    Centres are the wall-length-weighted means of the wall midpoints; spoke rest
    lengths are the current centre–vertex distances.
    """
    topo = tissue.topology
    centres = compute_cell_centroids(tissue)
    spoke = centres[topo.triangle_cells] - tissue.vertices[topo.cell_vertices]
    distances = np.sqrt(
        spoke[:, 0] * spoke[:, 0]
        + spoke[:, 1] * spoke[:, 1]
        + spoke[:, 2] * spoke[:, 2]
    )
    return CentreTriangulation(
        centres=centres,
        spokes=np.column_stack([distances, distances[topo.triangle_next]]),
        wall_extensions=np.zeros(len(distances)),
    )
