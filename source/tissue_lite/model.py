"""The meristem model and its per-call evaluation (Tissue's reaction list).

Tissue evaluates its reactions in model-file order, and several of them write cell
state as a side effect of computing derivatives. `ModelState` reproduces the order
of every retained model file:

- forces: TRBS membrane, pressure, angle diagnostics, volume diagnostic, boundary,
  vertex velocity;
- per-step updates: fibres, cell stress and strain, pressure ramp, microtubules,
  growth.
"""

import logging
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from tissue_lite import diagnostics, feedback, forces, growth, stress
from tissue_lite.arrays import BoolArray, FloatArray, IntArray
from tissue_lite.feedback import (
    FibreKernel,
    FibreStiffness,
    MicrotubuleAlignment,
    MicrotubuleKernel,
)
from tissue_lite.growth import StrainGrowth
from tissue_lite.material import (
    Boundary,
    ConcentrationSoftening,
    Material,
    Pressure,
    TRBSMembrane,
)
from tissue_lite.rand import make_rand_state
from tissue_lite.tissue import (
    CellState,
    CentreTriangulation,
    Tissue,
    initiate_triangulation,
)
from tissue_lite.triangles import TriangleMesh, build_triangle_mesh

logger = logging.getLogger(__name__)

type TomlValue = (
    bool | int | float | str | None | tuple[TomlValue, ...] | dict[str, TomlValue]
)


@dataclass(frozen=True)
class MeristemModel:
    """Everything that defines a simulation except the mesh and the solver.

    Without `track_velocity` the cell velocity keeps its initial value, as in
    Tissue models lacking `Calculate::VertexVelocity`.
    """

    membrane: TRBSMembrane
    pressure: Pressure
    boundary: Boundary = Boundary()
    fibres: FibreStiffness | None = None
    microtubules: MicrotubuleAlignment | None = None
    growth: StrainGrowth | None = None
    track_velocity: bool = True

    def __post_init__(self) -> None:
        if self.growth is not None and not self.membrane.double_rest_lengths:
            raise ValueError("StrainGrowth needs double_rest_lengths")

    @classmethod
    def from_toml(cls, path: Path) -> MeristemModel:
        """Reads a model written by `to_toml`."""
        tables = tomllib.loads(path.read_text())
        membrane = dict(tables["membrane"])
        if "softening" in membrane:
            membrane["softening"] = ConcentrationSoftening(**membrane["softening"])
        boundary = tables.get("boundary", {})
        return cls(
            membrane=TRBSMembrane(**membrane),
            pressure=Pressure(**tables["pressure"]),
            boundary=Boundary(fixed_axes=tuple(boundary.get("fixed_axes", (0, 1, 2)))),
            fibres=FibreStiffness(**tables["fibres"]) if "fibres" in tables else None,
            microtubules=MicrotubuleAlignment(**tables["microtubules"])
            if "microtubules" in tables
            else None,
            growth=StrainGrowth(**tables["growth"]) if "growth" in tables else None,
            track_velocity=tables.get("track_velocity", True),
        )

    def to_toml(self, path: Path) -> None:
        """Writes the model as TOML (None-valued fields are omitted)."""
        path.write_text(_format_toml(asdict(self)))


def _format_toml(tables: dict[str, TomlValue]) -> str:
    """Formats a nested dict of scalars, tuples and dicts as TOML."""
    return "\n".join(_toml_table(None, tables)) + "\n"


def _toml_table(name: str | None, table: dict[str, TomlValue]) -> list[str]:
    """Returns the lines of a TOML table (the root if name is None) and subtables."""
    lines = [] if name is None else ["", f"[{name}]"]
    lines += [
        f"{k} = {_toml_value(v)}"
        for k, v in table.items()
        if not isinstance(v, dict) and v is not None
    ]
    for k, v in table.items():
        if isinstance(v, dict):
            lines += _toml_table(k if name is None else f"{name}.{k}", v)
    return lines


def _toml_value(value: TomlValue) -> str:
    """Formats one TOML scalar or array."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, tuple | list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    if isinstance(value, float):
        return repr(value)
    return str(value)


@dataclass
class ModelState:
    """A model bound to a tissue, with the hidden state Tissue's reactions carry.

    The tissue is modified in place. `fixed` marks the vertex components the
    boundary holds still.
    """

    model: MeristemModel
    tissue: Tissue
    rand_state: IntArray = field(default_factory=make_rand_state)
    pressure_factor: float = 0.0
    rest_positions: FloatArray = field(init=False)
    mesh: TriangleMesh = field(init=False, repr=False)
    fixed: BoolArray = field(init=False, repr=False)
    _material: Material = field(init=False, repr=False)
    _fibres: FibreKernel | None = field(init=False, repr=False)
    _microtubules: MicrotubuleKernel | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initiates the reactions in Tissue's model-file order."""
        model, tissue = self.model, self.tissue
        membrane = model.membrane
        if model.fibres is not None:
            feedback.initiate_fibres(model.fibres, membrane, tissue.cells)
        tissue.triangulation = initiate_triangulation(tissue)
        if model.microtubules is not None:
            feedback.initiate_microtubules(tissue.cells)
        self.rest_positions = tissue.vertices.copy()
        self.mesh = build_triangle_mesh(tissue)
        self.fixed = np.zeros(tissue.vertices.shape, dtype=bool)
        edge = np.flatnonzero(tissue.topology.boundary)
        axes = np.array(model.boundary.fixed_axes, dtype=np.int64)
        self.fixed[np.ix_(edge, axes)] = True
        self._material = membrane.to_kernel()
        self._fibres = (
            None if model.fibres is None else model.fibres.to_kernel(membrane)
        )
        self._microtubules = (
            None if model.microtubules is None else model.microtubules.to_kernel()
        )

    @property
    def triangulation(self) -> CentreTriangulation:
        """The tissue's centre triangulation (always set after initiation)."""
        if self.tissue.triangulation is None:
            raise ValueError("Tissue has no centre triangulation")
        return self.tissue.triangulation

    def compute_forces(
        self,
        cells: CellState,
        centres: FloatArray,
        vertices: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        """Returns the (vertex, centre) forces, which are the position derivatives.

        Writes the diagnostics (angles, volume change, velocity) into `cells` and
        zeroes the fixed boundary components.
        """
        vertex_forces = np.zeros_like(vertices)
        centre_forces = np.zeros_like(centres)
        forces.compute_trbs_forces(
            self.mesh,
            centres,
            vertices,
            cells,
            self._material,
            self.rand_state,
            vertex_forces,
            centre_forces,
        )
        forces.compute_pressure_forces(
            self.mesh,
            centres,
            vertices,
            self.pressure_factor * self.model.pressure.pressure,
            vertex_forces,
            centre_forces,
        )
        diagnostics.update_angle_diagnostics(cells)
        diagnostics.update_volume_change(
            cells.volume_change, vertices, self.rest_positions, vertex_forces
        )
        vertex_forces[self.fixed] = 0.0
        if self.model.track_velocity:
            diagnostics.update_cell_velocity(cells, self.mesh, vertex_forces)
        return vertex_forces, centre_forces

    def update(self, h: float) -> None:
        """Applies the per-step updates of length h in Tissue's order."""
        model, tissue = self.model, self.tissue
        cells = tissue.cells
        centres = self.triangulation.centres
        if self._fibres is not None:
            feedback.update_fibres(cells, h, self._fibres)
        stress.update_cell_stress(
            self.mesh,
            centres,
            tissue.vertices,
            cells,
            self._material,
            self.rand_state,
        )
        if self.pressure_factor < 1.0:
            self.pressure_factor += h / model.pressure.ramp_time
        if self.pressure_factor > 1.0:
            self.pressure_factor = 1.0
        if self._microtubules is not None:
            feedback.update_microtubules(cells, h, self._microtubules)
        if model.growth is not None:
            growth.update_growth(
                self.mesh,
                centres,
                tissue.vertices,
                cells,
                h,
                model.growth.rate,
                model.growth.threshold,
            )
