"""The meristem model and its per-call evaluation (Tissue's reaction list).

Tissue evaluates its reactions in model-file order, and several of them write cell
state as a side effect of computing derivatives. `ModelState` reproduces the order
of every retained model file:

- forces: TRBS membrane, pressure, angle diagnostics, volume diagnostic, boundary,
  vertex velocity;
- per-step updates: fibres, cell stress and strain, pressure ramp, microtubules,
  growth.
"""

import math
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from tissue_lite import diagnostics, feedback, forces, growth, stress
from tissue_lite.arrays import BoolArray, FloatArray, IntArray
from tissue_lite.feedback import (
    FibreParameters,
    FibreStiffness,
    MicrotubuleAlignment,
    MicrotubuleParameters,
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
from tissue_lite.tissue import CellState, Tissue, compute_cell_centroids
from tissue_lite.triangles import TriangleMesh, build_triangle_mesh

type TomlValue = (
    bool | int | float | str | None | tuple[TomlValue, ...] | dict[str, TomlValue]
)


@dataclass(frozen=True)
class MeristemModel:
    """Everything that defines a simulation except the mesh and the solver.

    Without `track_velocity` the cell velocity keeps its initial value, as in
    Tissue models lacking `Calculate::VertexVelocity`, so the velocity gates see a
    constant (KNOWN_ISSUES F3).
    """

    membrane: TRBSMembrane
    pressure: Pressure
    boundary: Boundary = Boundary()
    fibres: FibreStiffness | None = None
    microtubules: MicrotubuleAlignment | None = None
    growth: StrainGrowth | None = None
    track_velocity: bool = True

    @classmethod
    def from_toml(cls, path: Path) -> MeristemModel:
        """Reads a model written by `to_toml`.

        Raises:
            KeyError: If the `membrane` or `pressure` table is missing.
            TypeError: If a table has an unknown or missing key.
            ValueError: If the TOML is malformed or a parameter fails its validation.
        """
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

    `positions` (V + C, 3) holds the integrated variables, the vertices followed by the
    cell centres; `tissue.vertices` is a view of its first V rows. The tissue is
    modified in place and its `vertices` must not be reassigned. Construction also
    initiates the fibres and microtubules, writing `tissue.cells`. `rest_positions` are
    the vertices at construction and `fixed` (V, 3) marks the components the boundary
    holds still. `pressure_factor` ramps from 0 to 1 by `h / ramp_time` per accepted
    step (KNOWN_ISSUES P2). The force and stress evaluations consume `rand_state` (T1).
    """

    model: MeristemModel
    tissue: Tissue
    rand_state: IntArray = field(default_factory=make_rand_state)
    pressure_factor: float = 0.0
    positions: FloatArray = field(init=False, repr=False)
    rest_positions: FloatArray = field(init=False, repr=False)
    mesh: TriangleMesh = field(init=False, repr=False)
    fixed: BoolArray = field(init=False, repr=False)
    _material: Material = field(init=False, repr=False)
    _fibres: FibreParameters | None = field(init=False, repr=False)
    _microtubules: MicrotubuleParameters | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initiates the reactions in Tissue's model-file order."""
        model, tissue = self.model, self.tissue
        membrane = model.membrane
        if model.fibres is not None:
            feedback.initiate_fibres(model.fibres, membrane, tissue.cells)
        centres = compute_cell_centroids(tissue)
        self.mesh = build_triangle_mesh(tissue, centres)
        if model.microtubules is not None:
            feedback.initiate_microtubules(tissue.cells)
        self.positions = np.concatenate([tissue.vertices, centres])
        tissue.vertices = self.positions[: tissue.topology.num_vertices]
        self.rest_positions = tissue.vertices.copy()
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
    def centres(self) -> FloatArray:
        """The cell centres, (C, 3), a view of `positions`."""
        return self.positions[self.tissue.topology.num_vertices :]

    def compute_forces(self) -> FloatArray:
        """Evaluates `compute_forces_at` on the real state, keeping its diagnostics."""
        return self.compute_forces_at(self.tissue.cells, self.positions)

    def compute_forces_at(
        self, cells: CellState, positions: FloatArray, record_diagnostics: bool = True
    ) -> FloatArray:
        """Returns the (V + C, 3) derivative of `positions` (vertices, then centres).

        `positions` may be a trial state. Writes the diagnostics (angles, volume change,
        velocity) and, without softening, the transverse moduli into `cells`, so stage
        evaluations pass a copy of the cells (KNOWN_ISSUES S5). The diagnostics do not
        feed back into the derivative; `record_diagnostics=False` skips them where the
        cell state is discarded. Uses the current `pressure_factor`. Zeroes the fixed
        boundary components after the volume diagnostic (L2). A vanishing fibre
        direction consumes `rand_state` (T1).
        """
        num_vertices = self.tissue.topology.num_vertices
        vertices, centres = positions[:num_vertices], positions[num_vertices:]
        derivatives = np.zeros_like(positions)
        vertex_forces, centre_forces = (
            derivatives[:num_vertices],
            derivatives[num_vertices:],
        )
        forces.compute_forces(
            self.mesh,
            centres,
            vertices,
            cells,
            self._material,
            self.rand_state,
            self.pressure_factor * self.model.pressure.pressure,
            2.0 if self.model.pressure.fidelity else 3.0,
            vertex_forces,
            centre_forces,
        )
        if record_diagnostics:
            diagnostics.update_angle_diagnostics(cells)
            diagnostics.update_volume_change(
                cells.volume_change, vertices, self.rest_positions, vertex_forces
            )
        vertex_forces[self.fixed] = 0.0
        if record_diagnostics and self.model.track_velocity:
            diagnostics.update_cell_velocity(
                cells.velocity, self.mesh.offsets, self.mesh.vertex_a, vertex_forces
            )
        return derivatives

    def compute_trial_forces(self, positions: FloatArray) -> FloatArray:
        """`compute_forces_at` as a pure function of `positions`.

        The diagnostics go to a scratch copy of the cell state and the random
        fibre fallback (KNOWN_ISSUES T1) restarts from the current seed, so the
        state is left untouched.
        """
        rand_state = self.rand_state.copy()
        derivatives = self.compute_forces_at(
            self.tissue.cells.copy(), positions, record_diagnostics=False
        )
        self.rand_state[:] = rand_state
        return derivatives

    def update(self, h: float) -> None:
        """Applies the per-step updates of length h in Tissue's order.

        Evaluates at the current `positions` and writes `tissue.cells` and the mesh rest
        lengths in place. Cash–Karp passes the trial step, not the accepted one
        (KNOWN_ISSUES S2), unless `CashKarp(fidelity=False)`.
        """
        model, tissue = self.model, self.tissue
        cells = tissue.cells
        if self._fibres is not None:
            feedback.update_fibres(cells, h, self._fibres)
        stress.update_cell_stress(
            self.mesh,
            self.centres,
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
                self.centres,
                tissue.vertices,
                cells,
                h,
                model.growth.rate,
                model.growth.threshold,
                math.inf
                if model.growth.velocity_threshold is None
                else model.growth.velocity_threshold,
            )
