"""The meristem model and its per-call evaluation (Tissue's reaction list).

Tissue evaluates its reactions in model-file order, and several of them write cell
state as a side effect of computing derivatives. `ModelState` reproduces the order
of every retained model file:

- derivatives: TRBS forces, pressure, angle diagnostics, volume diagnostic,
  boundary, vertex velocity;
- per-step updates: fibres, TRBS stress/strain, pressure ramp, microtubules, growth.
"""

import logging
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numba
import numpy as np
from numpy.typing import NDArray

from tissue_lite import feedback, growth, layout, mechanics
from tissue_lite.feedback import FibreStiffness, MicrotubuleAlignment
from tissue_lite.growth import StrainGrowth
from tissue_lite.mechanics import (
    Boundary,
    CellMechanics,
    ConcentrationSoftening,
    MaterialArgs,
    MaterialParameters,
    Pressure,
    TRBSMembrane,
    Triangles,
)
from tissue_lite.rand import make_rand_state
from tissue_lite.tissue import (
    CentreTriangulation,
    FloatArray,
    IntArray,
    Tissue,
    cell_centroids,
)

logger = logging.getLogger(__name__)

type Layout = Literal["sweep", "growth"]
type TomlValue = (
    bool | int | float | str | None | tuple[TomlValue, ...] | dict[str, TomlValue]
)


@dataclass(frozen=True)
class MeristemModel:
    """Everything that defines a simulation except the mesh and the solver.

    `layout` selects where secondary stress and strain axes are stored ("growth"
    stores them, overwriting the normal with the second stress axis, as Tissue's
    growth models do). Without `track_velocity` the velocity slot keeps its initial
    value, as in Tissue models lacking `Calculate::VertexVelocity`.
    """

    membrane: TRBSMembrane
    pressure: Pressure
    boundary: Boundary = Boundary()
    fibres: FibreStiffness | None = None
    microtubules: MicrotubuleAlignment | None = None
    growth: StrainGrowth | None = None
    track_velocity: bool = True
    layout: Layout = "sweep"

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
            layout=tables.get("layout", "sweep"),
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


def initiate_triangulation(tissue: Tissue, double: bool) -> CentreTriangulation:
    """Creates the centre triangulation from scratch, as Tissue does without one stored.

    Centres are the wall-length-weighted means of the wall midpoints; spoke rest
    lengths are the current centre–vertex distances.
    """
    topo = tissue.topology
    centres = cell_centroids(tissue)
    spoke = centres[topo.triangle_cells] - tissue.vertices[topo.cell_vertices]
    distances = np.sqrt(
        spoke[:, 0] * spoke[:, 0]
        + spoke[:, 1] * spoke[:, 1]
        + spoke[:, 2] * spoke[:, 2]
    )
    if not double:
        return CentreTriangulation(centres=centres, spokes=distances)
    return CentreTriangulation(
        centres=centres,
        spokes=np.repeat(distances, 2),
        wall_extensions=np.zeros(len(distances)),
    )


@dataclass
class ModelState:
    """A model bound to a tissue, with the hidden state Tissue's reactions carry.

    The tissue is modified in place.
    """

    model: MeristemModel
    tissue: Tissue
    rand_state: NDArray[np.int64] = field(default_factory=make_rand_state)
    pressure_factor: float = 0.0
    rest_positions: FloatArray = field(init=False)
    triangles: Triangles = field(init=False)
    _material_args: MaterialArgs = field(init=False, repr=False)
    _extensions: FloatArray = field(init=False, repr=False)  # empty if single
    _fixed: tuple[IntArray, IntArray] = field(init=False, repr=False)
    stress_tensor: FloatArray = field(init=False)
    strain_tensor: FloatArray = field(init=False)
    energy_isotropic: FloatArray = field(init=False)
    mechanics_valid: bool = False

    def __post_init__(self) -> None:
        """Initiates the reactions in Tissue's model-file order."""
        model, tissue = self.model, self.tissue
        membrane = model.membrane
        n_cells = tissue.topology.num_cells
        if tissue.cell_variables.shape[1] < layout.NORMAL_START + 3:
            raise ValueError("Tissue needs at least 35 cell variables")
        if model.fibres is not None:
            feedback.initiate_fibres(
                model.fibres,
                *model.fibres.moduli(membrane.young_matrix, membrane.young_fibre),
                tissue.cell_variables,
            )
        tissue.triangulation = initiate_triangulation(
            tissue, membrane.double_rest_lengths
        )
        if model.microtubules is not None:
            feedback.initiate_microtubules(tissue.cell_variables)
        self.rest_positions = tissue.vertices.copy()
        self.triangles = Triangles.from_tissue(tissue, membrane.double_rest_lengths)
        self._material_args = MaterialParameters.from_membrane(membrane).as_tuple()
        ext = tissue.triangulation.wall_extensions
        self._extensions = np.empty(0) if ext is None else ext
        edge = np.flatnonzero(tissue.topology.boundary)
        self._fixed = np.ix_(edge, np.array(model.boundary.fixed_axes, dtype=np.int64))
        self.stress_tensor = np.zeros((n_cells, 3, 3))
        self.strain_tensor = np.zeros((n_cells, 3, 3))
        self.energy_isotropic = np.zeros(n_cells)

    @property
    def triangulation(self) -> CentreTriangulation:
        """The tissue's centre triangulation (always set after initiation)."""
        if self.tissue.triangulation is None:
            raise ValueError("Tissue has no centre triangulation")
        return self.tissue.triangulation

    def derivatives(
        self,
        cell_vars: FloatArray,
        centres: FloatArray,
        vertices: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        """Returns (vertex, centre) derivatives; writes diagnostics into cell_vars."""
        model = self.model
        tri = self.triangles
        topo = self.tissue.topology
        vd = np.zeros_like(vertices)
        cd = np.zeros_like(centres)
        mechanics.trbs_forces(
            topo.offsets,
            tri.v2,
            tri.v3,
            tri.wall,
            tri.spoke_start,
            tri.spoke_end,
            topo.wall_vertices,
            self.tissue.wall_lengths,
            self.triangulation.spokes,
            self._extensions,
            centres,
            cell_vars,
            vertices,
            *self._material_args,
            self.rand_state,
            vd,
            cd,
        )
        mechanics.pressure_forces(
            topo.offsets,
            tri.v2,
            tri.v3,
            tri.wall,
            topo.wall_vertices,
            centres,
            vertices,
            self.pressure_factor,
            model.pressure.pressure,
            vd,
            cd,
        )
        _angle_diagnostic(
            cell_vars,
            layout.MICROTUBULE.start,
            layout.STRESS_START,
            layout.ANGLE_MT_STRESS,
        )
        _angle_diagnostic(
            cell_vars,
            layout.STRESS_START,
            layout.STRAIN_START,
            layout.ANGLE_STRESS_STRAIN,
        )
        _volume_diagnostic(cell_vars, vertices, self.rest_positions, vd)
        vd[self._fixed] = 0.0
        if model.track_velocity:
            _velocity_diagnostic(cell_vars, topo.offsets, topo.cell_vertices, vd)
        return vd, cd

    def update(self, h: float) -> None:
        """Applies the per-step updates in Tissue's order."""
        model, tissue = self.model, self.tissue
        membrane = model.membrane
        cell_vars = tissue.cell_variables
        triangulation = self.triangulation
        tri = self.triangles
        topo = tissue.topology
        if model.fibres is not None:
            fibres = model.fibres
            feedback.update_fibres(
                cell_vars,
                h,
                fibres.rate,
                feedback.COUPLING_CODES[fibres.coupling],
                fibres.k_hill,
                fibres.n_hill,
                fibres.velocity_threshold,
                *fibres.moduli(membrane.young_matrix, membrane.young_fibre),
            )
        mechanics.trbs_update(
            topo.offsets,
            tri.v2,
            tri.v3,
            tri.wall,
            tri.spoke_start,
            tri.spoke_end,
            topo.wall_vertices,
            tissue.wall_lengths,
            triangulation.spokes,
            self._extensions,
            triangulation.centres,
            cell_vars,
            tissue.vertices,
            *self._material_args,
            model.layout == "growth",
            self.rand_state,
            self.stress_tensor,
            self.strain_tensor,
            self.energy_isotropic,
        )
        self.mechanics_valid = True
        if self.pressure_factor < 1.0:
            self.pressure_factor += h / model.pressure.ramp_time
        if self.pressure_factor > 1.0:
            self.pressure_factor = 1.0
        if model.microtubules is not None:
            microtubules = model.microtubules
            feedback.update_microtubules(
                cell_vars, h, microtubules.rate, *microtubules.gate()
            )
        if model.growth is not None:
            growth.update_growth(
                topo.offsets,
                tri.v2,
                tri.v3,
                tri.wall,
                tri.spoke_start,
                tri.spoke_end,
                topo.wall_vertices,
                tissue.wall_lengths,
                triangulation.spokes,
                self._extensions,
                triangulation.centres,
                cell_vars,
                tissue.vertices,
                h,
                model.growth.rate,
                model.growth.threshold,
            )

    def cell_mechanics(self) -> CellMechanics:
        """Returns the stress and strain of the last update, with current slots.

        Raises:
            ValueError: If no update has run yet.
        """
        if not self.mechanics_valid:
            raise ValueError("No update has computed the mechanics yet")
        return build_cell_mechanics(
            cell_vars=self.tissue.cell_variables,
            stress_tensor=self.stress_tensor,
            strain_tensor=self.strain_tensor,
            energy_isotropic=self.energy_isotropic,
            membrane=self.model.membrane,
            layout_name=self.model.layout,
        )


def build_cell_mechanics(
    cell_vars: FloatArray,
    stress_tensor: FloatArray,
    strain_tensor: FloatArray,
    energy_isotropic: FloatArray,
    membrane: TRBSMembrane,
    layout_name: Layout,
) -> CellMechanics:
    """Collects the mechanics record from the cell-variable slots and tensors."""
    v = cell_vars.copy()
    growth_layout = layout_name == "growth"
    nan3 = np.full((len(v), 3), np.nan)
    nan1 = np.full(len(v), np.nan)
    young_t = (
        2 * membrane.young_matrix
        + membrane.young_fibre
        - v[:, layout.YOUNG_LONGITUDINAL]
    )
    return CellMechanics(
        stress_tensor=stress_tensor.copy(),
        strain_tensor=strain_tensor.copy(),
        normal=nan3 if growth_layout else v[:, layout.NORMAL],
        stress_direction=v[:, layout.STRESS_DIRECTION],
        stress_max=v[:, layout.STRESS_MAX],
        stress_direction_second=v[:, layout.STRESS_DIRECTION_SECOND]
        if growth_layout
        else nan3,
        stress_second=v[:, layout.STRESS_SECOND] if growth_layout else nan1,
        stress_anisotropy=v[:, layout.STRESS_ANISOTROPY],
        von_mises_stress=v[:, layout.VON_MISES_STRESS],
        strain_direction=v[:, layout.STRAIN_DIRECTION],
        strain_max=v[:, layout.STRAIN_MAX],
        strain_perpendicular=v[:, layout.STRAIN_PERPENDICULAR]
        if growth_layout
        else nan3,
        strain_second=v[:, layout.STRAIN_SECOND] if growth_layout else nan1,
        strain_anisotropy=v[:, layout.STRAIN_ANISOTROPY],
        rest_area=v[:, layout.REST_AREA],
        energy_isotropic=energy_isotropic.copy(),
        energy_anisotropic=v[:, layout.YOUNG_TRANSVERSE],
        young_transverse=young_t,
    )


@numba.njit(cache=True)
def _angle_diagnostic(
    cell_vars: FloatArray, first: int, second: int, store: int
) -> None:
    """Port of `Calculate::AngleVectors`: normalises both vectors in place."""
    for c in range(cell_vars.shape[0]):
        for start in (first, second):
            x, y, z = (
                cell_vars[c, start],
                cell_vars[c, start + 1],
                cell_vars[c, start + 2],
            )
            norm = np.sqrt(x * x + y * y + z * z)
            if norm > 0:
                for d in range(3):
                    cell_vars[c, start + d] = cell_vars[c, start + d] / norm
        dot = 0.0
        for d in range(3):
            dot += cell_vars[c, first + d] * cell_vars[c, second + d]
        cell_vars[c, store] = abs(dot)


@numba.njit(cache=True)
def _volume_diagnostic(
    cell_vars: FloatArray,
    vertices: FloatArray,
    rest_positions: FloatArray,
    vd: FloatArray,
) -> None:
    """Port of `Calculate::TissueVolumeChange` (stores into cells 1, 2, 3)."""
    change = 0.0
    total = 0.0
    for v in range(vertices.shape[0]):
        dx = vertices[v, 0] - rest_positions[v, 0]
        dy = vertices[v, 1] - rest_positions[v, 1]
        dz = vertices[v, 2] - rest_positions[v, 2]
        change += np.sqrt(dx * dx + dy * dy + dz * dz)
        total += np.sqrt(
            vd[v, 0] * vd[v, 0] + vd[v, 1] * vd[v, 1] + vd[v, 2] * vd[v, 2]
        )
    delta = change - cell_vars[1, layout.VOLUME_CHANGE]
    cell_vars[1, layout.VOLUME_CHANGE] = change
    cell_vars[2, layout.VOLUME_CHANGE] = delta
    cell_vars[3, layout.VOLUME_CHANGE] = total


@numba.njit(cache=True)
def _velocity_diagnostic(
    cell_vars: FloatArray,
    offsets: IntArray,
    cell_vertices: IntArray,
    vd: FloatArray,
) -> None:
    """Port of `Calculate::VertexVelocity`: mean vertex speed per cell."""
    for c in range(len(offsets) - 1):
        total = 0.0
        for k in range(offsets[c], offsets[c + 1]):
            v = cell_vertices[k]
            total += np.sqrt(
                vd[v, 0] * vd[v, 0] + vd[v, 1] * vd[v, 1] + vd[v, 2] * vd[v, 2]
            )
        cell_vars[c, layout.VELOCITY] = total / (offsets[c + 1] - offsets[c])
