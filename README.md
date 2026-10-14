# tissue-lite

A minimal Python reimplementation of the [Tissue](https://gitlab.com/slcu/teamHJ/tissue) vertex simulator, limited to what the meristem mechanics work uses: the flower-experiments models and the dome-shape results of Paper IV (thesis Fig. 5.4).

It reproduces Tissue e74e457 **bit for bit** on every regression case and on the full 963-cell paraboloid (x86-64, glibc 2.39; the tests allow rtol 1e-9 for other libm versions), and keeps Tissue's quirks; see [KNOWN_ISSUES.md](KNOWN_ISSUES.md). On that mesh a run is about 1.4× faster than Tissue (Numba kernels, one core).

## Scope

| Kept | Tissue reaction | tissue-lite |
|---|---|---|
| Anisotropic TRBS membrane on the centre triangulation (single and double rest lengths) | `VertexFromTRBScenterTriangulationMT` (MF=1) | `material.TRBSMembrane`, `forces.compute_trbs_forces` |
| Concentration-driven softening (e.g. auxin) | same, MF=10 | `material.ConcentrationSoftening` |
| Turgor pressure with linear ramp | `Pressure3D::CenterTriangulation::Linear` | `material.Pressure`, `forces.compute_pressure_forces` |
| Fixed boundary (all axes or some) | `VertexNoUpdateBoundary` | `material.Boundary` |
| Stress-feedback fibre stiffness | `FiberModel` | `feedback.FibreStiffness` |
| Microtubule alignment to stress | `UpdateMTDirectionEquilibrium` | `feedback.MicrotubuleAlignment` |
| Strain-driven wall growth | `CenterTriangulation::WallGrowth::StrainTRBS` | `growth.StrainGrowth` |
| Stress and strain outputs | TRBS `update` | `stress.update_cell_stress` |
| Diagnostics: velocity, angles, volume change | `Calculate::*`, `TemplateVolumeChange` | `diagnostics` |
| Adaptive Cash–Karp RK5 | `RK5Adaptive` | `solvers.cash_karp.CashKarp`, `solvers.simulate.simulate` |
| `.init` IO, VTU/PVD output (print flag 2) | `Tissue::readInit`, `VTUostream` | `legacy.init_file`, `output.VtuWriter` |

Everything else in Tissue (other reactions, division, other solvers and output formats) is dropped. Legacy `.model` files using anything outside this subset are rejected.

## Install

```bash
uv sync
```

## Use

As a drop-in for Tissue's `simulator` (same arguments, so `TISSUE_BIN=tissue-lite` works with flower-experiments' `prerunAndSim.sh`):

```bash
uv run tissue-lite model.model templ.init solver.rk5 -init_output final.init -vtk_output vtk
```

From Python:

```python
from pathlib import Path

from tissue_lite.feedback import FibreStiffness, MicrotubuleAlignment
from tissue_lite.legacy.init_file import read_init
from tissue_lite.material import Boundary, Pressure, TRBSMembrane
from tissue_lite.model import MeristemModel
from tissue_lite.solvers.simulate import simulate

model = MeristemModel(
    membrane=TRBSMembrane(young_matrix=120.0, young_fibre=120.0),
    pressure=Pressure(pressure=0.01, ramp_time=0.001),
    boundary=Boundary(fixed_axes=(2,)),
    fibres=FibreStiffness(rate=0.01, k_hill=0.4, n_hill=2.0),
    microtubules=MicrotubuleAlignment(rate=0.2),
)
result = simulate(model, read_init(Path("templ.init")), t_end=20.0, n_frames=10)
# Named per-cell arrays: stress/strain tensors, principal axes, anisotropies, moduli.
cells = result.final.cells
```

Models can be stored as TOML (`model.to_toml`, `MeristemModel.from_toml`) or read from legacy files (`legacy.model_file.read_model_file`). Per-cell state is named (`tissue.CellState`); Tissue's slot numbering lives only in `legacy.slots.SlotLayout`, used by the `.init` reader and writer and by `output.VtuWriter` (pass one as `writer=` to `simulate` for VTU frames), so files and readers such as `flower.output` work unchanged.

## Modules

| Module | Owns |
|---|---|
| `topology`, `tissue`, `triangles` | Mesh connectivity, state (`Tissue`, `CellState`), centre triangles (`TriangleMesh`) and their kinematics |
| `material`, `forces`, `stress` | Membrane, pressure and boundary parameters; force kernels; stress and strain update |
| `feedback`, `growth`, `diagnostics` | Fibre and MT feedback, wall growth, velocity/angle/volume diagnostics |
| `model` | `MeristemModel` and `ModelState` (positions, centre triangles, reaction state), which evaluates the reactions in Tissue's order |
| `solvers` | `Stepper` protocol, `CashKarp`, `simulate` |
| `legacy` | Slot numbering, `.init`, `.model` and solver files |
| `output`, `cli`, `rand` | VTU/PVD writer, command line, glibc `rand()` |

## Tests and reference data

| Case (`tests/data/cases/`) | Covers |
|---|---|
| `sweep_pre`, `sweep_120_0`, `sweep_120_120` | Prerun → main run of the Ym × Yf sweep; fibre and MT feedback |
| `sweep_rejections` | Tolerance 1e-8: rejected steps, `update(h)` |
| `iso`, `aniso` | `models/iso`, `models/aniso` (fibre flags, all-axes boundary) |
| `aniso_growth`, `iso_growth` | Double rest lengths, strain growth |
| `auxin_softening`, `auxin_softening_hill` | MF=10 as used (K = n = 0) and with K = n = 2 |
| `patch100` (`slow`) | 100-cell end-to-end run |

Each case stores Tissue's inputs, its `-init_output` state, its VTU frames and a `provenance.json` (commit, compiler, flags, command, SHA-256s); meshes are cut from flower-experiments' `templ.init` (`tests/data/meshes/provenance.json`). States are compared with rtol 1e-9, VTU files token by token within `%g` rounding.

```bash
uv run pytest            # fast suite
uv run pytest -m ""      # including slow cases
```

To regenerate the reference data (needs g++, make, the Boost headers and a flower-experiments checkout for the model files):

```bash
reference/build_reference.sh
uv run python reference/generate.py --models ../flower-experiments
```

## Development

```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest
```

pre-commit runs the same checks; CI runs them on every push and pull request.
