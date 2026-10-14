# Known issues

tissue-lite reproduces Tissue e74e457 as-is by default. Status: **Resolved** (tissue-lite's API does not carry the quirk), **Open**, **Kept** (needs a modelling decision), **Superseded** (a replacement solver avoids it), **By design**, **Won't fix**, **Not an issue**. ✔ marks items exercised by the regression data (fixing one fails the suite); the others come from code reading. Sources are `gitlab.com/slcu/teamHJ/tissue` at e74e457 (`src/…`).

## Solver (`solvers/cash_karp.py`, Tissue `rungeKutta.cc`)

| # | Issue | Tissue source | Effect | Status |
|---|---|---|---|---|
| S1 ✔ | Step size is capped at the initial step h1 after every step. | `simulate` loop, `if (h > h1_) h = h1_` | Error control can only shrink steps; t = 600 needs ≥ 120k steps. | Superseded by `QuasiStatic`; lifting the cap would let the explicit feedback (S4) take large steps |
| S2 ✔ | Per-step updates receive the trial step `h`, not the accepted `hDid`. | `rungeKutta.cc:225` | After a rejection, fibres, MTs, growth and the pressure ramp advance too far. | Fixed by `CashKarp(fidelity=False)` |
| S3 | Stage 3 writes `ak2W` instead of `ak3W`. | `rungeKutta.cc:486` | Wrong tableau for wall variables; latent, as wall derivatives are always 0. Not ported (walls are not integrated). | Won't fix: not ported |
| S4 ✔ | Feedback (fibres, MTs, growth) is explicit Euler outside the error control. | `updateReactions` | First order in time; results depend on the step history. | Kept: every solver feeds back by explicit Euler; `QuasiStatic` only decouples it from the Cash–Karp step history (N1) |
| S5 ✔ | Stage derivatives see copies of the cell variables, but the top-of-loop call writes the real state. | `derivs` vs `rkck` | Diagnostics (velocity, angles, var 20, var 25) describe the state at the start of the step. The velocity gates in `update` therefore read the velocity of the previous accepted state, one step stale. | Fixed by `CashKarp(fidelity=False)` |
| S6 ✔ | With more than two frames, t_end is printed twice (just before and after the last step); `tissue.pvd` lists only n frames. | print-time clipping | n + 1 VTU frames for n > 2. | Fixed for non-fidelity steppers (`QuasiStatic`, `Implicit`, `CashKarp(fidelity=False)`) |
| S7 | No convergence criterion. | solver file `t_end` | "Equilibrium" is assumed at a fixed time. | Fixed by `simulate(steady_tolerance=…)`, off by default |

The solver replacement (quasi-static Newton, compared against an implicit stiff integrator) is tracked as a separate issue.

## TRBS membrane (`forces.py`, `stress.py`, Tissue `mechanicalTRBS.cc`, `VertexFromTRBScenterTriangulationMT`)

| # | Issue | Tissue source | Effect | Status |
|---|---|---|---|---|
| T1 | When the MT vector in the rest frame is shorter than 1e-4, a random direction from `rand()` is used, inside `derivs`. | `:4143`, `:5047` | The right-hand side is not a function of the state; breaks RK error control. Reproducible only because glibc `rand()` is seeded with 1 (ported in `rand.py`). | Open |
| T2 | The 1e-4 threshold is applied to `adj(F)·a`, without 1/det F, and to the unnormalised in-plane MT. | `:4143` | The fallback triggers depending on deformation and on how far the MT points out of plane. | Open |
| T3 | Sliver-triangle "fix" adds the accumulated centre and vertex derivatives onto all three nodes; only angles near 0° are detected, not near 180°. | `:4507–4535` | Spurious forces on degenerate triangles. Not applied in `update`. | Open |
| T4 | The stress Jacobi eigen loop applies one extra rotation (pivot recomputed at the loop top) and loses the rotation sign; the strain loop does not. | `:5529` | Eigenvector signs and, rarely, the selected axis differ from `numpy.linalg.eigh`. | Won't fix: signs are arbitrary (the MT update re-aligns them), and another axis is picked only between eigenvalues within 1e-6, where either is principal |
| T5 | Principal stress is selected by \|value\| here but signed in the (dead) neighbour branch. | `:5606` vs `:6365` | Compressive strain can be reported as the maximum. | Kept: needs a modelling decision (signed or \|value\| ranking) |
| T6 | Update's isotropic stress uses SVK λT, μT while forces come from biquadratic springs. | `:4275` | — | Not an issue: Delingette's springs are SVK (forces agree to 1e-16), and forces and stress share `compute_anisotropic_stress` |
| T7 | Var 20 holds Y_T in `derivs` (MF=1 only) and the energy in `update` (isotropic, then overwritten by anisotropic). Printed frames call `derivs` last, so VTU var 20 is Y_T unless the model softens. | index layout | `CellState.young_transverse` mirrors the slot; `energy_isotropic` and `energy_anisotropic` keep the energies. | Open |
| T8 | Var 19 ("area ratio" in model files) is the total rest area. | `update` | Mislabelled output. | Resolved: `CellState.rest_area` |
| T9 | In the growth layout the second stress axis (31–34) overwrites the normal (32–34); the perpendicular strain (26–29) is unnormalised and its slot 29 holds e2. | index layout | Normals are lost in growth-layout files; `CellState.normal` keeps them and `SlotLayout(second_axes=True)` reproduces the overwrite on output. | Resolved: `CellState.normal`; the overwrite is kept on output |
| T10 | Var 21 ("MT stress") is the von Mises stress √(s1² + s2² − s1 s2). | `update` | Mislabelled output. | Resolved: `CellState.von_mises_stress` |
| T11 | Header documents p10 as "unused"; it selects double rest lengths (p10 == 1). | constructor | Misleading model files. tissue-lite always keeps both; the legacy reader checks p10 against the Initiate flag and rejects growth without it. | Resolved: the reader checks p10 |

## Concentration softening, MF=10 (`ConcentrationSoftening`)

| # | Issue | Tissue source | Effect | Status |
|---|---|---|---|---|
| C1 ✔ | K and n are read from p9 and p10, not from the "hill parameters" p11 and p12. | `:3520` | `celly/test_auxin_decay` sets p9 = p10 = 0, so K = 0, n = 0 and the factor is 1 − 0^0/(0^0 + c^0) = 1/2 everywhere: both moduli halved, independent of auxin. The intended (0.005, 2) are ignored. | Resolved in the API (`ConcentrationSoftening` takes K and n); the reader keeps p9, p10 and warns when p11, p12 differ |
| C2 | p9 also selects the TETA branch when 1, and p10 == 1 also means double rest lengths. | constructor | K = 1 or n = 1 cannot be expressed without side effects (the legacy reader rejects p9 = 1). | Resolved: the reader rejects p9 = 1 |
| C3 | `derivs` does not write Y_T (var 20) for MF=10. | `:3540` | Var 20 keeps the energy from the last update. | Open |

## Pressure (`Pressure`, Tissue `mechanical.cc`)

| # | Issue | Tissue source | Effect | Status |
|---|---|---|---|---|
| P1 | Each triangle node gets P·A/2; the consistent share is P·A/3. | `mechanical.cc:2454` | Effective turgor is 1.5 P; shapes unchanged, fitted moduli 1.5× off. | Open |
| P2 ✔ | The "linear ramp-up" parameter is a ramp *time*: the factor grows by h/p2 per step. | `update` | p2 = 0.001 reaches full pressure on the first step (h = 5e-3), i.e. no ramp. | Resolved: `Pressure.ramp_time` |

## Feedback (`feedback.py`, Tissue `fiberModel.cc`, `directionUpdate.cc`)

| # | Issue | Tissue source | Effect | Status |
|---|---|---|---|---|
| F1 ✔ | FiberModel flag 2 (`hill_direct`, used by `models/aniso`) sets Y_L = ½(1 + h(a))·Y_f, dropping Y_m and the velocity gate; its initialisation includes Y_m. | `fiberModel.cc:200–205` | With Y_m = 50, Y_f = 1000 the material is never isotropic and the MT direction is the softer one for a < 0.92. | Resolved: `coupling="hill"` is the gated form with Y_m; `hill_direct` reproduces Tissue |
| F2 ✔ | FiberModel has its own Y_m, Y_f, which may differ from the membrane's. | model files | `models/iso-growth` uses (100, 0) in FiberModel and (50, 1000) in TRBS. The legacy reader warns and keeps both. | Resolved: the moduli default to the membrane's |
| F3 | `models/*` lack `Calculate::VertexVelocity`, so var 22 is never written and every velocity gate (300) is inert. | model files | MT and fibre feedback act during transients. | Resolved: `MeristemModel.track_velocity`, True by default |
| F4 | FiberModel returns early when its rate is 0, so `initialise` still runs but nothing updates. | `fiberModel.cc` | Expected, but initialisation alone can change Y_L. | By design |

## Growth (`growth.py`, Tissue `growth.cc`, `CenterTriangulation::WallGrowth::StrainTRBS`)

| # | Issue | Tissue source | Effect | Status |
|---|---|---|---|---|
| G1 ✔ | The equilibrium gate is commented out (`if(true)`); the velocity threshold is unused. | `growth.cc:866–870` | `*-growth` models grow every step, away from equilibrium. | Open |
| G2 ✔ | Hard-coded inputs: var 3 as the strain along the growth axis (never written by any retained reaction) and var 7, the maximal **stress**, compared with a strain threshold. | `growth.cc` | Growth is driven by stress against a strain threshold; the var 3 branch depends on the init file. | Kept: needs a modelling decision |
| G3 ✔ | Growth counts are written to var 30. | `growth.cc` | Hidden slot use. | Resolved: `CellState.growth_count` |

## State layout and model files

| # | Issue | Effect | Status |
|---|---|---|---|
| L1 | All state lives in magic cell-variable slots, some hard-coded (3, 7, 22, 25, 30) and some from model files. | The legacy reader accepts only the layouts used by the project (`legacy/slots.py`) and rejects others. | Resolved: `CellState`, `SlotLayout` |
| L2 | `TemplateVolumeChange` stores into cells 1–3 of var 25 (cells, not variables), with the total derivative taken before boundary zeroing. | Needs at least four cells; mixes per-tissue scalars into per-cell data. | Resolved: `CellState.volume_change` |
| L3 | `StaticDirection` / `ParallellDirection` are no-ops without division. | Ignored by the reader. | Resolved: the reader drops them |
| L4 | Print flags other than 2 (VTU) are not supported; `models/*` use flag 6 (PLY). | The CLI warns and writes VTU. | Won't fix |

## Replacement solvers (`solvers/quasi_static.py`, `solvers/implicit.py`)

Both apply one feedback update of length `step` (default 0.1) and then a mechanical solve: `QuasiStatic` moves the positions to where the free forces vanish (pseudo-transient Newton, finite-difference Jacobian coloured per `solvers/sparsity.py`), `Implicit` integrates the overdamped dynamics over the window with scipy's Radau or BDF. They differ from Tissue's Cash–Karp as follows; the regression data exercises none of it.

| # | Difference | Effect | Status |
|---|---|---|---|
| N1 | Feedback (fibres, microtubules, growth, pressure ramp) advances once per `step`, before the mechanics of the step, instead of once per Cash–Karp step (≤ 5e-3) after it. | Explicit Euler on the feedback with a larger step: `rate * step` must stay well below 1 and growth per step scales with `step`. The end state of `aniso_growth` and `iso_growth` differs from the `step = 0.02` result by 46 % and 57 % of the movement at `step = 1`, 5 % and 2 % at the default 0.1, 1 % and 0.6 % at 0.05; `aniso` 0.2 % at 0.1. The fixed point is unchanged. | Open |
| N2 | A boundary that leaves rigid motions free (`fixed_axes` shorter than three) lets the pressure on its non-planar edge push the tissue rigidly: translation along each free axis and rotation about a single fixed one. Cash–Karp and `Implicit` integrate the drift (sweep: 1e-3 per time unit, `auxin_softening`: 0.9 per time unit); `QuasiStatic` discards it and holds the tissue in place. | Positions differ by a rigid motion; shapes and stresses do not, except that the fibres are fixed in space, so a rotation drift re-orients the tissue against them. | By design |
| N3 | Trial evaluations are pure functions of the positions (`ModelState.compute_trial_forces`): the random fibre fallback (T1) restarts from the current seed on every call and the diagnostics go to a scratch copy. | A cell whose microtubule axis vanishes gets the same pseudo-random fibre direction in every evaluation of a step; the real stream advances only in the frame loop's own evaluation. | By design |
| N4 | `QuasiStatic` skips the mechanical transient: every printed frame is an equilibrium of the current feedback state, and the velocity diagnostic (var 22) is the residual force, i.e. the drift of N2 or 0. | Velocity gates (300) are always open, as they are in practice in Tissue (F3). Shapes differ from Tissue's where its feedback acted during the transient (issue #2). The inflation from a flat rest state is a snap-through that costs the first step 60–200 Newton iterations; later steps take 1–5. | By design |
| N5 | The Jacobian pattern ignores Tissue's sliver branch (T3). | More Newton iterations on a sliver, never a wrong equilibrium: the residual is exact. | Open (follows T3) |
| N6 | S1 and S2 do not apply and S4 remains as N1; the frame loop is shared, so S6 and S7 applied too. | Exactly `n_frames` frames and no sliver step (S6 fixed); runs to t_end unless `steady_tolerance` is set (S7). | Fixed |

## tissue-lite limitations (not Tissue quirks)

| # | Issue | Status |
|---|---|---|
| X1 | Results are bit-identical to Tissue on the reference platform (x86-64, glibc 2.39); other libm versions may differ in the last bits of `pow`, `acos`, `cos` and `sin`, hence the tolerances in the tests. | Won't fix |
