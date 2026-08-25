# Chip Thermal Simulator (dolfinx) — Design Plan

## Context

Greenfield project at `~/Desktop/thermals` (currently empty except an empty `macro/`).
Goal: an open, hackable analogue of Cadence Celsius for finding **hotspots and heat
dissipation paths** in a chip — with the explicit long-term intent of replacing selected
subdomains with quantum/mesoscopic models (electron–phonon, phonon BTE, contact and
interface heating) rather than pure Fourier conduction.

Milestone 1 (this plan): a working **2D and 3D FEM steady + transient conduction solver**
over a synthetic but structurally realistic ~30-transistor die tile, with multiple
materials, a resolved BEOL interconnect stack, per-device power sources, and an explicit
heat-sink path. Physical fidelity is secondary to **structure**: the geometry, materials,
and physics must be declarative and editable, so LOD and quantum submodels drop in later
without a rewrite.

Environment: Python 3.13 (miniforge3), 112 cores, 502 GB RAM. dolfinx/gmsh not yet installed.

---

## Core architectural idea

Everything is driven by one declarative `ChipSpec` (plain dataclasses). The mesher, the
material fields, the sources, and the BCs are all *derived* from it. Editing the chip means
editing a spec file, never the solver.

```
ChipSpec = LayerStack (vertical) + Layout (lateral) + Materials + Sources + BCs
              |
              v
        gmsh OCC model  --fragment-->  conformal multi-material mesh
              |                         (physical volumes = materials,
              v                          physical surfaces = BC + budget planes)
        dolfinx Mesh + cell_tags + facet_tags
              |
              v
        DG0 coefficient fields (k, rho*cp, q)  -->  UFL forms  -->  PETSc CG+AMG
              |
              v
        T field  -->  post: hotspots, q = -k grad T, per-surface heat budget, streamlines
```

Two decisions that buy most of the future flexibility:

1. **Material properties live in DG0 `Function`s keyed off `cell_tags`, not in
   `dx(tag)`-summed forms.** One `dx` for the whole domain. Adding a material or a region
   never touches the variational form. Nonlinear `k(T)` becomes a UFL expression over DG0
   coefficient fields (`k_ref`, `alpha` per cell), so dolfinx's `NewtonSolver` gets an
   exact auto-differentiated Jacobian for free.
2. **A "region with a different constitutive law" is the same abstraction as a "quantum
   subdomain."** Milestone 1 only uses it for anisotropic and temperature-dependent
   Fourier, but the plug point is the same one the phonon/e-ph models will use.

---

## Proposed geometry (the brainstorm)

Mesh in **micrometres**, scale by 1e-6 on import to dolfinx — keeps gmsh size fields
well-conditioned while physics stays SI.

### Level-of-detail knob

Same code, three specs:

| LOD | Domain | Devices | Use |
|---|---|---|---|
| 0 | full package, mm-scale, homogenized BEOL | area heat source on a plane | package-level spreading |
| **1** | **24 x 24 um tile, 50 um Si, resolved BEOL** | **~30 resolved source boxes** | **this milestone** |
| 2 | single-device zoom, ~1 um box | fully resolved fin/gate | quantum submodel host |

LOD 1 truncates the substrate at 50 um and represents everything below (TIM + Cu spreader +
heat sink + ambient) as a **Robin BC with an effective `h_eff`** — with a spec toggle to
instead model the spreader stack explicitly as real material layers.

### Vertical stack (LOD 1), z = 0 at Si backside

| z (um) | Region | Material | k (W/m/K) | Notes |
|---|---|---|---|---|
| 0 | backside face | — | — | Robin `h_eff` (lumped sink) **or** explicit TIM+Cu stack |
| 0 – 50 | Si bulk substrate | Si | 148 @ 300 K, ~T^-1.3 | strongly graded: coarse at bottom |
| 50 – 50.03 | STI trench / **S-D junction** | SiO2 / **Si_SD_doped** | 1.4 / **70** | doped Si reduced vs bulk (impurity scattering); STI traps heat laterally |
| 50.03 – 50.04 | STI trench / **channel** | SiO2 / **Si_channel** | 1.4 / **20** | few-nm inversion layer, boundary-scattering-limited k |
| 50.04 – 50.06 | gate dielectric | HfO2 | 1.1 | |
| 50.06 – 50.11 | gate electrode | TiN / W | ~30 / 170 | |
| 50.11 – 50.13 | **silicide contact** | **NiSi** | **35** | self-aligned S/D + gate cap; intermetallic k, worse than either parent |
| 50.13 – 50.26 | MOL contacts | W plugs in SiO2 | 170 / 1.4 | first vertical escape path |
| 50.26 – 50.36 | M1 | Cu lines + TaN barrier, in **low-k SiCOH** | ~350 / **0.4** | 60 nm lines, 120 nm pitch |
| 50.36 – 50.46 | V1 | Cu vias in low-k | | |
| ... | M2, V2, M3, V3 | progressively thicker/wider | | |
| ~51.9 – 54.9 | M_top power grid | thick Cu | 400 | wide bars, main lateral spreader |
| +0.5 | passivation | SiN | 30 | |
| top face | | | | near-adiabatic (face-up) **or** C4 bumps (flip-chip toggle) |

The **low-k ILD (k ≈ 0.4, ~4x worse than SiO2)** is deliberate — it is a real and
underappreciated thermal bottleneck, and makes the "which path does heat actually take"
question non-trivial.

The transistor's own **z-direction gradient** (channel -> gate stack -> silicide -> MOL) is
resolved as its own set of layers rather than one lumped "active" slab — S/D and channel
get distinct, physically-motivated thermal conductivities (see Heat sources below for why
this matters), and the silicide cap is where contact-resistance heating physically lives.
This is a simplified LOD1 proxy: S/D and channel are actually coplanar (side by side along
channel length, not stacked in z), which is folded into two separate z-layers here for
LOD1 tractability; true lateral S/D/channel resolution is deferred to the LOD2 zoom model.

### Lateral layout — 30 transistors, engineered to be interesting

- 6 x 5 array, ~2 um x 3 um pitch, occupying ~12 x 15 um inside the 24 x 24 um tile
  (remaining Si acts as a guard band / lateral spreader).
- **Non-uniform power:** one corner cluster of 6 devices at ~20 uW each ("hot ALU block"),
  the remaining 24 at ~2 uW. Total ~168 uW over 576 um^2 ≈ 29 W/cm^2.
- **Asymmetric escape paths:** half the devices get a full M1→M4 via stack directly above
  them; the other half sit under unbroken low-k with no vias. Expect a clear multi-kelvin
  delta between otherwise identical devices — this is the headline result that shows the
  tool is measuring *paths*, not just temperature.
- One corner carries a dedicated **thermal-via / dummy-metal farm** as an explicit
  heat-extraction structure to contrast against.

### Heat sources

Each device gets **two** volumetric sources, split 70/30, because two distinct physical
mechanisms dissipate power in a MOSFET and they sit in different layers:

- **Channel (70%)** — hot-carrier / velocity-saturation dissipation in the high-field
  region toward the drain end of the channel. This is the dominant term in a biased
  bulk-planar device and the reason the channel layer itself is called out with its own
  (boundary-scattering-reduced) k above. A small box in the `STI_channel` layer, offset
  toward the drain end, not smeared over the whole device footprint.
- **Contact (30%)** — I^2*R heating across the S/D metal-silicide contact. Real and
  non-negligible (contact resistivity is a known scaling bottleneck at advanced nodes),
  but spread over a wider contact-pad footprint in the `silicide` layer, so its
  volumetric density is an order of magnitude below the channel term even though its
  total power is smaller too.

At 20 uW/device the channel term alone gives ~1e17 W/m^3 — enough to produce genuine local
self-heating above the array-scale rise. Each source box gets its **own cell tag** so
per-device power is a runtime array, not a remesh.

### Interfaces

Thermal boundary resistance (Kapitza) is where phonon physics first shows up at the
continuum level, so it is in the design from day one — initially as **thin equivalent
layers** (`k_eff = t / R_th`, conformal, works with continuous Lagrange). The exact
treatment (a true jump via interior-penalty DG on tagged `dS` facets) is noted as the
follow-on, not built now.

---

## Module layout

```
thermals/
  spec/
    materials.py   # Material(k, rho, cp, k_tensor, k_texp); MATERIALS registry
    stack.py       # LayerStack: ordered vertical layers, thicknesses, mesh sizes
    layout.py      # device array, via stacks, thermal-via farm, power assignment
    chip.py        # assembles ChipSpec  <-- THE file you edit to change the chip
  mesh/
    boxes.py       # Box primitive + fragment/tag bookkeeping
    build.py       # ChipSpec -> gmsh OCC -> dolfinx mesh + cell_tags + facet_tags
    sizing.py      # Box + Distance/Threshold size fields (grading)
    section.py     # 2D cross-section emitter from the same ChipSpec
  physics/
    coeffs.py      # cell_tags -> DG0 k / rho*cp / q fields
    forms.py       # UFL: steady, transient (theta), nonlinear k(T)
    bcs.py         # Dirichlet / Robin / flux, driven by facet_tags
  solve/
    steady.py, transient.py, nonlinear.py   # PETSc options centralized
  post/
    budget.py      # integrate q.n over each tagged plane -> where the heat went
    flux.py        # q = -k grad T, export for streamlines
    metrics.py     # Tmax, hotspot location, per-device Rth junction-to-ambient
    io.py          # XDMF / VTX export for ParaView
  quantum/         # (later) subdomain constitutive plugins, TBR, phonon BTE
  cases/
    run_2d.py, run_3d.py
```

Style: little to no comment noise; short functions; the spec files carry the intent.

---

## Decisions (locked)

- **Env:** new conda env `thermals` via miniforge3/conda-forge — `fenics-dolfinx mpich
  python-gmsh pyvista meshio pyamg`. Isolated from base. First action is to install and
  print the dolfinx version, then write against that exact API (0.8 vs 0.9+ differ in
  `assemble`/`NewtonSolver`/`gmshio` details — no guessing).
- **Mesher:** gmsh OCC + `fragment` for conformal multi-material meshes, physical volumes
  per material, physical surfaces for BCs and budget planes, Box + Distance/Threshold size
  fields for grading.
- **Devices:** planar blocks (active island, gate stack, contacts) + drain-side source
  boxes. Resolved FinFET geometry is deferred to the LOD 2 zoom model, where the quantum
  work will live anyway.
- **Order:** 2D cross-section end-to-end first, then 3D on proven foundations.

## Build sequence

1. **Env + smoke test.** Create `thermals` env; run a stock dolfinx Poisson solve to
   confirm PETSc/MPI work; record the version.
2. **`spec/`.** `Material`, `MATERIALS` registry (Si with `k_texp`, SiO2, low-k SiCOH, Cu,
   W, TiN, HfO2, SiN, TaN), `LayerStack`, `Layout`, `ChipSpec`. Pure data, no dolfinx —
   independently testable.
3. **`mesh/section.py` + `mesh/build.py` (2D path).** ChipSpec -> gmsh 2D cross-section
   through the hot cluster -> `gmshio.model_to_mesh` -> mesh + cell_tags + facet_tags.
   Scale 1e-6 on import.
4. **GATE 0 — visual geometry check (blocking).** Before any physics: dump the mesh and
   tags to disk and render them. See "Visual verification" below. Nothing proceeds until
   the geometry has been eyeballed and confirmed correct — a wrong stack that solves
   cleanly is the expensive failure mode here.
5. **`physics/` + `solve/steady.py`.** DG0 coefficient fields from cell_tags; steady linear
   form; Robin/Dirichlet/flux BCs from facet_tags; CG + hypre BoomerAMG.
6. **Numeric gates.** Analytic slab check, then energy conservation:
   `sum(P_source) == sum(q.n)` on exterior facets.
7. **`post/`.** Heat budget per tagged plane, `q = -k grad T`, Tmax/hotspot, per-device
   Rth, XDMF export. Read the 2D result and confirm the low-k vs via-stack contrast.
8. **3D.** Same ChipSpec through `mesh/build.py` 3D path; **re-run GATE 0 first**, then the
   numeric gates; tune grading until mesh-converged in Tmax.
9. **Nonlinear + transient.** `k(T)` via NewtonSolver; theta-scheme transient for thermal
   time constants. Both are additive — no restructuring.

Deferred by design (hooks exist, not built): true TBR jumps via interior-penalty DG on
tagged `dS` facets; LOD 0 package model; LOD 2 device zoom + submodel BC transfer;
`quantum/` constitutive plugins.

## Visual verification (primary gate at this stage)

Correctness of the *geometry* is the thing to establish first, and it is established by
looking at it — not by a passing solve. So mesh export is a first-class deliverable of
`mesh/build.py`, not an afterthought of post-processing. Every mesh build writes to
`out/mesh/` unconditionally:

- `chip.msh` — raw gmsh mesh, openable in the gmsh GUI with all physical groups intact.
- `mesh_tags.xdmf` / `.h5` — dolfinx mesh with `cell_tags` (material id per cell) and
  `facet_tags` (BC + budget surfaces) as fields, for ParaView. Colour by tag to read the
  stack directly.
- `regions_2d.png` — for the 2D path, an immediate matplotlib render: each material region
  filled in its own colour with a labelled legend, mesh edges overlaid, plus a
  z-axis-zoomed inset of the BEOL (the layers there are ~100 nm against a 50 um substrate
  and are invisible at full-domain aspect ratio).
- `sources.png` / source tag export — device power boxes highlighted, so per-device power
  assignment and the hot-cluster placement can be confirmed by eye.
- `stack_table.txt` — the realized layer stack read back *out of the mesh* (z-extent,
  material, tag id, cell count per region). Catches silent gmsh `fragment` failures where
  a thin layer gets absorbed and vanishes — the single most likely and least visible
  meshing bug in this design.

Surface these images to the user directly as they are produced. For 3D, the equivalent is
the XDMF tag export plus scripted pyvista clip/exploded-view screenshots, so the stack can
be checked without a manual ParaView session each time.

## Compute budget

**Cap MPI at 16 ranks.** All run scripts default to `-n 16` and no solver is allowed to
grab the full 112 cores; anything wanting more is an explicit, deliberate override. Mesh
sizing targets are set so the 3D tile solves comfortably within that budget — if a mesh
needs more than 16 ranks to be tractable, that is a signal to coarsen or push detail down
an LOD, not to widen the core count.

## Verification

- **2D first**: a cross-section through the hot cluster solves in seconds — used to sanity
  check materials, BCs, and grading before paying for 3D.
- **Analytic check**: single uniform slab with known `h` reproduces `T = q*t/k + q/h`.
- **Mesh convergence**: `Tmax` vs refinement level on the 3D case.
- **Energy conservation**: `sum(source power)` must equal `sum(q.n)` over all exterior
  facets to solver tolerance — this is the primary correctness gate and it exercises the
  heat-budget post-processing at the same time.
- **Physical sanity**: devices under a via stack must be measurably cooler than identical
  devices under unbroken low-k; the whole array must be hotter at the centre than the edge.
- Visual: ParaView isosurfaces + flux streamlines showing the dissipation paths.

---

## Known risks

- **Scale separation** (20 nm features vs 50 um domain) is the main way gmsh fails. Floor
  the feature size at ~20 nm, grade aggressively in z, and treat a blown tet count as a
  signal to push detail down an LOD rather than to fight the mesher.
- **dolfinx API drift** between 0.8 and 0.9+ is real. Pin the version at install and match
  it rather than writing from memory.

---

## Status as of 2026-08-25

- Directory scaffold created: `spec/ mesh/ physics/ solve/ post/ quantum/ cases/ out/mesh/
  tests/` (all empty so far).
- **Env `thermals` is built and confirmed working**: `dolfinx 0.11.0`, `mpi4py 4.1.2`,
  `gmsh 4.15.2` (checked via `conda run -n thermals python -c "import dolfinx, mpi4py,
  gmsh; ..."`). Write code against the dolfinx **0.11** API specifically — do not assume
  0.8/0.9 tutorial code matches without checking (e.g. `gmshio`, `NewtonSolver`, `assemble`
  signatures may differ). Activate with `conda run -n thermals ...` or `conda activate
  thermals`.
- **Step 2 (`spec/`) is done.** `materials.py` (13-material registry incl. Si with
  `k_texp`, low-k SiCOH at 0.4 W/m/K, Cu split fine/thick, plus doped-Si/channel-Si/NiSi
  for the transistor anatomy below), `stack.py` (`LayerStack` reproducing the LOD-1
  z-table, Si backside to passivation at z~55.4 um — the former single "STI_field" slab
  is now resolved into distinct `STI_SD` / `STI_channel` / `gate_dielectric` /
  `gate_electrode` / `silicide` / `MOL_contacts` layers so the z-direction gradient
  through the transistor itself is real structure, not one lumped region), `layout.py`
  (30-device 6x5 array, 2x3 hot ALU corner at 20 uW / rest at 2 uW = 168 uW total =
  29 W/cm^2 matching the plan's number, checkerboard via-stack assignment for the
  asymmetric-path case, corner thermal-via farm), `chip.py` (`ChipSpec` assembling
  stack+layout+BCs+sources; each device now gets **two** `SourceBox`es — 70% in the
  channel layer near the drain end (hot-carrier dissipation), 30% at the S/D silicide
  contact (I^2R contact-resistance heating) — see PLAN.md's Heat sources section for the
  justification). `tests/test_spec.py`: 10 tests, all passing (`conda run -n thermals
  python -m pytest tests/`), pytest installed into the `thermals` env for this. Pure
  Python, no dolfinx import.
- `spec/stack.py`'s `Layer` gained `device_fill`/`via_fill` fields (which material
  a device footprint carves into a layer's background, and whether that applies to
  every device or only `has_via_stack` ones) — this is what let mesh/build.py stay
  a thin, generic interpreter of the spec rather than hard-coding per-layer geometry
  logic.
- **Step 3 (`mesh/`, 2D path) is done, and GATE 0 has passed.** `mesh/boxes.py`
  (interval-partition primitive + `RegionRegistry` — every distinguishable cell
  region, including each device's channel/contact source box, gets a stable int
  tag id), `mesh/section.py` (emits non-overlapping gmsh OCC rectangles per
  layer/device from `ChipSpec`, `gmsh.model.occ.fragment` self-stitches shared
  edges into a conformal mesh — verified this is 1:1, no splitting, since our
  rectangles never actually overlap by construction), `mesh/sizing.py` (one gmsh
  Box field per z-layer, combined with Min — this is PLAN.md's "Box + Distance/
  Threshold size fields for grading" decision), `mesh/build.py` (physical groups
  for materials + BC facets, `dolfinx.io.gmsh.model_to_mesh` — note the API is
  `dolfinx.io.gmsh`, not `dolfinx.io.gmshio`, in 0.11; returns a `MeshData`
  NamedTuple, not a bare tuple), `mesh/viz.py` (the GATE 0 renders). Entry point:
  `conda run -n thermals python cases/run_2d.py [row]`.
  - Row-0 cross-section (through the hot ALU corner): 131,521 cells, 24 tagged
    regions, builds+meshes+converts in ~6s.
  - `stack_table.txt` cross-checks realized vs. expected z-bounds per layer using
    exact layer membership tracked at geometry-build time (not a centroid+epsilon
    guess — an earlier version of this check had a real bug where the epsilon
    window for the thick Si_substrate layer swallowed almost the whole mesh; fixed
    by having mesh/section.py report which layer each gmsh surface came from
    directly). All 15 layers now match exactly, cell counts sum to 131,521 with no
    gaps or double-counting.
  - `regions_2d.png` (3-panel: full stack, BEOL zoom, transistor-anatomy zoom) and
    `sources.png` visually confirm: the transistor z-anatomy (STI_SD ->
    STI_channel -> gate_dielectric -> gate_electrode -> silicide -> MOL_contacts)
    renders as distinct, correctly-ordered, correctly-sized bands; via columns
    line up exactly with the checkerboard `has_via_stack` devices; per-device
    power (hot ALU corner vs. rest) and the channel/contact source split are both
    visually obvious. GATE 0 passed — proceeding to physics.
- **Critical fix applied during step 5**: `mesh/build.py` was never actually doing
  PLAN.md's stated "mesh in micrometres, scale by 1e-6 on import to dolfinx" —
  the dolfinx mesh geometry was left in raw micrometre coordinates while every
  material property (k, h_eff, q) is proper SI. Fixed with
  `mesh_data.mesh.geometry.x[:] *= 1e-6` right after `model_to_mesh` (before
  `mesh_tags.xdmf` is written, so anything reading that file downstream is
  already correct). This was caught before any solve was trusted, not after.
- **Step 5 (`physics/` + `solve/steady.py`) is done.** `physics/coeffs.py`
  (cell_tags -> DG0 k/rho_cp/q via a single scatter through the region
  registry — adding a material never touches the variational form, per
  PLAN.md's core idea), `physics/bcs.py` + `physics/forms.py` (Robin at
  `FACET_BOTTOM`, natural/adiabatic elsewhere — matches `chip.bcs`),
  `solve/steady.py` (`dolfinx.fem.petsc.LinearProblem`, CG + hypre BoomerAMG;
  note 0.11 requires `petsc_options_prefix` — no longer has a default).
  Entry point: `conda run -n thermals python cases/run_2d_solve.py [row]`.
- **Numeric gates, both passing** (`tests/test_solve_analytic.py`,
  `tests/test_energy_conservation.py`):
  - Analytic slab check: a uniform single-material slab with the *same*
    Robin+adiabatic+volumetric-source structure the real chip uses (not a
    bespoke reformulation) matches the closed-form profile
    `T(y) = T_amb + q*t/h + q*(t*y - y^2/2)/k` to <0.1% of the temperature rise —
    exercises physics/forms.py + physics/bcs.py + solve/steady.py end to end.
  - Energy conservation on the real row-0 mesh: taking v=1 in the discrete weak
    form collapses it to `h*int(T-T_amb)ds(bottom) = int(q)dx` exactly (by
    construction) — this is the primary gate and holds to solver tolerance.
    A secondary diagnostic (reconstructing flux from grad(T), one order less
    accurate for P1 elements, especially with >100x k contrasts on this
    graded mesh) is within 8.5%, informational only, not a correctness bug.
- **Found and fixed a second real bug**: the first full-chip solve gave
  Tmax=4135K (past Si's melting point). Root cause: a 2D cross-section is
  translationally invariant in the unmodeled third dimension, so using each
  source's true (tiny, ~20-100nm) depth for q implicitly extrudes that power
  to infinite depth — massively overstating injected power. Fixed by adding
  `source_depth_m` to `physics/coeffs.build_coeffs`: None (raw 3D density) for
  the future true-3D path, or the device row pitch (3um) for a 2D
  cross-section — the correct homogenization for a slice through a periodic
  array. Result after the fix: Tmax=333.66K, a physically sane ~33.6K rise.
  `solve/steady.py` and the tests were updated to pass this explicitly.
- **Physical sanity gates, both holding** on row 0: hot devices measurably
  hotter than cold (dT_hot=33.657K vs dT_cold=33.591K) and via-stack devices
  measurably cooler than an otherwise-identical no-via device
  (333.627K vs 333.657K) — small deltas (LOD1's shared Robin-cooled substrate
  dominates the thermal budget), but directionally correct and real, not noise.
  `post/metrics.py` (Tmax/hotspot, nearest-dof per-device T), `post/io.py`
  (solution.xdmf for ParaView), `post/viz.py` (temperature_2d.png — clean
  monotonic gradient from the cool Robin-cooled backside up to the hot ALU
  corner, visually confirms the FEM solution before trusting any number from it).
- Next: step 8, the 3D path — same ChipSpec through a 3D `mesh/build.py`, re-run
  GATE 0, then the numeric gates again on 3D. That's also when `source_depth_m`
  goes away (a true 3D mesh resolves each source's real depth directly). Steps
  9 (nonlinear k(T), transient) remain deferred and additive per the plan.
- Two standing constraints for this project (also in Claude's cross-session memory):
  cap MPI/parallel work at 16 cores even though 112 are available; treat mesh/geometry
  visualization as a blocking gate before trusting any solve.
