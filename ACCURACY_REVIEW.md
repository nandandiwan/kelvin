# Kelvin: approach and accuracy review

Review date: 2026-09-15. Reviewed checkout: `ce0f1e265833996eeed19b6c80cbf2b6d59a83e7`, including the untracked GDS import adapter and its documentation. This is a review of the current implementation, not the earlier version discussed in chat. No solver, source-model, or geometry code was changed for this review.

Subsequent update (2026-09-16): the transistor-instance power duplication in
finding 3 has been repaired and independently tested. Its original diagnosis
is retained below as history. See [SRAM_INSTANCE_MAPPING.md](SRAM_INSTANCE_MAPPING.md)
for the repair and [SRAM_SIMULATION_OVERVIEW.md](SRAM_SIMULATION_OVERVIEW.md)
for current workflow status; other historical findings are not implicitly resolved.

## Assessment

Kelvin implements a reasonable linear finite-element heat-conduction model. All **16 existing tests pass**, and independent checks support its core equation assembly, source-unit conversion, and backward-Euler time integration. They do **not** establish that the current SRAM temperature predictions are physically accurate: important errors remain in the older GDS material geometry and transistor-power mapping, and several electrical and boundary-condition assumptions need qualification. The new import path preserves polygon geometry, but its documented example currently fails on a material-name mismatch; with an explicit diagnostic override, its saved inverter mesh solves and conserves energy.

There are two distinct GDS workflows. The older `mesh/gds_build.py` path constructs simplified geometry from layout masks. The newer `read_gds.ipynb` → tagged MSH/JSON → `mesh/gds_import.py` path imports geometry built upstream. Findings about the older emitter do not automatically apply to the notebook/import path.

## What the code models

### Geometry and material fields

The synthetic example uses declarative layer, layout, material, and source specifications under `spec/`. The older layout-derived workflow reads GDS using `gdstk`, identifies channels from diffusion intersecting polysilicon, and contacts from diffusion intersecting LICON1. `gds/techmap.py` supplies vertical layer thicknesses and material identities; these are extra process assumptions, not information recovered from GDS itself. Gmsh creates tagged 2D triangles or 3D tetrahedra, then coordinates are converted from micrometres to metres before DOLFINx assembly.

The new notebook constructs actual polygonal prisms, subtracts occupied regions from the background, and meshes conformal interfaces. Its import adapter reads the resulting material manifest and tagged 3D MSH. It checks material names, source-candidate tags, agreement between mesh and manifest volume tags, complete cell tagging, and connected components touching the backside. Only selected physical backside faces receive Kelvin's cooling tag. These are useful safeguards against floating, uncooled material islands. Callers must invoke `case.assert_solver_ready()` before solving, as shown in `GDS_IMPORT.md`.

`physics/coeffs.py` assigns elementwise constant conductivity `k`, volumetric heat capacity `rho*cp`, and source density `q`. Conductivities are evaluated at the fixed reference values in the material registry. The existing `k_at(T)` function is not used to update the solve. Interfaces are perfectly bonded; no explicit thermal interface resistance or electron/phonon nonequilibrium is solved.

### Heat-generation models

| Workflow | Power supplied to the thermal model | Important assumption |
|---|---|---|
| Synthetic chip | Prescribed device powers, split 70% channel / 30% contact | The previous 97% transistor / 3% wire allocation is absent from this checkout. |
| Generic GDS / whole macro | User-specified power, or Liberty leakage plus access/idle energy times frequency; then 70% channel / 30% contact | Spatial allocation is heuristic, not extracted transistor activity. |
| Compact-model bitcell/gallery | BSIM3 DC channel power, approximately `abs(Id*Vds)`; operation-dependent scaling | Contacts are assigned zero power; DC snapshots and assumed duty/charge budgets are not measured switching waveforms. |
| Notebook/MSH import | Explicit `source_density_by_tag` in W/m³ | The documented 1 µW inverter source is a demonstration input, not extracted workload power. |

The compact READ model sets energy to `C_BL * sense_margin * VDD`, with assumed 1 fF per row and 100 mV sense margin. For its 16-row cell context this is 2.88 fJ per access. CROWBAR forces the two internal nodes to midrail and scales that snapshot using an assumed transition duration. Settled WRITE is a DC endpoint and omits the energy of the preceding switching event.

There is currently no general interconnect-current/resistance extraction and no distributed wire `I²R` calculation. Warm wires can simply be conducting heat away from transistor sources. The importer permits explicitly supplied wire heating when intentionally enabled, but does not calculate it.

### Thermal equations and boundary conditions

The steady equation is

```text
-div(k grad T) = q.
```

The transient equation is

```text
rho*cp * dT/dt - div(k grad T) = q(t).
```

Temperature uses continuous first-order finite elements. The steady weak form includes `k*grad(T)·grad(v)` and Robin boundary terms. Transients use implicit backward Euler:

```text
rho*cp * (T[n+1] - T[n])/dt - div(k grad T[n+1]) = q[n+1].
```

PETSc solves the linear systems with conjugate gradients and hypre BoomerAMG, requesting relative tolerance `1e-10`. A small linear-system residual is not a bound on geometry, material, spatial-discretization, or time-discretization error.

Default cooling is `-k grad(T)·n = h*(T - T_ambient)` at the backside, with `T_ambient=300 K` and `h=20,000 W/m²/K`. Other faces are insulated unless top cooling, an approximate lateral `h=k/r` closure, or coarse-model Dirichlet data are selected. The coefficient `h` represents an assumed package cooling resistance; it is not derived from this layout.

In 3D, source density is `P/volume`. A 2D row solve replaces each source's unmodeled depth by an explicit representative depth, using `q=P/(width*thickness*depth)`. Its raw source integral has units W/m; multiplying by the same depth recovers watts. This preserves the selected row power but loses out-of-plane spreading and localized 3D self-heating.

The coarse whole-die workflow bins geometry and power into tiles, homogenizes layer properties, and solves a structured 3D mesh. Fine submodels interpolate that global temperature onto lateral cut boundaries. This is one-way submodeling: the detailed window does not correct the outer solution.

## Checks performed in this review

The complete existing suite passed: **16 passed in 680.76 s**. This covers specification consistency, an analytic steady slab, synthetic-row source power/energy balance, generic GDS source allocation, a GDS cross-section mesh, and single-/dual-sink GDS energy conservation. It does not cover the newly added importer or the asymmetric SPICE-to-channel mapping. The initial non-GDS subset also passed (12 tests).

An independent 3D transient check used a uniform slab with a Robin bottom and insulated other faces. The initial 1 K temperature-rise profile was the exact mode `cos(mu*(1-z/L))`, with `mu*tan(mu)=1`; its exact decay is exponential. Parameters were `L=10 µm`, `k=10 W/m/K`, `rho*cp=10^6 J/m³/K`, and `h=k/L`, on a 2×2×40 hexahedral mesh. Over one analytic decay time:

| Timesteps | Maximum temperature error |
|---:|---:|
| 20 | 0.00899646 K |
| 40 | 0.00453758 K |
| 80 | 0.00227374 K |

Halving the timestep approximately halves the error, as expected for backward Euler. The maximum relative residual in **stored-energy change plus Robin outflow** was `1.27e-9`. Successful runs had positive PETSc convergence reasons. This verifies the core transient implementation on this problem, not the SRAM pulse assumptions.

On the real bundled `sram_sp_cell`, a separate OCC geometry check found 190 volumes and 190 matching labels, with no missing, stale, or duplicate tags. Total CAD volume was 101.1823152 µm³, equal to the footprint times stack thickness. All 16 source CAD volumes matched their declared box volumes to about `2e-13` relative error. These checks can pass even when the material footprints are wrong, as demonstrated next.

The new notebook's pinned inverter was checked separately. Its 24 pinned bundle checksums and source/geometry hashes matched; reconstructed current region records matched the saved manifest, and the positive-volume overlap check passed. Actual polygon footprints were preserved: poly 0.4689 µm², LI1 1.6457 µm², and MET1 1.3248 µm². The saved minimum mesh-quality measure `minSICN=0.08485` exceeded the notebook's 0.025 threshold. Geometry fidelity and acceptable element quality are useful evidence, but are not a temperature-convergence study.

Importing the saved inverter **failed by default** on `poly_Si`. With the explicit, in-memory override `material_by_mask={(66, 20): "PolySi"}`, the diagnostic imported 76,482 cells and 19 material tags, with one connected/backside-anchored component. A fresh steady solve with the documented 1 µW demonstration source reproduced:

| Imported inverter quantity | Fresh result |
|---|---:|
| Minimum temperature | 308.8777883 K |
| Maximum temperature | 308.8849187 K |
| Integrated source power | 0.9999999999999997 µW |
| Integrated backside outflow | 0.9999999997287813 µW |
| Relative source/outflow mismatch | 2.71e-10 |

This reproduces the numerical example after resolving the material name for that diagnostic only. No notebook, manifest, or source file was changed. The backside footprint is 5.632 µm², so `P/(h*A)=8.87784 K`: the approximately 8.88 K rise is largely imposed by the assumed boundary resistance. It is not evidence that a real inverter dissipating 1 µW reaches that temperature.

The Liberty parser reproduced 547.065 nW leakage, 14.984645 pJ/write, and 13.182923 pJ/read under the code's rail-summing convention. These are parser results, not independently measured macro energies. No fresh ngspice simulation was possible: `gds/spice_power.py:67` hardcodes a missing executable at `/home/nandan_diwan/miniforge3/envs/thermals/bin/ngspice`, and no PATH executable was available.

Historical temperature, mesh-convergence, and Oprins reproduction numbers in `VALIDATION.md` were not re-established by those checks. The prior `kelvin/out/` result directory was absent. Agreement with one reported benchmark ratio does not by itself validate other geometries, workloads, or boundary conditions.

## Accuracy findings

### 1. High priority: the new notebook/import example is not currently compatible with the material registry

The notebook emits material `poly_Si` for polysilicon (notebook JSON line 1086 and the saved inverter manifest's tag 14), while `spec/materials.py:113` registers only `PolySi`. `mesh/gds_import.py:100` correctly rejects the unknown material. Consequently, the documented `GDS_IMPORT.md` example fails before solving in this checkout. Align the producer's material key and the registry, or explicitly document/pass the supported override. The fresh solve above demonstrates that this compatibility error, rather than an unanchored mesh or thermal assembly failure, blocks this particular example.

The new notebook also makes different physical approximations from the older SRAM path: its working inverter substrate extends to z = -2 µm; the channel source occupies the full 0.12 µm diffusion depth, rather than the older 0.01 µm source layer; and the complement of active diffusion is filled with bulk silicon rather than explicitly modeled STI. Gate oxide and detailed interface resistance are not resolved. These choices need justification or sensitivity checks before interpreting local transistor temperature contrast. Polygon fidelity does not validate these vertical/material assumptions.

### 2. High priority: the older 3D emitter changes material footprints substantially

At `mesh/gds_volume.py:40`, clipped/merged polygons are replaced by their axis-aligned bounding rectangles. Connected nonrectangular diffusion and metal shapes therefore acquire material in regions that should be voids or dielectric. Measuring the **union** of emitted rectangles, not double-counting overlaps, gives:

| Bitcell layer | Exact GDS area (µm²) | Emitted footprint (µm²) | Excess |
|---|---:|---:|---:|
| DIFF | 0.489900 | 0.991700 | 102.4% |
| LI1 | 0.894900 | 1.424200 | 59.1% |
| MET1 | 0.747225 | 1.169200 | 56.5% |
| MET2 | 0.753400 | 0.828000 | 9.9% |

This changes thermal paths and potentially connectivity. Refining the mesh converges to the altered geometry; neither power conservation nor total-volume conservation detects the error. Preserve polygonal footprints or use an exact nonoverlapping rectangle decomposition. These percentages measure geometry error, **not** a demonstrated percentage error in temperature. They apply to the older emitter, not automatically to the notebook-generated MSH.

### 3. Resolved 2026-09-16: asymmetric transistor powers were duplicated by device class

**Update:** SRAM cases now use the original-mask, netlist-assisted per-instance
map in `gds/bitcell_mapping.py`. Every X0–X7 power is preserved separately,
including through mirrored array placement. New per-device FEM audits compare
against the original electrical dictionary, not just the already-mapped boxes.
See [SRAM_INSTANCE_MAPPING.md](SRAM_INSTANCE_MAPPING.md) for evidence, tests,
the cell-specific mapping contract and remaining limitations. The paragraphs
below describe the original finding; old cached asymmetric results remain stale.

The earlier `cases/run_bitcell_gallery.py` and `cases/run_bitcell_transient.py` kept only the powers for X0, X1, X5, and X3, then assigned each to both geometrical members of its class. This was valid only when the pair powers actually agreed, as intended for the symmetric crowbar snapshot. READ, HOLD, and settled WRITE are asymmetric. For example, the principal stored-1 READ path includes X0 and X7, but that mapping discarded X7 and duplicated X1 instead.

This could alter both hotspot location and total power after upstream normalization. The original `post/budget.py::power_balance` compares the assembled source against the already-mapped source boxes, so its agreement alone does not prove SPICE-to-device correspondence. The new `verify_device_source_powers` check additionally compares the original electrical totals and per-device powers with the final FEM sources.

The generic GDS allocation has a separate orientation error: `gds/sources.py:81` weights by Y extent as if it were device width. In this SRAM bitcell, X is width and Y is gate length, also recognized by `cases/run_bitcell_compact.py`. Equal-length 0.21 µm latch and 0.14 µm access devices therefore receive equal heuristic powers despite the documented width-weighting rule.

### 4. High priority for validation claims: leakage agreement likely includes numerical conductance

The supplied cell instances specify W/L but no diffusion areas/perimeters; the wrappers default AD/AS/PD/PS to zero. The version-matched [ngspice BSIM3v3.2 implementation](https://raw.githubusercontent.com/ngspice/ngspice/master/src/spicelib/devices/bsim3v32/b3v32ld.c) uses a fallback junction saturation current for that case and includes numerical `GMIN*V` in junction currents. [Ngspice documents default GMIN as 1e-12 S](https://nmg.gitlab.io/ngspice-manual/analysesandoutputcontrol_batchmode/simulatorvariables__options/dcsolutionoptions.html).

Using the default conductance and fallback `Is=1e-14 A`, seven reverse-biased 1.8 V junctions give

```text
7 * (GMIN * 1.8² + Is * 1.8) = 22.806 pW.
```

That almost exactly explains the documented HOLD difference `23.23 - 0.42 = 22.81 pW`. Six junctions give 19.548 pW, likewise matching settled WRITE's documented missing power. This is a **source-backed quantitative inference**, not a rerun of the unavailable historical ngspice binary. A GMIN/tolerance sweep with extracted diffusion geometry is needed before treating the claimed leakage agreement with Liberty as physical validation. The active thermal runs omit these junction terms, so this primarily undermines the validation claim rather than directly changing those active temperatures.

### 5. Coarse material fields do not consistently represent the fine model

`gds/upscale.py:65` converts raw polygons to boxes and its rasterizer sums overlaps before clamping at one. For a whole-bitcell tile, LI1 becomes 100% fill (1.896 µm²) although its actual union occupies only 0.8949 µm². MCON area is overstated by 25% even though its fine footprints are rectangular.

At `gds/upscale.py:133`, repeated `field*(1-f)+material*f` also fails to implement the stated geometrical overwrite priority. A simple disjoint 50% PolySi / 50% W tile returns 92.85 W/m/K instead of the correct vertical parallel-path mixture of 100 W/m/K. Heat-capacity mixing has the same issue.

Fine channel sources additionally use `Si_channel` conductivity 20 W/m/K, while coarse channel material retains `Si_SD_doped` at 70 W/m/K. Scalar coarse conductivity is applied in every direction, even though directional metal connectivity generally requires anisotropic effective properties. Fix the geometry fractions and material consistency before attributing coarse/fine differences solely to mesh resolution.

### 6. Pulse definitions and boundary interpretations are not interchangeable

The steady crowbar model uses the Liberty clock-high minimum, 0.239519 ns, as a transition-duration proxy. `run_bitcell_transient.py:181` instead applies its instantaneous crowbar power for 3.86681 ns: **16.14 times the energy** at the same power. The timing file belongs to a different-sized macro from the bitcell extraction, and neither clock timing constraint establishes the actual crowbar current waveform. Label these as distinct assumed stimuli or derive one common per-device transient energy model.

Fixed-step array pulse sampling introduces additional energy error. The default 50 ps / three-cycle settings in `run_sram_array_transient.py` supply 0.750 ns of on-time rather than 0.718557 ns, about 4.38% extra. Split timesteps at switching edges or use step-averaged sources and check temporal convergence.

`VALIDATION.md:44` describes an insulated one-cell result as one active row with other rows idle. The modeled domain contains no idle neighboring rows. Zero lateral flux describes insulated/symmetry boundaries; it is not a general periodic boundary constraint and does not establish that mixed-activity interpretation.

The lateral `h=k/r` condition in `physics/bcs.py:60` is a far-field approximation based on a spherical `1/r` temperature rise. It is not exact on all planar faces of a layered rectangular transient domain. The default tiled array places active edge cells next to lateral cuts, so padding/domain/boundary sensitivity must be established for that actual workload.

The ~28 ms silicon diffusion explanation in `solve/transient.py` is also numerically inconsistent with the current properties: `L²/(k/(rho*cp))` for 50 µm Si is about **28 µs**, while its approximate lumped backside-cooling time `rho*cp*L/h` is **4.15 ms**, before other layers add heat capacity.

### 7. Electrical energy accounting needs signed source power

`gds/spice_power.py:348` sums `abs(V*I)` over ideal voltage sources. Conservation requires signed net delivered power, conventionally `-sum(V*I)` with SPICE source-current orientation. A direct mocked call returns 2.7 µW when one source delivers 1.8 µW and another absorbs 0.9 µW; correct net delivery is 0.9 µW. The defect matters whenever a forced internal-node source absorbs energy. It does not prove every historical named-point ratio is wrong.

The separate read/write transient decks also integrate only VDD energy, omitting the independently driven bitline/wordline sources and changes in stored capacitor energy. Their floating read capacitors receive `.nodeset` hints rather than an explicit precharge circuit. Those results should not be labeled complete operation energies without correcting the testbench and energy accounting.

### 8. Failure detection, cached results, and import integration need strengthening

Neither steady nor transient solve enforces a positive PETSc convergence reason. A deliberately restricted transient solve with `max_it=0` returned normally with reason `-3` and temperatures 299.13–302.58 K despite positive heating from 300 K. This demonstrates missing failure handling, not failure of the normally converged checks above. Enable `ksp_error_if_not_converged` or check the termination reason explicitly.

`cases/render_sram_heatmap.py:146` reuses its saved field without validating changed frequency, activity, stack, layout, or material/source settings. `run_gds_submodel.py` assumes fixed 1 GHz / 50% / frontside settings without checking the coarse cache. A changed input can therefore leave an old temperature picture or give the fine solve incompatible boundary data. Cache a complete configuration fingerprint and reject mismatches.

`mesh/gds_import.py` sources contain a density and name, whereas `post/budget.py` expects box dimensions and `power_uw`. Calling that helper on the imported inverter reproduced `AttributeError: ImportedHeatSource ... power_uw`. Imported cases need a volume-integrated source budget with an independently declared intended power; the fresh inverter check above assembled both integrals directly. The importer also does not remap lateral faces to Kelvin's lateral cooling tags, so its supported boundary contract should be stated explicitly.

The new `cases/visualize_gds_mesh.py` CLI is also disconnected from its implementation: even `--help` raises `ImportError` because `mesh/gds_viz.py` does not define its imported `render_gds_mesh` function. This is a visualization integration failure, separate from thermal accuracy.

## What would make the accuracy claim stronger

First align the notebook/import material contract, correct the older emitter's material footprints, and fix transistor-instance mapping; then compare electrical power with the final meshed source integrals. Extend automated checks to the new importer and its actual boundary areas. Establish the electrical GMIN sensitivity and one consistent, energy-normalized pulse definition. Finally repeat mesh, timestep, domain-size, and cooling-coefficient sweeps on the corrected workloads, retaining raw results and configuration metadata.

Material values, omitted gate oxide, perfect interfaces, fixed-temperature properties, heuristic contact power, and one-way electrical coupling remain modeling assumptions after those corrections. For absolute temperature predictions, package cooling and workload power need calibration or defensible uncertainty ranges. Current checks support Kelvin as a research/prototyping conduction solver; they do not supply a universal temperature-accuracy percentage.

## Reproduction and review scope

Run the existing suite from `kelvin/` with a DOLFINx-compatible Python environment:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/tmp/kelvin-review-deps \
XDG_CACHE_HOME=/tmp/kelvin-review-cache \
MPLCONFIGDIR=/tmp/kelvin-review-mpl \
UCX_TLS=self \
/global/home/krishnabhattaram/Neural-Network-Materials/Electrostats/env/bin/python \
  -m pytest tests -q
```

`gdstk==1.0.1` was installed only into `/tmp/kelvin-review-deps` to supply the missing dependency; the existing environment was not modified. Omit that `PYTHONPATH` once `gdstk` is available in the chosen environment. Temporary diagnostic scripts retained for this session are `/tmp/kelvin_geometry_audit_20260915.py`, `/tmp/kelvin_solver_review.py`, `/tmp/kelvin_power_review.py`, and `/tmp/kelvin_bridge_audit_20260915.py`; run them from `kelvin/` with the same environment. The geometry script accepts `--occ` for the volume/tag checks. The bridge script reads the saved pinned inverter under the workspace's `out/gds_geometry/` and applies the material-name override in memory. Temporary files are not durable project artifacts.

The review combined code inspection, existing tests, independent analytic/dimensional checks, actual GDS polygon/OCC comparisons, and controlled failure probes. It did not rerun every expensive application case, reproduce the full published BSPDN study, validate the process stack against fabrication metrology, or measure device temperatures.
