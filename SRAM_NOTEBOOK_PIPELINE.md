# SRAM notebook mesh → Kelvin simulation

`read_gds.ipynb` now generates the complete **single-bitcell thermal mesh**,
not just disconnected layer previews. The compact steady and general transient
SRAM drivers use this polygon-preserving backend by default. Array, gallery,
isolation and other legacy entry points have not been migrated.

## From the notebook

1. Open `../read_gds.ipynb`, restart its kernel, and run the staged meshing cells.
   The default is `kelvin/data/sram22_64x22m4w22.gds`, cell `sram_sp_cell`.
   The historical comparison section is optional.
2. The notebook writes `out/sram_notebook_mesh/` **inside Kelvin**, including
   `sram_sp_cell_material_regions.msh` and its `.json` material/source manifest.
   One operation snapshots the current region/material state, validates a
   staged mesh, and publishes the mesh files followed by their sealed manifest.
   The earlier manifest cell is preview-only; the mesh cell never seals a
   previously saved JSON. After changing region/material settings, rerun region
   construction and the mesh cell; running the manifest-preview cell is optional.
3. Use that manifest in a thermal run below. Kelvin imports the saved MSH;
   it does **not** regenerate the mesh. Electrical powers can change between runs.

The final notebook cell can launch a steady run when `RUN_KELVIN_SRAM=True`.
`KELVIN_PYTHON` selects its external DOLFINx interpreter, so a geometry-only
notebook kernel remains usable. Mesh generation itself does not require SPICE.
`GDS_CASE=generic` restores the earlier inverter/custom-layout configuration;
`GDS_REFINE=0.5` requests half the default target mesh lengths.

## Command-line use

From `kelvin/`, using its DOLFINx/Gmsh/ngspice environment:

```bash
# Same geometry backend and settings as the notebook; generate only.
python cases/build_sram_mesh.py --out-dir out/sram_notebook_mesh

# Reuse that mesh with powers averaged from a transient-SPICE READ event.
python cases/run_bitcell_compact.py --point read_1 \
  --mesh-manifest out/sram_notebook_mesh/sram_sp_cell_material_regions.json \
  --out-dir out/sram_notebook_read_steady

# Follow the actual per-transistor SPICE READ waveform, then cool down.
python cases/run_bitcell_transient.py --point read_1 --instantaneous \
  --mesh-manifest out/sram_notebook_mesh/sram_sp_cell_material_regions.json \
  --out-dir out/sram_notebook_read_transient
```

Omitting `--mesh-manifest` generates a new polygon mesh in that run's output
directory. `--mesh-method legacy --interconnect none` explicitly selects the old bounding-box
mesher. `--lateral-bc insulating` restores the former sidewall condition;
periodic x/y is the default. Use separate output directories for comparisons.
`--refine` affects new mesh generation only, not an imported mesh.
The default power model is `spice-transient`; READ/WRITE event energies drive
both the steady average and the time-resolved pulse. Use `--thermal-step-ps`
for pulse resolution and `--row-hit-rate`/`--period-ns` for averaged activity.
Old crowbar/HOLD/settled-WRITE cases require `--power-model dc-surrogate`.
See [SPICE_TRANSIENT.md](SPICE_TRANSIENT.md) for the full interface and limits.
The notebook flow now includes `--interconnect layout` by default: locally
extracted sheet/contact resistances generate SPICE Joule heat, conservatively
mapped into the same saved thermal mesh. Use `--interconnect none` for the
previous ideal-wire model. See [INTERCONNECT_HEATING.md](INTERCONNECT_HEATING.md).

On this host the tested interpreter is
`/global/home/krishnabhattaram/Neural-Network-Materials/Electrostats/env/bin/python`.
Add `$PWD/.deps/dolfinx-mpc-0.10/python` to `PYTHONPATH` when running there;
the locally installed ngspice is discovered automatically. The complete
environment setup is in [SRAM_SIMULATION_OVERVIEW.md](SRAM_SIMULATION_OVERVIEW.md#7-running-the-integrated-bitcell-model).

## What changed

`read_gds.ipynb` → shared `mesh/gds_notebook.py` → tagged MSH + manifest →
`mesh/sram.py` / `mesh/gds_import.py` → Kelvin heat solver.

- GDS polygons are unioned and extruded **without replacing them by bounding
  boxes**. Substrate, field material, dielectric fill and contact plugs form
  one conforming thermal domain.
- Each contact must overlap its upper and lower landing layers. LICON is
  classified as a whole cut; its full footprint is retained. Partial enclosure
  is accepted only within the explicit memory-core marker. See
  [SRAM_CONTACT_GEOMETRY.md](SRAM_CONTACT_GEOMETRY.md) for evidence and limitations.
- All eight source footprints are matched to the verified X0–X7 map by polygon
  equality, not physical-tag order or transistor class. Each density is
  `q_i = P_i / V_i`, using polygon area × source thickness.
- Mesh and manifest fingerprints prevent accidental stale/mismatched imports,
  including changed material/source-tag assignments. Imported source volumes
  and independently prescribed per-device powers are checked by integration.
  Mesher-produced region and file hashes are required to seal a new bundle;
  a stale on-disk manifest cannot be blindly resealed against a new mesh.
- Source audits, mesh-import reports and steady simulation summaries are saved
  with the results. Transient metadata records the exact manifest used;
  its renderer samples actual polygon prisms at the correct z coordinates.

## Thermal assumptions intentionally retained

| Input | SRAM setting |
|---|---|
| Active-top to backside | 50.3262 µm |
| Channel heat-deposition depth | 10 nm |
| Active-region background | SiO₂ |
| Top passivation | Nominal 1 µm SiN |
| Cooling | Backside Robin, 20,000 W/(m²·K), 300 K ambient |
| Lateral faces | Periodic x/y unless explicitly changed |
| Interconnect/contact heat | Layout-linked local resistor waveforms by default; `--interconnect none` disables them |

The coordinate origin changes: active top is z=0 and backside is z=−50.3262 µm.
The previous cooling/material/source assumptions are not recalibrated by this
mesh upgrade. A saved manifest retains its own settings; inspect them before
comparing a deliberately modified notebook mesh with another run.

## Verification

The current SPICE-to-thermal pipeline has been exercised on this saved mesh
for READ/WRITE pulses and steady event averages. Every transistor's deposited
energy is checked independently. See [the coupled verification results](SPICE_TRANSIENT.md#coupled-notebook-mesh-verification)
for current powers/temperatures, timestep refinement, and the remaining
nonphysical P1 undershoot/cooling caveats. The results below document the
earlier geometry integration and are historical where explicitly labeled.

Fresh CLI-generated bundle: `out/sram_notebook_integration/`.
It has 444,518 tetrahedra, 81,736 nodes, full material ownership and one
backside-anchored thermal component. Every region passed volume and minimum
`minSICN ≥ 0.025` checks. LI1 retains its six connected polygon components
and 0.8949 µm² area.

The updated notebook was also executed end to end: its generated geometry,
materials and source tags exactly matched the CLI pipeline. Its independently
saved bundle is in `out/sram_notebook_mesh/`; minimum observed `minSICN` was
0.0912. A historical raw-rectangle READ/cooldown CLI run reused that notebook mesh
and saved 39 temperature fields in `out/sram_notebook_integration/read_transient/`.
Their manifest identity and all 57 displayed polygon-prism samples were checked.
Those pre-fix pulse fields do not use the current energy model and must not be
used as accurate access predictions. The default now follows the SPICE waveform
and preserves its integrated energy in the steady average; see
[the workload limitations](SRAM_SIMULATION_OVERVIEW.md#5-other-sram-studies-in-the-repository).

The following **historical, pre-SPICE-unit-fix** results demonstrate numerical
integration only; their electrical loads and temperatures are superseded and
must be regenerated. See [SPICE_TRANSIENT.md](SPICE_TRANSIENT.md).

| Historical periodic steady case | Generated power | Peak temperature | Relative heat-balance error |
|---|---:|---:|---:|
| Duty-averaged crowbar | 1.18420285 µW | 331.458005 K | 6.36 × 10⁻¹⁰ |
| Duty-averaged READ | 0.74479998 µW | 319.806204 K | 1.08 × 10⁻⁹ |

An additional two-step transient check used eight deliberately unequal test
powers: energy errors were below 2.4 × 10⁻¹⁰ and periodic constraint residuals
were zero. These checks establish integration and conservation, **not mesh
convergence or experimentally validated SRAM temperatures**. Package cooling,
workload timing/power, gate oxide/interface resistance and material calibration
remain important accuracy limitations.

Tests cover contact classification, exact material/source geometry, importer
normalization, instance identity, saved-data integrity, periodic transients and
polygon-based temperature rendering. `tests/test_sram_notebook_pipeline.py`
uses an existing generated bundle; set `KELVIN_SRAM_TEST_MANIFEST` to test a
different notebook output without rebuilding a mesh during every unit-test run.
The current broad regression run passed 427 tests, excluding the four
pre-existing full-macro legacy GDS tests. This includes the shared SPICE event
model, time-varying source audits, steady/transient wiring and cache validation.
