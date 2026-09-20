# Kelvin SRAM pipeline: code-review guide and proposed PR series

Reviewed working-tree snapshot: **2026-09-17**.

This document explains the implementation that currently exists, then divides it into **13 proposed, dependency-ordered pull requests**. Each numbered section is both a conceptual explanation and a PR review packet. These are proposed PRs, not PRs already opened or commits already created. No simulation implementation is changed by this document.

**Publication decision (2026-09-20):** the current updates are being prepared as
**one draft PR with 13 review sections**, not 13 separate GitHub PRs. The numbered
packets and dependency graph below remain the conceptual review guide; proposed
branch names are retained only as a possible future split. See
[SRAM_REVIEW_PR.md](SRAM_REVIEW_PR.md) for the single-PR scope and checklist.
F01–F06 remain outstanding accuracy work, not completed changes in this PR.
The repository-local notebook is included for standalone-clone use. Legacy array
examples added upstream are preserved and adapted to per-instance source IDs;
this does not migrate their electrical or geometry models to the canonical flow.

The scope is the SRAM simulation. Shared Kelvin infrastructure is included only where the SRAM path calls it. Synthetic-device studies, Oprins benchmarks, backside-power-delivery experiments, and unrelated material experiments are not part of this review series.

## 1. Scope, terminology, and reading order

### 1.1 The primary pipeline being reviewed

The current integrated case is the `sram_sp_cell` bitcell selected from [data/sram22_64x22m4w22.gds](data/sram22_64x22m4w22.gds), using the bundled SKY130 compact models, local layout-linked resistor network, notebook-derived conforming mesh, and Kelvin finite-element heat solver.

It is a **single-bitcell electrical/thermal model with assumed external loading and periodic lateral thermal boundaries**. It is not an extracted electrical simulation of the entire 64×22 SRAM macro. The macro name describes 64 logical words of 22 bits; the electrical loading constants identify 16 physical rows and 88 columns for this organization. Do not substitute 64 for the physical row count when reviewing bitline loading.

The primary supported routes are:

| Route | Entry point | Actual meaning |
|---|---|---|
| Mesh only | [cases/build_sram_mesh.py](cases/build_sram_mesh.py), `main()` | Generate the same polygon-based SRAM mesh used by the notebook; no SPICE or thermal solve. |
| Electrical characterization | [cases/characterize_spice_access.py](cases/characterize_spice_access.py), `main()` | Run READ/WRITE in either initial state; optionally refine SPICE timestep; no thermal solve. |
| Steady temperature | [cases/run_bitcell_compact.py](cases/run_bitcell_compact.py), `run()` / `main()` | Integrate SPICE event energies, convert them to average powers, then solve a steady heat equation. |
| Single-access thermal transient | [cases/run_bitcell_transient.py](cases/run_bitcell_transient.py), `main()` with `--instantaneous` | Transfer the time-resolved SPICE powers conservatively, then apply zero-source cooldown. |
| Sustained-average thermal transient | Same driver, without `--instantaneous` | Turn on the event-averaged powers for a prescribed duration, then turn them off. This is not a sequence of individually resolved electrical accesses. |
| Notebook | [read_gds.ipynb](read_gds.ipynb), SRAM configuration and final launch cell | Construct and inspect geometry, publish a sealed mesh bundle, optionally launch the steady CLI using that mesh. |

The steady route is the main route for the current steady-state study. The transient route remains in this document because it shares the electrical, geometry, and source-transfer implementation and is important for reviewing conservation.

**Important defaults:** the thermal CLIs and electrical characterization CLI select layout interconnects. Lower-level `run_access_transient()` and `build_spice_workload()` default to `interconnect="none"` for compatibility; callers must explicitly request `"layout"`. A notebook mesh is the default thermal geometry. These are different defaults at different API layers, not interchangeable assumptions.

### 1.2 Suggested reading order

1. Read the end-to-end diagram and contracts below.
2. Follow PRs 01–05 to understand where the electrical heat comes from.
3. Follow PRs 06–08 to understand where the thermal materials and mesh come from.
4. Read PR 09 carefully: it is the interface between electrical watts and spatial heat density.
5. Read PRs 10–11 for the mathematical thermal problem and solver.
6. Read PRs 12–13 for orchestration, saved evidence, and interpretation of figures.
7. Review the separate accuracy backlog before approving any claim of calibrated SRAM temperature.

### 1.3 The two branches and their join

```text
GDS + original bitcell netlist + verified X0–X7 identity
                       |
           +-----------+------------------------+
           |                                    |
    ELECTRICAL BRANCH                     GEOMETRY BRANCH
    local sheet/contact R                 physical material/source regions
           |                                    |
    compact models + access deck          conforming Gmsh tetrahedra
           |                                    |
    ngspice waveform                      sealed MSH + region manifest
           |                                    |
    completion/state validation           import, units, tags, materials
           |                                    |
    per-instance and per-resistor P(t)           |
           |                                    |
    event energy / workload averaging           |
           +----------------+-------------------+
                            |
                  conservative spatial projection
                  q(x,t) or average q(x), W/m³
                            |
                  materials + boundary conditions
                            |
                 +----------+-----------+
                 |                      |
              steady                  transient
             temperature             temperature history
                 +----------+-----------+
                            |
                   budgets, saved fields, figures
```

The geometry branch does **not** manufacture a power budget. The electrical branch does **not** know the thermal tetrahedral DOF numbering. Their common identifiers are transistor instance IDs, resistor IDs, physical polygons, and the validated manifest.

## 2. Contracts that every reviewer should understand

### 2.1 Quantities and units

| Quantity | Representation and units | Principal conversion boundary |
|---|---|---|
| GDS polygons and working stack heights | µm | Geometry extraction and meshing use micrometres. |
| SPICE transistor dimensions | Micrometre-valued instance parameters with `.option scale=1u` | Deck generation; waveform validation checks resulting physical dimensions in metres. |
| Imported thermal coordinates | m | `load_tagged_msh()` multiplies mesh coordinates by `geometry_length_scale_to_m`, normally `1e-6`. |
| Polygon-prism volume | m³ | Polygon area in µm² × height in µm × `1e-18`. |
| Electrical power | W | Named dictionaries; never infer units from plotting labels. |
| Electrical energy | J | Piecewise-linear time integration of power in seconds. |
| Thermal source `q` | W/m³ | Device normalization or conservative resistor projection. |
| `SourceBox.power_uw` | µW | Compatibility container; explicitly convert from W with `×1e6`. |
| Conductivity `k` | W/(m·K) | [spec/materials.py](spec/materials.py) through `build_coeffs()`. |
| Volumetric heat capacity `rho_cp` | J/(m³·K) | `Material.rho_cp`; used in the transient mass matrix, not the steady equation. |
| Boundary coefficient `h` | W/(m²·K) | Robin boundary form. |
| Solved temperature | K | Celsius is used only for SPICE's electrical temperature setting. |

For this **3D** path, `source_depth_m=None`. The homogenization-depth option belongs to older 2D cross-section models and must not be inserted into this pipeline.

### 2.2 Key Python objects

| Object | Defined in | What the next stage relies on |
|---|---|---|
| `MappedChannel` | [gds/bitcell_mapping.py](gds/bitcell_mapping.py) | Instance ID, exact channel polygon, gate/drain/source/body net names; class is descriptive only. |
| Interconnect network dictionary | [gds/interconnect.py](gds/interconnect.py) | Nodes, resistor IDs/ohms, device terminals, external ports, heat-region polygons/weights, fingerprint. |
| `SpiceAccessResult` | [gds/spice_transient.py](gds/spice_transient.py) | Validated timestamps/curves, channel/resistor power arrays, separate junction/source reports, integration methods. |
| `SpiceThermalWorkload` | [gds/thermal_workload.py](gds/thermal_workload.py) | Per-device and resistor energies, average powers, period/activity, and electrical provenance. |
| Region manifest | [mesh/gds_notebook.py](mesh/gds_notebook.py), specialized by [mesh/sram.py](mesh/sram.py) | Polygon prisms, material names, physical tags, source roles/IDs, units, mesh/contract hashes. |
| `GDSKelvinCase` | [mesh/gds_import.py](mesh/gds_import.py) | `mesh_data`, `registry`, `chip`, `manifest`, import/volume/connectivity report. |
| `DeviceHeatSources` / `CombinedHeatSources` | [physics/device_sources.py](physics/device_sources.py), [physics/interconnect_sources.py](physics/interconnect_sources.py) | Reusable DG0 source field and independent budget checks. |
| Temperature `Function` | [solve/steady.py](solve/steady.py), [solve/transient.py](solve/transient.py) | Scalar continuous P1 nodal values; use its own function-space coordinates/connectivity for rendering. |

DG0 means one constant source/material value per tetrahedron. P1 means temperature varies linearly within each tetrahedron and is continuous across shared nodes. These are different spaces: electrical IDs, mesh cell indices, DG0 DOFs, and temperature DOFs must not be treated as the same numbering.

### 2.3 Baseline assumptions, not established process facts

The current SRAM adapter specifies a 50.3262 µm active-top-to-backside distance, a 10 nm heat-source depth, SiO₂ field/background material, nominal 1 µm SiN passivation, periodic x/y, an insulated top, and backside `h=20000 W/(m²·K)` to 300 K. Electrical operation uses 1.8 V, fixed 25 °C models, estimated 16 fF per bitline, and prescribed driver edges. Preserve these assumptions when packaging the existing implementation; change them only in an explicitly labeled scientific-model PR.

## 3. Proposed PR index and dependency graph

| PR | Proposed title | Main review boundary | Prerequisite PRs |
|---|---|---|---|
| [01](#pr-01) | `SRAM: bind layout channels to verified compact-model instances` | Input identity and device correspondence | Existing project skeleton/data infrastructure |
| [02](#pr-02) | `SRAM: extract the local layout resistor and contact network` | GDS connectivity → electrical network | 01 |
| [03](#pr-03) | `SRAM: generate and execute validated compact-model access decks` | Circuit/testbench → raw ngspice output | 01, 02 for layout mode |
| [04](#pr-04) | `SRAM: validate waveforms and extract channel and resistor heat` | Raw electrical output → physical power/energy records | 03 |
| [05](#pr-05) | `SRAM: define the shared event-to-thermal workload contract` | Access energy → steady and transient source contract | 04 |
| [06](#pr-06) | `SRAM: construct polygon-preserving physical material regions` | GDS masks → non-overlapping 3D region specification | 01 |
| [07](#pr-07) | `SRAM: build and seal conforming notebook mesh bundles` | Region specification → validated mesh artifacts | 06 |
| [08](#pr-08) | `SRAM: import tagged meshes and assign thermal materials` | Mesh bundle → solver-ready case | 01, 07 |
| [09](#pr-09) | `SRAM: conservatively map transistor and wire heat onto the mesh` | Named electrical powers → audited DG0 heat density | 02, 05, 08 |
| [10](#pr-10) | `SRAM: apply backside Robin and lateral periodic boundaries` | Thermal domain → constrained boundary-value problem | 08 |
| [11](#pr-11) | `SRAM: solve steady and transient heat equations with budget checks` | Coefficients/sources/BCs → temperature | 08, 10; 09 for coupled evidence |
| [12](#pr-12) | `SRAM: integrate CLI and notebook end-to-end runs` | Public workflow and persisted run contract | 05, 07–11 |
| [13](#pr-13) | `SRAM: render verified thermal fields and publish review evidence` | Saved results → interpretable figures and regression evidence | 12 |

This is a **logical dependency graph**, not a promise that whole current files can be cherry-picked in that order. Several files currently contain responsibilities belonging to multiple PRs. Section 5 gives the symbol-level ownership and stacking rules needed to make each proposed PR buildable.

## 4. Conceptual sections and formal PR review packets

<a id="pr-01"></a>

### PR 01 — Input identity and transistor-to-layout correspondence

**Proposed branch:** `sram/01-instance-identity`

**Purpose:** establish that heat named `X0` is applied to the physical channel of `X0`, not to whichever channel happens to be enumerated first or classified as an access transistor.

**Read these files/functions:**

| File | Functions/data to review | Responsibility |
|---|---|---|
| [data/sram_sp_cell.spice](data/sram_sp_cell.spice), [data/sram_sp_cell_exposed.spice](data/sram_sp_cell_exposed.spice) | Subcircuit pins and X0–X7 instances | Circuit identity and terminal connectivity. |
| [data/sram_sp_cell_channel_map.json](data/sram_sp_cell_channel_map.json) | `channels`, input hashes, stated contracts | Persisted cell-specific mapping evidence. |
| [cases/verify_bitcell_mapping.py](cases/verify_bitcell_mapping.py) | `_read_devices()`, `verify()`, `check_manifest()`, `main()` | Independently reconstruct electrical connectivity from masks and the netlist, then compare against the manifest. |
| [gds/bitcell_mapping.py](gds/bitcell_mapping.py) | `geometry_fingerprint()`, `mapping_revision()`, `map_bitcell_channels()`, `validate_device_powers()`, `channel_sources_for()` | Runtime identity validation and per-instance source construction. |
| [gds/read.py](gds/read.py), [gds/sources.py](gds/sources.py) | `flatten_by_layer()`, `extract_channels()`, `_dims_um()` | Flatten the selected cell; identify DIFF∩POLY channel footprints. |

**Conceptual operation.** The SRAM has six functional cell transistors plus two D=S PMOS instances in the supplied netlist. All eight retain identity. The connectivity verifier uses conducting masks, contacts, port labels, and explicit netlist contracts to associate channel polygons with gate/drain/source/body nets. This is deliberately a cell-specific correspondence check, not general-purpose LVS. Runtime mapping rechecks input hashes and exact polygon matching; a changed GDS, label, netlist, or source footprint must fail rather than reuse a stale mapping.

The historical class-averaging problem was caused by sharing power among devices of the same class. Here, `validate_device_powers()` requires exactly the expected IDs, and `channel_sources_for()` transfers each instance's own power. Its `SourceBox` compatibility route also verifies rectangularity before normalizing by box dimensions. The notebook import later uses manifest polygon volumes instead.

**Input → output:** original local cell masks + netlist + mapping manifest → eight `MappedChannel` records and an exact-ID power contract.

**PR scope / non-goals:** own the mapping manifest and verification logic. Do not change compact-model parameters, source thickness, resistor values, or thermal cooling. In a future cleanup, the reusable verifier should live below `cases/`; currently `gds/interconnect.py` imports it from that CLI module, which is a real cross-layer dependency to preserve or move explicitly.

**Acceptance checklist:**

- [ ] Unequal powers remain unequal and correctly located, including READ-0 versus READ-1.
- [ ] Missing/extra IDs, stale input hashes, ambiguous channels, and nonfinite/negative powers fail.
- [ ] X3/X4 D=S identity is preserved; they are not deleted merely because channel dissipation is zero.
- [ ] Mapping does not depend on GDS polygon order, physical-tag order, or class ordering.

**Tests/evidence:** [tests/test_bitcell_mapping.py](tests/test_bitcell_mapping.py); run `python cases/verify_bitcell_mapping.py --check-manifest` against the supplied manifest. Attach a device-ID/geometry/terminal comparison, not only a total-power number.

<a id="pr-02"></a>

### PR 02 — Local interconnect electrical resistance

**Proposed branch:** `sram/02-layout-resistance`

**Purpose:** derive wire/contact voltage drops and heat from the same electrical network as the transistor currents.

**Code:** [gds/interconnect.py](gds/interconnect.py): `sheet_network()`, `_line_intervals()`, `_shared_length()`, `_Union`, `build_interconnect_network()`, `network_fingerprint()`; constants `SHEET_OHM`, `CONTACT_OHM`, `CONTACT_SIZE_UM`.

**Conceptual operation.** A polygon conductor is clipped into a Cartesian electrical grid. Neighboring pieces receive a sheet resistor proportional to centroid-normal separation divided by shared interface length:

```text
R_branch = R_sheet × distance / shared_width
```

This electrical segmentation is independent of the later thermal tetrahedral mesh. Bends and polygon boundaries are retained rather than replaced by bounding boxes. Source/drain diffusion connectivity remains ideal in this network. Gate terminals and external ports attach to explicitly selected conductor cross sections, which are treated as equipotential at that attachment. The two local WL pieces are externally tied according to the circuit contract.

For a contact, the electrical conducting core is `cut ∩ lower landing ∩ upper landing`. Resistance scales inversely with retained core area. An equipotential contact hub and area-weighted parallel half-resistances distribute current between the landing sheets. A coincident duplicate MCON is counted once. A floating unlabelled M2 feature is excluded electrically but remains thermal material.

Each resistor carries one or more `heat_regions`: a layer, polygon, and positive weight. The weights sum to one. These are spatial instructions for later heat deposition; they are not a thermal mesh or a prescribed fraction of total cell power.

**Input → output:** verified GDS connectivity + electrical step size → serializable nodes, resistors, terminal/port maps, heat supports, assumptions, fingerprint.

**PR scope / non-goals:** own resistance extraction and its geometry-to-network contract. Deck serialization belongs to PR 03. Do not introduce an arbitrary wire-power percentage, extracted capacitance claims, or full-array rail loading. Nominal resistance values are assumptions; the current local ports can leave M2 carrying negligible current even though it is thermally present.

**Acceptance checklist:**

- [ ] Rectangular-sheet analytical resistance and contact parallel combinations match expected values.
- [ ] No resistor shorts distinct logical nets; all intended nodes connect to the correct external port.
- [ ] Duplicate cuts are not double-counted; missing landings and invalid core areas fail.
- [ ] Every resistor has positive resistance and conservative, geometry-bound heat weights.
- [ ] Electrical grid refinement is assessed separately from thermal mesh refinement.

**Tests/evidence:** [tests/test_interconnect_network.py](tests/test_interconnect_network.py); electrical network portions of [tests/test_spice_interconnect.py](tests/test_spice_interconnect.py). Archive representative path resistances and the generated network JSON. The current default is 0.05 µm electrical segmentation, not a 0.05 µm thermal mesh guarantee.

<a id="pr-03"></a>

### PR 03 — Compact models, access testbench, and ngspice execution

**Proposed branch:** `sram/03-spice-testbench`

**Purpose:** turn a specified READ/WRITE experiment into a reproducible, unit-correct electrical simulation.

| File | Functions/data | Review focus |
|---|---|---|
| [gds/spice_netlist.py](gds/spice_netlist.py) | `_INCLUDES`, `_preamble()`, `_write_interconnect_cell()`, `write_access_deck()`, `_write()` | Model includes, unit scaling, generated circuit, stimulus, saved vectors. |
| [gds/spice_power.py](gds/spice_power.py) | `find_ngspice()`, `_run()`, `_DEVICES`, `_DEVICE_BULK`, `SPICE_MODEL_REVISION` | Binary discovery, execution/error handling, model-instance vector paths. Other DC helpers in this file are not the modern switching model. |
| [gds/spice_transient.py](gds/spice_transient.py) | Execution/archive portion of `run_access_transient()` | File lifecycle, timeout, simulator log, raw output handed to PR 04. |
| [data/sky130_fd_pr](data/sky130_fd_pr) | Files enumerated by `_INCLUDES`, including `ngspice_fixed/` | Actual bundled model source, curated wrappers, global/mismatch parameters. |

**Conceptual operation.** The deck uses `.option scale=1u` so the micrometre-valued instance dimensions have the intended physical size. Electrical temperature is fixed at 25 °C. The default supply is 1.8 V. Two estimated 16 fF bitline loads and prescribed driver impedances/edges represent external loading; they are not extracted array RC.

For READ, the DC initialization has precharge switches on. They release before WL rises, and the cell then develops a differential bitline voltage. There is no sense-amplifier feedback or recharge phase. For WRITE, the target bitline values are already driven before WL assertion, and the cell is expected to change state. `.nodeset` guides initialization; it is not accepted as proof that the intended state actually occurred.

Layout mode writes a separate `interconnect.spice` subcircuit using PR 02's network and preserves transistor identities. It saves each transistor's local terminal voltages, intrinsic current and junction currents, plus each local resistor's branch voltage. Using ideal logical-net voltages after inserting resistors would invalidate the heating calculation.

`find_ngspice()` checks `KELVIN_NGSPICE`, PATH, the isolated local installation, then the legacy fallback. `_run()` checks both process status and known failure text: ngspice can emit incomplete data even when a control script exits with status zero. Layout runs receive a longer bounded timeout. Existing output directories with electrical evidence are not overwritten.

**Input → output:** operation/state/timing + compact models + optional resistor network → deck, metadata, raw waveform, log; successful completion is established by PR 04, not by process exit alone.

**PR scope / non-goals:** package deck generation and execution without silently retuning device models to make traces look plausible. A reusable process runner may remain in its current file for compatibility. A new full-cycle testbench is an accuracy follow-up, not an undocumented change in this packaging PR.

**Acceptance checklist:**

- [ ] Physical L/W values are checked, including 25 nm short-channel instances.
- [ ] Both READ states and both WRITE directions are exercised.
- [ ] Failure text, timeout, missing binary, malformed output, and nonempty destinations fail explicitly.
- [ ] Device-local voltage/current vectors and resistor voltage columns correspond to the generated circuit.

**Tests:** [test_spice_discovery.py](tests/test_spice_discovery.py), [test_spice_execution.py](tests/test_spice_execution.py), [test_spice_model_units.py](tests/test_spice_model_units.py), [test_spice_transient_decks.py](tests/test_spice_transient_decks.py), and deck portions of [test_spice_interconnect.py](tests/test_spice_interconnect.py).

<a id="pr-04"></a>

### PR 04 — Waveform validation, dissipation, and event energy

**Proposed branch:** `sram/04-electrical-heat`

**Purpose:** establish which electrical quantities are physical local heat and reject incomplete/nonfunctional electrical experiments.

**Code:** [gds/spice_transient.py](gds/spice_transient.py): `read_access_waveform()`, `validate_access_waveform()`, `access_result_from_waveform()`, `integrate_trace()`, `SpiceAccessResult` and its energy/average-power methods. [cases/characterize_spice_access.py](cases/characterize_spice_access.py): `plot_access()`, `main()`.

**Conceptual operation.** Parsing checks the complete header, finite values, strictly increasing timestamps starting at zero, and the requested final timestamp. Functional validation checks dimensions, initial complementary state, actual WL activity, final state, and READ polarity/disturb. A nonzero bitline differential alone is not a complete READ validation.

The current heat definitions are:

```text
P_channel,i(t) = intrinsic_id_i(t) × |local_Vd_i(t) − local_Vs_i(t)|
P_resistor,r(t) = [Va_r(t) − Vb_r(t)]² / R_r
E_i            = integral P_i(t) dt over the complete saved event
```

The code's BSIM3 current convention is checked before using a nonnegative magnitude. Junction-current powers are reported separately and are not deposited by the current thermal path. Signed power delivered by external voltage sources is also reported separately. Supply-terminal power is not automatically local instantaneous heat: some energy is stored in capacitors or dissipated outside the bitcell.

`integrate_trace()` inserts interpolated interval endpoints and integrates the piecewise-linear samples. The same routine supplies full-event energies and per-thermal-step average powers, preventing point-sampling from missing narrow pulses. The characterization CLI can rerun at half and quarter electrical step sizes and compare integrated channel and resistor energies. That is temporal convergence of this electrical model, not validation of its loading assumptions.

**Input → output:** validated raw vectors → `SpiceAccessResult`, per-device/resistor power archives, functional/energy summary.

**PR scope / non-goals:** own waveform semantics and energy integration, not source placement or thermal normalization. Do not replace the measured event energy with an independent `CV²f` budget or a normalized target energy.

**Acceptance checklist:**

- [ ] Truncated, nonmonotonic, nonfinite, wrong-column, wrong-state, and READ-disturb traces fail.
- [ ] Local VDS is used with layout resistance enabled.
- [ ] Integration across nonuniform samples and partial windows is conservative.
- [ ] Channel, junction, external-driver and wire energies remain distinguishable.

**Tests/evidence:** [tests/test_spice_transient.py](tests/test_spice_transient.py), power extraction portions of [tests/test_spice_interconnect.py](tests/test_spice_interconnect.py); characterization waveforms for all four access cases and an electrical timestep study.

<a id="pr-05"></a>

### PR 05 — Shared workload and the meaning of steady power

**Proposed branch:** `sram/05-workload-contract`

**Purpose:** prevent steady and transient solvers from using different electrical energies or incompatible activity conventions.

**Code:** [gds/thermal_workload.py](gds/thermal_workload.py): `SpiceThermalWorkload`, `SPICE_ACCESS_POINTS`, `build_spice_workload()`, `_power_trace_sha256()`, `THERMAL_POWER_REVISION`. [gds/power.py](gds/power.py): `AccessTiming`, `mine_access_timing()`, `_first_constraint_value()`. The `LIB_PATH` constant currently lives in [gds/spice_power.py](gds/spice_power.py).

**Conceptual operation.** One validated event is the common source of transistor and resistor energies. For steady or sustained-average loading:

```text
access_rate_hz = row_hit_rate / period_s
average_power = event_energy × access_rate_hz
```

`row_hit_rate` is the fraction of cycles selecting the modeled row/cell workload, not an arbitrary fraction of power assigned to wires. It changes the averaged frequency, not the waveform or energy of an individual selected access. The period must contain the complete electrical observation window. The thermal instantaneous CLI rejects a nonunit row-hit rate because its purpose is one selected event.

Defaults borrow period and WL high time from the supplied Liberty file, currently for `sram22_2048x8m8w1`, not the target 64×22 macro. Explicit overrides are recorded. Electrical traces, model inputs, mapping/network data, and archived files are fingerprinted so later thermal rendering can identify stale evidence.

**Current limitation to preserve visibly:** the event is not a complete repeating recharge/access/idle cycle. Holding leakage outside the trace is omitted; zero activity therefore gives zero modeled average heat. WRITE state sequences are not combined into a state-consistent repeating workload. These are scientific limitations of the current contract, not reasons to silently redistribute the extracted energy.

**Input → output:** one access result + period + activity → per-ID event joules, per-ID average watts, trace access, provenance/omissions metadata.

**PR scope / non-goals:** own the common workload object and its provenance. Complete-cycle/leakage accounting belongs to follow-up F01. Do not include old `named_bias_point_power_w()` crowbar approximations as an alternative implementation of a real switching event.

**Acceptance checklist:**

- [ ] Each ID independently satisfies `P_average × period = E_event × row_hit_rate`.
- [ ] Changing rate/period leaves the single-event trace and energy unchanged.
- [ ] Invalid periods/rates and incomplete electrical results fail.
- [ ] Model, timing source, omitted heat mechanisms, and artifact hashes are persisted.

**Tests:** [tests/test_thermal_workload.py](tests/test_thermal_workload.py), workload-selection portions of [tests/test_compact_workload.py](tests/test_compact_workload.py).

<a id="pr-06"></a>

### PR 06 — Physical regions: masks, contacts, substrate, and dielectric

**Proposed branch:** `sram/06-physical-regions`

**Purpose:** convert the selected GDS cell into a complete thermal material domain without treating every mask as a separate physical solid.

| File | Functions | Responsibility |
|---|---|---|
| [mesh/sram.py](mesh/sram.py) | `thermal_context_options()`, `prepare_sram_regions()`, `prepare_sram_manifest()`, `match_source_tags()` | SRAM-specific stack/source assumptions and X0–X7 tag correspondence. |
| [mesh/gds_notebook.py](mesh/gds_notebook.py) | `extract_layout()`, `normalized_components()`, `boolean_components()`, `make_region()`, `build_sky130_regions()`, `assert_no_material_overlaps()` | Polygon normalization, material subtraction/fill, non-overlapping 3D region descriptions. |
| [mesh/gds_contacts.py](mesh/gds_contacts.py) | `classify_sky130_licon()`, `validate_sky130_contacts()` and their landing checks | Preserve contact identity and verify lower/upper landing relationships. |
| [mesh/gds_notebook.py](mesh/gds_notebook.py) | `install_helpers()`, `create_context()` | Bind shared functions to an isolated notebook/CLI context. |

**Conceptual operation.** Geometry is assembled bottom-up: substrate, p/n-well partitions, source/drain silicon, channel underlayers, individual 10 nm channel-source regions, poly, LICON, LI1, MCON, metals/vias, dielectric fill, and passivation. Channel footprints are DIFF∩POLY, with well overlap distinguishing NMOS/PMOS geometry. Device names are subsequently matched by polygon equality; geometric source-tag numbering is not electrical identity.

For each vertical slab, conductors are subtracted from the domain to obtain dielectric fill. Area conservation and material non-overlap are checked. Implant, pin-purpose, memory-core and other annotation layers may identify/classify geometry without becoming additional material volumes. This explains why displaying every GDS layer is not the same as meshing every layer as a thermal solid.

LICON is classified as an entire cut landing on either diffusion/tap or poly, with different vertical spans. Whole drawn contact footprints are retained as thermal solids. Partial enclosure is accepted/reported only under the explicit memory-core policy. **Do not confuse that thermal footprint with PR 02's conducting overlap core**, where electrical contact resistance and Joule-source support are defined. Both descriptions must remain geometrically compatible without silently changing one into the other.

The shared helper module does not read/execute the notebook at runtime. It binds functions into a context dictionary so staged notebook values are visible. Functions such as `build_sky130_regions()` therefore depend on the bound context, not only explicit arguments. Review context isolation and stale-state invalidation carefully.

**Input → output:** selected cell + SRAM thermal assumptions → region list, process-model record, material/source tags, layout identity; no powers yet.

**PR scope / non-goals:** own geometry semantics and stack settings. GDS is not a source of material conductivity, actual wafer thickness, package cooling, or gate-oxide/interface resistance. Those currently omitted details must remain disclosed, not described as recovered process truth.

**Acceptance checklist:**

- [ ] Concavities, disconnected LI1 components, and channel footprints survive normalization.
- [ ] Solids do not overlap; conductor plus dielectric fills each intended slab.
- [ ] Contact landing/type errors and inappropriate memory-core exceptions fail.
- [ ] Exact source shapes match the verified instance map; all material names resolve.

**Tests/evidence:** [tests/test_gds_contacts.py](tests/test_gds_contacts.py), region/mapping parts of [tests/test_sram_mesh.py](tests/test_sram_mesh.py). Attach physical-region tables and the GDS-versus-region view, not only a shaded mesh image.

<a id="pr-07"></a>

### PR 07 — Conforming mesh generation and sealed bundle publication

**Proposed branch:** `sram/07-mesh-bundle`

**Purpose:** make a conforming, uniquely tagged finite-element mesh and ensure its manifest describes the geometry actually meshed.

**Code:** [mesh/gds_notebook.py](mesh/gds_notebook.py): `add_polygon_prism()`, `build_and_validate_mesh()`, `region_contract_sha256()`, `region_layout_identity()`, `build_region_manifest()`, `build_notebook_mesh_bundle()`; [mesh/sram.py](mesh/sram.py): `_contract_hash()`, `validated_sram_manifest()`, `seal_sram_manifest()`, `generate_sram_mesh()`; [cases/build_sram_mesh.py](cases/build_sram_mesh.py): `main()`.

**Conceptual operation.** Each polygon is extruded over its physical z interval. Gmsh OCC fragmentation creates shared material interfaces. The fragment map assigns every resulting volume to exactly one input physical region; ambiguous ownership is rejected. Exterior surfaces and interfaces are tagged distinctly so an interior contact face cannot accidentally become a cooling boundary.

The mesh uses first-order tetrahedra, deterministic seed/settings, and refinement around source/conductor surfaces. Validation checks material ownership, nominal-versus-meshed volumes, connectivity, and minimum tetrahedral quality. Mesh target length and element quality do not establish thermal convergence; the resulting solution still requires refinement studies.

`build_notebook_mesh_bundle()` snapshots **current** regions and metadata, builds in a staging directory, and publishes artifacts with the manifest last. This is staged multi-file publication, not a single filesystem-atomic transaction. If publication is interrupted, hash checks reject an inconsistent bundle. A successful earlier bundle remains untouched when meshing fails before publication. Sealing requires mesher-produced hashes for both actual region input and actual mesh bytes; an arbitrary old JSON cannot be certified against a new MSH.

**Input → output:** current region specification → `.msh`, quality mesh, `.brep`, mesh statistics, optional geometry SVG, sealed material-region JSON.

**PR scope / non-goals:** no SPICE, power assignment, or heat solve. Keep notebook previews separate from authoritative bundle creation. A source-code-only review must include the manifest schema and failure paths, not only successful Gmsh output.

**Acceptance checklist:**

- [ ] Every tetrahedron has one material owner; source volumes and region hashes match.
- [ ] Disconnected or insufficient-quality meshes are rejected before thermal solving.
- [ ] Changing notebook regions/materials without rerunning the preview still seals the current state correctly.
- [ ] Changed layout/top-cell state, stale manifests, tampered mesh bytes and blind resealing fail.
- [ ] A failed staged build does not overwrite the prior valid bundle.

**Tests:** [tests/test_notebook_mesh_bundle.py](tests/test_notebook_mesh_bundle.py), mesh-generation parts of [tests/test_sram_mesh.py](tests/test_sram_mesh.py). Preserve one generated smoke-test bundle separately from small unit fixtures.

<a id="pr-08"></a>

### PR 08 — Mesh import, SI units, material coefficients, and readiness

**Proposed branch:** `sram/08-thermal-import`

**Purpose:** turn a sealed geometry bundle into a well-specified thermal domain with auditable units and material/source ownership.

**Code:** [mesh/sram.py](mesh/sram.py): `load_sram_mesh()`, `build_sram_mesh()`; [mesh/gds_import.py](mesh/gds_import.py): `ImportedHeatSource`, `ImportedRegionRegistry`, `GDSKelvinCase.assert_solver_ready()`, `load_tagged_msh()`, `_resolve_material()`, `_manifest_source_volume_m3()`, `_source_volume_report()`, `_remap_boundary_tags()`, `_connectivity_report()`; [physics/coeffs.py](physics/coeffs.py): `_tag_by_cell()`, `build_coeffs()`; [spec/materials.py](spec/materials.py): `Material`, `MATERIALS`, `get()`; [mesh/boxes.py](mesh/boxes.py): shared `Region` record.

**Conceptual operation.** The SRAM loader verifies the sealed contract, source identity, and mesh hash. The importer reads Gmsh physical tags, converts coordinates to metres, verifies complete tag coverage, and assigns material records. Channel watts supplied independently by the electrical workload are divided by manifest polygon volume. The actual tagged mesh volume is then checked against that declared volume; simply renormalizing to whatever volume happened to mesh would conceal a geometry error.

Only global-bottom surfaces become the backside cooling tag. The tops/bottoms of embedded individual conductors must not become external sinks. Connected components are checked for connection to the backside anchor, avoiding an unconstrained steady thermal component.

`build_coeffs()` supplies scalar DG0 conductivity, volumetric heat capacity, and initial source fields. It currently uses `mat.k` at the reference temperature, not `Material.k_at(T)`, and does not enable temperature feedback. This must be explicit in the PR description even though a temperature-dependent helper exists elsewhere in the repository.

**Input → output:** sealed MSH/manifest + per-instance watts + BC configuration → `GDSKelvinCase`, import report, material/source coefficient fields.

**PR scope / non-goals:** no new cooling model, nonlinear materials, or source redistribution. Shared generic importer options may remain, but the reviewed SRAM route must use explicit independent watts and exactly eight device IDs. Source-candidate protections must not be disabled globally merely to support the separate resistor projector.

**Acceptance checklist:**

- [ ] Micrometre-to-metre conversion occurs exactly once.
- [ ] Unknown material, missing material tags, invalid source units, and conflicting power/density assignments fail.
- [ ] The imported source integral equals independent prescribed watts, not a budget derived from that same integral.
- [ ] Backside tag covers the intended external footprint, not buried interfaces.

**Tests:** [tests/test_gds_import_sources.py](tests/test_gds_import_sources.py), import portions of [tests/test_sram_mesh.py](tests/test_sram_mesh.py), [tests/test_sram_notebook_pipeline.py](tests/test_sram_notebook_pipeline.py).

<a id="pr-09"></a>

### PR 09 — Conservative spatial heat-source projection

**Proposed branch:** `sram/09-spatial-heat`

**Purpose:** deliver the correct number of watts/joules at the correct physical locations, even when electrical resistor tiles do not match thermal tetrahedra.

| File | Functions/classes | Responsibility |
|---|---|---|
| [physics/device_sources.py](physics/device_sources.py) | `DeviceHeatSources.__init__()`, `set_powers()`, `integrated_powers()`, `integrated_energy()`, `audit_energy()` | Channel volume validation, per-ID DG0 updates, independent integral checks. |
| [physics/interconnect_sources.py](physics/interconnect_sources.py) | `_convex_parts()`, `_prism()`, `_tetra_prism_volume()`, `_layer_region()` | Concave support decomposition and polygon-prism/tetrahedron intersection. |
| Same file | `InterconnectHeatSources`, `_project_region()`, `set_powers()`, `_assert_actual_field()`, `audit_energy()` | Sparse resistor-to-cell projection and spatial/energy checks. |
| Same file | `CombinedHeatSources.set_powers()`, `audit_powers()`, `audit_energy()` | Combined source field with independently preserved channel and wire budgets. |
| [post/budget.py](post/budget.py) | `verify_device_source_powers()` | Independently prescribed per-device watts versus actual FEM integrals. |

**Channels.** Each channel has an individual physical source tag. Its uniform density is `q_i = P_i / V_i`. The adapter validates the eight names, checks nominal versus meshed volumes, and maps cells through their actual DG0 DOF map. All old source values are cleared before an update, so a previously active device cannot leave stale power behind.

**Interconnects.** One thermal layer tag can contain several disconnected wires, so assigning a resistor's watts to the whole tag would be wrong. Instead, each weighted heat-support prism is intersected with tetrahedra belonging to the correct material tag. For a support `s` of resistor `r` and thermal cell `K`:

```text
q_K contribution = P_r × weight_rs × volume(K ∩ support_s)
                   / [volume(support_s) × volume(K)]
```

The expensive intersections are precomputed into a sparse operator. Runtime updates multiply that operator by the resistor power vector. Support volumes and column integrals are checked; missing material overlap is an error, not something to hide by silently renormalizing away missing geometry. The contact's lower landing selects the appropriate LICON vertical family.

**Auditing.** For transients, the driver accumulates the actual `solver.q × dt` consumed at every step. It checks device energy, layer energy, total energy, and spatial agreement against independently expected electrical energies projected onto the mesh. The wire-only error is checked against the wire budget, so small wire heat cannot be lost behind the much larger transistor budget. Overlapping resistor supports cannot be uniquely recovered from a summed DG0 field; reports must not claim independent recovery of every overlapping branch.

**Input → output:** per-ID W plus mesh/manifest/network → combined DG0 W/m³ and volume/power/energy audit reports.

**PR scope / non-goals:** preserve energy and geometry; do not tune a transistor/wire split. An exact source-overlap calculation does not imply that temperature gradients below a tetrahedron are resolved. Keep heat-source validation separate from the final PDE heat balance.

**Acceptance checklist:**

- [ ] Equal total power with swapped device locations is detected.
- [ ] Disconnected wires sharing a layer tag do not receive unrelated resistor heat.
- [ ] Concave polygons, partial contacts, overlapping supports and invalid volumes are covered.
- [ ] Actual solver-load corruption fails, including corruption confined to the smaller wire budget.
- [ ] Combined source updates preserve each channel and the resistor-field budget independently.

**Tests:** [tests/test_device_heat_sources.py](tests/test_device_heat_sources.py), [tests/test_interconnect_heat_sources.py](tests/test_interconnect_heat_sources.py), integration wiring in [tests/test_interconnect_thermal_driver.py](tests/test_interconnect_thermal_driver.py) and [tests/test_spice_thermal_driver.py](tests/test_spice_thermal_driver.py).

<a id="pr-10"></a>

### PR 10 — Boundary conditions and periodic constraints

**Proposed branch:** `sram/10-thermal-boundaries`

**Purpose:** specify what region the bitcell represents and where heat can leave it.

**Code:** [spec/chip.py](spec/chip.py): `BoundaryConditions`; [physics/bcs.py](physics/bcs.py): `robin_terms()`; [physics/periodic.py](physics/periodic.py): `_check_face_areas()`, `_flatten_constraints()`, `build_periodic_constraint()`; [mesh/build.py](mesh/build.py): shared facet-tag constants. The actual SRAM defaults are instantiated in the two bitcell thermal drivers, not inferred from GDS.

**Mathematical conditions:**

```text
Backside:      -k grad(T) · n = h (T - T_ambient)
Top:           -k grad(T) · n = 0
Lateral x/y:   T(slave point) = T(translated master point)
```

Robin terms contribute `h u v` to the matrix and `h T_ambient v` to the load on the exterior backside. An unspecified exterior flux is the natural zero-flux condition of the weak form. The top is therefore insulated when no top Robin term is requested.

Periodic conditions constrain both trial and test spaces with `dolfinx_mpc`; they are not post-solve averaging. Nonmatching opposing meshes use master-side interpolation. Doubly periodic edges require constraint dependencies to be flattened so a slave does not remain a dependent master of another slave. The implementation checks face coverage, preserves constants, rejects conflicting lateral Robin/Dirichlet settings, and verifies that every intended slave is constrained.

The constraint search temporarily normalizes coordinates because library search tolerances combine geometric and interpolation uses. Original metre coordinates are restored before assembling physics. This restoration and corner handling deserve focused review: an unnoticed unit change here would corrupt conductivity and source integrals.

**Physical interpretation.** One-cell periodicity repeats identical orientation and activity. It is not reflection symmetry, not an isolated heated cell in cold silicon, and not automatically the true mirrored macro repeat unit. Periodic faces remove no net heat. With the current insulated top, the mean backside rise obeys `P_total/(h A_backside)`.

**PR scope / non-goals:** implement the specified BC problem without claiming the assumed h or repeat unit is calibrated. Preserve the explicit insulating compatibility option; do not silently fall back if the MPC dependency is absent.

**Acceptance checklist:**

- [ ] Translation residuals and constant preservation pass, including x/y corners and nonmatching faces.
- [ ] Geometry is restored on success and failure of constraint construction.
- [ ] Periodic constraints enter mass, stiffness and load assembly, not only the final temperature vector.
- [ ] Boundary heat balance and manufactured/analytical cases are tested separately from the SRAM rendering.

**Tests/evidence:** [tests/test_periodic.py](tests/test_periodic.py), relevant [tests/test_energy_conservation.py](tests/test_energy_conservation.py); [PERIODIC_BOUNDARIES.md](PERIODIC_BOUNDARIES.md) documents dependency setup and interpretation.

<a id="pr-11"></a>

### PR 11 — Steady/transient finite-element solves and thermal budgets

**Proposed branch:** `sram/11-heat-solvers`

**Purpose:** solve the stated thermal equations and expose failures without confusing solver convergence with physical validation.

| File | Functions/classes | Responsibility |
|---|---|---|
| [physics/forms.py](physics/forms.py) | `steady_form()` | P1 conduction weak form plus boundary terms. |
| [solve/steady.py](solve/steady.py) | `_solve()`, `solve_steady()`, `solve_steady_from_fields()` | Standard/MPC solve, supplied versus registry-built source fields, KSP checks. |
| [solve/transient.py](solve/transient.py) | `TransientHeatSolver.__init__()`, `_assemble_matrix()`, `step()` | Backward-Euler time stepping, mass/stiffness/BC assembly, changing dt, periodic reconstruction. |
| [post/budget.py](post/budget.py) | `power_balance()`, `print_power_balance()` | Intended versus meshed watts and steady boundary heat-out check. |
| [post/metrics.py](post/metrics.py) | `tmax()` | Peak temperature and location. |

**Steady equation:** `-div(k grad T) = average_q`. There is no heat-capacity term. `solve_steady_from_fields()` is the current layout-interconnect route because PR 09 has already constructed the combined spatial source; reconstructing q from channel-only registry entries would drop wire heat. The linear system uses CG with hypre BoomerAMG and checks PETSc's convergence reason.

**Transient equation:** `rho_cp (T_new - T_old)/dt - div(k grad T_new) = q_step`. The solver uses a consistent P1 mass matrix and backward Euler. It reuses the matrix when dt is unchanged, rebuilds it when dt changes, updates ghosts, and back-substitutes periodic constraints before the next step. The driver, not the solver, is responsible for calculating each interval's energy-conserving source.

**Budget interpretation.** At steady state, source watts should equal total Robin heat out. In a transient, their difference also changes stored thermal energy, so a steady equality must not be asserted before equilibrium. Moreover, steady conservation follows from the discrete equation for any imposed q: it cannot prove q was the intended electrical source. That is why PR 09's independent source checks are also necessary. `print_power_balance()` documents a small-power numerical floor for the relative boundary check, not permission to ignore incorrect prescribed source watts.

**Current numerical limitation:** the consistent-mass transient discretization can undershoot below ambient with nonnegative heat. Backward-Euler stability is not a positivity guarantee. The saved READ undershoot is about 0.143 K. Do not clamp it in plots or call solver tolerance a mesh-convergence study. Steady spatial convergence also remains unestablished.

**PR scope / non-goals:** preserve the existing linear PDE/discretization. Nonlinear conductivity, electrothermal feedback, interface jumps, and positivity fixes need separately measured behavior-changing PRs. Shared solver code still needs compatibility regression for other callers, even though the science reviewed here is SRAM-only.

**Acceptance checklist:**

- [ ] Solver failures raise; changes in dt replace the matrix without leaking its prior PETSc allocation.
- [ ] Periodic state reconstruction occurs before reuse of the old-temperature term.
- [ ] Manufactured/simple-domain solutions and energy balances pass.
- [ ] Per-device source audits are not replaced by the global PDE heat-balance check.
- [ ] Numerical limitations and convergence status are visible in run reports.

**Tests:** [tests/test_energy_conservation.py](tests/test_energy_conservation.py), [tests/test_periodic.py](tests/test_periodic.py), [tests/test_transient_cache.py](tests/test_transient_cache.py), and coupled-driver tests. If this PR is too large for one review, split steady solve/budget from transient stepping without changing their common source/BC contracts.

<a id="pr-12"></a>

### PR 12 — Public CLI orchestration, notebook handoff, and run artifacts

**Proposed branch:** `sram/12-integrated-runs`

**Purpose:** make the approved stages one reproducible user-facing SRAM workflow, with modes that cannot silently change physical meaning.

**Steady driver:** [cases/run_bitcell_compact.py](cases/run_bitcell_compact.py): `_check_output_directory()`, `_build_power_workload()`, `_audit_averaged_energy()`, `run()`, `_argument_parser()`, `main()`.

`run()` obtains the workload, maps channels, builds/imports the notebook mesh, assigns BCs, constructs combined channel/wire q, solves steady temperature, audits watts and energy-to-average consistency, checks heat out, and saves a summary. It requires a single MPI rank for the present SPICE/output orchestration. Legacy mesh and DC-surrogate modes remain explicit opt-ins; layout resistor heat requires the notebook mesh and real SPICE access path.

**Transient driver:** [cases/run_bitcell_transient.py](cases/run_bitcell_transient.py): `main()`, `build_bitcell_mesh()`, `spice_thermal_time_grid()`, `_dt_schedule()`, `_select_uniform_dt_frames()`.

For a single access, `spice_thermal_time_grid()` spans the complete electrical trace with uniform thermal intervals no longer than the requested maximum. Each step receives the exact piecewise-linear SPICE energy for that interval divided by dt. It saves the actual temperature history and source-step data, then uses increasing timesteps for zero-source cooling. For sustained-average mode it instead applies constant event-averaged powers over a schedule. The schedule length is not a measured thermal time constant.

**Notebook handoff:** [read_gds.ipynb](read_gds.ipynb), sections “Configuration,” “Construct non-overlapping 3D material and source regions,” “Build and validate the conformal Gmsh mesh,” and “Run Kelvin with this saved SRAM mesh.” `install_helpers()` provides the shared implementation; the final cell passes the sealed manifest to the steady CLI. `RUN_KELVIN_SRAM=False` is the current guard, and `KELVIN_PYTHON` can select an external DOLFINx interpreter. The notebook is not executed by the solver to recover geometry code.

**Artifacts reviewers must trace:**

```text
mesh bundle/
  sram_sp_cell_material_regions.msh
  sram_sp_cell_material_regions.json   # tags/materials/source identity + hashes
  sram_sp_cell_mesh_stats.json

run/
  spice/
    access.sp, metadata.json, waveform.txt, ngspice.log, summary.json
    device_power.npz
    interconnect.spice, interconnect_network.json, interconnect_power.npz
  power_workload.json
  mesh_import_report.json
  interconnect_projection.json
  source_power_audit.json
  source_energy_audit.json
  simulation_summary.json             # steady driver
  thermal_source_steps.npz             # time-resolved SPICE transient
  fields/                             # transient driver
    meta.json, dof_coords.npy, connectivity.npy
    times_s.npy, tmax_hist.npy, T_000.npy, ...
    tmin_hist.npy                      # time-resolved SPICE branch
```

The current steady driver does **not** routinely save its full temperature field. Recent steady plots needed a reconstruction from archived powers. Production steady field persistence is a specific integration improvement for PR 12/13, not an existing capability to assume in review. Large workload metadata embeds many resistor records; inspect selected JSON keys rather than dumping entire files into a review.

**PR scope / non-goals:** wire together approved interfaces, mode validation, output protection, provenance, and notebook execution. Do not mix in full-array migration or new workload physics. If adding steady-field persistence, label it as an output-only change and prove the solve is unchanged.

**Acceptance checklist:**

- [ ] Default steady, instantaneous transient and sustained-average transient are distinguished in CLI help and metadata.
- [ ] Saved-mesh reuse does not remesh or reinterpret `--refine`; input powers may change independently.
- [ ] Existing run evidence is not overwritten; mismatched modes/flags and unsupported MPI invocation fail.
- [ ] A notebook-generated mesh and CLI-generated mesh obey the same material/source contract.
- [ ] An end-to-end layout-READ run passes source audits and records the actual inputs needed to reproduce it.

**Tests:** [tests/test_compact_workload.py](tests/test_compact_workload.py), [tests/test_spice_thermal_driver.py](tests/test_spice_thermal_driver.py), [tests/test_interconnect_thermal_driver.py](tests/test_interconnect_thermal_driver.py), [tests/test_sram_notebook_pipeline.py](tests/test_sram_notebook_pipeline.py). The latter can use `KELVIN_SRAM_TEST_MANIFEST` to select a saved bundle.

<a id="pr-13"></a>

### PR 13 — Scientific visualization, reproducibility, and review evidence

**Proposed branch:** `sram/13-results-and-rendering`

**Purpose:** make figures faithful to verified fields and make their temperature scale, geometry scope and provenance unambiguous.

| File | Functions | Current responsibility |
|---|---|---|
| [cases/render_transient_layers.py](cases/render_transient_layers.py) | `_validate_mapping_metadata()`, `_dof_indices_per_volume()`, `main()` | Verify archived model/workload/mesh identity; sample true polygon prisms; render transient layers. |
| [mesh/gds_svg_viz.py](mesh/gds_svg_viz.py) | `render_layer_regions_svg()`, `render_regions_geometry_svg()`, `build_layer_regions()` | Shared GDS-left/isometric-right rendering. |
| [mesh/viz3d.py](mesh/viz3d.py) | `_temperature_grid()`, `render_temperature_xy_slice()`, `render_temperature_xz_slice()` | Slices of the actual FEM temperature field, using the temperature function's own DOF map. |
| [mesh/gds_notebook.py](mesh/gds_notebook.py) | `make_svg()`, `read_gmsh_mesh_for_preview()`, `make_mesh_diagnostics_svg()`, `region_table_html()` | Region/mesh inspection, not thermal solution. |
| [cases/render_sram_notebook_comparison.py](cases/render_sram_notebook_comparison.py), [cases/audit_sram_layers.py](cases/audit_sram_layers.py), [cases/check_sram_li1.py](cases/check_sram_li1.py) | `main()` and rendering/audit helpers | SRAM geometry comparisons and layer/connectivity evidence. |

**Current layer rendering.** The modern transient sampler selects nodes inside actual polygon prisms, including interfaces, with a numerical-scale coordinate tolerance. It fails when no nodes are found rather than borrowing a nearest unrelated node. Each polygon is colored by its hottest sampled node. That can make a whole gate stripe look hot even though its interior is not uniformly at the maximum. It is a presentation convention, not another thermal solve.

The original transient renderer subtracts each frame's coolest polygon-peak value and uses one common excess scale across selected frames. This is not temperature above ambient. Its frame selection emphasizes changes in global peak temperature, and GIF playback is not proportional to physical timestep. Small, delayed interconnect changes can be hard to see. A separately scaled interconnect view is valid only if that scale is explicit. The generic `mesh.gds_svg_viz.dof_indices_per_volume()` still uses an older bounding-box sampler for legacy callers; it is **not** the stricter helper in `cases/render_transient_layers.py`.

**Recent steady/full-substrate views.** The following scripts currently live under generated output, not a supported `cases/` entry point:

- [read_steady/visualization/render_steady.py](out/spice_interconnect_integration/read_steady/visualization/render_steady.py): reconstructs the field from unchanged archived powers, checks the archived peak and heat balance, saves fields and polygon views.
- [read_steady/visualization/render_substrate.py](out/spice_interconnect_integration/read_steady/visualization/render_substrate.py): uses those fields for exterior substrate triangles and a full-depth cross-section/profile; below-active-plane z is compressed 25× only in the 3D display.

They are useful local evidence, but an ignored `out/` directory and an embedded host-specific rasterization command are not a production rendering API. This PR should promote/generalize them into a versioned renderer, coordinate with PR 12 on a steady-field schema, and add field/mesh association checks. Their current links can be absent in a fresh clone; they are not required source inputs for the physics pipeline.

**Acceptance checklist:**

- [ ] Figures distinguish absolute K, rise above ambient, and local contrast; identify per-polygon maximum versus continuous field rendering.
- [ ] All full-depth views state the physical depth and any display compression; the whole bitcell substrate column is not labeled the full die.
- [ ] Device/wire-only views state what was hidden and whether their color scale differs.
- [ ] Metadata tampering, changed mesh/mapping, concave gaps, and neighboring-layer sampling are rejected.
- [ ] A supported steady renderer can consume saved fields without rerunning SPICE or solving heat again.
- [ ] Numeric data and audit summaries accompany screenshots; figure appearance is not the regression oracle.

**Tests/evidence:** [tests/test_transient_polygon_sampling.py](tests/test_transient_polygon_sampling.py), renderer/cache portions of [tests/test_transient_cache.py](tests/test_transient_cache.py), plus **new proposed** steady-rendering/provenance tests. Publish a READ steady view, transistor-only view, interconnect view, full-substrate view, and quantitative scale descriptions from the same field.

## 5. How to turn these review packets into real PRs

### 5.1 Do not split by filename alone

The working tree already contains modified and untracked files, including changes outside this SRAM path. Choose an agreed baseline and preserve that snapshot before constructing the series. Do not stage the whole workspace, reset unrelated changes, or assume every line in a shared file belongs to this project slice.

| Shared file | Ownership split in this proposal |
|---|---|
| `gds/spice_power.py` | PR 03: discovery/runner/model identity used by the new path. PR 05: timing dependency. Leave legacy DC/surrogate helpers explicitly separate. |
| `gds/spice_transient.py` | PR 03: execution lifecycle; PR 04: parsing/validation/heat result. These require a small interface boundary or a stacked PR with the consumer added later. |
| `mesh/gds_notebook.py` | PR 06: regions/context; PR 07: meshing/bundles; PR 13: previews. Generic non-SRAM code is not automatically part of each PR. |
| `mesh/sram.py` | PR 06: SRAM configuration/source semantics; PR 07: sealing/generation; PR 08: import/build wrapper. |
| `post/budget.py` | PR 09: independent source checks; PR 11: PDE/boundary budget. |
| `spec/chip.py` | PR 10: BC fields and validation. Keep unrelated synthetic `default_chip()` changes out. |
| Thermal drivers | PR 12: integration; only necessary prerequisite API definitions should move earlier. |
| Notebook | PR 12: workflow/configuration/handoff; geometry and numerical implementation stay in reviewed Python modules. |

Each PR should contain its own tests and relevant contract documentation, not defer all tests until PR 13. If an early PR cannot import because a later helper is in the same module, move the minimal behavior-preserving interface into the earlier PR or adjust the stack. Do not land fake source fields or silent fallback solvers to make intermediate commits pass.

### 5.2 Recommended merge strategy

1. Review and land the identity foundation.
2. Review the electrical chain (02–05) and geometry chain (06–08) independently against that foundation.
3. Join them in source projection (09), then boundary/solver integration (10–11).
4. Land public CLI/notebook integration (12) and supported visualization/evidence (13).
5. Submit the accuracy changes below as distinct behavior-changing PRs with new comparison evidence.

This is a review/merge plan, not permission to remove the old paths. Decide whether legacy modes remain supported, are explicitly deprecated, or are removed in a separately approved cleanup. A review of the modern SRAM flow should not quietly certify old array/gallery outputs.

### 5.3 Required body for every formal PR

```markdown
## Purpose
What physical/computational concept this PR owns; link to its PR packet.

## Base and dependencies
Parent PRs/commits and the exact comparison baseline.

## Included code
Files AND functions/hunks; identify shared infrastructure changes.

## Inputs, outputs, and units
Schemas/IDs, validation rules, artifacts, and unchanged caller contracts.

## Behavior change
Packaging/refactor only, numerical-method change, or scientific-model change?
State which powers, temperatures, hashes, and output schemas should change.

## Verification
Commands, dependency versions, tests run/skipped, numerical tolerances,
independent analytical checks, and artifact locations.

## Physical limitations
What this PR does not model or validate.

## Compatibility and migration
Old caller behavior, cache/version invalidation, output-directory policy.

## Acceptance checklist
Copy and resolve the stage-specific criteria from the review guide.
```

Keep generated multi-megabyte waveform/network/field files out of ordinary code diffs unless the repository deliberately chooses versioned fixtures. Prefer small deterministic test inputs plus an artifact manifest for full-run evidence. Do not commit `.deps/` binaries, machine-specific caches, or unrelated notebook outputs as implementation changes.

## 6. Accuracy follow-ups: separate from packaging existing code

The 13 PRs above establish the current implementation and review boundaries. They do not fix all its physical limitations. The following are **proposed future changes**, not functions or modes already present:

| Follow-up | Proposed PR purpose | What must be demonstrated |
|---|---|---|
| F01 — Complete repeating workload | Model READ recharge/access/idle, state-consistent WRITE sequences, mixed activity, and holding loss outside access windows. Depends on 03–05. | Nonzero standby heat where appropriate; no double counting of hold power already included in an event; periodic electrical-state/energy closure; local versus external losses separated. |
| F02 — Target-macro electrical loading/timing | Replace borrowed timing and estimated capacitance with justified target-SRAM inputs or extracted RC; improve shared-wire port/loading representation. Depends on 01–04. | Parameter provenance, both-state functionality, extraction/step convergence, sensitivity of event energy and rail loss. No universal expected increase/decrease should be asserted in advance. |
| F03 — Cooling/domain and array repeat unit | Establish package/backside resistance, substrate truncation, actual mirrored repeat geometry and activity distribution. Depends on 06–08, 10–12. | Domain/boundary sensitivity, consistent per-area cooling allocation, correct repeating unit or macro-to-cell boundary handoff. Required user/process/package information must be explicit. |
| F04 — Electrothermal consistency | Iterate electrical power/resistance and thermal material properties at temperature. Depends on 03–05, 08–12. | Converged temperature/power fixed point, state/functionality checks at the operating temperature, bounded failure handling. Existence of `Material.k_at()` alone is not implementation of this coupling. |
| F05 — Local process/heat-path fidelity | Add justified gate-oxide/interface resistance, calibrated layer properties, and assess the assumed channel deposition depth/contact geometry. Depends on 06–09, 11. | Mesh/interface/source conservation plus sensitivity of local channel/wire temperatures; no unsupported claim of foundry-accurate 3D reconstruction. |
| F06 — Numerical accuracy and positivity | Establish thermal spatial/time/domain convergence; address transient sub-ambient undershoot without hiding it. Depends on 07–11. | Refinement tables for peaks and selected local temperatures; nonnegative-heating/simple-domain tests; transient energy storage balance. Clamping colors or temperature arrays is not a valid fix. |

For the present steady-state objective, F01/F02 establish the imposed watts, F03 establishes the dominant heat-removal path, and F04 makes a high-temperature prediction self-consistent. Local hotspot detail additionally needs F05/F06. These may be separate PRs even when the same module is touched; scientific changes should not be buried inside code-movement PRs.

## 7. Evidence, commands, and reproducibility requirements

### 7.1 Useful existing evidence, with its limits

The saved layout-resistor READ steady run is [out/spice_interconnect_integration/read_steady](out/spice_interconnect_integration/read_steady). It uses the notebook mesh with 444,518 tetrahedra and 81,736 nodes. Its recorded average power is approximately 4.519594 µW, including 0.105798 µW of local resistor heat, and its peak is 420.176176 K. Its steady heat-balance relative error is about `2.5e-10`.

This is an **implementation reproduction baseline**, not a silicon-temperature target. Roughly 119.19 K of the 120.18 K peak rise comes from `P/(hA)` at the assumed backside. Model changes should legitimately change the result. Matching 420.176 K must never be used to tune a supposedly more accurate model.

Other evidence directories include `out/spice_interconnect_integration/read_pulse`, `write_pulse`, and `out/spice_interconnect_validation`. They are generated artifacts and may not exist in another checkout. A prior broad test run reported 523 passes while excluding `tests/test_gds_pipeline.py`; that is historical evidence, not an unqualified “entire repository passes” claim. The immediately preceding focused review reran 102 workload/periodic/energy tests successfully. This document does not turn either count into a physical-accuracy certification.

### 7.2 Reproduction commands

Run from `kelvin/` with the configured Python environment. On the current host, the tested interpreter is `/global/home/krishnabhattaram/Neural-Network-Materials/Electrostats/env/bin/python`; the local MPC/gdstk prefix may need `$PWD/.deps/dolfinx-mpc-0.10/python` on `PYTHONPATH`. Set `KELVIN_NGSPICE` only if discovery should use a specific executable. See [SRAM_NOTEBOOK_PIPELINE.md](SRAM_NOTEBOOK_PIPELINE.md) and [PERIODIC_BOUNDARIES.md](PERIODIC_BOUNDARIES.md) for environment details.

Every destination below must be new or empty where required; choose another review identifier for reruns.

```bash
# Geometry only; this may be the expensive mesh-generation step.
python cases/build_sram_mesh.py --out-dir out/review_sram_mesh

# Electrical operation/energy checks; no thermal solve.
python cases/characterize_spice_access.py \
  --operation all --initial-state both --interconnect layout \
  --check-convergence --no-plots --out-dir out/review_sram_electrical

# Steady event-average temperature, using the saved mesh unchanged.
python cases/run_bitcell_compact.py \
  --point read_1 --power-model spice-transient --interconnect layout \
  --mesh-manifest out/review_sram_mesh/sram_sp_cell_material_regions.json \
  --period-ns 3.86681 --row-hit-rate 1.0 \
  --no-renders --out-dir out/review_sram_read_steady

# One time-resolved access, for checking the transfer contract separately.
python cases/run_bitcell_transient.py \
  --point read_1 --instantaneous --interconnect layout \
  --mesh-manifest out/review_sram_mesh/sram_sp_cell_material_regions.json \
  --spice-step-ps 1 --thermal-step-ps 5 \
  --no-renders --out-dir out/review_sram_read_pulse
```

The explicit period above reproduces the current assumption; it does not validate that frequency for the target SRAM. `--interconnect none` changes the electrical network and therefore transistor powers too. Comparing its temperature directly with `layout` is **not** a pure wire-heat ablation. To isolate wire-generated warming, retain the same electrical traces and selectively zero one thermal source contribution in a separately labeled experiment.

### 7.3 Test layers and what they establish

| Test layer | Examples | Establishes | Does not establish |
|---|---|---|---|
| Pure input/geometry/unit tests | Mapping, deck text, contact polygons, analytic sheet resistance | IDs, units, connectivity and local algorithms | Target-chip electrical accuracy |
| Real ngspice integration | Completed READ/WRITE and step refinement | Functional behavior and convergence of the specified circuit | Correct extracted macro loading or complete-cycle power |
| Mesh/import/projector tests | Region ownership, hashes, volume/integral checks | Spatial/unit consistency and conservative source transfer | Resolved thermal hotspot gradients |
| PDE/constraint tests | Analytical/manufactured cases, periodic tests, heat balance | Correct solution of the specified discretized problem | Correct package/cooling assumptions |
| Coupled saved-mesh smoke tests | READ steady plus one READ/WRITE pulse | End-to-end interfaces, evidence and failure handling | Silicon calibration or thermal mesh convergence |
| Scientific validation studies | Thermal mesh/domain/time studies and calibrated inputs | Accuracy within their demonstrated scope | Universal validity across another process/workload |

Do not drop a test merely because it passes with mocks; do not treat a mocked test as an end-to-end solve either. PR bodies must list actual executed tests, skips, external requirements and generated evidence. In particular, saved-mesh tests need a valid bundle and electrical integration tests need ngspice/model files.

## 8. Other SRAM code: explicit routing, not silent equivalence

These files are relevant to a repository-wide SRAM review but are **not automatically part of the modern SPICE → notebook-mesh pipeline above**. Some file headers describe earlier behavior and should not override the actual call graph.

| Existing route/files | Implementation to inspect | Status relative to this PR series |
|---|---|---|
| [cases/run_bitcell_gallery.py](cases/run_bitcell_gallery.py), [cases/fig_bitcell_operating_points.py](cases/fig_bitcell_operating_points.py) | `build_bitcell_geometry()`, `solve_operating_point()` / `solve_point()`; `named_bias_point_power_w()` and old GDS mesher | Legacy DC/surrogate operating-point figures, not real transient-event averages by default. Keep separately labeled. |
| [cases/run_bitcell_access_transient.py](cases/run_bitcell_access_transient.py), [cases/fig_bitcell_access_layers.py](cases/fig_bitcell_access_layers.py) | `main()`; bias-point powers applied as prescribed access pulses | Historical access visualization, not a substitute for `run_bitcell_transient.py --instantaneous`. |
| [cases/run_sram_array_transient.py](cases/run_sram_array_transient.py), [cases/run_array_full.py](cases/run_array_full.py), [gds/tile_array.py](gds/tile_array.py) | `main()`, `tile_array()`, `_transform_points()`; named per-cell sources carried through mirroring | Explicit tiled-array studies still use older electrical/geometry paths. Their array geometry is not implicitly used by the periodic single-cell solve. |
| [cases/fig_array_transient_layers.py](cases/fig_array_transient_layers.py), [cases/fig_sram_array_transient.py](cases/fig_sram_array_transient.py) | Array-history renderers | Must inherit the validity and scale caveats of the array producer, not those of the new single-cell run. |
| [cases/run_bitcell_isolation.py](cases/run_bitcell_isolation.py), [cases/run_bitcell_heff_sensitivity.py](cases/run_bitcell_heff_sensitivity.py) | `_worker()` / `main()`; geometry/cooling parameter studies | Useful historical sensitivity tools; verify source-model and mesh choices before reusing results. |
| [cases/run_gds_2d_solve.py](cases/run_gds_2d_solve.py), [cases/run_gds_convergence.py](cases/run_gds_convergence.py) | 2D cutline meshing with homogenization depth | Different mathematical reduction. Its power normalization is not the 3D per-device contract. |
| [cases/run_gds_3d.py](cases/run_gds_3d.py), [mesh/gds_build.py](mesh/gds_build.py), [mesh/gds_volume.py](mesh/gds_volume.py) | `build_gds_3d_mesh()`, `emit_gds_3d_geometry()` | Older bounding-box/window route; also the explicit `--mesh-method legacy` path. |
| [cases/run_gds_coarse_solve.py](cases/run_gds_coarse_solve.py), [cases/render_sram_heatmap.py](cases/render_sram_heatmap.py), [gds/upscale.py](gds/upscale.py), [mesh/gds_coarse.py](mesh/gds_coarse.py) | Tile-grid effective material/source fields and coarse whole-die solve | Macro-scale approximation with different power inputs; not local layout-resistor heating. |
| [cases/run_gds_submodel.py](cases/run_gds_submodel.py) | `build_coarse_interpolator()`, `run_submodel()` | Coarse-to-local lateral Dirichlet handoff, not the present periodic bitcell BC. |
| [cases/build_sram_polygon_preview.py](cases/build_sram_polygon_preview.py), [cases/compare_sram_meshes.py](cases/compare_sram_meshes.py), [cases/visualize_gds_mesh.py](cases/visualize_gds_mesh.py) | Layer previews / old-new comparisons / mesh inspection | Visualization utilities; disconnected preview layer meshes are not the full solver-ready thermal domain. |

**Recommended disposition:** PR 12 should document/gate these route differences. Migrating the tiled-array or coarse-macro routes to the new electrical/mesh contract is a separate follow-up project, with its own workload and boundary validation. Do not include that migration accidentally just because those filenames contain “SRAM.” Conversely, do not omit them from a repository inventory and let reviewers assume all SRAM entry points use the same implementation.

## 9. Final review checklist

- [ ] Can every displayed hotspot be traced to a saved temperature field, mesh bundle, workload, electrical trace, and named physical source?
- [ ] Are all conversions between µm, m, W, µW, J and W/m³ explicit and tested?
- [ ] Does each transistor retain its own power through every layer of the pipeline?
- [ ] Are wire powers computed from branch voltages/resistance and mapped only to their physical supports?
- [ ] Are steady averaging, a single electrical access, and sustained-average thermal response distinguished?
- [ ] Are geometry/material assumptions separated from facts in the GDS and compact models?
- [ ] Are periodic repeat-unit/activity assumptions and the cooling-area interpretation stated?
- [ ] Are source conservation, PDE conservation, numerical convergence, and physical calibration treated as separate claims?
- [ ] Are incomplete/failed/stale artifacts rejected rather than rendered as successful runs?
- [ ] Does each proposed PR have a minimal, buildable diff with owned tests and no unrelated workspace changes?
- [ ] Are remaining scientific limitations and accuracy follow-ups visible in approval decisions?

The review goal is not simply “the solver runs” or “the plot looks plausible.” It is that each conversion—from transistor experiment to resistor/device heat, from layout to material domain, and from watts to solved temperature—has a small, explicit, independently reviewable contract.
