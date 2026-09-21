# Layout-linked interconnect heating in the SRAM flow

The main notebook-mesh READ/WRITE drivers now include local interconnect and
contact heating by default. Transistor and resistor powers come from the
**same transient electrical solve**. There is no prescribed transistor/wire
power split, and no additional `CV²f` heat budget.

This is a reproducible, cell-specific **resistance extraction approximation**,
not foundry-qualified RC parasitic extraction or a full SRAM-array simulation.
The existing estimated 16 fF external capacitance per bitline remains unchanged.
No layout-derived interconnect capacitances have been added.

## Electrical model

`gds/interconnect.py` first rechecks the original GDS/netlist connectivity and
the verified X0–X7 correspondence. It constructs a clipped, two-dimensional
finite-volume sheet network for poly, LI1, M1 and connected M2. A branch has
`R = R_sheet * normal centroid separation / shared interface length`.
This preserves real polygon bends and branches; it is not a bounding-box wire
model. Clipped diagonal cells and terminal locations introduce discretization
error, so electrical grid refinement is an independent accuracy check.

Nominal values come from the public [SKY130 resistance table](https://skywater-pdk.readthedocs.io/en/main/rules/rcx.html#resistance-values):

| Element | Nominal resistance |
|---|---:|
| Poly | 48.2 Ω/□ |
| LI1 | 12.8 Ω/□ |
| M1 / M2 | 0.125 Ω/□ |
| LICON | 15 Ω per nominal 0.17 µm square cut |
| MCON | 152 Ω per nominal 0.17 µm square cut |
| VIA | 4.5 Ω per nominal 0.15 µm square cut |

The table's milliohm entries are converted to ohms. Contact resistance scales
inversely with the retained conducting core area: cut ∩ lower landing ∩ upper
landing. Contacts use an equipotential middle plane connected by area-weighted
parallel half-resistances. A half cut at the cell boundary is not treated as a
complete nominal cut. These public values are not separately calibrated for
the specialised SRAM process geometry.

External pins attach to conductor cross sections at GDS label coordinates.
The two locally disconnected WL stripes share the external WL according to the
existing netlist contract. Compact gate terminals attach at the channel-centre
cross section. Source/drain diffusion is still ideal locally; its resistance,
well/substrate resistance and additional compact-device series loss are not
extracted here. One floating M2 strip has no local electrical connection and
gets no Joule source; it remains a thermal conductor. The duplicate coincident
MCON cut is counted once.

`gds/spice_netlist.py` writes a new subcircuit without changing the bundled
original netlist or its mapping fingerprint. All eight compact-model instances
retain their identity. Their voltages are their **local** terminal voltages,
not the original ideal-net voltages. `gds/spice_transient.py` computes:

```text
P_channel,i(t) = intrinsic_id_i(t) * abs(local_Vd_i(t) - local_Vs_i(t))
P_resistor,r(t) = (Va_r(t) - Vb_r(t))² / R_r
```

The resistor current includes charging/discharging current. External driver,
precharge and sense-amplifier losses are not deposited in this bitcell. An
array electrical model is still needed for realistic distributed shared-line
loading and for losses elsewhere in the array. Electrical resistances and
compact models stay at their prescribed temperature; no temperature feedback
has been added.

The event starts with READ bitlines already precharged, and with WRITE target
bitlines already driven. Loss during the omitted READ recharge or preceding
WRITE driver setup is not included, even if some of it would occur locally.

## Mapping into the existing notebook mesh

The saved thermal mesh has conductor tags per layer/family, not per electrical
resistor. Assigning watts directly to a layer tag would incorrectly heat all
of that layer's disconnected conductors.

`physics/interconnect_sources.py` instead intersects each resistor's actual
polygon-prism heat support with tetrahedra of the correct conductor tag.
Nominal support volume must equal the summed intersection volume before the
mapping is accepted. Heat is projected conservatively into a reusable scalar
DG0 field. Sheet branch heat is split between its two adjacent sheet tiles;
contact heat is distributed within its conducting core and thermal height.
Geometry is precomputed once; each time step only updates source values.

The mapping reuses the notebook mesh without remeshing. It resolves thermal
cell-average heating, not gradients smaller than a tetrahedron. Finer electrical
tiles therefore do not substitute for thermal mesh convergence.

Every thermal step receives the piecewise-linear SPICE power integral divided
by that step's duration. The source audit accumulates the **actual solver load**,
then checks each transistor, conductor layer, total energy, and the complete
spatial source field against the electrical energies' projection. Several
resistors can overlap a thermal cell; the audit does not falsely claim to
recover each overlapping resistor independently from the summed field.

Steady and sustained-average sources use each channel/resistor event energy
times `row_hit_rate / period_s`. Cooldown has zero imposed source power.
The existing material properties, periodic lateral boundaries and backside
Robin cooling are unchanged.

## Run

From `kelvin/`, using the configured DOLFINx/ngspice environment:

```bash
python cases/run_bitcell_transient.py --point read_1 --instantaneous \
  --interconnect layout --interconnect-step-um 0.05 \
  --mesh-manifest out/sram_notebook_mesh/sram_sp_cell_material_regions.json \
  --out-dir out/my_read_with_wires --no-renders

python cases/run_bitcell_compact.py --point read_1 --interconnect layout \
  --mesh-manifest out/sram_notebook_mesh/sram_sp_cell_material_regions.json \
  --out-dir out/my_read_with_wires_steady --no-renders
```

Both thermal drivers select `layout` automatically for transient-SPICE power.
Use `--interconnect none` for the previous ideal-wire/channel-only model.
Layout heating requires the notebook mesh and does not silently fall back when
used with a legacy mesh or DC-surrogate power. Existing geometry-only bundles
remain valid. Always use a new output directory to preserve previous evidence.

`--interconnect-step-um` changes electrical segmentation (default 0.05 µm;
supported 0.0125–0.2 µm), `--spice-step-ps` changes electrical time resolution,
and `--thermal-step-ps` changes the thermal time resolution. These are separate
controls. Thermal spatial refinement requires a finer notebook mesh.

For electrical-only checks, `cases/characterize_spice_access.py` now defaults
to layout resistances and supports the same `--interconnect` controls. Its
`--check-convergence` tests time resolution, not electrical grid resolution.
The lower-level Python `run_access_transient` and `build_spice_workload` APIs
retain `interconnect="none"` for backward compatibility; pass `"layout"`
explicitly there.

## Saved evidence

Each layout run additionally archives:

- `spice/interconnect_network.json`: geometry, connectivity, resistances,
  thermal supports, assumptions and a content fingerprint;
- `spice/interconnect.spice`: the generated electrical subcircuit;
- `spice/interconnect_power.npz`: every resistor's power waveform in watts;
- `interconnect_projection.json`: nominal-versus-meshed support-volume checks;
- `thermal_source_steps.npz`: interval-averaged device and resistor powers;
- `source_energy_audit.json`: actual deposited energy and spatial audit.

The workload fingerprints generated artifacts, waveforms and the geometry
mapping; the renderer checks those archived files before using saved fields.
No old runs or GIFs are overwritten.

## Verification and remaining limits

Electrical validation outputs are in `out/spice_interconnect_validation/`;
coupled READ/WRITE pulse outputs are in `out/spice_interconnect_integration/`.

At 0.05 µm electrical grid size, 1 ps maximum SPICE step, and approximately
5 ps thermal steps on the unchanged 444,518-tetrahedron notebook mesh:

| Event | Channel energy | Wire/contact energy | Wire/contact fraction of local heat | Peak temperature |
|---|---:|---:|---:|---:|
| READ-1 | 17.067310 fJ | 0.409100 fJ | 2.34% | 301.560415 K at 0.28485 ns |
| WRITE-0→1 | 3.809843 fJ | 0.054133 fJ | 1.40% | 300.791956 K at 0.15492 ns |

Both actual deposited-energy audits pass. In READ, MCON accounts for about
0.326 fJ of the 0.409 fJ resistor heat; LI1 contributes 0.042 fJ and LICON
0.039 fJ. Adding resistance changes the transistor waveforms too: total heat
is not the old channel energy plus an independently assigned wire budget.
The earlier ideal-wire pulse peaks were 301.628154 K and 300.795684 K.

The averaged READ solve supplies 4.519594 µW and reaches 420.176176 K, with
relative steady heat-balance error 2.52e-10. This temperature remains strongly
dependent on applying backside `h=20,000 W/(m² K)` to one cell's footprint;
it is not a calibrated operating temperature for a packaged SRAM.

Refining the electrical sheet grid from 0.05 to 0.025 µm changes READ channel
energy by 0.0021% and wire/contact energy by 0.083%; WRITE wire/contact energy
changes by 0.177%. Three representative
bitline/storage-node path resistances change by less than 0.4% between 0.05
and 0.0125 µm. These checks address electrical discretization, not thermal
mesh convergence or uncertainty in the nominal resistance parameters.
The successful fine-grid electrical archives are `read_grid0.025_long/` and
`write_grid0.025_long/` under the validation directory. The earlier
`read_grid0.025/` retains a failed 60-second timeout attempt, not a successful
result. Layout runs now have a bounded 300-second simulator timeout; incomplete
waveforms still fail validation and never feed the thermal solver.

The 0.05 µm network has approximately 1,339 nodes, 2,452 resistor branches and
16 distinct contacts. Tests cover analytic rectangular sheet resistance,
polygon/connectivity preservation, contact conductance, invalid inputs,
functional READ/WRITE in both directions, conservative waveform integration,
geometric overlap, disconnected-wire isolation, and actual-load corruption.

Final regression: **523 tests passed**, using
`python -m pytest -q --ignore=tests/test_gds_pipeline.py`. The four pre-existing
legacy full-macro GDS tests are outside this run. The notebook JSON and
`git diff --check` also pass validation. Both new GIFs retain the original
GDS-left/isometric-right rendering and include the exact saved thermal peaks.

This extension does not fix the existing P1 consistent-mass sub-ambient
temperature undershoot, establish thermal mesh convergence, calibrate cooling,
extract full-array parasitic capacitance, or add electrothermal feedback.
