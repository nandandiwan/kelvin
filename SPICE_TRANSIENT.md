# SRAM SPICE transient repair

READ and WRITE now complete successfully for both stored states. The compact
steady and general transient thermal drivers now use these **per-transistor
SPICE channel-power waveforms by default**, with one shared event-energy model.
The default operation is READ-1, not the old forced crowbar snapshot.

The notebook-mesh drivers and electrical characterization CLI now also default
to local layout-derived resistor heating. See [INTERCONNECT_HEATING.md](INTERCONNECT_HEATING.md).
Use `--interconnect none` to reproduce the channel-only numbers below. The
low-level Python APIs retain ideal wires unless `interconnect="layout"` is passed.

## What was wrong

1. **Missing geometry units:** the extracted cell specifies `l=0.150`,
   `w=0.140` in micrometres. Without `.option scale=1u`, ngspice interpreted
   them as metres. DC currents could still look plausible because W/L was
   unchanged, but capacitances were enormous and transients aborted near t=0.
   Every HOLD, forced-DC, self-consistent-DC and transient deck now specifies
   the scale and 25 °C. Tests query actual L/W for all eight devices.
2. **READ had no precharge circuit:** `.nodeset` is an operating-point guess,
   not a capacitor initial condition. The repaired testbench precharges both
   bitlines through switches during the initial DC solution, releases them
   before WL rises, then lets their capacitors discharge through the cell.
3. **Incomplete data could look usable:** ngspice can leave a partial output
   file after failure. Kelvin now checks simulator diagnostics, column names,
   finite data, strictly increasing timestamps, completion time, dimensions,
   initial/final complementary states, WL activity and READ precharge/signal.

No compact-model parameter changes, artificial internal capacitors or relaxed
convergence tolerances were used. Old SPICE-derived DC powers and thermal
results must be regenerated; cached transient rendering now rejects metadata
predating the unit correction. Existing geometry-only meshes remain usable.

## Run it

From `kelvin/`, with NumPy and ngspice available:

```bash
python cases/characterize_spice_access.py \
  --check-convergence --out-dir out/my_spice_access
```

Use a new/empty output directory. This runs READ-0, READ-1, WRITE-0→1 and
WRITE-1→0 at 1 ps maximum timestep, then repeats at 0.5 and 0.25 ps. Use
`--operation read --initial-state 1` for just READ-1, `--pulse-width-ns` to
set the high plateau, or `--no-plots` to omit figures. The repository-local
`.deps/ngspice/bin/ngspice` is discovered automatically. On this host:

```bash
/global/home/krishnabhattaram/Neural-Network-Materials/Electrostats/env/bin/python \
  cases/characterize_spice_access.py --out-dir out/my_spice_access
```

Each run saves its deck, raw headered waveform, log, metadata, verification
summary, waveform figure, and `device_power.npz` (`time_s`, `X0`…`X7`, SI units).
The ngspice `wrdata` path must not contain whitespace or quotes. Failed runs
raise an error; a success summary is written only after validation. Existing
output directories are not overwritten.

Python interface: `gds.spice_transient.run_access_transient("read",
stored_one=True)`. `stored_one` is the **initial** state; WRITE targets its
opposite. `characterize_bitcell()` now uses these validated traces, not final-row
VDD-energy extraction. Its READ/WRITE energy fields mean integrated **channel
heat**, with supply/junction energies available separately in the reports.

## Heat extraction and its limits

For this ngspice BSIM3v3.2 implementation, `@device[id]` is the intrinsic
nonnegative channel-current magnitude, including reversed D/S operation. Thus:

`P_channel,i(t) = id_i(t) × |Vd_i(t) − Vs_i(t)|`.

The saved intrinsic current excludes displacement current. Kelvin does not
take the absolute value of total terminal power or label VDD energy as cell
heat. Body-junction loss and **signed** external-source energy are reported
separately. READ obtains much of its energy from initially charged bitline
capacitors. A complete energy balance must also include intrinsic stored
charge and driver/switch losses; source energy alone is not that balance.

The relevant primary references are the [ngspice manual, SCALE option §15.1.5](https://ngspice.sourceforge.io/docs/ngspice-42-manual.pdf),
the [BSIM3v3.2 current parameter mapping](https://github.com/ngspice/ngspice/blob/master/src/spicelib/devices/bsim3v32/b3v32.c),
and the [channel/charge calculation](https://github.com/ngspice/ngspice/blob/master/src/spicelib/devices/bsim3v32/b3v32ld.c).
Real-ngspice tests additionally check current signs for NFET/PMOS in both
drain/source orientations.

`average_device_power_w(t0, t1)` integrates piecewise-linear electrical samples
over exact interval boundaries. The thermal driver uses this energy-conserving
interface, avoiding narrow pulses being missed by point sampling.

## Run the coupled electrical → thermal pipeline

From `kelvin/`, in the DOLFINx/ngspice environment described in
[SRAM_SIMULATION_OVERVIEW.md](SRAM_SIMULATION_OVERVIEW.md#7-running-the-integrated-bitcell-model):

```bash
# One time-resolved READ access followed by zero-source cooldown.
python cases/run_bitcell_transient.py --point read_1 --instantaneous \
  --mesh-manifest out/sram_notebook_mesh/sram_sp_cell_material_regions.json \
  --out-dir out/my_read_pulse --no-renders

# Steady heating from repeated accesses with the SAME integrated event energy.
python cases/run_bitcell_compact.py --point read_1 --row-hit-rate 1 \
  --mesh-manifest out/sram_notebook_mesh/sram_sp_cell_material_regions.json \
  --out-dir out/my_read_steady --no-renders

# An actual switching WRITE, not a settled DC bias point.
python cases/run_bitcell_transient.py --point write_0_to_1 --instantaneous \
  --mesh-manifest out/sram_notebook_mesh/sram_sp_cell_material_regions.json \
  --out-dir out/my_write_pulse --no-renders
```

Supported accesses: `read_0`, `read_1`, `write_0_to_1`, `write_1_to_0`.
`--spice-step-ps` controls electrical sampling (default 1 ps);
`--thermal-step-ps` controls thermal resolution for a single access (default
5 ps maximum). The complete electrical observation window, including its
settling tail, is divided into equal thermal intervals. Changing the thermal
step changes temperature discretization but **does not change event energy**.

For every device and thermal interval, `P_step = integral(P_spice dt) / dt`,
then `q_step = P_step / channel_volume`. Frozen geometry and materials are
reused; only source values change. A separate DG0 field accumulates the exact
load consumed by the heat solver, and its per-device/whole-domain integrals
must reproduce the independent SPICE event energies.

Steady and sustained-average runs use `P_avg = E_device * row_hit_rate /
period_s`. `--period-ns` defaults to the borrowed 3.86681 ns period;
`--pulse-width-ns` sets the electrical WL high plateau. A period shorter than
the full electrical observation window is rejected. Row-hit rate scales
repetition frequency, not a single event's pulse amplitude. Holding leakage
outside the event and additional unmodeled accesses are not added.

Repeated directional WRITE averaging is a conditional event-rate model, not
a complete logical traffic sequence: the opposite transition/reset operation
must be included separately for a sustained toggling workload.

Each thermal run archives the electrical traces under `spice/`, plus
`power_workload.json`, material/source audits and temperature fields. Input
models, netlist, timing file and electrical artifacts are fingerprinted.
Transient rendering checks saved workload/artifact identity. The source-step
table is in `thermal_source_steps.npz`; deposited energy is checked in
`source_energy_audit.json`. Use new result directories; geometry-only meshes
can be reused without remeshing.

The legacy mode remains explicit:
`--power-model dc-surrogate --point crowbar` (or `hold_1`, `write_1_settled`,
`read_1`). Its optional pulse is a labeled rectangle, not an actual SPICE
access. Other array/gallery drivers retain their prior power models.

## Previous channel-only verification results

Saved reproducible outputs: `out/spice_transient_repair/`.

| Operation | Functional result | Channel heat at 1 ps max step | Energy change, 1 ps → 0.25 ps |
|---|---|---:|---:|
| READ-0 / READ-1 | Stored state retained; ≈0.781 V differential | 17.766945 fJ | 0.00162% |
| WRITE-0→1 / WRITE-1→0 | Both storage nodes switch and settle | 3.844647 fJ | 0.00665% |

Every run reaches 0.939519 ns. READ starts with both bitlines at approximately
1.8 V. Complementary directions have matching total energies and swapped
per-instance waveforms. The largest aggregate-peak change in the timestep
comparison is about 0.126%. These checks establish functioning testbenches
and temporal convergence of the specified model, not silicon calibration.

The broader regression run passed **427 tests**, excluding the four existing
full-macro legacy GDS tests. Focused SPICE tests include all deck families,
physical model dimensions/current conventions, malformed-output rejection,
functional accesses, timestep convergence and conservative energy integration.

### Coupled notebook-mesh verification

Full runs reuse `out/sram_notebook_mesh/sram_sp_cell_material_regions.json`
(444,518 tetrahedra), with periodic x/y, 300 K ambient, and the unchanged
backside Robin coefficient of 20,000 W/(m²·K). Results are preserved in
`out/spice_thermal_integration/`.

| Run | Applied heat | Peak temperature |
|---|---:|---:|
| `read_pulse` — ≤5 ps thermal step | 17.766945 fJ per access | 301.628154 K |
| `read_pulse_dt2p5` — ≤2.5 ps thermal step | Same 17.766945 fJ | 301.630428 K |
| `write_pulse` — 0→1, ≤5 ps thermal step | 3.844647 fJ per access | 300.795684 K |
| `read_steady` — full row-hit rate | 4.594729 µW | 422.178136 K |
| `write_steady` — 0→1 event-rate average | 0.994268 µW | 326.420151 K |

All eight deposited pulse-energy integrals match their independent SPICE
values to floating-point accuracy. Halving the READ thermal timestep changes
peak temperature rise by about 0.14%, without changing energy. Steady source
and backside heat balance agree to better than 10⁻⁹ relative error. These
are energy-transfer and limited temporal-resolution checks, not a spatial
mesh-convergence study.

The pulse runs also expose nonphysical minimum temperatures: 299.8511 K
(READ) and 299.8935 K (WRITE) at the default step. The existing consistent-mass
P1 scheme is not positivity-preserving; halving the timestep alone does not
remove the undershoot. The high full-activity steady READ temperature is
strongly dependent on one-cell-footprint cooling, and fixed-25 °C electrical
powers have no temperature feedback. Do not interpret these extrema as
calibrated device-temperature predictions.

Remaining assumptions:

- Isolated extracted eight-transistor cell, TT models, fixed 25 °C; no thermal feedback.
- Estimated 16 fF per bitline, 1 Ω WL/WRITE drivers, prescribed 50 ps edges.
- WL high plateau 0.239519 ns borrowed from another macro's Liberty file.
- READ has no sense-amplifier feedback or recharge phase. Its ≈0.781 V swing
  differs from the legacy 0.1 V charge-limited surrogate; their energies should
  not be compared as the same event.
- WRITE starts with target bitlines already driven. Preceding driver setup
  energy is outside the simulated event.
- The historical results above use ideal interconnects. Current default runs
  add local layout-derived resistance, but not extracted interconnect capacitance
  or diffusion areas/perimeters. Junction leakage
  includes ngspice's numerical `gmin` floor and is not a calibrated leakage model.

Thermal coupling is one-way: fixed-temperature electrical traces drive a
linear thermal model, with no SPICE temperature feedback. Existing cooling,
channel depth and material assumptions are unchanged. The consistent-mass P1
thermal discretization can produce small sub-ambient undershoots even with
nonnegative heat; the transient driver saves minimum temperature and warns
when this occurs. Energy conservation alone is not a proof of accurate local
temperature extrema or experimentally validated SRAM temperatures.
