# SRAM transistor-to-heat-source mapping

Implemented and checked: **2026-09-16**.

## What changed

Kelvin now preserves the separate SPICE powers of **X0 through X7** when
assigning heat to the bundled `sram_sp_cell` layout. The former assignment
copied X0's power to both access devices, X1's to both pull-downs, and X5's to
both pull-ups. That happened to work for symmetric crowbar power, but corrupted
asymmetric READ/HOLD/WRITE sources and could change their total power.

The implementation is shared by the compact, gallery, general transient,
access-transient, sensitivity, operating-point figure and runtime cases.
Tiled-array cases map the original bitcell first, then carry each instance name
through translation/mirroring (`r0c0_X0`, etc.). They retain their existing
selected-row workload, pulse timing and boundary conditions.

No electrical bias, duty-factor formula, boundary condition, material, layer
thickness or bounding-box thermal geometry was changed in this repair.

## How the correspondence was established

[cases/verify_bitcell_mapping.py](cases/verify_bitcell_mapping.py) derives the
mapping from the **original polygon masks**, not the old thermal boxes:

1. Intersect diffusion and poly to find channels; subtract poly from diffusion
   to identify source/drain regions.
2. Trace connected poly, diffusion, LI1, M1 and M2 through overlapping LICON,
   MCON and via polygons.
3. Use the GDS's BL, BR, WL, supply and ground labels as anchors. BL identifies
   the Q-side access device and BR the QB-side device; cross-coupled gate
   connections distinguish the remaining functional transistors.
4. Match each channel's terminal nets, body/type and W/L to the extracted
   [bitcell netlist](data/sram_sp_cell.spice), allowing source/drain interchange
   for matching but retaining the netlist's original terminal ordering.

| Instance | Role | Channel center x, y (µm) | Gate | Drain / source |
| --- | --- | --- | --- | --- |
| X0 | Access | −0.260, −0.195 | WL | QB / BR |
| X1 | Pull-down | −0.295, −0.985 | QB | Q / VSS |
| X2 | Access | −0.260, −1.385 | WL | BL / Q |
| X3 | Extracted parasitic | −0.940, −1.3225 | WL | Q / Q |
| X4 | Extracted parasitic | −0.940, −0.2575 | WL | QB / QB |
| X5 | Pull-up | −0.940, −0.595 | Q | VDD / QB |
| X6 | Pull-up | −0.940, −0.985 | QB | Q / VDD |
| X7 | Pull-down | −0.295, −0.595 | Q | VSS / QB |

This is a **cell-specific, netlist-assisted connectivity verification, not a
general-purpose LVS extractor**. Its explicit contracts are:

- The two access-gate stripes are disconnected within the local cell drawing;
  the bundled netlist identifies both as logical WL. Only the bottom stripe has
  a local WL label. Their external stitching is not independently established
  by this local mask trace.
- Each short PMOS edge overlap touches one adjacent storage diffusion. Its
  extracted D=S interpretation comes from the bundled netlist, not two
  independently observed physical terminals. Both devices still receive zero
  power for the current electrical model.
- PMOS channels lie inside nwell; NMOS channels lie outside it in the implicit
  p-body. The small `(64,44)` shape is a VNB port marker, not a full pwell mask.

These assumptions are saved with the correspondence in
[data/sram_sp_cell_channel_map.json](data/sram_sp_cell_channel_map.json).

## Runtime safeguards and power accounting

[gds/bitcell_mapping.py](gds/bitcell_mapping.py) checks:

- The original GDS file digest, including label/port semantics.
- Relevant physical/pin polygon fingerprints, independent of enumeration,
  starting vertex and winding direction.
- Both bundled normal/exposed bitcell netlist digests.
- Exactly eight distinct instance-to-channel assignments, matching exact
  rectangular source footprints.
- A complete power dictionary with finite, nonnegative values for every X0–X7.

Changed or unsupported geometry is rejected instead of silently falling back
to class-based assignment. A changed cell needs a new connectivity verification;
do not merely replace the saved hashes to suppress the check.

Source tags now identify the instance (`src_X7`, for example). The new
`post/budget.py::verify_device_source_powers` compares, for every device:

```text
original electrical power → assigned source-box power → integral(q over tagged mesh)
```

The expected powers come from the original full electrical dictionary, not
from the already-mapped source list. The audit also rejects unexpected heat
outside those channel regions. Updated simulation entry points save the
results in `source_power_audit.json`.

## Verification results

- **24 new tests passed** in `tests/test_bitcell_mapping.py`: independent
  connectivity reconstruction; instance identities; asymmetric powers;
  stale geometry/netlist/GDS rejection; all four array mirror combinations;
  invalid inputs; and actual tetrahedral source integration.
- The mesh tests deliberately swap equal-volume devices' powers while keeping
  total heat unchanged. The per-device audit correctly rejects the swap.
- **3 cache-regression tests passed**, rejecting absent/stale transient mapping
  revisions while accepting current metadata.
- **30 existing tests passed**, covering specifications, analytical/energy
  checks, periodic boundaries and ngspice discovery. The four slow generic-GDS
  tests were not rerun for this mapping change.
- All four named ngspice DC points executed successfully and retained every
  instance's returned power. This is not a validation of electrical transients.

Fresh default-row-hit-rate power checks:

| Operating point | Old class-copy total | Correct full SPICE / assigned total |
| --- | ---: | ---: |
| Crowbar | 1.184202849 µW | 1.184202849 µW |
| READ stored 1 | 1.295365596 µW | 0.744799977 µW |
| HOLD stored 1 | 0.840660744 pW | 0.420330466 pW |
| Settled WRITE 1 | 0.091685170 pW | 0.045842616 pW |

In READ, X0 receives approximately 0.647683 µW and X7 0.0971172 µW. X7 is no
longer replaced by the nearly unpowered X1, and X0 is no longer duplicated into
X2. The correction restores the energy budget already computed upstream; it
does not introduce a new READ-energy model.

Fresh **periodic x/y** thermal solves on the existing legacy geometry gave:

| Case | Peak temperature | Relative source/backside imbalance |
| --- | ---: | ---: |
| Crowbar | 331.455670 K | 1.10 × 10⁻⁹ |
| READ stored 1 | 319.798474 K | 1.65 × 10⁻⁹ |

Both passed all per-instance meshed-power checks. Crowbar power is unchanged;
the small difference from the previous peak (about 0.000025 K) accompanies
remeshing after source-tag ordering changed. These are numerical checks of the
mapping repair, not calibrated physical temperature predictions.

Fresh meshes and source audits are saved under
[`out/instance_power_verification__sqt1e0n/`](out/instance_power_verification__sqt1e0n/).
The output directory is local/generated and is not a committed prerequisite.

## Reproduce and use

Use the compatible Python environment described in
[SRAM_SIMULATION_OVERVIEW.md](SRAM_SIMULATION_OVERVIEW.md), from `kelvin/`:

```bash
"$KELVIN_PYTHON" cases/verify_bitcell_mapping.py --check-manifest
"$KELVIN_PYTHON" -m pytest tests/test_bitcell_mapping.py -q
"$KELVIN_PYTHON" cases/run_bitcell_compact.py \
    --lateral-bc periodic --out-dir out/bitcell_compact_instances
"$KELVIN_PYTHON" cases/run_bitcell_transient.py \
    --point read_1 --lateral-bc periodic --out-dir out/bitcell_read_instances
```

Old asymmetric temperature fields and figures must be regenerated. The
operating-point NPZ cache and general transient field metadata now carry the
mapping revision; their renderers reject missing/stale revisions. Old outputs
were not deleted or overwritten by this verification.

The known bounding-box geometry error, new polygon-flow contact blocker,
assumed timing/cooling and omitted wire heat remain separate limitations.
This change does not make the new polygon previews solver-ready or turn the
bitcell workflow into a coupled electrothermal simulation.
