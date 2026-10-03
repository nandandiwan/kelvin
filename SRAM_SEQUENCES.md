# Continuous SRAM sequences

`cases/run_sram_sequence.py` generates and runs one continuous ngspice transient
from a JSON sequence. It uses the bundled SKY130 SRAM transistor models, with
shared wordlines per row and shared bitlines, precharge and masked drivers per
column. Storage state is initialized only once, then carried through all
operations. No standard-cell power library is introduced.

```sh
python cases/run_sram_sequence.py data/sequence_2x2.json --out-dir out/sequence
python -m pytest -q tests/test_spice_sequence.py
```

Limit CPU affinity and numerical-library threads at launch. With the separate
`perf/transient-reuse` branch available, use its `cases/limited_run.py` launcher
with `--cores 4 --threads 1 --` before the command above. All concurrently
running jobs must fit the aggregate 32-core allocation.

The JSON requires a rectangular binary `initial_bits` array, explicit
`bitline_cap_f` (per bitline per column, not per cell), and `operations`.
Each operation has `start_ns`, `operation` (`read` or `write`), and zero-based
`row`. Writes additionally require one binary `data` and `mask` entry per
column. A mask bit of zero preserves that column's stored value. Operations
are at least 1 ns apart; the first starts no earlier than 0.1 ns. Array size
is deliberately limited to 256 cells for this reference implementation.

Timing is currently a fixed access template: precharge releases before WL
assertion, followed by write/read and recovery. This accepts arbitrary legal
*operation sequences*, not arbitrary pin waveforms or a CPU netlist. Direct
PWL/VCD inputs require a separate interface and functional checker.

## Evidence and scope

The checked-in 2x2 example exercises seven operations: reads, writes, masked
writes, and writing an already-held value. Each run checks initial state,
before/after state of every cell, read differential, precharge, wordline
activity, disturbance of unchanged cells, and physical transistor dimensions.
Truncated/nonfinite traces are rejected. Raw waveforms, deck/model hashes,
metadata, simulator log, per-device channel-power traces and integrated
energies are saved in a fresh output directory.

On ngspice 41, the example's channel energy is 1.7933890e-13 J at 300 K with
a 1 ps maximum electrical timestep and 1.7933606e-13 J at 0.5 ps: a relative
difference of 1.586e-5 (0.001586%). Both functional checks pass; at 305 K the
1 ps run also passes, with channel energy 1.7831316e-13 J. The tests check
total and individual-source timestep sensitivity, not just positive power.

These results are **not a multi-cell electrothermal validation**. Routing and
drivers are idealized; the example's 16 fF bitline load is an explicit estimate,
not extracted capacitance. Heat reported here is intrinsic channel
`Id * abs(Vds)` only, excluding driver/precharge, junction and wire losses.
Temperature is globally fixed during each complete SPICE run. Supply energy
must not be substituted for local dissipated heat.

Before closing phase 1, connect cell/device identities to physical tile source
volumes, add extracted/shared routing losses, preserve interval energy, and
verify temperature/power feedback, thermal time/mesh convergence and the
multi-cell heat budget. A bank of whole-sequence global-temperature runs is
only an approximation to evolving local transistor temperatures; validate it
against a temperature-aware electrical reference rather than labeling it
fully local self-consistency.
