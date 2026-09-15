# Correctness audit: thermal solve + SPICE→FEM self-consistency

One-day audit before the demo, covering the two claims that need to survive scrutiny:
**(1) the absolute temperatures** and **(2) the self-consistency** between real SPICE
compact-model power and the FEM thermal solve. Every number below is either verified
against an independent check (with the check shown), or explicitly labeled a stated
assumption with its sensitivity — nothing is presented as a bare number.

## Headline result

**The solver arithmetic is right. The boundary condition it was asked to solve needs to
be stated explicitly, and now is.**

| Check | Result |
|---|---|
| FEM vs. independent 1D analytic resistance model | 31.4402 K vs 31.4418 K (0.005% agreement) |
| SPICE→FEM power delivery (`∫q dx` vs. intended power) | exact, 1.000000, for every one of 24 tags checked (8 gallery cases × up to 16 source tags) |
| Electrical energy conservation (Tellegen's theorem, all ideal sources) | 1.0000 for READ and CROWBAR (the two headline points) |
| 3D mesh convergence (91k → 749k cells, 8x range) | dT varies 0.003% |
| Max-principle check (61 saved transient fields) | zero violations; min T = 300.000000 K exactly |
| Published-literature benchmark (Oprins/ITherm BSPDN ratios) | see below |

## 1. The absolute-temperature claim depends entirely on one boundary condition

The FEM is exact given its inputs — an independent 1D series-resistance calculation
(`R = t/(k·A) + 1/(h·A)`) reproduces the crowbar dT to 0.005%. But **99.33% of the total
thermal resistance is the Robin boundary condition** `1/(h·A)` over the cell's own bare
1.896 µm² footprint; only 0.67% is conduction through 50 µm of silicon. All four lateral
mesh faces are adiabatic — physically, this solves "every cell in the array is equally
hot" (array-periodic), not "one hot cell in a quiet array" (isolated).

Both are legitimate solves of *different physical questions*. We now have real solver
numbers for both, not just an analytic estimate:

| Configuration | Lateral padding | dT (crowbar) | Mesh cells |
|---|---|---|---|
| As-built (array-periodic) | 0 µm | **31.456 K** | 346,785 |
| Isolated, modest guard ring | 1 µm | 5.219 K | 631,296 |
| Isolated, modest guard ring | 2 µm | 2.070 K | 934,490 |
| Isolated, wider guard ring | 5 µm | 0.475 K | 2,612,155 |
| Isolated, analytic (semi-infinite spreading) | ∞ | ~0.0026 K | — |

The trend is monotonic and consistent with the analytic bound as padding increases —
confirming F2 is real, not a modeling artifact. **For the demo: report the array-periodic
number (31.5K) as "this row hammered continuously, every other row idle" and be explicit
that it is NOT a single isolated cell's temperature** — that number is under 1K.

### h_eff sensitivity (since it sets 99.3% of the as-built answer)

| h_eff (W/m²/K) | dT (K) |
|---|---|
| 2,000 | 312.52 |
| 5,000 | 125.14 |
| 10,000 | 62.69 |
| **20,000 (stated value)** | **31.46** |
| 40,000 | 15.84 |
| 80,000 | 8.03 |
| 160,000 | 4.13 |

Near-exact 1/h scaling, as expected from F2. `backside_h_eff=20000 W/m²/K` is a *stated
assumption* (lumped TIM + spreader + sink), not a prediction — present it with this
table, not as a bare number. Figure: `out/bitcell_heff_sensitivity/dT_vs_heff.png`.

## 2. Self-consistency: SPICE power → FEM source term

`post/budget.py::power_balance` (new — factors out what were three independent
copy-pasted `assemble_scalar` blocks) checks two things the prior tests never did:

- **`∫q dx` vs. intended `sum(SourceBox.power_uw)`**: exact (ratio 1.000000) for all 4
  operating points × 2 access scenarios × up to 16 source tags each. The prior energy-
  conservation tests only checked `∫q dx == ∫h(T−T_amb)ds`, which holds by construction
  for *any* q — this is the check that was actually missing.
- **Per-tag meshed volume vs. nominal box volume**: every single tag (access, latch,
  pullup, parasitic, all 8 contacts) matches 1.000000. The real GDS channel geometry is
  meshed at exactly the volume `physics/coeffs.py` assumed when computing `q`.
- **Transient path checked too**: `p_gen_vs_intended` = 1.000000 exactly at the end of the
  active phase. The separate Robin-identity check (`p_gen == p_out_robin`) is a
  STEADY-STATE identity and does not apply exactly to a transient step still approaching
  equilibrium — confirmed directly: it shows a small (0.046%) residual there, which is
  real physics (some injected power is still raising stored thermal energy, not yet
  leaving through the boundary), not a bug. `post/budget.py` now documents this
  distinction explicitly (`check_robin=False` for non-equilibrium steps).

## 3. Electrical-side energy conservation (new)

No check previously existed for whether the 8 per-device SPICE powers actually balance
against the circuit's own sources. First attempt (checking against `V_DD·I_VDD` alone)
gave nonsensical ratios (47x, 3,000,000x) — a bug in the *check*, not the physics: BL/BR/
WL are separately-driven ideal sources in this single-cell testbench, and the dominant
READ current path (bitline → access transistor → internal node → pulldown → VSS) never
touches VDD at all. Fixed to sum power over **every** ideal source (Tellegen's theorem):

| Operating point | channel only | + bulk junctions | sum(all sources) | ratio (channel) | ratio (full) |
|---|---|---|---|---|---|
| **READ** | 132.0028 µW | 132.0028 µW | 132.0027 µW | 1.0000 | **1.0000** |
| **CROWBAR** | 19.1178 µW | 19.1178 µW | 19.1179 µW | 1.0000 | **1.0000** |
| HOLD (leakage) | 0.42 pW | 23.23 pW | 23.23 pW | 0.018 | **1.0000** |
| WRITE (settled) | 0.046 pW | 19.59 pW | 19.59 pW | 0.002 | **1.0000** |

**The HOLD/WRITE shortfall was diagnosed and closed, not written off.** The original
suspicion (that `P=|Id·Vds|` neglects junction leakage) was correct and is now *measured*:
BSIM3 exposes the bulk-junction currents as `ibd`/`ibs`, and adding
`|I_bd·V_db| + |I_bs·V_sb|` closes every point to **1.0000**. At an ON point the channel
dominates by ~7 orders so the omission is invisible; in HOLD the channel current is itself
leakage-scale and the reverse-biased S/D–bulk diodes carry **98%** of the dissipation.
(BSIM3 in this ngspice build exposes no `ig`/`is`/`ib` instance vectors — checked directly,
"Error: no such parameter ig" — so `ibd`/`ibs` are the available junction terms, and they
are sufficient.)

The junction term is reported (`gds.spice_power.device_power_breakdown_w`) but deliberately
**not** folded into the power that drives the thermal solve: that power is mapped onto
*channel* geometry, whereas junction heat belongs at the S/D diffusions. Folding it in would
move heat to the wrong place to correct an error 8 orders of magnitude below the crowbar
power the thermal results actually use (4.2e-13 W vs 1.9e-5 W). No reported temperature
changes.

Independent confirmation that the *full* accounting is the right one: the vendor's own
Liberty file gives **33.39 pW** of static leakage per bit (547 nW over 2048×8 bits). The
channel-only number (0.42 pW) is 80× below that; the full number (23.23 pW) is within
**1.44×** — and on the correct side, since the macro total also carries periphery.

### Read energy is a weak test, and is reported as such

The same Liberty file gives 13.18 pJ per read access. Comparing a single-cell model against
that as a *point* value is not meaningful, for two independent reasons, both now handled:

1. **The published number is itself ambiguous by ~2×.** Liberty splits internal energy
   between the vdd and vss rails and reports nearly equal values (6.99 vs 7.28 pJ); summing
   them vs treating them as two views of one energy is a convention choice (`PG_PIN_SUM`).
   Subtracting the `ce=0` clock/control baseline, the array-access energy is **5.86–11.67 pJ**.
2. **The array shape was being read wrongly.** 2048×8 is the *logical* shape. The macro name
   encodes an 8:1 column mux, so the physical array is **256 rows × 64 columns** — a read
   asserts one word line across all 64 columns, and each bit line carries 256 cells, not
   2048. (Rule checked against the other macro in `data/`: it predicts 16×88, exactly what
   `gds/spice_netlist.py` counted from that macro's real netlist.)

The bit-line charge model gives **1.5–11.8 pJ** across plausible bit-line loading
(0.5–2 fF/row) and sense margin (100–200 mV). The two ranges overlap, so the comparison is
*consistent* — but it is a bracket, not a sharp test, and it only covers bit lines, not
decoders, sense amps or output drivers. `fig5c` plots it as overlapping bands for that
reason. Leakage is the sharp check; access energy is a sanity bound.

Symmetric-pair assumption (`X0==X2`, `X1==X7`, `X5==X6` at the crowbar bias) — previously
asserted only in a code comment — is now directly verified: exact equality, and `X3==X4`
both zero as expected (parasitic devices, Vds=0 by construction).

## 4. Mesh convergence (new — no 3D sweep existed before this audit)

`cases/run_gds_convergence.py` only ever swept the 2D cross-section path. The bitcell's
actual 3D mesh — the one every headline number comes from — had never been shown
grid-independent, despite the 50 µm substrate carrying visibly few cells (2,423) relative
to the 10 nm channel layer (89,781).

| refine | cells | dT (K) |
|---|---|---|
| 2.0 (coarsest) | 91,619 | 31.4555 |
| 1.4 | 176,143 | 31.4561 |
| 1.0 (default) | 346,785 | 31.4564 |
| 0.7 (finest) | 748,829 | 31.4563 |

0.003% variation over an 8x cell-count range — converged, and converged already at the
coarsest level tested.

## 5. A bug the audit caught in itself

Running `post/budget.py` against the existing 2D-homogenized tests (not just the new 3D
bitcell path) surfaced a real bug — in the new audit code, not the solver. On a 2D
cross-section, `∫q dx` integrates a W/m³ density over an AREA, giving **watts per metre**
of the unmodeled depth, not real watts. The first version compared that raw number
directly against `sum(power_uw)` in real watts and printed a "333,333x failure" — purely a
missing `×source_depth_m` unit conversion in the check, not the solver: the underlying
`p_gen == p_out_robin` Galerkin identity held throughout (both sides in W/m), which is
exactly why it never showed up before. The same units mismatch also broke
`test_energy_conservation_row0`'s secondary flux-reconstruction diagnostic once its
reference `p_gen` became real-watt-scaled; fixed there too (same `×source_depth_m`
factor). Fixed and reverified — full suite green (16 passed) after; the 3D bitcell
numbers above were unaffected throughout (they all use `source_depth_m=None`, where the
conversion factor is 1). This is the audit process working as intended — new checks get
checked too.

## 6. Other checks

- **Max-principle**: every one of 61 saved transient temperature fields (crowbar
  active→idle) satisfies T ≥ 300.000000 K exactly and Tmax stays far below Si's melting
  point (1687 K) — no sub-ambient numerical artifact of the kind seen previously on the
  structured-hex coarse-mesh path.
- **k(T) linearization**: the solve uses `k` at a fixed 300 K reference throughout;
  `Material.k_at(T)` exists but is never called in the solve path. Quantified impact: at
  Tmax=331.5K, k is ~12% below its 300K value, but since conduction is only 0.67% of the
  as-built case's total resistance (§1), the effect on dT is ~0.09% (31.456 → ~31.486K) —
  negligible. Even smaller for the isolated-cell numbers, whose absolute temperatures sit
  closer to the 300K reference. Documented linearization, not a live bug.
- **Published-literature benchmark**: `cases/run_bspdn_benchmark.py` re-run to confirm no
  regression from this session's changes. Result: **matches the previously-documented
  values almost exactly** (B/A Si-thinning-alone 2.352x vs prior 2.36x; backside-metal
  reduction 5.4% vs prior 5.4%; dual-sided recovery 19.6% vs prior 19.6%; thin-film-k
  contribution 9.3% vs prior 9.2%) — no regression. Two of four ratios land within ~10% of
  Oprins/ITherm's published numbers; the other two are directionally correct but weaker,
  a limitation already documented in `SRAM_THERMAL_REPORT.md` §7.4 (this tool's 2D
  cross-section homogenizes one lateral axis, under-capturing a benefit that in the
  published study comes from discrete, individually-resolved 3D nTSV/BPR spreading) — a
  known, named, and unchanged gap, not a new finding.

## 7. Reproducing a published paper with that paper's own inputs

The strongest solver evidence available: feed [Oprins] iTherm 2022's **own published
inputs** into our solver and see whether it returns their published answer. This tests the
SOLVER (same inputs → same answer?), as distinct from testing our own material choices.
`cases/run_oprins_reproduction.py`, two-scale (100µm homogenized outer cell + 5µm discrete
window — mirroring their own package-cell + 5×5µm detail model, their Fig. 7).

| Metric | This solver | [Oprins] published | Error |
|---|---|---|---|
| **Net BS-PDN penalty vs FS-PDN** | **1.645x** | **1.6x (+60%)** | **+2.8%** |
| Si thinning alone (200µm → 500nm) | 1.952x | 2.2x | −11% |
| Backside metal's own reduction | 15.7% | 27% | −42% |

Their headline conclusion number — the net BS-PDN penalty — reproduces to **within 3%**.
For context, the same comparison on the previous 2D cross-section gave 2.226x against 1.6x
(+39% error).

### What it took (each a real defect, not tuning)

Read from the paper (Section III, Table 1, Fig. 9) rather than inferred:

1. **Backside metal was drawn as disconnected posts** in 3D — `_synthetic_grid_rects` made
   width×width islands for *every* SYNTHETIC band. A metal rail that isn't laterally
   continuous cannot spread heat, which is the entire mechanism being measured. Now
   `synthetic_shape="lines"` for rails, `"posts"` for vias.
2. **Configs weren't mesh-equivalent** (B: 4.33M cells vs C: 2.04M) purely because one had
   via features to grade around and the other didn't — making the ratio partly a resolution
   artifact. Fixed with an explicit `uniform_background` sizing flag.
3. **µTSV footprint** was a 250nm square (25% area); [Oprins] Table 1 specifies 180×250nm
   (18%, matching their stated 16.7% density). Non-square posts now supported.
4. **Material properties** were ours, not theirs: Si@500nm 107.6 vs their 84.0 W/m-K, Cu 400
   vs their 330. Their values come from dedicated Monte-Carlo BTE (their Fig. 9); ours from
   the Liu & Asheghi *measured* fit. Both are defensible, but reproducing their result
   requires their inputs — kept as separate `Si_oprins_500nm` / `Cu_oprins` entries rather
   than overwriting ours, since the 28% gap between a measured fit and a BTE value at 500nm
   is itself a result.
5. **Reference thickness** is 200µm Si in the paper, not our 50µm.

### Convergence of the method itself

- **Outer domain**: 100µm → their full 200µm cell moves the answer 1.952 → 1.962x (0.5%).
- **Window size**: converged for ≥4µm; below that the reported peak *equals* the hottest
  Dirichlet BC value, i.e. the window was measuring its own boundary condition.
- **Outer tile size**: halving it changes stage-1's peak 64.6 → 94.8K (that peak is a known
  artifact of binning a sub-tile source) but moves the fine answer by 0.01% — the BC sits in
  the smooth far field where the coarse model is reliable.
- **Energy**: delivered power equals intended power to 1.000000 in every run.

### The remaining gap, named

The backside metal's own benefit (15.7% vs 27%) is the weakest match. [Oprins] Table 1 also
includes **Ru buried power rails** (30nm wide, 180nm pitch) which the paper calls
instrumental in getting heat into the µTSVs, and which we do not faithfully model. Adding
them as a band made the match *worse* (net penalty 1.645x → 2.674x) because a band whose
background is SiO₂ puts a near-continuous oxide barrier under the source, whereas real BPRs
sit embedded laterally in STI with silicon continuing around them. Modelling that needs
lateral STI/diffusion structure this synthetic test case doesn't carry. Reported as a named,
quantified limitation — not tuned away (`--bpr` reproduces the experiment).

## Explicitly out of scope this session

The NEGF/electron-phonon physics-extension work is deliberately dropped — no new physics
until the existing numbers were trustworthy. Full mesh/parsing code review is in the
session transcript; this file covers only what changes the demo's credibility.
