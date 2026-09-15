"""ngspice deck generator for the real sram_sp_cell bitcell (data/sram_sp_cell.spice
— the actual 8-device netlist extracted from sram22_sky130_macros, using real SKY130
special_nfet_pass/special_nfet_latch/special_pfet_pass BSIM3 models; see
data/sky130_fd_pr/ and PLAN.md's [[optimized-singing-creek]] plan Phase 0/1).

Three testbenches, all driven by real .lib-mined timing (gds.power.mine_access_timing),
not an assumed generic waveform:
  - HOLD: DC operating point, cell idle (WL=0, BL=BR=VDD precharged) -- static leakage.
  - WRITE: force a known initial state via .ic, pulse WL for the real min_pulse_width
    while driving BL/BR to the opposite state, verify the write actually took.
  - READ: force a known initial state, precharge BL=BR=VDD, pulse WL for the real
    min_pulse_width, no forced BL/BR drive -- the cell's own access+pull-down current
    discharges whichever bit line matches its stored '0' through a lumped bitline
    capacitance (see BITLINE_CAP_F below for why, and what it approximates).

Results are extracted via `wrdata` to a plain text file, not console `print`/
`.measure` output -- verified empirically that ngspice's `[time=X]` vector
indexing is not valid syntax here, and that console `.measure`/`print` output
is easy to misparse; `wrdata`'s file is unambiguous column data.
"""

import os

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_PDK_DIR = os.path.join(_DATA_DIR, "sky130_fd_pr")

_INCLUDES = [
    os.path.join(_PDK_DIR, "globals.spice"),
    os.path.join(_PDK_DIR, "specialized_cells_curated.spice"),
    os.path.join(_PDK_DIR, "cells/special_nfet_pass/sky130_fd_pr__special_nfet_pass__mismatch.corner.spice"),
    os.path.join(_PDK_DIR, "cells/special_nfet_latch/sky130_fd_pr__special_nfet_latch__mismatch.corner.spice"),
    os.path.join(_PDK_DIR, "cells/special_pfet_pass/sky130_fd_pr__special_pfet_pass__mismatch.corner.spice"),
    os.path.join(_PDK_DIR, "ngspice_fixed/special_nfet_pass.spice"),
    os.path.join(_PDK_DIR, "ngspice_fixed/special_nfet_latch.spice"),
    os.path.join(_PDK_DIR, "ngspice_fixed/special_pfet_pass.spice"),
    os.path.join(_PDK_DIR, "ngspice_fixed/special_pfet_pass_shortl.spice"),
    os.path.join(_DATA_DIR, "sram_sp_cell.spice"),
]

VDD = 1.8

# Real array dimensions for data/sram22_64x22m4w22.gds/.spice -- the macro this
# project's bitcell (data/sram_sp_cell.spice) is actually extracted from. Counted
# directly from the real netlist's net names, not assumed:
#   grep -o 'wl\[[0-9]*\]' data/sram22_64x22m4w22.spice | sort -u | wc -l  -> 16
#   grep -o 'bl\[[0-9]*\]' data/sram22_64x22m4w22.spice | sort -u | wc -l  -> 88
# A previous version of BITLINE_CAP_F hardcoded 2048 -- the row count of a
# DIFFERENT macro (sram22_2048x8m8w1) this project also uses elsewhere for its
# validated Liberty numbers, copied in by mistake. 128x too large for this one.
N_ROWS = 16
N_COLS = 88

# Real bitline capacitance isn't meaningful for an isolated single-cell testbench --
# a real access discharges the FULL shared bitline (this row's cell plus every other
# row's access-transistor junction capacitance hanging off the same line, plus real
# metal capacitance), not just this one cell's own terminals. Approximated here as
# ~1fF/cell (a commonly cited order-of-magnitude figure for advanced-node SRAM bitline
# loading per row) times this macro's REAL row count (N_ROWS, above) -- a labeled
# ESTIMATE, not a measurement, but now at least the right array's row count.
BITLINE_CAP_F = 1e-15 * N_ROWS


def _preamble():
    return [f'.include "{p}"' for p in _INCLUDES] + [
        f"VVDD VDD 0 {VDD}",
        "VVSS VSS 0 0",
    ]


def write_hold_deck(out_path: str, stored_one: bool = True) -> str:
    """DC operating point, cell idle -- static leakage power.

    Needs a .nodeset bias toward one real stable corner: verified directly
    that an UNBIASED .op on this symmetric cross-coupled latch converges to
    Q=QB=0.731V, the latch's symmetric METASTABLE point (not a real stored
    '0' or '1'), which is not what real HOLD leakage means.

    `.op` is a scalar result, not a time-series vector -- wrdata silently
    produces no file for it (verified empirically); use `print`, parsed as
    "name = value" text, same as this project's first standalone sanity
    check.
    """
    q0, qb0 = (VDD - 0.01, 0.01) if stored_one else (0.01, VDD - 0.01)
    lines = ["* HOLD: static leakage, cell idle"] + _preamble() + [
        "VWL WL 0 0",
        f"VBL BL 0 {VDD}",
        f"VBR BR 0 {VDD}",
        "X0 BL BR VDD VSS WL VSS VDD sram_sp_cell",
        f".nodeset v(x0.q)={q0} v(x0.qb)={qb0}",
        ".op",
        ".control",
        "run",
        "let ivdd = -i(vvdd)",
        "let pvdd = ivdd * vdd",
        "print pvdd",
        ".endc",
        ".end",
    ]
    _write(out_path, lines)
    return out_path


_EPS = 0.01  # nodeset guess kept slightly off-rail; exact-rail guesses can sit at a zero-gradient point


def write_write_deck(out_path: str, data_path: str, pulse_width_ns: float, write_one: bool = True) -> str:
    """Bias Q/QB toward the OPPOSITE of the target via .nodeset (a starting GUESS
    for the DC solve that precedes a non-uic .tran, not a hard force), pulse WL
    while driving BL/BR to the target, integrate VDD supply energy over the
    pulse, and record Q/QB so the caller can verify the write actually happened.

    Why .nodeset instead of .ic+uic: verified empirically that .ic+uic does NOT
    reliably set node voltages here (Q/QB have no explicit capacitor -- only
    implicit BSIM-internal device capacitance -- and .ic+uic left QB sitting
    above VDD regardless of instance naming or an added explicit capacitor).
    .nodeset biases the Newton iteration of the DC solve toward the intended
    stable corner instead of the cross-coupled latch's symmetric metastable
    point, and reliably converges there -- confirmed directly against a plain
    .op (Q=QB=0.731V, the symmetric point, with no bias at all)."""
    t0 = 0.1
    t_end = t0 + pulse_width_ns + 0.5
    bl, br = (VDD, 0.0) if write_one else (0.0, VDD)
    # nodeset guess is the OPPOSITE of the target -- the pulse must actually flip it
    q0, qb0 = (_EPS, VDD - _EPS) if write_one else (VDD - _EPS, _EPS)
    lines = [f"* WRITE {'1' if write_one else '0'}: pulse_width={pulse_width_ns}ns (real, .lib-mined)"] \
        + _preamble() + [
        # Small series resistors on the driven lines: an ideal (zero-impedance)
        # source fighting the cell's own cross-coupled state at t=0 is a
        # classic SPICE stiffness/non-convergence trigger ("timestep too
        # small" -- hit this empirically) once WL turns the access
        # transistors on; 1 ohm is electrically negligible but breaks it.
        f"VWL wl_src 0 PULSE(0 {VDD} {t0}n 50p 50p {pulse_width_ns}n {2*t_end}n)",
        "RWL wl_src WL 1",
        f"VBL bl_src 0 {bl}",
        "RBL bl_src BL 1",
        f"VBR br_src 0 {br}",
        "RBR br_src BR 1",
        "X0 BL BR VDD VSS WL VSS VDD sram_sp_cell",
        f".nodeset v(x0.q)={q0} v(x0.qb)={qb0}",
        f".tran 1p {t_end}n",
        ".control",
        "run",
        "let ivdd = -i(vvdd)",
        "let pvdd = ivdd * vdd",
        "let evdd = integ(pvdd)",
        f'wrdata {data_path} evdd v(x0.q) v(x0.qb)',
        ".endc",
        ".end",
    ]
    _write(out_path, lines)
    return out_path


def write_read_deck(out_path: str, data_path: str, pulse_width_ns: float, stored_one: bool = True) -> str:
    """Bias Q/QB toward `stored_one` via .nodeset (see write_write_deck's
    docstring for why nodeset, not .ic+uic), precharge BL=BR=VDD (standard),
    pulse WL for the real access pulse width, no forced BL/BR drive -- only a
    lumped bitline load capacitance (BITLINE_CAP_F) on each line, so the
    cell's own access current can discharge whichever side matches its
    stored '0'."""
    t0 = 0.1
    t_end = t0 + pulse_width_ns + 0.5
    q0, qb0 = (VDD - _EPS, _EPS) if stored_one else (_EPS, VDD - _EPS)
    lines = [f"* READ (stored={'1' if stored_one else '0'}): pulse_width={pulse_width_ns}ns (real, .lib-mined)"] \
        + _preamble() + [
        f"VWL wl_src 0 PULSE(0 {VDD} {t0}n 50p 50p {pulse_width_ns}n {2*t_end}n)",
        "RWL wl_src WL 1",
        f"CBL BL 0 {BITLINE_CAP_F}",
        f"CBR BR 0 {BITLINE_CAP_F}",
        f".nodeset v(bl)={VDD} v(br)={VDD} v(x0.q)={q0} v(x0.qb)={qb0}",
        "X0 BL BR VDD VSS WL VSS VDD sram_sp_cell",
        f".tran 1p {t_end}n",
        ".control",
        "run",
        "let ivdd = -i(vvdd)",
        "let pvdd = ivdd * vdd",
        "let evdd = integ(pvdd)",
        f'wrdata {data_path} evdd v(bl) v(br)',
        ".endc",
        ".end",
    ]
    _write(out_path, lines)
    return out_path


def write_bias_op_deck(out_path: str, wl: float, bl: float, br: float, q: float, qb: float) -> str:
    """DC-only (no transient) snapshot of the cell at an EXPLICITLY FORCED
    (WL, BL, BR, Q, QB) bias, on the variant with Q/QB exposed as extra ports
    (data/sram_sp_cell_exposed.spice). This deliberately is NOT a transient:
    no real circuit passes through an arbitrary (Q, QB) pair continuously,
    it's a forced snapshot of one instant -- exactly what a hotspot map
    needs, without the (separately unresolved, see optimized-singing-
    creek.md's Phase 1 notes) transient convergence issue on this circuit.

    Covers every operating point cases/run_bitcell_gallery.py uses:
      - HOLD:  wl=0,    bl=br=VDD,        q/qb = a real stored state
      - WRITE: wl=VDD,  bl/br = write target, q/qb = the SETTLED state
               (real currents here are tiny -- the access transistors have
               already finished charging/discharging by the settled state)
      - READ:  wl=VDD,  bl=br=VDD (precharged), q/qb = the stored state
      - CROWBAR: wl=bl=br=VDD, q=qb=0.9 -- the worst-case mid-switching
               instant, both pull-down NMOS partially on simultaneously
               (real currents symmetric when q==qb and bl==br, confirmed
               empirically: X0==X2, X1==X7, X5==X6 exactly)
    """
    exposed_path = os.path.join(_DATA_DIR, "sram_sp_cell_exposed.spice")
    includes = _INCLUDES[:-1] + [exposed_path]  # swap in the Q/QB-exposed variant
    lines = [f"* forced bias snapshot (DC op only, no transient): "
             f"wl={wl} bl={bl} br={br} q={q} qb={qb}"] \
        + [f'.include "{p}"' for p in includes] + [
        f"VVDD VDD 0 {VDD}",
        "VVSS VSS 0 0",
        f"VWL WL 0 {wl}",
        f"VBL BL 0 {bl}",
        f"VBR BR 0 {br}",
        f"VQ Q 0 {q}",
        f"VQB QB 0 {qb}",
        "X0 BL BR VDD VSS WL VSS VDD Q QB sram_sp_cell_exposed",
        ".op",
        ".control",
        "run",
        "print i(vvdd)",
        "print i(vvss)",
        "print i(vwl)",
        "print i(vbl)",
        "print i(vbr)",
        "print i(vq)",
        "print i(vqb)",
    ] + _device_print_lines() + [
        ".endc",
        ".end",
    ]
    _write(out_path, lines)
    return out_path


def write_selfconsistent_op_deck(out_path: str, wl: float, bl: float, br: float, q_guess: float, qb_guess: float) -> str:
    """DC-only snapshot for a REAL, physically-sustained state (HOLD, a
    settled WRITE, or READ) -- unlike write_bias_op_deck, Q/QB are NOT
    forced by ideal sources here; `.nodeset` only supplies a starting guess
    for the Newton iteration, so the solver finds whatever Q/QB the network
    actually implies (respecting KCL at those nodes), using the standard
    (non-exposed) sram_sp_cell subckt.

    This distinction is load-bearing, not stylistic: hard-forcing Q/QB via
    ideal sources while the access transistors are ALSO externally driven
    (WL+BL/BR asserted) creates an artificial current path between two
    competing ideal sources through the access transistor -- verified
    directly: a forced READ snapshot (wl=bl=br=VDD, q/qb forced to a stored
    state) gave 212uW through one access transistor alone, ~1e4x every
    other device, propagating into an 11562K "hotspot" once mapped onto
    geometry. The nodeset version lets that transistor settle to whatever
    tiny current a real self-consistent read actually draws.

    Only usable for bias points that describe a genuinely stable network
    state (a nodeset is a mere hint, not a lock) -- NOT for the crowbar
    snapshot, whose whole point is an intentionally-impossible instant no
    self-consistent solve would ever reach on its own; that one still needs
    write_bias_op_deck's hard forcing.
    """
    lines = [f"* self-consistent bias snapshot (DC op, nodeset-guided): "
             f"wl={wl} bl={bl} br={br}"] \
        + _preamble() + [
        f"VWL WL 0 {wl}",
        f"VBL BL 0 {bl}",
        f"VBR BR 0 {br}",
        "X0 BL BR VDD VSS WL VSS VDD sram_sp_cell",
        f".nodeset v(x0.q)={q_guess} v(x0.qb)={qb_guess}",
        ".op",
        ".control",
        "run",
        "print i(vvdd)",
        "print i(vvss)",
        "print i(vwl)",
        "print i(vbl)",
        "print i(vbr)",
    ] + _device_print_lines() + [
        "print v(x0.q)",
        "print v(x0.qb)",
        ".endc",
        ".end",
    ]
    _write(out_path, lines)
    return out_path


# (instance, ngspice model suffix) for the 8 devices in sram_sp_cell, in
# netlist order. X3/X4 are declared as *_shortl in data/sram_sp_cell.spice but
# ngspice reports them under the plain special_pfet_pass name (verified against
# real output, not assumed) -- gds/spice_power.py::_DEVICES relies on the same.
_DEVICE_MODELS = [
    ("x0", "nfet_pass"), ("x1", "nfet_latch"), ("x2", "nfet_pass"),
    ("x3", "pfet_pass"), ("x4", "pfet_pass"), ("x5", "pfet_pass"),
    ("x6", "pfet_pass"), ("x7", "nfet_latch"),
]


def _device_print_lines():
    """`id` plus the two bulk-junction currents for every device.

    `ibd`/`ibs` are NOT redundant with `id`: `id` is the CHANNEL current only,
    so `|id * vds|` misses the power dissipated in the reverse-biased
    drain-bulk and source-bulk junctions. At an ON operating point that
    omission is invisible (channel current dominates by ~7 orders), but in
    HOLD the channel current IS leakage-scale and the junctions carry ~98% of
    the dissipation -- a channel-only sum closes Tellegen's theorem to a ratio
    of 0.018, not 1.0. See gds/spice_power.py::device_power_breakdown_w.

    BSIM3 exposes no `ig`/`is`/`ib` instance vectors in this ngspice build
    (checked directly: "Error: no such parameter ig"), so `ibd`/`ibs` are the
    available junction terms and they are sufficient -- they close the balance
    to 1.0000 at every named bias point.
    """
    lines = []
    for inst, model in _DEVICE_MODELS:
        dev = f"@m.x0.{inst}.msky130_fd_pr__special_{model}"
        lines += [f"print {dev}[{term}]" for term in ("id", "ibd", "ibs")]
    return lines


def _write(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
