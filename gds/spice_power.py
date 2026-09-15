"""Runs the gds.spice_netlist decks through ngspice and parses out per-operation
energy/power -- the real, first-principles counterpart to gds.power.parse_lib's
vendor-published Liberty numbers (see optimized-singing-creek.md's Phase 1 gate:
these should agree with the macro-level 14.98pJ/write, 13.18pJ/read to within an
order of magnitude before anything downstream is trusted).

Results come from `wrdata` files, not console `print`/`.measure` output --
verified empirically that ngspice's `[time=X]` vector indexing is not valid
syntax, and console print/measure formatting is easy to misparse; a wrdata
file's column data is unambiguous.
"""

import re
import subprocess
import tempfile

import numpy as np

from gds.power import mine_access_timing, parse_lib
from gds.spice_netlist import (
    BITLINE_CAP_F, N_ROWS, VDD, write_bias_op_deck, write_hold_deck, write_read_deck,
    write_selfconsistent_op_deck, write_write_deck,
)

LIB_PATH = "data/sram22_2048x8m8w1_tt_025C_1v80.lib"

# Real differential sense amps trip on a bitline swing well under VDD --
# ~50-150mV is typical for advanced-node SRAM (the amp is there specifically
# so the access can end, and the bitline droop stop, long before a full-VDD
# swing). 100mV is a labeled, stated assumption (this design's own sense-amp
# margin isn't in anything we have), used only to BOUND the charge a read can
# actually move -- see read_access_energy_j's docstring for why that bound
# matters and what it replaces.
SENSE_MARGIN_V = 0.1

# name -> (ngspice @device[id] path, drain node, source node) for computing
# P = |Id * (Vd - Vs)| from the crowbar .op snapshot. Matches
# data/sram_sp_cell_exposed.spice's device list exactly.
_DEVICES = {
    "X0": ("m.x0.x0.msky130_fd_pr__special_nfet_pass", "qb", "br"),
    "X1": ("m.x0.x1.msky130_fd_pr__special_nfet_latch", "q", "vss"),
    "X2": ("m.x0.x2.msky130_fd_pr__special_nfet_pass", "bl", "q"),
    "X3": ("m.x0.x3.msky130_fd_pr__special_pfet_pass", "q", "q"),
    "X4": ("m.x0.x4.msky130_fd_pr__special_pfet_pass", "qb", "qb"),
    "X5": ("m.x0.x5.msky130_fd_pr__special_pfet_pass", "vdd", "qb"),
    "X6": ("m.x0.x6.msky130_fd_pr__special_pfet_pass", "q", "vdd"),
    "X7": ("m.x0.x7.msky130_fd_pr__special_nfet_latch", "vss", "qb"),
}

# Bulk node per device, from data/sram_sp_cell.spice's 4th terminal (VNB for
# the NMOS, VPB for the PMOS) and the instance line in gds/spice_netlist.py,
# which ties VNB->VSS and VPB->VDD.
_DEVICE_BULK = {
    "X0": "vss", "X1": "vss", "X2": "vss", "X7": "vss",
    "X3": "vdd", "X4": "vdd", "X5": "vdd", "X6": "vdd",
}

# Roles, for figures and for mapping onto channel geometry. Matches
# cases/run_bitcell_compact.py::_classify's width/length-based classification.
DEVICE_ROLE = {
    "X0": "access", "X2": "access",
    "X1": "latch (pull-down NMOS)", "X7": "latch (pull-down NMOS)",
    "X5": "pull-up PMOS", "X6": "pull-up PMOS",
    "X3": "parasitic (D=S)", "X4": "parasitic (D=S)",
}

NGSPICE = "/home/nandan_diwan/miniforge3/envs/thermals/bin/ngspice"


def _run(deck_path: str) -> str:
    result = subprocess.run([NGSPICE, "-b", deck_path], capture_output=True, text=True, timeout=60, check=True)
    return result.stdout + result.stderr


def _last_print_value(output: str, varname: str) -> float:
    matches = re.findall(rf"^\s*{re.escape(varname)}\s*=\s*([\-\d.eE+]+)", output, re.M)
    if not matches:
        raise ValueError(f"could not find printed value for {varname!r} in ngspice output:\n{output[-2000:]}")
    return float(matches[-1])


def _read_wrdata(data_path: str, num_vectors: int) -> list:
    """`wrdata` writes `t0 v0 t1 v1 ... tN-1 vN-1` per row (each vector paired
    with its own time column, all identical for one .tran run) -- returns the
    LAST row's value for each of the `num_vectors` value columns."""
    last = np.loadtxt(data_path)[-1]
    return [last[2 * i + 1] for i in range(num_vectors)]


def characterize_bitcell(lib_path: str = "data/sram22_2048x8m8w1_tt_025C_1v80.lib"):
    """Runs HOLD/WRITE/READ, returns a dict of real per-operation energies (J) and
    leakage power (W), plus the mined timing used to drive the testbenches."""
    timing = mine_access_timing(lib_path)
    pulse_ns = timing.min_pulse_width_high_ns

    with tempfile.TemporaryDirectory() as tmp:
        hold_out = _run(write_hold_deck(f"{tmp}/hold.sp"))
        p_leak = _last_print_value(hold_out, "pvdd")

        _run(write_write_deck(f"{tmp}/write.sp", f"{tmp}/write.txt", pulse_ns, write_one=True))
        e_write, q_final, qb_final = _read_wrdata(f"{tmp}/write.txt", 3)
        write_ok = q_final > VDD / 2

        _run(write_read_deck(f"{tmp}/read.sp", f"{tmp}/read.txt", pulse_ns, stored_one=True))
        e_read, bl_final, br_final = _read_wrdata(f"{tmp}/read.txt", 3)

    return {
        "timing": timing,
        "leakage_w": p_leak,
        "write_energy_j": e_write,
        "write_verified": write_ok,
        "read_energy_j": e_read,
    }


def bias_device_power_w(wl: float, bl: float, br: float, q: float, qb: float) -> dict:
    """{device_name: power_w} from one HARD-FORCED DC bias snapshot (Q/QB
    pinned by ideal sources — see gds.spice_netlist.write_bias_op_deck).

    Only valid for a deliberately-artificial instant (the crowbar snapshot):
    forcing Q/QB while the access transistors are ALSO externally driven
    (WL+BL/BR asserted) creates an artificial current path between two
    competing ideal sources through the access transistor whenever the
    forced Q/QB doesn't closely match what BL/BR implies. Verified directly
    -- see selfconsistent_device_power_w's docstring for the concrete
    number this produced when misused for a READ point (212uW through one
    access transistor, ~1e4x every other device). Do not use this for HOLD/
    WRITE/READ; use selfconsistent_device_power_w for those.
    """
    node_v = {"vdd": VDD, "vss": 0.0, "bl": bl, "br": br, "wl": wl, "q": q, "qb": qb}

    with tempfile.TemporaryDirectory() as tmp:
        out = _run(write_bias_op_deck(f"{tmp}/bias.sp", wl, bl, br, q, qb))

    power = {}
    for name, (path, d_node, s_node) in _DEVICES.items():
        current = _last_print_value(out, f"@{path}[id]")
        vds = node_v[d_node] - node_v[s_node]
        power[name] = abs(current * vds)
    return power


def crowbar_device_power_w(q_bias: float = 0.9) -> dict:
    """Mid-switching crowbar snapshot: WL/BL/BR at VDD, Q=QB=q_bias -- the
    worst-case instant both pull-down NMOS are simultaneously partially on.
    Deliberately hard-forced (see bias_device_power_w's docstring for why
    that's the right tool ONLY here, not for HOLD/WRITE/READ)."""
    return bias_device_power_w(VDD, VDD, VDD, q_bias, q_bias)


def selfconsistent_device_power_w(wl: float, bl: float, br: float, q_guess: float, qb_guess: float) -> dict:
    """{device_name: power_w} from a REAL, self-consistent DC state (HOLD, a
    settled WRITE, or READ) -- Q/QB are NOT forced, only nodeset-guided (see
    gds.spice_netlist.write_selfconsistent_op_deck). This is the correct
    tool for any bias point meant to represent an actual sustained state a
    real cell sits in, as opposed to the crowbar snapshot's deliberately
    impossible instant.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out = _run(write_selfconsistent_op_deck(f"{tmp}/sc.sp", wl, bl, br, q_guess, qb_guess))

    q_actual = _last_print_value(out, "v(x0.q)")
    qb_actual = _last_print_value(out, "v(x0.qb)")
    node_v = {"vdd": VDD, "vss": 0.0, "bl": bl, "br": br, "wl": wl, "q": q_actual, "qb": qb_actual}

    power = {}
    for name, (path, d_node, s_node) in _DEVICES.items():
        current = _last_print_value(out, f"@{path}[id]")
        vds = node_v[d_node] - node_v[s_node]
        power[name] = abs(current * vds)
    return power


def device_power_breakdown_w(name: str) -> dict:
    """{device: {"channel_w": .., "junction_w": ..}} for one named bias point.

    Splits each device's dissipation into the two physically distinct paths a
    BSIM3 device has at DC:

      channel  = |Id * Vds|            -- the inversion-layer current, the
                 term every other function in this file uses, and the one
                 that maps onto the channel GEOMETRY in the thermal solve.
      junction = |Ibd * Vdb| + |Ibs * Vsb|  -- reverse-biased drain-bulk and
                 source-bulk diode leakage. Dissipated at the S/D diffusions,
                 not in the channel.

    Why this exists: `verify_energy_balance` originally summed only the
    channel term and found ratio=1.0000 for READ and CROWBAR but 0.0181 for
    HOLD and 0.0023 for a settled WRITE. That was not a solver bug and not
    noise -- it is exactly this missing term. At an ON point the channel
    dominates by ~7 orders so the omission is invisible; in HOLD the channel
    current is itself leakage-scale, and the junctions carry ~98% of the
    (femtowatt) total. Adding them closes Tellegen's theorem to 1.0000 at
    ALL FOUR points.

    Deliberately NOT folded into named_bias_point_power_w: that function feeds
    the thermal solve, which places power on channel geometry, and junction
    power belongs at the diffusions instead. Folding it in would move heat to
    the wrong place to fix an error that is 8 orders of magnitude below the
    crowbar power the thermal results actually use (4.2e-13 W vs 1.9e-5 W).
    Kept separate, reported honestly, and used for the conservation figure.
    """
    wl, bl, br, q, qb, kind = NAMED_BIAS_POINTS[name]
    with tempfile.TemporaryDirectory() as tmp:
        if kind == "forced":
            out = _run(write_bias_op_deck(f"{tmp}/bias.sp", wl, bl, br, q, qb))
            q_actual, qb_actual = q, qb
        else:
            out = _run(write_selfconsistent_op_deck(f"{tmp}/sc.sp", wl, bl, br, q, qb))
            q_actual = _last_print_value(out, "v(x0.q)")
            qb_actual = _last_print_value(out, "v(x0.qb)")

    node_v = {"vdd": VDD, "vss": 0.0, "bl": bl, "br": br, "wl": wl,
              "q": q_actual, "qb": qb_actual}

    breakdown = {}
    for dev, (path, d_node, s_node) in _DEVICES.items():
        b_node = _DEVICE_BULK[dev]
        i_d = _last_print_value(out, f"@{path}[id]")
        i_bd = _last_print_value(out, f"@{path}[ibd]")
        i_bs = _last_print_value(out, f"@{path}[ibs]")
        breakdown[dev] = {
            "channel_w": abs(i_d * (node_v[d_node] - node_v[s_node])),
            "junction_w": (abs(i_bd * (node_v[d_node] - node_v[b_node]))
                           + abs(i_bs * (node_v[s_node] - node_v[b_node]))),
        }
    return breakdown


# Named operating points for cases/run_bitcell_gallery.py -- (wl, bl, br,
# q_guess, qb_guess, kind). "selfconsistent" points use the guess only as a
# nodeset hint (the solver finds the real Q/QB); "forced" points (crowbar
# only) pin Q/QB exactly -- see the two power functions' docstrings for why
# that distinction matters.
_EPS = 0.01
NAMED_BIAS_POINTS = {
    "hold_1": (0.0, VDD, VDD, VDD - _EPS, _EPS, "selfconsistent"),          # idle, cell stores '1'
    "write_1_settled": (VDD, VDD, 0.0, VDD - _EPS, _EPS, "selfconsistent"),  # WL asserted, write '1' just completed
    "read_1": (VDD, VDD, VDD, VDD - _EPS, _EPS, "selfconsistent"),           # WL asserted, precharged, reading a stored '1'
    "crowbar": (VDD, VDD, VDD, 0.9, 0.9, "forced"),                         # worst-case mid-switching instant
}

# HOLD is the only genuinely SUSTAINED state (idle almost all the time, by
# construction -- the cell is not idle only during the brief access window).
# Every other named point is real but BRIEF: it only happens while (a) this
# row is selected -- 1-in-N_ROWS on average, not every cycle -- and (b) only
# for the real access duration within that cycle, not the whole period.
SUSTAINED_POINTS = {"hold_1"}


def read_access_energy_j(n_rows: int = N_ROWS) -> float:
    """Charge-BOUNDED read energy: a real differential sense amp ends the
    access once the bitline has drooped by SENSE_MARGIN_V, not once it's
    fully discharged -- E = C_BL * dV_sense * VDD, C_BL from the array's
    real row count (gds.spice_netlist.BITLINE_CAP_F, n_rows-scaled).

    Why this replaces "peak DC current x pulse width": a `.op` READ snapshot
    holds BL/BR at an IDEAL, unmoving voltage source, so it reports the
    current at the very first instant of access, before the bitline droops
    at all -- treating that peak as sustained for a full duty-cycle-scaled
    pulse width overstates the real energy transferred, since the real
    current falls as the bitline capacitor charges/discharges (a `.op`
    snapshot can't see that; getting it directly needs a working transient,
    separately unresolved -- see optimized-singing-creek.md Part 1(e) for
    the transient-based cross-check instead). Bounding by the actual charge
    a real sense amp lets move is the correct fix at the `.op`-only level.
    """
    cap = BITLINE_CAP_F if n_rows == N_ROWS else 1e-15 * n_rows
    return cap * SENSE_MARGIN_V * VDD


def named_bias_point_power_w(name: str, row_hit_rate: float = 1.0, lib_path: str = LIB_PATH) -> dict:
    """{device_name: power_w}, REAL AVERAGE power for cases/run_bitcell_gallery.py's
    steady-state solve -- i.e. already scaled for how often this row's access
    actually happens, not the peak instantaneous power a `.op` snapshot reports.

    `row_hit_rate`: fraction of accesses that land on THIS row (1.0 = worst
    case, same row hammered every cycle; 1/N_ROWS = average case, uniform
    random access across rows) -- see optimized-singing-creek.md Part 1(a).
    Applies to every point except HOLD (SUSTAINED_POINTS), which by
    definition isn't gated on this row's own access frequency at all.
    """
    wl, bl, br, q, qb, kind = NAMED_BIAS_POINTS[name]
    raw = bias_device_power_w(wl, bl, br, q, qb) if kind == "forced" \
        else selfconsistent_device_power_w(wl, bl, br, q, qb)

    if name in SUSTAINED_POINTS:
        return raw

    timing = mine_access_timing(lib_path)
    access_hz = row_hit_rate / (timing.min_period_ns * 1e-9)

    if name == "read_1":
        # Charge-bounded average power, redistributed across devices in the
        # SAME proportion the raw (peak-instantaneous) snapshot showed --
        # preserves the compact model's spatial answer ("which device"),
        # replaces only the magnitude ("how much, on average").
        raw_total = sum(raw.values()) or 1.0
        p_avg_total = read_access_energy_j() * access_hz
        return {k: v / raw_total * p_avg_total for k, v in raw.items()}

    if name == "crowbar":
        # CROWBAR is a brief TRANSITION instant (both pull-downs momentarily
        # both-on while Q/QB cross each other), not a state that persists
        # for a full access window -- unlike WRITE-settled below, using the
        # multi-cycle clk->Q access time as its duty basis here saturates at
        # 1.0 and overstates it by >10x (confirmed directly: regressed the
        # validated 31.5K crowbar result toward ~500K). min_pulse_width_high
        # (~0.24ns, the clock's own minimum high time) is the short,
        # transition-scale proxy this needs instead.
        duty = min(1.0, timing.min_pulse_width_high_ns / timing.min_period_ns) * row_hit_rate
        return {k: v * duty for k, v in raw.items()}

    # WRITE (settled): genuinely persists close to the full access window
    # once WL is asserted, so the clk->Q access time (the better access-
    # DURATION proxy per optimized-singing-creek.md Part 1(d), vs. the
    # clock's own min_pulse_width constraint) is the right basis, times how
    # often this row is actually hit. It exceeds one period here (~4.4ns vs
    # ~3.9ns, a pipelined access), so the ratio saturates at 1.0: "active
    # for essentially its whole cycle whenever this row IS selected," with
    # row_hit_rate alone doing the work of "how often that happens."
    duty = min(1.0, timing.clk_to_q_access_ns / timing.min_period_ns) * row_hit_rate
    return {k: v * duty for k, v in raw.items()}


def supply_power_w(name: str) -> float:
    """sum(|V_src * I_src|) over EVERY ideal source in the deck (VDD, VSS,
    WL, BL, BR, and for the forced/crowbar variant also Q, QB) -- this, not
    "current through VDD alone," is the correct conservation law here
    (Tellegen's theorem: total power delivered by all independent sources
    equals total power dissipated in the rest of the network at a DC
    operating point). Checking VDD alone was tried first and is WRONG for
    this circuit: BL/BR/WL are separately-driven ideal sources in this
    single-cell testbench, and the dominant current path for e.g. a READ
    (precharged bitline -> access transistor -> internal node -> pulldown
    -> VSS) never touches the VDD net at all -- confirmed directly: a
    VDD-only check gave nonsensical ratios (3,000,000x for read_1) purely
    from this modeling error, not from any real bug in the power numbers.
    """
    wl, bl, br, q, qb, kind = NAMED_BIAS_POINTS[name]
    with tempfile.TemporaryDirectory() as tmp:
        if kind == "forced":
            out = _run(write_bias_op_deck(f"{tmp}/bias.sp", wl, bl, br, q, qb))
            sources = {"vvdd": VDD, "vvss": 0.0, "vwl": wl, "vbl": bl, "vbr": br, "vq": q, "vqb": qb}
        else:
            out = _run(write_selfconsistent_op_deck(f"{tmp}/sc.sp", wl, bl, br, q, qb))
            sources = {"vvdd": VDD, "vvss": 0.0, "vwl": wl, "vbl": bl, "vbr": br}
    return sum(abs(v * _last_print_value(out, f"i({name_})")) for name_, v in sources.items())


def verify_energy_balance() -> dict:
    """P0.2 audit check: does sum(per-device |Id*Vds|) match the total power
    delivered by every ideal source (supply_power_w, Tellegen's theorem),
    at the RAW (un-duty-scaled) level where the electrical physics actually
    lives -- duty-cycle scaling (named_bias_point_power_w) applies
    identically to every device and to every source alike, so it would
    cancel in the ratio and hide nothing; checking pre-scaling is simpler
    and more direct.

    Reports TWO ratios, because the channel-only one is incomplete by
    construction (see device_power_breakdown_w):

      ratio_channel  sum(|Id*Vds|) / sum(sources). Closes at ON points
                     (READ, CROWBAR) and fails badly at leakage-scale points
                     (HOLD 0.018, WRITE 0.002) -- the missing power is
                     bulk-junction leakage, not a bug.
      ratio_full     channel + junction terms. Closes at ALL FOUR points.

    `bias_device_power_w`'s own docstring (crowbar/forced path) warns of an
    artificial current path through the ideal Q/QB sources when they fight
    externally-driven access transistors -- this check catches exactly
    that: crowbar is the one point where sum(device) may legitimately fall
    short of sum(all sources) (some current flows through VQ/VQB itself,
    not through the 8 counted channel devices), so its ratio is reported
    but not held to the same bar. HOLD/WRITE/READ (self-consistent, no
    forced Q/QB) are real KCL nodes with no such escape hatch and SHOULD
    match closely.
    """
    results = {}
    for name, (wl, bl, br, q, qb, kind) in NAMED_BIAS_POINTS.items():
        bd = device_power_breakdown_w(name)
        p_channel = sum(d["channel_w"] for d in bd.values())
        p_junction = sum(d["junction_w"] for d in bd.values())
        p_supply = supply_power_w(name)
        r_ch = p_channel / p_supply if p_supply else float("nan")
        r_full = (p_channel + p_junction) / p_supply if p_supply else float("nan")
        results[name] = {
            "p_channel_w": p_channel, "p_junction_w": p_junction,
            "p_devices_w": p_channel + p_junction, "p_supply_w": p_supply,
            "ratio_channel": r_ch, "ratio_full": r_full, "kind": kind,
            "breakdown": bd,
        }
        print(f"{name:18s} kind={kind:16s} channel={p_channel:11.4e} W  "
              f"junction={p_junction:11.4e} W  sources={p_supply:11.4e} W  "
              f"ratio(channel)={r_ch:8.4f}  ratio(full)={r_full:.4f}")
    return results


def gate_check(lib_path: str = "data/sram22_2048x8m8w1_tt_025C_1v80.lib", num_rows: int = 2048):
    """Phase 1 gate: single-cell SPICE energy, scaled by the real row count this
    macro's own name states, vs. the already-validated Liberty numbers. Orders-of-
    magnitude agreement is the bar (see plan) -- print both sides plainly rather
    than asserting, since this is a sanity check to read, not a pass/fail unit test.
    """
    result = characterize_bitcell(lib_path)
    macro = parse_lib(lib_path)

    print(f"mined timing: {result['timing']}")
    print(f"single-cell leakage: {result['leakage_w']*1e9:.4f} nW "
          f"(macro leakage / row: {macro.leakage_w*1e9/num_rows:.4f} nW)")
    print(f"single-cell write energy: {result['write_energy_j']*1e15:.4f} fJ "
          f"(write verified: {result['write_verified']})")
    print(f"single-cell read energy: {result['read_energy_j']*1e15:.4f} fJ")
    print(f"macro write energy (published): {macro.energy_per_access_j(True)*1e12:.4f} pJ")
    print(f"macro read energy (published): {macro.energy_per_access_j(False)*1e12:.4f} pJ")
    return result


if __name__ == "__main__":
    gate_check()
