"""Compact-model DC power and explicitly approximate averaged access models.

Validated transient electrical waveforms live in ``gds.spice_transient``;
``characterize_bitcell`` delegates READ/WRITE characterization there. An
isolated loaded cell is not the complete macro including drivers/sense amps.
All decks use micrometre-scaled instance dimensions and 25 C. Results produced
before that unit correction must be regenerated, including DC thermal loads.
"""

import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from gds.power import mine_access_timing, parse_lib
from gds.spice_netlist import (
    BITLINE_CAP_F, N_ROWS, VDD, write_bias_op_deck, write_hold_deck,
    write_selfconsistent_op_deck,
)

LIB_PATH = "data/sram22_2048x8m8w1_tt_025C_1v80.lib"

# Bump when event-energy, rectangle-duration, or per-device allocation
# semantics change; saved transient fields must not reuse older load models.
ACCESS_ENERGY_MODEL_REVISION = "shared-access-energy-v2-micron-25c"
SPICE_MODEL_REVISION = "sky130-micron-25c-v1"

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

# Descriptive roles for figures only; gds.bitcell_mapping preserves each
# individual instance's power when assigning heat to channel geometry.
DEVICE_ROLE = {
    "X0": "access", "X2": "access",
    "X1": "latch (pull-down NMOS)", "X7": "latch (pull-down NMOS)",
    "X5": "pull-up PMOS", "X6": "pull-up PMOS",
    "X3": "parasitic (D=S)", "X4": "parasitic (D=S)",
}

_LOCAL_NGSPICE = Path(__file__).resolve().parents[1] / ".deps" / "ngspice" / "bin" / "ngspice"
_LEGACY_NGSPICE = Path("/home/nandan_diwan/miniforge3/envs/thermals/bin/ngspice")


def find_ngspice() -> str:
    """Find an executable without changing the process or shared environment.

    An explicit KELVIN_NGSPICE override wins and must be valid. Otherwise
    prefer PATH, then Kelvin's isolated local installation, then the original
    developer's installation if it still exists. Resolve at call time so an
    override set after importing this module also takes effect.
    """
    configured = os.environ.get("KELVIN_NGSPICE")
    if configured is not None:
        executable = shutil.which(configured)
        if executable is None:
            raise FileNotFoundError(
                f"KELVIN_NGSPICE={configured!r} does not name an executable ngspice binary"
            )
        return executable
    executable = shutil.which("ngspice")
    if executable is not None:
        return executable
    for candidate in (_LOCAL_NGSPICE, _LEGACY_NGSPICE):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    raise FileNotFoundError(
        "ngspice was not found. Install ngspice on PATH, set KELVIN_NGSPICE to its "
        "executable, or install it in Kelvin's .deps/ngspice/bin/ngspice."
    )


def _run(deck_path: str, timeout_s: float = 60) -> str:
    result = subprocess.run(
        [find_ngspice(), "-b", deck_path], capture_output=True, text=True, timeout=timeout_s, check=False
    )
    output = result.stdout + result.stderr
    # A .control script can write a partial wrdata file and even exit with code
    # zero after an aborted analysis. Neither a file nor a zero status proves
    # completion. Transient callers also validate the final timestamp.
    failure = re.search(
        r"(?im)^\s*(?:error\b|fatal\b)|timestep too small|simulation\s+aborted|"
        r"doanalyses:\s*.*failed|run simulation\(s\) aborted", output,
    )
    if result.returncode != 0 or failure:
        raise RuntimeError(f"ngspice failed for {deck_path} (exit {result.returncode}):\n{output[-6000:]}")
    return output


def _last_print_value(output: str, varname: str) -> float:
    matches = re.findall(rf"^\s*{re.escape(varname)}\s*=\s*([\-\d.eE+]+)", output, re.M)
    if not matches:
        raise ValueError(f"could not find printed value for {varname!r} in ngspice output:\n{output[-2000:]}")
    return float(matches[-1])


def characterize_bitcell(lib_path: str = "data/sram22_2048x8m8w1_tt_025C_1v80.lib"):
    """Validated WRITE-1/READ-1 channel heat, not supply-energy surrogates.

    Energies integrate intrinsic channel dissipation over the completed
    electrical event including its post-access settling. Junction and signed
    external-source energies are retained separately in the detailed reports.
    ``leakage_w`` retains the legacy HOLD VDD-only measurement; it is not a
    calibrated full-cell leakage estimate (junction geometry is not extracted).
    """
    from gds.spice_transient import run_access_transient

    timing = mine_access_timing(lib_path)
    pulse_ns = timing.min_pulse_width_high_ns
    with tempfile.TemporaryDirectory() as tmp:
        hold_out = _run(write_hold_deck(f"{tmp}/hold.sp"))
        p_leak = _last_print_value(hold_out, "pvdd")
    write = run_access_transient("write", stored_one=False, pulse_width_ns=pulse_ns)
    read = run_access_transient("read", stored_one=True, pulse_width_ns=pulse_ns)
    return {
        "timing": timing, "leakage_w": p_leak,
        "write_energy_j": write.report["channel_energy_j"], "write_verified": True,
        "read_energy_j": read.report["channel_energy_j"], "read_verified": True,
        "write_report": write.report, "read_report": read.report,
        "energy_definition": "integrated per-device intrinsic channel heat over completed transient",
        "spice_model_revision": SPICE_MODEL_REVISION,
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
    """Legacy charge-limited READ surrogate: C_BL * assumed sense swing * VDD.

    This is not measured channel heat. The validated transient bench instead
    integrates channel dissipation with a prescribed WL pulse and no sense-amp
    feedback, so its bitline swing and event energy are different. Keep this
    explicit model for existing averaged/rectangular thermal cases until their
    workload is deliberately migrated to the actual electrical traces.
    """
    cap = BITLINE_CAP_F if n_rows == N_ROWS else 1e-15 * n_rows
    return cap * SENSE_MARGIN_V * VDD


def _validate_row_hit_rate(row_hit_rate: float) -> float:
    rate = float(row_hit_rate)
    if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
        raise ValueError("row_hit_rate must be finite and between 0 and 1")
    return rate


def _positive_timing_s(value_ns: float, name: str) -> float:
    value_s = float(value_ns) * 1e-9
    if not math.isfinite(value_s) or value_s <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return value_s


def _raw_named_device_power_w(name: str) -> dict:
    """Validate the complete per-instance DC result before using its ratios."""
    wl, bl, br, q, qb, kind = NAMED_BIAS_POINTS[name]
    raw = bias_device_power_w(wl, bl, br, q, qb) if kind == "forced" \
        else selfconsistent_device_power_w(wl, bl, br, q, qb)
    if set(raw) != set(_DEVICES):
        raise ValueError(f"{name}: DC powers must contain exactly X0 through X7")
    result = {device: float(power) for device, power in raw.items()}
    if any(not math.isfinite(power) or power < 0.0 for power in result.values()):
        raise ValueError(f"{name}: DC device powers must be finite and nonnegative")
    if not math.isfinite(sum(result.values())):
        raise ValueError(f"{name}: total DC device power must be finite")
    return result


@dataclass(frozen=True)
class AccessPulse:
    """Energy-equivalent rectangular thermal surrogate for ONE selected access.

    This is not a simulated electrical waveform. The DC operating point sets
    the relative device dissipation; the named energy model and timing proxy
    set the event energy and rectangle duration. There is no row-hit scaling
    within one selected event. Averaged workloads use ``average_power_w``.
    """

    name: str
    device_energy_j: dict
    device_power_w: dict
    duration_s: float
    period_s: float
    description: str

    @property
    def total_energy_j(self) -> float:
        return sum(self.device_energy_j.values())

    def average_power_w(self, row_hit_rate: float = 1.0) -> dict:
        rate = _validate_row_hit_rate(row_hit_rate)
        return {device: energy * rate / self.period_s
                for device, energy in self.device_energy_j.items()}


def named_access_pulse(name: str, lib_path: str = LIB_PATH) -> AccessPulse:
    """Return the shared per-device event energy and rectangular pulse model.

    CROWBAR uses its deliberately forced DC power for the minimum clock-high
    time, capped at one period. READ uses ``C_BL * sense_margin * VDD`` in
    the DC devices' power proportions, distributed uniformly over the capped
    clk-to-Q timing proxy; its rectangle is NOT the raw DC peak. Settled WRITE
    uses raw DC power over that same capped clk-to-Q proxy, not a switching
    write waveform. These choices retain the existing averaged energy model.

    HOLD is sustained leakage, not an access; use
    ``named_bias_point_power_w('hold_1')`` for it. The Liberty timings and READ
    capacitance/sense margin remain modeling assumptions, not calibration of
    this GDS cell's actual temporal electrical behavior.
    """
    if name not in NAMED_BIAS_POINTS:
        raise KeyError(name)
    if name in SUSTAINED_POINTS:
        raise ValueError(f"{name} is a sustained state, not a single-access pulse")

    timing = mine_access_timing(lib_path)
    period_s = _positive_timing_s(timing.min_period_ns, "min_period_ns")
    if name == "crowbar":
        duration_s = min(period_s, _positive_timing_s(
            timing.min_pulse_width_high_ns, "min_pulse_width_high_ns"))
    else:
        duration_s = min(period_s, _positive_timing_s(
            timing.clk_to_q_access_ns, "clk_to_q_access_ns"))

    raw = _raw_named_device_power_w(name)
    if name == "read_1":
        raw_total = sum(raw.values())
        if raw_total <= 0.0:
            raise ValueError("read_1: positive total DC power is required to distribute READ energy")
        energy_j = float(read_access_energy_j())
        if not math.isfinite(energy_j) or energy_j <= 0.0:
            raise ValueError("read_1: charge-bounded access energy must be finite and positive")
        energies = {device: power / raw_total * energy_j for device, power in raw.items()}
        powers = {device: energy / duration_s for device, energy in energies.items()}
        description = (
            "Charge-bounded READ energy C_BL * sense_margin * VDD, allocated by "
            "DC device-power proportions; uniform energy-equivalent rectangle "
            "over min(clk_to_Q, period), not the raw DC peak or an electrical waveform."
        )
    else:
        powers = raw
        energies = {device: power * duration_s for device, power in powers.items()}
        description = (
            "Synthetic forced-DC crowbar rectangle over min(clock_high, period); "
            "not an electrical switching waveform."
            if name == "crowbar" else
            "Settled-WRITE DC rectangle over min(clk_to_Q, period); "
            "not the energy or waveform of a write transition."
        )
    if any(not math.isfinite(value) for value in (*powers.values(), *energies.values())):
        raise ValueError(f"{name}: derived access energies and powers must be finite")
    return AccessPulse(name, energies, powers, duration_s, period_s, description)


def named_bias_point_power_w(name: str, row_hit_rate: float = 1.0, lib_path: str = LIB_PATH) -> dict:
    """Per-device averaged thermal-model power, consistent with single events.

    Access states use the SAME energy definition as ``named_access_pulse``:
    ``P_avg[device] = E_access[device] * row_hit_rate / period``. The hit rate
    is the fraction of accesses selecting this row (1 = every cycle, 1/N_ROWS
    = uniform random access). HOLD remains a sustained DC state, independent
    of hit rate. The compact DC/timing model is a surrogate, not a measured
    or transient-SPICE electrical waveform.
    """
    rate = _validate_row_hit_rate(row_hit_rate)
    if name in SUSTAINED_POINTS:
        return _raw_named_device_power_w(name)
    return named_access_pulse(name, lib_path=lib_path).average_power_w(rate)


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
    """Print electrical characterization and separate macro context, not calibration."""
    result = characterize_bitcell(lib_path)
    macro = parse_lib(lib_path)

    print(f"mined timing: {result['timing']}")
    print(f"single-cell leakage: {result['leakage_w']*1e9:.4f} nW "
          f"(macro leakage / row: {macro.leakage_w*1e9/num_rows:.4f} nW)")
    print("Cell channel heat and full-macro Liberty supply energy are different quantities.")
    print(f"single-cell write channel heat: {result['write_energy_j']*1e15:.4f} fJ "
          f"(write verified: {result['write_verified']})")
    print(f"single-cell read channel heat: {result['read_energy_j']*1e15:.4f} fJ")
    print(f"macro write energy (published): {macro.energy_per_access_j(True)*1e12:.4f} pJ")
    print(f"macro read energy (published): {macro.energy_per_access_j(False)*1e12:.4f} pJ")
    return result


if __name__ == "__main__":
    gate_check()
