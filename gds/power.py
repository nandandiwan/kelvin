"""Real total-power numbers for a macro, read out of its own Liberty (.lib)
characterization instead of assumed.

This replaces "pick 1mW and see what happens" with a number the macro's own
vendor-published characterization supports:

    P_total = P_leakage + f_clk * E_per_cycle

For sram22_2048x8m8w1 at tt/25C/1.8V that is 547nW of leakage against
~13-15pJ per access — i.e. **>99% of the power is access-driven**, not
leakage. (It also retroactively justifies the old 1mW default: that's about
a 70MHz write rate.)

Deliberately a small targeted parser, not a general Liberty reader — it
needs exactly two things (`cell_leakage_power` and the clock pin's
`internal_power` tables) out of a file whose full grammar is enormous.

Two conventions worth stating explicitly, because both are choices:

1. **Energy per cycle sums `rise_power + fall_power`.** A memory access
   spans a full clock cycle: the rising edge does the work, the falling edge
   completes it, and Liberty books them separately on the clock pin.
2. **Energy sums over `related_pg_pin`.** Liberty splits internal energy
   between the vdd and vss rails; the physically dissipated total is their
   sum. This is the common convention in power-analysis flows, but note the
   two rails here report nearly equal values (6.99 vs 7.28 pJ), so a reader
   who instead treats them as two views of the *same* energy would land a
   factor of ~2 lower. `PG_PIN_SUM = False` switches to that reading.
"""

import re
from dataclasses import dataclass
from typing import Dict

# Liberty units in these files: voltage_unit 1V, current_unit 1mA,
# time_unit 1ns  =>  internal_power energies are in 1V*1mA*1ns = 1 pJ.
# leakage_power_unit is stated directly as 1nW.
_ENERGY_UNIT_J = 1e-12
_LEAKAGE_UNIT_W = 1e-9

PG_PIN_SUM = True


@dataclass(frozen=True)
class MacroPower:
    leakage_w: float
    energy_by_condition_j: Dict[str, float]   # e.g. {"we&ce": 1.5e-11, ...}

    def energy_per_access_j(self, write: bool = True) -> float:
        """Energy for one enabled access (ce true). Write costs more than read."""
        key = "we&ce" if write else "!we&ce"
        return self.energy_by_condition_j[key]

    def energy_per_idle_cycle_j(self, write: bool = True) -> float:
        """Clock still toggling, chip disabled (ce false) — not free."""
        key = "we&!ce" if write else "!we&!ce"
        return self.energy_by_condition_j[key]

    def total_power_w(self, freq_hz: float, activity: float = 1.0, write: bool = True) -> float:
        """`activity` = fraction of clock cycles that are real accesses; the
        remainder still burn the (much smaller) disabled-clock energy.
        """
        e_active = self.energy_per_access_j(write)
        e_idle = self.energy_per_idle_cycle_j(write)
        e_avg = activity * e_active + (1.0 - activity) * e_idle
        return self.leakage_w + freq_hz * e_avg


def _first_value(block: str, kind: str) -> float:
    """First number out of a `<kind>_power (...) { ... values ( "a, b, ..." ) }`
    table. These tables are constant across the input-transition index in this
    file (all seven entries identical), so the first value is the value.
    """
    m = re.search(kind + r"_power[^{]*\{.*?values \(\s*\\?\s*\n?\s*\"([\d.eE+-]+)", block, re.S)
    return float(m.group(1)) if m else 0.0


def parse_lib(path: str) -> MacroPower:
    text = open(path).read()

    leak = re.search(r"cell_leakage_power\s*:\s*([\d.eE+-]+)\s*;", text[text.find("cell ("):])
    leakage_w = float(leak.group(1)) * _LEAKAGE_UNIT_W if leak else 0.0

    energies: Dict[str, float] = {}
    for m in re.finditer(r"internal_power \(\)\s*\{(.*?)\n      \}", text, re.S):
        block = m.group(1)
        when = re.search(r'when\s*:\s*"([^"]+)"', block)
        if not when:
            continue
        e = (_first_value(block, "rise") + _first_value(block, "fall")) * _ENERGY_UNIT_J
        if PG_PIN_SUM:
            energies[when.group(1)] = energies.get(when.group(1), 0.0) + e
        else:
            energies[when.group(1)] = max(energies.get(when.group(1), 0.0), e)

    if not energies:
        raise ValueError(f"no internal_power tables found in {path!r}")
    return MacroPower(leakage_w=leakage_w, energy_by_condition_j=energies)


@dataclass(frozen=True)
class AccessTiming:
    """Real, characterized clock/word-line timing (ns) — mined, not guessed,
    for gds/spice_netlist.py's bitcell testbenches. `time_unit` in this file
    is "1ns" (checked, not assumed).

    `min_pulse_width_high_ns` is the CLOCK pin's own minimum-high-time
    constraint -- a constraint on the clock waveform, not a direct
    measurement of how long a wordline stays asserted during one access.
    `clk_to_q_access_ns` (the `rising_edge` clk->Q arc's `cell_rise` delay)
    is a real measured access-completion time and the better proxy for
    "how long is this row actually active" -- callers should prefer it and
    keep min_pulse_width_high_ns only as a secondary cross-check (see
    optimized-singing-creek.md Part 1(d)).
    """
    min_period_ns: float
    min_pulse_width_high_ns: float
    min_pulse_width_low_ns: float
    clk_to_q_access_ns: float


def _first_constraint_value(text: str, timing_type: str, constraint: str) -> float:
    """First `values` entry inside one `timing() { timing_type: X; ...
    <constraint> (...) { ... values (...) } }` block — these constraint
    tables are ~flat across the input-slew index in this file (see
    min_pulse_width/minimum_period rows), so the first entry represents it.
    """
    m = re.search(
        r"timing_type\s*:\s*" + timing_type + r"\s*;.*?" + constraint +
        r"\s*\([^)]*\)\s*\{.*?values \(\s*\\?\s*\n?\s*\"([\d.eE+-]+)",
        text, re.S,
    )
    if not m:
        raise ValueError(f"no {timing_type}/{constraint} table found")
    return float(m.group(1))


def mine_access_timing(path: str) -> AccessTiming:
    """Pull real word-line pulse-width and minimum-period constraints out of
    the clock pin's `min_pulse_width`/`minimum_period` timing arcs — the
    same characterization flow that produced the bitcell netlist, so the
    testbench this drives is internally consistent with the design instead
    of an arbitrary generic waveform (see optimized-singing-creek.md's
    Phase 1)."""
    text = open(path).read()
    return AccessTiming(
        min_period_ns=_first_constraint_value(text, "minimum_period", "rise_constraint"),
        min_pulse_width_high_ns=_first_constraint_value(text, "min_pulse_width", "rise_constraint"),
        min_pulse_width_low_ns=_first_constraint_value(text, "min_pulse_width", "fall_constraint"),
        clk_to_q_access_ns=_first_constraint_value(text, "rising_edge", "cell_rise"),
    )
