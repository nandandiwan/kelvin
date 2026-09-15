"""Slide 5 (device physics) figures -- three, each making exactly one point.

Where the heat numbers come from, and why they should be believed:

  fig5a_device_power.png   WHERE the power is. Per-device dissipation at four
                           named bias points, from real SKY130 BSIM3 models in
                           ngspice on the real extracted 8-device bitcell.
                           The spatial non-uniformity is the whole reason a
                           3D thermal solve is needed at all.
  fig5b_conservation.png   That the circuit solution CONSERVES energy
                           (Tellegen's theorem), and that it only does so once
                           bulk-junction leakage is counted alongside channel
                           current.
  fig5c_liberty.png        That the absolute magnitudes agree with the
                           vendor's own published Liberty characterization --
                           an INDEPENDENT number this project did not produce.

    python cases/fig_device_physics.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gds.power import parse_lib
from gds.spice_netlist import BITLINE_CAP_F, N_ROWS, VDD
from gds.spice_power import (
    DEVICE_ROLE, NAMED_BIAS_POINTS, SENSE_MARGIN_V, device_power_breakdown_w,
    supply_power_w,
)

OUT = Path("out/presentation")
LIB_PATH = "data/sram22_2048x8m8w1_tt_025C_1v80.lib"

# Physical array shape of sram22_2048x8m8w1, from the macro naming convention
#   <words> x <word_width> m<column mux> w<write-mask granularity>
#   rows = words / mux ,  physical columns = word_width * mux
# That rule is CHECKED, not assumed: applied to the other macro in data/
# (sram22_64x22m4w22) it gives rows=16, cols=88 -- exactly the values
# gds/spice_netlist.py counted directly out of that macro's real netlist.
#
# This matters for read energy: 2048x8 is the LOGICAL shape. Reading one
# 8-bit word asserts a word line spanning all 64 PHYSICAL columns, so 64
# bit lines droop (the 8:1 column mux only selects which 8 are sensed), and
# each bit line carries the capacitance of 256 cells, not 2048.
LIB_WORDS, LIB_WORD_BITS, LIB_MUX = 2048, 8, 8
LIB_ROWS = LIB_WORDS // LIB_MUX          # 256
LIB_COLS = LIB_WORD_BITS * LIB_MUX       # 64

# Bit-line loading per row and sense-amp trip margin are both stated
# ASSUMPTIONS, not measurements (see gds/spice_netlist.py::BITLINE_CAP_F and
# gds/spice_power.py::SENSE_MARGIN_V). Their plausible ranges are what set the
# width of the predicted band in fig5c.
C_PER_ROW_RANGE_F = (0.5e-15, 2.0e-15)
SENSE_MARGIN_RANGE_V = (0.10, 0.20)

POINT_LABEL = {
    "hold_1": "HOLD",
    "write_1_settled": "WRITE (settled)",
    "read_1": "READ",
    "crowbar": "CROWBAR",
}
POINT_COLOR = {
    "hold_1": "#9e9e9e",
    "write_1_settled": "#4c9f70",
    "read_1": "#1f77b4",
    "crowbar": "#d62728",
}
# X3/X4 are LVS-extracted parasitics with drain tied to source, so Vds == 0 and
# their power is identically zero. Excluded from the figure (two permanently
# empty slots read as a bug); stated in the talk instead.
DEVICES = ["X0", "X1", "X2", "X5", "X6", "X7"]


def fig_device_power(data):
    """Per-device channel power. Channel, not channel+junction, because this
    is the term that gets mapped onto channel GEOMETRY in the thermal solve --
    the figure should show the quantity that actually drives the temperature
    field, not a bookkeeping total (see gds/spice_power.py's breakdown docstring)."""
    fig, ax = plt.subplots(figsize=(10, 5.2))
    names = list(NAMED_BIAS_POINTS)
    width = 0.8 / len(names)
    # A device that is exactly OFF reports 0 W, which a log axis cannot draw.
    # Floor it below the visible range rather than dropping the bar silently.
    floor = 1e-13   # nW

    for i, name in enumerate(names):
        vals = [max(data[name]["breakdown"][d]["channel_w"] * 1e9, floor) for d in DEVICES]
        ax.bar(np.arange(len(DEVICES)) + i * width - 0.4 + width / 2, vals, width,
               label=POINT_LABEL[name], color=POINT_COLOR[name])

    ax.set_yscale("log")
    ax.set_ylim(1e-12, 1e7)
    ax.set_xticks(range(len(DEVICES)))
    ax.set_xticklabels([f"{d}\n{DEVICE_ROLE[d].split(' (')[0]}" for d in DEVICES], fontsize=9)
    ax.set_ylabel("channel dissipation $|I_d V_{ds}|$  (nW)")
    ax.set_title("Per-device power in the real SRAM bitcell\n"
                 "SKY130 BSIM3 compact models, DC operating points", fontsize=12)
    ax.grid(axis="y", alpha=0.3, which="major")
    ax.legend(fontsize=9, ncol=4, loc="upper center")
    fig.tight_layout()
    fig.savefig(OUT / "fig5a_device_power.png", dpi=160)
    plt.close(fig)
    print(f"wrote {OUT}/fig5a_device_power.png")


def fig_conservation(data):
    """Tellegen's theorem: at a DC operating point, the power delivered by
    every independent source equals the power dissipated in the rest of the
    network. Shown as a ratio so all four points share one axis despite
    spanning 7 orders of magnitude in absolute power."""
    fig, ax = plt.subplots(figsize=(9, 5))
    names = list(NAMED_BIAS_POINTS)
    x = np.arange(len(names))

    r_ch = [data[n]["ratio_channel"] for n in names]
    r_full = [data[n]["ratio_full"] for n in names]
    ax.bar(x - 0.19, r_ch, 0.36, label="channel current only,  $\\Sigma|I_dV_{ds}|$",
           color="#c9c9c9", edgecolor="#777")
    ax.bar(x + 0.19, r_full, 0.36, label="+ bulk-junction currents  $I_{bd}$, $I_{bs}$",
           color="#1f77b4")
    ax.axhline(1.0, color="k", lw=1.2, ls="--", zorder=5)

    ax.set_yscale("log")
    ax.set_ylim(1e-3, 2.2)
    ax.set_xticks(x)
    ax.set_xticklabels([POINT_LABEL[n] for n in names])
    ax.set_ylabel(r"$\Sigma\,P_{\mathrm{devices}}\ /\ \Sigma\,P_{\mathrm{sources}}$")
    ax.set_title("Power balance: devices vs. sources", fontsize=13)
    ax.grid(axis="y", alpha=0.3, which="major")
    ax.legend(fontsize=9.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "fig5b_conservation.png", dpi=160)
    plt.close(fig)
    print(f"wrote {OUT}/fig5b_conservation.png")


def _lib_read_array_energy_j():
    """Published energy for the ARRAY part of one read, as a RANGE.

    Two corrections to the naive "13.18 pJ" headline, both real:

    1. Subtract the ce=0 condition. Liberty books `!we&!ce` (clock still
       toggling, chip disabled) separately; that is the clock tree and
       control logic, which happens with or without an array access. What a
       bitline model could possibly explain is the difference.
    2. The pg_pin convention is genuinely ambiguous by ~2x. Liberty splits
       internal energy between the vdd and vss rails, reporting nearly equal
       values (6.99 vs 7.28 pJ). Summing them (the common power-analysis
       convention) vs treating them as two views of one energy differ by a
       factor of two -- see gds/power.py's PG_PIN_SUM note. Neither reading
       is obviously wrong, so both bound the answer rather than one being
       picked to flatter the comparison.
    """
    import gds.power as power_mod
    out = []
    original = power_mod.PG_PIN_SUM
    try:
        for convention in (False, True):
            power_mod.PG_PIN_SUM = convention
            m = power_mod.parse_lib(LIB_PATH)
            out.append(m.energy_per_access_j(write=False) - m.energy_per_idle_cycle_j(write=False))
    finally:
        power_mod.PG_PIN_SUM = original
    return min(out), max(out)


def _bitline_read_energy_j(c_per_row_f, sense_v):
    """E = N_cols * C_BL * dV_sense * VDD, with C_BL = rows * c_per_row."""
    return LIB_COLS * (LIB_ROWS * c_per_row_f) * sense_v * VDD


def fig_liberty(data):
    """Absolute-magnitude check against the vendor's own Liberty file -- numbers
    this project did not produce and cannot tune.

    Two panels making deliberately DIFFERENT claims. Leakage is a sharp test:
    both sides are per-bit, the array dominates the macro, and the comparison
    lands within 1.44x. Access energy is not a sharp test, and the figure says
    so: the published value is itself ambiguous by 2x, and a single-cell model
    only covers the bit lines, not the decoders, sense amps or output drivers.
    Showing it as overlapping BANDS rather than two bars is the honest form --
    presented as two bars it reads as a 4x failure, which misstates what was
    actually tested.
    """
    macro = parse_lib(LIB_PATH)
    n_bits = LIB_ROWS * LIB_COLS

    lib_leak_per_cell = macro.leakage_w / n_bits
    ours_leak = data["hold_1"]["p_supply_w"]
    ours_leak_channel = data["hold_1"]["p_channel_w"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.9))

    # ---- panel 1: leakage, a sharp per-bit comparison ----
    vals = [ours_leak_channel, ours_leak, lib_leak_per_cell]
    ax1.bar([0, 1, 2], [v * 1e12 for v in vals],
            color=["#c9c9c9", "#1f77b4", "#e6a700"], width=0.6)
    ax1.set_xticks([0, 1, 2])
    ax1.set_xticklabels(["BSIM3 .op\nchannel only", "BSIM3 .op\nfull", "vendor Liberty\n(per bit)"],
                        fontsize=9.5)
    ax1.set_yscale("log")
    for i, v in enumerate(vals):
        ax1.text(i, v * 1.35e12, f"{v*1e12:.2f} pW", ha="center", fontsize=10, fontweight="bold")
    ax1.set_ylim(0.1, 400)
    ax1.set_ylabel("static leakage per bit (pW)")
    ax1.set_title(f"HOLD leakage — within "
                  f"{max(ours_leak, lib_leak_per_cell)/min(ours_leak, lib_leak_per_cell):.2f}× "
                  f"of vendor", fontsize=12)
    ax1.grid(axis="y", alpha=0.3, which="major")

    # ---- panel 2: read energy, overlapping bands ----
    lib_lo, lib_hi = _lib_read_array_energy_j()
    ours = [_bitline_read_energy_j(c, v)
            for c in C_PER_ROW_RANGE_F for v in SENSE_MARGIN_RANGE_V]
    ours_lo, ours_hi = min(ours), max(ours)
    ours_nom = _bitline_read_energy_j(BITLINE_CAP_F / N_ROWS, SENSE_MARGIN_V)

    bands = [("vendor Liberty\narray access\n(read, minus ce=0)", lib_lo, lib_hi, None, "#e6a700"),
             (f"this model\nbit-line charge\n({LIB_COLS} columns × {LIB_ROWS} rows)",
              ours_lo, ours_hi, ours_nom, "#1f77b4")]
    for i, (lab, lo, hi, nom, col) in enumerate(bands):
        ax2.barh(i, (hi - lo) * 1e12, left=lo * 1e12, height=0.42, color=col, alpha=0.85)
        ax2.text((lo + hi) / 2 * 1e12, i + 0.32, f"{lo*1e12:.1f} – {hi*1e12:.1f} pJ",
                 ha="center", fontsize=9.5, fontweight="bold")
        if nom is not None:
            ax2.plot([nom * 1e12], [i], "|", ms=22, mew=2.5, color="#0b3c5d")
            ax2.text(nom * 1e12, i - 0.34, f"nominal {nom*1e12:.1f}", ha="center", fontsize=8.5,
                     color="#0b3c5d")
    overlap_lo, overlap_hi = max(lib_lo, ours_lo), min(lib_hi, ours_hi)
    if overlap_hi > overlap_lo:
        ax2.axvspan(overlap_lo * 1e12, overlap_hi * 1e12, color="#4caf50", alpha=0.13, zorder=0)
        ax2.text((overlap_lo + overlap_hi) / 2 * 1e12, 1.62, "consistent range",
                 ha="center", fontsize=9, color="#2e7d32", style="italic")

    ax2.set_yticks([0, 1])
    ax2.set_yticklabels([b[0] for b in bands], fontsize=9)
    ax2.set_ylim(-0.75, 1.85)
    ax2.set_xlim(0, max(lib_hi, ours_hi) * 1.15e12)
    ax2.set_xlabel("energy per read access (pJ)")
    ax2.set_title("READ energy — bands, not a point test", fontsize=12)
    ax2.grid(axis="x", alpha=0.3)

    fig.suptitle("Absolute cross-check against the vendor's published characterization",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "fig5c_liberty.png", dpi=160)
    plt.close(fig)
    print(f"wrote {OUT}/fig5c_liberty.png")
    print(f"  leakage: ours {ours_leak*1e12:.3f} pW/bit  vendor {lib_leak_per_cell*1e12:.3f} pW/bit"
          f"  ratio {ours_leak/lib_leak_per_cell:.3f}")
    print(f"  read:    ours {ours_lo*1e12:.2f}-{ours_hi*1e12:.2f} pJ (nominal {ours_nom*1e12:.2f})"
          f"  vendor array access {lib_lo*1e12:.2f}-{lib_hi*1e12:.2f} pJ")


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    data = {}
    for name in NAMED_BIAS_POINTS:
        bd = device_power_breakdown_w(name)
        p_ch = sum(d["channel_w"] for d in bd.values())
        p_j = sum(d["junction_w"] for d in bd.values())
        p_src = supply_power_w(name)
        data[name] = {"breakdown": bd, "p_channel_w": p_ch, "p_junction_w": p_j,
                      "p_supply_w": p_src,
                      "ratio_channel": p_ch / p_src, "ratio_full": (p_ch + p_j) / p_src}
        print(f"{name:18s} channel={p_ch:.4e} W  junction={p_j:.4e} W  "
              f"sources={p_src:.4e} W  ratio(full)={data[name]['ratio_full']:.4f}")

    fig_device_power(data)
    fig_conservation(data)
    fig_liberty(data)


if __name__ == "__main__":
    main()
