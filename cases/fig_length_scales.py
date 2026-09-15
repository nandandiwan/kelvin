"""Why a bitcell that shows only ~17 mK of transistor-to-transistor structure
nevertheless sits 31 K above ambient -- and what sets each of those numbers.

The two numbers live at different length scales and are set by different
resistances. Written as a series chain from the channel to the ambient:

    T_channel - T_ambient = P * (R_device + R_vertical + R_sink)

      R_device    channel -> the rest of the cell. Sub-micron. TINY, because
                  heat leaving a 0.15um source spreads into a solid angle
                  immediately. This is the only term that differs from device
                  to device, and it is the only reason a 3D mesh is needed.
      R_vertical  cell -> backside, through ~53um of silicon. Small.
      R_sink      backside -> ambient, through h over the cell's OWN
                  footprint. Dominant -- 99.3% of the total.

So "31 K" is not the silicon heating up. It is one 1.9um^2 patch of heat sink
being asked to carry 1.18uW by itself, which is a power density of 62 W/cm^2.
That is what adiabatic side walls MEAN: mirror symmetry, i.e. every cell in
the array doing the same thing at the same instant, forever.

    python cases/fig_length_scales.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("out/presentation")
CACHE = Path("out/bitcell_oppoints")
ISO = Path("out/bitcell_isolation")

H_EFF = 20000.0        # W/m^2/K, spec/chip.py's backside_h_eff for these runs
K_SI = 148.0           # W/m/K, Si_bulk
CELL_X_UM, CELL_Y_UM = 1.2, 1.58


def budget(point="crowbar"):
    """Decompose the solved field into the three series terms, and compare the
    two that have closed forms against those closed forms."""
    d = np.load(CACHE / f"{point}.npz")
    dof, dT, p_w = d["dof_um"], d["dT"], float(d["total_power_w"])
    area_m2 = CELL_X_UM * 1e-6 * CELL_Y_UM * 1e-6
    t_m = (dof[:, 2].max() - dof[:, 2].min()) * 1e-6

    bottom = np.isclose(dof[:, 2], dof[:, 2].min(), atol=1e-7)
    z_hot = dof[np.argmax(dT), 2]
    plane = np.isclose(dof[:, 2], z_hot, atol=1e-7)

    dt_sink = dT[bottom].mean()
    dt_dev = dT[plane].max() - dT[plane].min()
    dt_vert = dT.max() - dT[bottom].mean() - dt_dev
    return {
        "p_w": p_w, "area_m2": area_m2, "total": float(dT.max()),
        "solved": [dt_sink, dt_vert, dt_dev],
        # Closed forms for the two uniform-slab terms. The device term has no
        # clean analytic counterpart (the channel sits in a stack of four
        # different materials), so it is measured only -- stated, not faked.
        "analytic": [p_w / (H_EFF * area_m2), p_w * t_m / (K_SI * area_m2), None],
        "t_m": t_m,
    }


def panel_budget(ax, b):
    labels = ["$R_{\\mathrm{sink}}$\nbackside → ambient\n$1/(hA_{\\mathrm{cell}})$",
              "$R_{\\mathrm{vertical}}$\nchannel → backside\n$t/(kA)$, t = %.0f µm" % (b["t_m"] * 1e6),
              "$R_{\\mathrm{device}}$\ndevice → device\n(3D spreading)"]
    colors = ["#d62728", "#1f77b4", "#e6a700"]
    y = np.arange(3)[::-1]

    ax.barh(y, b["solved"], height=0.55, color=colors)
    for yi, v, a in zip(y, b["solved"], b["analytic"]):
        ax.text(v * 1.35, yi, f"{v:.4g} K   ({v/b['total']*100:.2f}%)",
                va="center", fontsize=10, fontweight="bold")
        if a is not None:
            ax.plot([a], [yi], "|", ms=26, mew=2.5, color="#111111", zorder=5)

    ax.plot([], [], "|", ms=14, mew=2.5, color="#111111", label="closed form")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xscale("log")
    ax.set_xlim(5e-3, 4e3)
    ax.set_xlabel("contribution to $T_{\\mathrm{channel}} - T_{\\mathrm{ambient}}$  (K)")
    ax.set_title(f"Where the {b['total']:.1f} K actually comes from\n"
                 f"three resistances in series, {b['p_w']*1e6:.2f} µW through all of them",
                 fontsize=12)
    ax.grid(axis="x", alpha=0.3, which="major")
    ax.legend(fontsize=9, loc="lower right")


def panel_isolation(ax, b):
    """dT vs how much heat-sink area the cell is allowed to use.

    Pad = 0 is the as-built model: adiabatic walls on the cell boundary, i.e.
    array-periodic, every neighbour equally hot. Growing the pad hands the
    cell its idle neighbours' share of the sink.
    """
    files = sorted(ISO.glob("pad_*.json"), key=lambda f: float(f.stem.split("_")[1]))
    if not files:
        ax.text(0.5, 0.5, "run cases/run_bitcell_isolation.py", transform=ax.transAxes,
                ha="center", fontsize=11, color="#888")
        ax.set_axis_off()
        return

    pads, dts = [], []
    for f in files:
        d = json.loads(f.read_text())
        pads.append(d["pad_um"])
        dts.append(d["tmax_k"] - 300.0)
    pads, dts = np.array(pads), np.array(dts)
    areas_m2 = (CELL_X_UM + 2 * pads) * (CELL_Y_UM + 2 * pads) * 1e-12
    areas = areas_m2 * 1e12

    # The same three-term series chain as the left panel, now as a PREDICTION
    # across the sweep. Only the device term is taken from the solve (measured
    # once, at pad 0); the other two are closed forms in A. It is pad-
    # independent because it is local physics -- which is exactly why the
    # measured curve pulls away from 1/(hA) at large pad: not lateral
    # conduction kicking in, but a FIXED term becoming a visible fraction as
    # the sink term shrinks.
    dt_sink = b["p_w"] / (H_EFF * areas_m2)
    dt_vert = b["p_w"] * b["t_m"] / (K_SI * areas_m2)
    dt_dev = b["solved"][2]
    model = dt_sink + dt_vert + dt_dev
    err = np.abs(model - dts) / dts * 100

    ax.loglog(areas, dts, "o", color="#1f77b4", ms=9, zorder=5, label="solved (3D FEM)")
    ax.loglog(areas, model, "-", color="#1f77b4", lw=2.0,
              label="$P/(hA) + Pt/(kA) + \\Delta T_{\\mathrm{device}}$")
    ax.loglog(areas, dt_sink, "--", color="#d62728", lw=1.6,
              label="sink term $P/(hA)$ alone")
    ax.axhline(dt_dev, color="#e6a700", lw=1.6, ls=":")
    ax.text(areas.max() * 0.95, dt_dev * 1.3,
            f"device term, {dt_dev*1e3:.1f} mK — does not shrink", fontsize=8.5,
            color="#8a6500", ha="right")

    l_heal_um = np.sqrt(K_SI * b["t_m"] / H_EFF) * 1e6
    ax.text(0.035, 0.05,
            f"3-term model, one measured constant:\n"
            f"max error {err.max():.2f}% over a {dts.max()/dts.min():.0f}× range in $\\Delta T$.\n"
            f"No lateral-conduction term is needed — "
            f"$\\sqrt{{kt/h}}$ = {l_heal_um:.0f} µm $\\gg$ {np.sqrt(areas.max()):.0f} µm here.",
            transform=ax.transAxes, fontsize=8.8, color="#2e7d32",
            bbox=dict(fc="w", ec="#2e7d32", alpha=0.92))

    ax.set_xlabel("heat-sink area available to this cell  (µm²)")
    ax.set_ylabel("$\\Delta T_{max}$  (K)")
    ax.set_title("What if the neighbours are idle?\n"
                 "same cell, same power, more sink to spread into", fontsize=12)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8.8, loc="upper right")
    print(f"  isolation sweep: 3-term model max error {err.max():.2f}% "
          f"over dT {dts.max():.4f} -> {dts.min():.4f} K")


def fig_activity():
    """What an actual ARRAY runs at, versus what the single-cell worst case assumes.

    The single-cell model's 62.5 W/cm^2 is "every cell crowbarring, every cycle,
    forever". A real macro is nothing like that: one word line selects 64 of
    16384 cells per access, and the vendor's own Liberty file states the energy
    per access, so the array's real power density can be computed rather than
    guessed.

    The array is 176um on a side -- still 3.6x SMALLER than the 629um healing
    length -- so it is isothermal end to end and dT = P/(h*A_array) applies to
    the whole block exactly as it did to one cell. That is why this arithmetic
    is legitimate without another 3D solve.
    """
    from gds.power import mine_access_timing, parse_lib
    lib_path = "data/sram22_2048x8m8w1_tt_025C_1v80.lib"
    m = parse_lib(lib_path)
    f_hz = 1e9 / mine_access_timing(lib_path).min_period_ns
    area_m2 = 2048 * 8 * (1.2e-6 * 1.58e-6)

    cases = [
        ("idle\n(clock gated,\nleakage only)", m.leakage_w, "#9e9e9e"),
        ("deselected\n(clock running,\nce = 0)", m.leakage_w + f_hz * m.energy_per_idle_cycle_j(False), "#4c9f70"),
        ("reading\nevery cycle\n(100% activity)", m.total_power_w(f_hz, 1.0, False), "#1f77b4"),
        ("single-cell\nCROWBAR model\n(all cells, always)", 62.458e4 * area_m2, "#d62728"),
    ]

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    dts = [p / (H_EFF * area_m2) for _, p, _ in cases]
    ax.bar(range(len(cases)), dts, 0.6, color=[c for _, _, c in cases])
    for i, ((lab, p, _), dt) in enumerate(zip(cases, dts)):
        ax.text(i, dt * 1.6, f"{dt:.3g} K\n{p/area_m2/1e4:.3g} W/cm²",
                ha="center", fontsize=9.5, fontweight="bold")
    ax.set_yscale("log")
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels([c[0] for c in cases], fontsize=9)
    ax.set_ylim(min(dts) / 8, max(dts) * 30)
    ax.set_ylabel("array temperature rise (K)")
    ax.set_title(f"A {2048*8//1024}k-bit SRAM macro at realistic duty\n"
                 f"vendor Liberty energy per access, {f_hz/1e6:.0f} MHz, "
                 f"{area_m2*1e12:.0f} µm² array", fontsize=12.5)
    ax.grid(axis="y", alpha=0.3, which="major")
    fig.tight_layout()
    fig.savefig(OUT / "fig6g_activity.png", dpi=165)
    plt.close(fig)
    for (lab, p, _), dt in zip(cases, dts):
        print(f"  {lab.splitlines()[0]:14s} P={p*1e3:9.4f} mW  "
              f"{p/area_m2/1e4:7.3f} W/cm^2  dT={dt:.4g} K")
    print(f"wrote {OUT}/fig6g_activity.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    b = budget("crowbar")
    print(f"P = {b['p_w']*1e6:.4f} uW over {b['area_m2']*1e12:.3f} um^2 "
          f"= {b['p_w']/b['area_m2']/1e4:.1f} W/cm^2")
    for lab, s, a in zip(("sink", "vertical", "device"), b["solved"], b["analytic"]):
        extra = "" if a is None else f"   closed form {a:.6g} K  ({abs(s-a)/a*100:.2f}% off)"
        print(f"  {lab:9s} {s:.6g} K  ({s/b['total']*100:6.3f}%){extra}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.6))
    panel_budget(ax1, b)
    panel_isolation(ax2, b)
    fig.suptitle("The bitcell's 31 K and its 17 mK live at different length scales",
                 fontsize=13.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig6e_length_scales.png", dpi=165)
    plt.close(fig)
    print(f"wrote {OUT}/fig6e_length_scales.png")
    fig_activity()


if __name__ == "__main__":
    main()
