"""Slide 6a: what the SRAM bitcell IS, and how each SPICE device maps to a
physical piece of silicon.

Left  -- the 6T schematic, with every device carrying the SAME instance name
         (X0..X7) that data/sram_sp_cell.spice uses, so the power numbers on
         slide 5 can be pointed at directly.
Right -- the REAL GDS layout of that same cell (sram22_64x22m4w22.gds /
         sram_sp_cell), with the eight extracted channel regions -- the actual
         DIFF n POLY overlaps that become heat-source volumes in the 3D mesh --
         outlined and labelled by role.

The point of putting them side by side: nothing in the thermal model is a
cartoon. The heat-source boxes are literally these polygons.

    python cases/fig_bitcell_anatomy.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gdstk
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from gds import techmap
from gds.read import flatten_by_layer
from gds.sources import _dims_um, extract_channels

OUT = Path("out/presentation")
GDS_PATH = "data/sram22_64x22m4w22.gds"
CELL_NAME = "sram_sp_cell"
NWELL_SPLIT_X_UM = -0.72   # same constant cases/run_bitcell_compact.py classifies on

ROLE_COLOR = {
    "access": "#1f77b4",
    "latch": "#d62728",
    "pullup": "#e6a700",
    "parasitic": "#9e9e9e",
}
ROLE_LABEL = {
    "access": "access (nfet_pass)",
    "latch": "pull-down (nfet_latch)",
    "pullup": "pull-up (pfet_pass)",
    "parasitic": "parasitic, $V_{ds}\\equiv0$",
}

NWELL = (64, 20)


def _classify(x_ext, y_ext, is_left):
    if abs(y_ext - 0.025) < 0.01:
        return "parasitic"
    if abs(x_ext - 0.21) < 0.01:
        return "latch"
    return "pullup" if is_left else "access"


# ---------------------------------------------------------------- schematic

def _mos_v(ax, x, y, pmos, name, gate_left, color):
    """Vertical MOSFET symbol. Channel bar on the x axis, gate bar offset to
    one side; drain lead exits up, source lead down. `gate_left` puts the gate
    bar (and its lead) on the -x side."""
    s = -1.0 if gate_left else 1.0
    h = 0.34
    ax.plot([x, x], [y - h, y + h], color=color, lw=3.0, solid_capstyle="butt", zorder=4)
    ax.plot([x + 0.15 * s, x + 0.15 * s], [y - h * 0.9, y + h * 0.9],
            color="k", lw=2.0, solid_capstyle="butt", zorder=4)
    lead_x0 = x + 0.15 * s
    lead_x1 = x + (0.42 if not gate_left else -0.42)
    if pmos:
        # inversion bubble sits on the gate lead, just outside the gate bar
        ax.add_patch(plt.Circle((x + 0.24 * s, y), 0.065, fc="w", ec="k", lw=1.6, zorder=5))
        lead_x0 = x + 0.31 * s
    ax.plot([lead_x0, lead_x1], [y, y], color="k", lw=1.4, zorder=3)
    ax.plot([x, x], [y + h, y + h + 0.26], color="k", lw=1.4, zorder=3)
    ax.plot([x, x], [y - h - 0.26, y - h], color="k", lw=1.4, zorder=3)
    ax.text(x - 0.13 * s, y + 0.0, name, fontsize=9.5, fontweight="bold",
            color=color, ha="right" if not gate_left else "left", va="center", zorder=6)


def _mos_h(ax, x, y, name, gate_down, color):
    """Horizontal NMOS symbol (the access transistors): channel bar on the y
    axis, gate bar offset below, source/drain leads exiting left and right."""
    s = -1.0 if gate_down else 1.0
    h = 0.34
    ax.plot([x - h, x + h], [y, y], color=color, lw=3.0, solid_capstyle="butt", zorder=4)
    ax.plot([x - h * 0.9, x + h * 0.9], [y + 0.15 * s, y + 0.15 * s],
            color="k", lw=2.0, solid_capstyle="butt", zorder=4)
    ax.plot([x, x], [y + 0.15 * s, y + (0.42 if not gate_down else -0.42)],
            color="k", lw=1.4, zorder=3)
    ax.plot([x - h - 0.26, x - h], [y, y], color="k", lw=1.4, zorder=3)
    ax.plot([x + h, x + h + 0.26], [y, y], color="k", lw=1.4, zorder=3)
    ax.text(x, y - 0.30 * s, name, fontsize=9.5, fontweight="bold", color=color,
            ha="center", va="top" if not gate_down else "bottom", zorder=6)


def draw_schematic(ax):
    wire = dict(color="k", lw=1.4, zorder=3)
    XL, XR = 1.20, 2.80          # Q column, QB column
    Y_PU, Y_PD = 2.80, 0.70      # pull-up / pull-down device centres
    Y_VDD, Y_VSS = 3.55, -0.05
    GB_L, GB_R = 1.62, 2.38      # gate buses: QB drives left pair, Q drives right
    Y_CROSS_Q, Y_CROSS_QB = 2.18, 1.32
    Y_ACC = 1.75
    X_BL, X_BR = 0.25, 3.75
    Y_WL = -0.85

    # supply rails, deliberately spanning only the inverter columns so the
    # word-line and bit-line leads never have to cross them
    for y, lab, col in ((Y_VDD, "$V_{DD}$", "#c0392b"), (Y_VSS, "$V_{SS}$", "#2c3e50")):
        ax.plot([0.85, 3.15], [y, y], color=col, lw=3.0, zorder=2)
        ax.text(0.78, y, lab, fontsize=11, ha="right", va="center", color=col, fontweight="bold")

    # cross-coupled inverter pair
    _mos_v(ax, XL, Y_PU, True, "X6", gate_left=False, color=ROLE_COLOR["pullup"])
    _mos_v(ax, XL, Y_PD, False, "X1", gate_left=False, color=ROLE_COLOR["latch"])
    _mos_v(ax, XR, Y_PU, True, "X5", gate_left=True, color=ROLE_COLOR["pullup"])
    _mos_v(ax, XR, Y_PD, False, "X7", gate_left=True, color=ROLE_COLOR["latch"])

    for x in (XL, XR):
        ax.plot([x, x], [Y_PU + 0.60, Y_VDD], **wire)       # pull-up to VDD
        ax.plot([x, x], [Y_VSS, Y_PD - 0.60], **wire)       # pull-down to VSS
        ax.plot([x, x], [Y_PD + 0.60, Y_PU - 0.60], **wire)  # the storage node

    # gate buses + the two cross-coupling wires
    ax.plot([GB_L, GB_L], [Y_PD, Y_PU], **wire)
    ax.plot([GB_R, GB_R], [Y_PD, Y_PU], **wire)
    ax.plot([XL, GB_R], [Y_CROSS_Q, Y_CROSS_Q], **wire)     # Q  -> right gates
    ax.plot([XR, GB_L], [Y_CROSS_QB, Y_CROSS_QB], **wire)   # QB -> left gates

    # access transistors, gates down to the word line
    X_A_L, X_A_R = 0.66, 3.34
    _mos_h(ax, X_A_L, Y_ACC, "X2", gate_down=True, color=ROLE_COLOR["access"])
    _mos_h(ax, X_A_R, Y_ACC, "X0", gate_down=True, color=ROLE_COLOR["access"])
    ax.plot([X_A_L + 0.60, XL], [Y_ACC, Y_ACC], **wire)
    ax.plot([XR, X_A_R - 0.60], [Y_ACC, Y_ACC], **wire)
    ax.plot([X_BL, X_A_L - 0.60], [Y_ACC, Y_ACC], **wire)
    ax.plot([X_A_R + 0.60, X_BR], [Y_ACC, Y_ACC], **wire)

    ax.plot([X_BL, X_BL], [Y_ACC, 4.05], color="#1f77b4", lw=2.2, zorder=2)
    ax.plot([X_BR, X_BR], [Y_ACC, 4.05], color="#1f77b4", lw=2.2, zorder=2)
    ax.text(X_BL, 4.15, "BL", fontsize=11, ha="center", color="#1f77b4", fontweight="bold")
    ax.text(X_BR, 4.15, "BR", fontsize=11, ha="center", color="#1f77b4", fontweight="bold")

    ax.plot([X_BL, X_BR], [Y_WL, Y_WL], color="#7b2fbe", lw=2.6, zorder=2)
    ax.plot([X_A_L, X_A_L], [Y_WL, Y_ACC - 0.42], color="#7b2fbe", lw=1.4, zorder=3)
    ax.plot([X_A_R, X_A_R], [Y_WL, Y_ACC - 0.42], color="#7b2fbe", lw=1.4, zorder=3)
    ax.text(X_BL - 0.12, Y_WL, "WL", fontsize=11, ha="right", va="center",
            color="#7b2fbe", fontweight="bold")

    for x, lab in ((XL, "Q"), (XR, "QB")):
        ax.add_patch(plt.Circle((x, Y_ACC), 0.055, fc="k", ec="k", zorder=6))
        ax.text(x + (-0.13 if lab == "Q" else 0.13), Y_ACC + 0.20, lab, fontsize=12,
                fontweight="bold", ha="right" if lab == "Q" else "left")

    # X3/X4 are real LVS-extracted devices with drain shorted to source, so
    # they carry no Vds and no power -- shown so the count matches the netlist.
    ax.text(2.0, -1.20,
            "X3, X4: LVS-extracted parasitics on Q / QB\n(drain tied to source, "
            "$V_{ds}\\equiv0$, no dissipation)",
            fontsize=8.5, ha="center", va="top", style="italic", color="#666")

    ax.set_xlim(-0.15, 4.25)
    ax.set_ylim(-1.75, 4.45)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("6T cell schematic — instance names are the real netlist's\n"
                 "(data/sram_sp_cell.spice)", fontsize=11.5)


# ------------------------------------------------------------------- layout

def draw_layout(ax):
    lib = gdstk.read_gds(GDS_PATH)
    cell = next(c for c in lib.cells if c.name == CELL_NAME)
    by_layer = flatten_by_layer(cell)

    def polys(key, **kw):
        for p in by_layer.get(key, []):
            ax.add_patch(mpatches.Polygon(p.points, closed=True, **kw))

    polys(NWELL, facecolor="#e8eef5", edgecolor="#5b7fa6", lw=1.2, ls="--", zorder=0)
    polys(techmap.DIFF, facecolor="#bfe3c6", edgecolor="#4a8a5c", lw=0.9, zorder=1)
    polys(techmap.POLY, facecolor="#f2b6b6", edgecolor="#b03a45", lw=0.9, alpha=0.85, zorder=2)
    polys(techmap.LICON1, facecolor="#4a4a4a", edgecolor="k", lw=0.5, zorder=3)

    for i, poly in enumerate(extract_channels(by_layer)):
        cx, cy, x_ext, y_ext = _dims_um(poly)
        role = _classify(x_ext, y_ext, is_left=cx < NWELL_SPLIT_X_UM)
        ax.add_patch(mpatches.Rectangle((cx - x_ext / 2, cy - y_ext / 2), x_ext, y_ext,
                                        facecolor=ROLE_COLOR[role], edgecolor="k",
                                        lw=1.3, alpha=0.92, zorder=5))
        # Parasitics are only 25nm tall -- a label centred on them is unreadable,
        # so nudge those outward instead of overlapping the box.
        dy = 0.0 if y_ext > 0.05 else 0.075
        ax.text(cx, cy + dy, f"ch{i}", fontsize=7.5, fontweight="bold", color="w" if not dy else "k",
                ha="center", va="center", zorder=6)

    ax.axvline(NWELL_SPLIT_X_UM, color="#5b7fa6", lw=1.4, ls=":", zorder=4)
    ax.text(NWELL_SPLIT_X_UM, -1.60, "nwell edge:  PMOS ←  |  → NMOS", fontsize=8.5,
            ha="center", va="bottom", color="#3c6690",
            bbox=dict(fc="w", ec="none", alpha=0.85, pad=1.5))

    ax.set_xlim(-1.26, 0.06)
    ax.set_ylim(-1.64, 0.06)
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title("Real GDS layout — highlighted boxes are the\n"
                 "DIFF ∩ POLY channels used as heat sources", fontsize=11.5)

    handles = [mpatches.Patch(facecolor=ROLE_COLOR[r], edgecolor="k", label=ROLE_LABEL[r])
               for r in ("access", "latch", "pullup", "parasitic")]
    handles += [
        mpatches.Patch(facecolor="#bfe3c6", edgecolor="#4a8a5c", label="diffusion (DIFF)"),
        mpatches.Patch(facecolor="#f2b6b6", edgecolor="#b03a45", label="gate (POLY)"),
        mpatches.Patch(facecolor="#4a4a4a", edgecolor="k", label="contact (LICON1)"),
    ]
    # Below the axes, not inside: the cell is only 1.2x1.6um and every inch of
    # it has a channel in it -- an inset legend covers real geometry.
    ax.legend(handles=handles, fontsize=8.2, ncol=2, framealpha=1.0,
              loc="upper center", bbox_to_anchor=(0.5, -0.10))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 7.0),
                                   gridspec_kw={"width_ratios": [1.35, 1.0]})
    draw_schematic(ax1)
    draw_layout(ax2)
    fig.tight_layout()
    fig.savefig(OUT / "fig6a_bitcell_anatomy.png", dpi=170)
    plt.close(fig)
    print(f"wrote {OUT}/fig6a_bitcell_anatomy.png")


if __name__ == "__main__":
    main()
