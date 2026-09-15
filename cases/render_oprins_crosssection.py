"""A cross-section styled to sit next to [Oprins] iTherm 2022 Fig. 5 (left)
for side-by-side comparison in a talk.

Styling only -- every rectangle is still taken from the real emitters
(mesh/gds_volume.py) and the real band table (gds/techmap.py), so this is the
geometry that actually gets meshed and solved, drawn to look like the paper's
schematic rather than like a mesh dump. The colours and labels imitate theirs
(purple buried rails, pink backside uTSV, grey wells) purely so the two
figures can be read against each other.

Cut ACROSS the buried rails (plane of constant x through a backside track and
a uTSV, plotted against y) -- the same view direction as their figure, which
is why their VDD/VSS rails appear as discrete blocks.

    python cases/render_oprins_crosssection.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from cases.run_bspdn_benchmark_3d import CENTER_UM
from cases.run_oprins_reproduction import thinned_stack
from gds import techmap
from mesh.gds_volume import _synthetic_grid_rects

OUT = Path("out/oprins_geometry")
SPAN_UM = 1.30            # y half-width shown, a few rail pitches like their figure

# [Oprins]-like palette
C_TSV = "#f1808a"
C_RAIL = "#7b2fbe"
C_PWELL = "#8e9298"
C_NWELL = "#b9c4cc"
C_PSUB = "#d3d6da"
C_DEVICE = "#cfe6f5"
C_METAL = "#ef4444"
C_OXIDE = "#eef4f8"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stack = thinned_stack(0.5, with_metal=True, si_material="Si_oprins_500nm")
    zb = list(techmap.z_bounds(stack))
    band = {b.name: (z0, z1, b) for z0, z1, b in zb}

    y_lo, y_hi = CENTER_UM - SPAN_UM, CENTER_UM + SPAN_UM
    x0, x1 = CENTER_UM - 1.0, CENTER_UM + 1.0
    anchor = (CENTER_UM, CENTER_UM)

    z_m0, z_m1, b_m = band["backside_metal"]
    z_p0, z_p1, b_p = band["p_substrate"]
    z_w0, z_w1, b_w = band["well"]
    z_r0, z_r1, b_r = band["buried_power_rail"]
    z_dev_top = band["poly"][1]

    fig, ax = plt.subplots(figsize=(10.5, 7.4))

    # device side above the rails
    ax.add_patch(mpatches.Rectangle((y_lo, z_r1), y_hi - y_lo, z_dev_top - z_r1,
                                     facecolor=C_DEVICE, edgecolor="#9fb6c4", lw=0.8))
    # p substrate (bottom 100nm) and the well region above it
    ax.add_patch(mpatches.Rectangle((y_lo, z_p0), y_hi - y_lo, z_p1 - z_p0,
                                     facecolor=C_PSUB, edgecolor="none"))
    ax.add_patch(mpatches.Rectangle((y_lo, z_w0), y_hi - y_lo, z_w1 - z_w0,
                                     facecolor=C_PWELL, edgecolor="none"))
    # rail band background: silicon, not empty space (see the band table)
    ax.add_patch(mpatches.Rectangle((y_lo, z_r0), y_hi - y_lo, z_r1 - z_r0,
                                     facecolor=C_PWELL, edgecolor="none"))
    # bonding oxide + backside metal
    ax.add_patch(mpatches.Rectangle((y_lo, z_m0 - 0.22), y_hi - y_lo, 0.22,
                                     facecolor=C_OXIDE, edgecolor="#9fb6c4", lw=0.8))
    ax.add_patch(mpatches.Rectangle((y_lo, z_m0), y_hi - y_lo, z_m1 - z_m0,
                                     facecolor=C_METAL, edgecolor="k", lw=0.8))

    # uTSVs on THIS track: they span p substrate + well, i.e. the full 500nm
    vias = _synthetic_grid_rects(x0, y_lo, x1, y_hi, *b_w.synthetic_grid,
                                  b_w.synthetic_shape, b_w.synthetic_width_y_um,
                                  anchor, b_w.synthetic_pitch_y_um, b_w.synthetic_stagger_y_um)
    on_track = sorted([v for v in vias if v[0] - 1e-9 <= CENTER_UM <= v[2] + 1e-9],
                       key=lambda v: v[1])
    via_spans = [(v[1], v[3]) for v in on_track]

    rails = sorted(_synthetic_grid_rects(x0, y_lo, x1, y_hi, *b_r.synthetic_grid,
                                          b_r.synthetic_shape, b_r.synthetic_width_y_um,
                                          anchor, b_r.synthetic_pitch_y_um,
                                          b_r.synthetic_stagger_y_um), key=lambda r: r[1])

    def over_via(ry0, ry1):
        c = (ry0 + ry1) / 2
        return any(v0 - 1e-9 <= c <= v1 + 1e-9 for v0, v1 in via_spans)

    # nwell where a rail sits on a via (that rail is VDD), pwell elsewhere --
    # [Oprins] Fig. 5: VDD lands on the uTSV, VSS sits on the pwell.
    for ry0, ry1 in [(r[1], r[3]) for r in rails]:
        if not over_via(ry0, ry1):
            continue
        # nwell spans ONE rail pitch centred on the VDD rail, so nwell and
        # pwell alternate 1:1 exactly as the rails do.
        half_w = b_r.synthetic_grid[0] / 2
        w0, w1 = (ry0 + ry1) / 2 - half_w, (ry0 + ry1) / 2 + half_w
        ax.add_patch(mpatches.Rectangle((w0, z_w0), w1 - w0, z_r1 - z_w0,
                                         facecolor=C_NWELL, edgecolor="none", zorder=2))

    # The via spans the full 500nm Si and protrudes only a LITTLE below it,
    # rather than running the whole backside-metal thickness.
    overhang = 0.05
    for vy0, vy1 in via_spans:
        ax.add_patch(mpatches.Rectangle((vy0, z_p0 - overhang), vy1 - vy0,
                                         z_w1 - (z_p0 - overhang),
                                         facecolor=C_TSV, edgecolor="#b03a45", lw=1.1,
                                         alpha=0.80, zorder=3))

    for ry0, ry1 in [(r[1], r[3]) for r in rails]:
        ax.add_patch(mpatches.Rectangle((ry0, z_r0), ry1 - ry0, z_r1 - z_r0,
                                         facecolor=C_RAIL, edgecolor="k", lw=0.5, zorder=5))
        if abs((ry0 + ry1) / 2 - CENTER_UM) <= 0.62:
            ax.text((ry0 + ry1) / 2, z_r1 + 0.010,
                     "VDD" if over_via(ry0, ry1) else "VSS",
                     ha="center", fontsize=8, color=C_RAIL, fontweight="bold", rotation=90)

    # ---- annotations, matching theirs ----
    y_ann = y_lo + 0.09
    ax.annotate("", xy=(y_ann, z_p0), xytext=(y_ann, z_w1),
                 arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=1.7))
    ax.text(y_ann + 0.03, (z_p0 + z_w1) / 2, f"{(z_w1-z_p0)*1000:.0f}nm",
             color="#c0392b", fontsize=12, va="center", fontweight="bold")
    ax.annotate("", xy=(y_hi - 0.30, z_p0), xytext=(y_hi - 0.30, z_p1),
                 arrowprops=dict(arrowstyle="<->", color="#2e7d32", lw=1.5))
    ax.text(y_hi - 0.33, (z_p0 + z_p1) / 2, f"~{(z_p1-z_p0)*1000:.0f}nm", color="#2e7d32",
             fontsize=10, va="center", ha="right", fontweight="bold")
    vy0, vy1 = via_spans[len(via_spans) // 2]
    ax.annotate("", xy=(vy0, z_p0 - 0.11), xytext=(vy1, z_p0 - 0.11),
                 arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=1.6))
    ax.text((vy0 + vy1) / 2, z_p0 - 0.16, f"{(vy1-vy0)*1000:.0f}nm", color="#c0392b",
             ha="center", fontsize=11, fontweight="bold")

    z_lab = z_w0 + (z_w1 - z_w0) * 0.16
    for ry0, ry1 in [(r[1], r[3]) for r in rails]:
        c = (ry0 + ry1) / 2
        if abs(c - CENTER_UM) > 0.40:
            continue
        ax.text(c, z_lab, "nwell" if over_via(ry0, ry1) else "pwell", ha="center",
                 va="center", fontsize=7.5, rotation=90,
                 color="#33383d" if over_via(ry0, ry1) else "white")

    ax.text(y_lo + 0.05, z_dev_top - 0.04, "PMOS / NMOS (FEOL)", fontsize=10,
             color="#2c5c77", va="top")
    ax.text(y_hi - 0.05, (z_p0 + z_p1) / 2, "p substrate", fontsize=9,
             color="#404040", ha="right", va="center", style="italic")
    ax.text(y_hi - 0.05, (z_m0 + z_m1) / 2, "backside metal M1 (Cu)", fontsize=9,
             color="white", ha="right", va="center", fontweight="bold")
    ax.annotate("Backside uTSV", xy=((vy0 + vy1) / 2, (z_m1 + z_w0) / 2),
                 xytext=(CENTER_UM + 0.70, z_m0 - 0.15), fontsize=12, color=C_TSV,
                 fontweight="bold", ha="center",
                 arrowprops=dict(arrowstyle="->", color=C_TSV, lw=1.8))
    ax.legend(handles=[
        mpatches.Patch(facecolor=C_NWELL, label="nwell (under the VDD rail / uTSV)"),
        mpatches.Patch(facecolor=C_PWELL, label="pwell (under the VSS rail)"),
        mpatches.Patch(facecolor=C_RAIL, label="buried power rail (Ru)"),
        mpatches.Patch(facecolor=C_TSV, label="backside uTSV (W)"),
    ], loc="upper right", fontsize=8.5, framealpha=0.92)

    ax.set_xlim(y_lo, y_hi)
    ax.set_ylim(z_m0 - 0.28, z_dev_top)
    ax.set_xlabel("y (um) — across the buried rails")
    ax.set_ylabel("z (um)")
    ax.set_title("BS-PDN cross section (matched to Oprins)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "oprins_crosssection_paperstyle.png", dpi=170)
    print(f"wrote {OUT}/oprins_crosssection_paperstyle.png")
    print(f"  Si total {(z_w1-z_p0)*1000:.0f}nm = {(z_p1-z_p0)*1000:.0f}nm p substrate "
          f"+ {(z_w1-z_w0)*1000:.0f}nm well; uTSV spans all of it")
    print(f"  uTSV {(vy1-vy0)*1000:.0f}nm along-track, {len(via_spans)} on this track in view")
    print(f"  rails {b_r.synthetic_grid[1]*1000:.0f}nm wide @ {b_r.synthetic_grid[0]*1000:.0f}nm "
          f"pitch, {(z_r1-z_r0)*1000:.0f}nm tall, sitting ON TOP of the Si")


if __name__ == "__main__":
    main()
