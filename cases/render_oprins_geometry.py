"""Render the ACTUAL emitted BS-PDN geometry for visual comparison against
[Oprins] iTherm 2022 Fig. 5 (cross-section + top view) and Table 1.

Deliberately draws the real output of mesh/gds_volume.py's emitters and
gds/techmap.py's band table -- NOT a separate illustration -- so what is shown
is exactly what gets meshed and solved. A pretty picture that agrees with the
paper while the mesh disagrees would be worse than useless.

    python cases/render_oprins_geometry.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from cases.run_bspdn_benchmark import SOURCE_L_UM, SOURCE_W_UM
from cases.run_bspdn_benchmark_3d import CENTER_UM
from cases.run_oprins_reproduction import thinned_stack
from gds import techmap
from mesh.gds_volume import _synthetic_grid_rects

OUT = Path("out/oprins_geometry")
WINDOW_UM = 2.5           # a few pitches, enough to read the pattern
MAT_COLOR = {"Cu_oprins": "#c87137", "W": "#5a5a66", "Ru_oprins": "#7d7f88",
             "SiO2": "#dce8f0", "Si_oprins_500nm": "#9fb8c8", "Si_bulk": "#b9cad6",
             "SiN": "#e3d6ef", "Al": "#d8d8de", "PolySi": "#c9a0dc",
             "TiN": "#c05a4a", "Si_SD_doped": "#8fae9f"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stack = thinned_stack(0.5, with_metal=True, si_material="Si_oprins_500nm")
    half = WINDOW_UM / 2
    x0, x1 = CENTER_UM - half, CENTER_UM + half
    y0, y1 = CENTER_UM - half, CENTER_UM + half
    anchor = (CENTER_UM, CENTER_UM)

    fig, (ax_top, ax_cut) = plt.subplots(1, 2, figsize=(15, 7))

    # ---------------- TOP VIEW (vs [Oprins] Fig. 5 right) ----------------
    metal = stack.find("backside_metal")
    si = stack.find("Si_substrate")
    m_rects = _synthetic_grid_rects(x0, y0, x1, y1, *metal.synthetic_grid,
                                     metal.synthetic_shape, metal.synthetic_width_y_um, anchor)
    v_rects = _synthetic_grid_rects(x0, y0, x1, y1, *si.synthetic_grid,
                                     si.synthetic_shape, si.synthetic_width_y_um, anchor, si.synthetic_pitch_y_um, si.synthetic_stagger_y_um)
    bpr = stack.find("buried_power_rail") if any(
        b.name == "buried_power_rail" for _, _, b in techmap.z_bounds(stack)) else None
    if bpr is not None:
        for rx0, ry0, rx1, ry1 in _synthetic_grid_rects(
                x0, y0, x1, y1, *bpr.synthetic_grid, bpr.synthetic_shape,
                bpr.synthetic_width_y_um, anchor, bpr.synthetic_pitch_y_um, bpr.synthetic_stagger_y_um):
            ax_top.add_patch(mpatches.Rectangle((rx0, ry0), rx1 - rx0, ry1 - ry0,
                                                 facecolor="#2a9d8f", lw=0, zorder=1))
    for rx0, ry0, rx1, ry1 in m_rects:
        ax_top.add_patch(mpatches.Rectangle((rx0, ry0), rx1 - rx0, ry1 - ry0,
                                             facecolor=MAT_COLOR["Cu_oprins"], alpha=0.85, lw=0))
    for rx0, ry0, rx1, ry1 in v_rects:
        ax_top.add_patch(mpatches.Rectangle((rx0, ry0), rx1 - rx0, ry1 - ry0,
                                             facecolor=MAT_COLOR["W"], lw=0))
    ax_top.add_patch(mpatches.Rectangle(
        (CENTER_UM - SOURCE_W_UM / 2, CENTER_UM - SOURCE_L_UM / 2), SOURCE_W_UM, SOURCE_L_UM,
        facecolor="none", edgecolor="#d81b60", lw=2.4, ls="--", zorder=5))

    # pitch/width annotations, read off the band table rather than retyped
    mp, mw = metal.synthetic_grid
    vp, vw = si.synthetic_grid
    vwy = si.synthetic_width_y_um
    ax_top.annotate("", xy=(CENTER_UM - mw / 2, y1 - 0.18), xytext=(CENTER_UM - mw / 2 + mp, y1 - 0.18),
                     arrowprops=dict(arrowstyle="<->", color="k", lw=1.3))
    ax_top.text(CENTER_UM - mw / 2 + mp / 2, y1 - 0.14, f"M1 pitch {mp*1000:.0f}nm",
                 ha="center", fontsize=9)
    ax_top.annotate("", xy=(CENTER_UM - mw / 2, y0 + 0.12), xytext=(CENTER_UM + mw / 2, y0 + 0.12),
                     arrowprops=dict(arrowstyle="<->", color="k", lw=1.3))
    ax_top.text(CENTER_UM, y0 + 0.16, f"M1 width {mw*1000:.0f}nm", ha="center", fontsize=9)
    ax_top.set_xlim(x0, x1); ax_top.set_ylim(y0, y1); ax_top.set_aspect("equal")
    ax_top.set_xlabel("x (um)"); ax_top.set_ylabel("y (um)")
    ax_top.set_title(f"TOP VIEW — backside metal + uTSVs  (vs [Oprins] Fig. 5 right)\n"
                      f"Cu lines {mw*1000:.0f}nm wide @ {mp*1000:.0f}nm pitch, "
                      f"uTSV {vw*1000:.0f}x{vwy*1000:.0f}nm @ {vp*1000:.0f}nm", fontsize=10)
    ax_top.legend(handles=[
        mpatches.Patch(color=MAT_COLOR["Cu_oprins"], label="backside metal M1 (Cu)"),
        mpatches.Patch(color=MAT_COLOR["W"], label="uTSV (W)"),
        mpatches.Patch(color="#2a9d8f", label="FS buried power rail (Ru)"),
        mpatches.Patch(facecolor="none", edgecolor="#d81b60", ls="--", lw=2,
                        label=f"heat source {SOURCE_W_UM*1000:.0f}x{SOURCE_L_UM*1000:.0f}nm"),
    ], loc="upper left", fontsize=8, framealpha=0.9)

    # ------------- CROSS-SECTION (vs [Oprins] Fig. 5 left) -------------
    # Cut ACROSS the buried rails (a plane of constant x, plotted against y),
    # which is how [Oprins] Fig. 5 left is taken: their VDD/VSS rails appear
    # as DISCRETE blocks, which only happens looking across them. Cutting
    # along a rail instead renders it as one solid band and hides the
    # structure being compared.
    for z0, z1, band in techmap.z_bounds(stack):
        ax_cut.add_patch(mpatches.Rectangle((y0, z0), y1 - y0, z1 - z0,
                                             facecolor=MAT_COLOR.get(band.background, "#eeeeee"),
                                             edgecolor="#8894a0", lw=0.6))
        for key, material in band.drawn:
            if key is not techmap.SYNTHETIC:
                continue
            p_, w_ = band.synthetic_grid
            for rx0, ry0, rx1, ry1 in _synthetic_grid_rects(
                    x0, y0, x1, y1, p_, w_, band.synthetic_shape,
                    band.synthetic_width_y_um, anchor, band.synthetic_pitch_y_um, band.synthetic_stagger_y_um):
                if not (rx0 - 1e-9 <= CENTER_UM <= rx1 + 1e-9):
                    continue    # this feature isn't cut by the x=CENTER plane
                ax_cut.add_patch(mpatches.Rectangle((ry0, z0), ry1 - ry0, z1 - z0,
                                                     facecolor=MAT_COLOR.get(material, "#888888"),
                                                     edgecolor="k", lw=0.4))
        ax_cut.text(y1 + 0.03, (z0 + z1) / 2, f"{band.name}  ({z1-z0:.3f}um)",
                     va="center", fontsize=7.5)

    zb = list(techmap.z_bounds(stack))
    z_metal = [a for a, _b, bd in zb if bd.name == "backside_metal"]
    z_lo = (z_metal[0] - 0.15) if z_metal else zb[0][1]
    z_hi = [b for _a, b, bd in zb if bd.name == "poly"][0]
    ch = [(a, b) for a, b, bd in zb if bd.name == "channel"][0]
    sw, sl = SOURCE_W_UM, SOURCE_L_UM
    ax_cut.add_patch(mpatches.Rectangle(
        (CENTER_UM - sl / 2, ch[0]), sl, max(ch[1] - ch[0], 0.012),
        facecolor="#d81b60", edgecolor="#d81b60", lw=2.0, zorder=6))
    ax_cut.annotate("heat source", xy=(CENTER_UM, ch[1]), xytext=(CENTER_UM + 0.5, ch[1] + 0.09),
                     fontsize=9, color="#d81b60",
                     arrowprops=dict(arrowstyle="->", color="#d81b60", lw=1.4))
    ax_cut.set_xlim(y0, y1); ax_cut.set_ylim(z_lo, z_hi)
    ax_cut.set_xlabel("y (um)  — across the buried rails"); ax_cut.set_ylabel("z (um)")
    ax_cut.set_title(f"CROSS-SECTION at x={CENTER_UM}um (through a track + uTSV)\n"
                      "BS metal -> uTSV through thinned Si -> buried rail -> device "
                      "(vs [Oprins] Fig. 5 left)", fontsize=10)

    fig.suptitle("BS-PDN geometry AS MESHED — drawn from mesh/gds_volume.py's own emitter output",
                  fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "oprins_geometry.png", dpi=160)
    print(f"wrote {OUT}/oprins_geometry.png")

    print("\nband table as meshed:")
    for z0, z1, b in techmap.z_bounds(stack):
        drawn = ",".join(m for _, m in b.drawn) or "-"
        print(f"  {b.name:20s} z={z0:9.4f}..{z1:9.4f}  t={z1-z0:7.4f}um  "
              f"bg={b.background:16s} drawn={drawn:12s} shape={b.synthetic_shape} "
              f"grid={b.synthetic_grid} wy={b.synthetic_width_y_um}")


if __name__ == "__main__":
    main()
