"""Top-down (X-Y plan) view of a raw GDS layout — the "standard" chip-layout
view (nwell/diff/poly, classic layout-viewer palette), as opposed to the
X-Z cross-section renders in mesh/gds_viz.py. Operates directly on a
flattened `by_layer` dict; no gmsh/dolfinx involved.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .techmap import DIFF, NWELL, POLY

# Classic IC-layout-viewer palette (klayout/magic-style), not the physical
# material colors mesh/viz.py uses — this is a plan-view of drawn mask
# layers, so the convention readers of a chip layout already recognize.
LAYER_COLORS = {
    "nwell": "#e8d9a0",
    "diff": "#4a9b4a",
    "poly": "#c23b3b",
}


def _add_layer(ax, by_layer, key, name, alpha=1.0, zorder=0):
    polys = by_layer.get(key, [])
    if not polys:
        return
    verts = [p.points for p in polys]
    ax.add_collection(PolyCollection(
        verts, facecolors=LAYER_COLORS[name], edgecolors="none", alpha=alpha, zorder=zorder
    ))


def render_top_view(by_layer, die_bounds_um, out_path, highlight_y_um=None, inset_center_um=None,
                     inset_size_um=12.0):
    """die_bounds_um = (x0, x1, y0, y1). `highlight_y_um` draws a line marking
    where a 2D cross-section cutline would be taken (ties this render to the
    cross-section render). `inset_center_um` (x, y) picks where the zoomed
    panel looks — defaults to (die center x, highlight_y_um) if omitted, so
    the inset shows real, individually-recognizable transistors rather than
    the unresolvable texture the full view shows at chip scale.
    """
    x0, x1, y0, y1 = die_bounds_um
    if inset_center_um is None:
        cy = highlight_y_um if highlight_y_um is not None else (y0 + y1) / 2
        inset_center_um = ((x0 + x1) / 2, cy)
    icx, icy = inset_center_um
    ix0, ix1 = icx - inset_size_um / 2, icx + inset_size_um / 2
    iy0, iy1 = icy - inset_size_um / 2, icy + inset_size_um / 2

    height_ratio = (y1 - y0) / (x1 - x0)
    fig_h = min(max(6.0, 6.0 * height_ratio), 24.0)
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(14, fig_h), gridspec_kw={"width_ratios": [1, 1]})

    for ax in (ax_full, ax_zoom):
        _add_layer(ax, by_layer, NWELL, "nwell", alpha=0.6, zorder=0)
        _add_layer(ax, by_layer, DIFF, "diff", zorder=1)
        _add_layer(ax, by_layer, POLY, "poly", zorder=2)

    ax_full.set_xlim(x0, x1)
    ax_full.set_ylim(y0, y1)
    ax_full.set_aspect("equal")
    ax_full.set_xlabel("x (um)")
    ax_full.set_ylabel("y (um)")
    ax_full.set_title(f"Full macro, top view ({x1-x0:.1f} x {y1-y0:.1f} um)")
    if highlight_y_um is not None:
        ax_full.axhline(highlight_y_um, color="black", linewidth=1.0, linestyle="--", zorder=5)
        ax_full.add_patch(plt.Rectangle((ix0, iy0), inset_size_um, inset_size_um,
                                         fill=False, edgecolor="black", linewidth=1.2, zorder=5))

    ax_zoom.set_xlim(ix0, ix1)
    ax_zoom.set_ylim(iy0, iy1)
    ax_zoom.set_aspect("equal")
    ax_zoom.set_xlabel("x (um)")
    ax_zoom.set_ylabel("y (um)")
    ax_zoom.set_title(f"Zoom: individual transistors, {inset_size_um:.0f}x{inset_size_um:.0f}um around ({icx:.1f},{icy:.1f})")
    if highlight_y_um is not None and iy0 <= highlight_y_um <= iy1:
        ax_zoom.axhline(highlight_y_um, color="black", linewidth=1.0, linestyle="--", zorder=5)

    handles = [Patch(facecolor=LAYER_COLORS[n], label=n) for n in ("nwell", "diff", "poly")]
    if highlight_y_um is not None:
        handles.append(Line2D([0], [0], color="black", linestyle="--", label="2D cutline"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9)
    fig.suptitle("SRAM macro — top-down layout view (nwell / diff / poly)")
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
