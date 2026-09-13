"""GATE 0 renders for the GDS 2D cross-section — same approach as mesh/viz.py
(raw gmsh mesh, before dolfinx touches it), reusing its MATERIAL_COLORS.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch

from .viz import MATERIAL_COLORS


def render_regions(xz, tri_nodes, tri_region, x_bounds, z_bounds, cut_y_um, out_path,
                    substrate_z0=None, feol_top=None, stack=None):
    x0, x1 = x_bounds
    z0, z1 = z_bounds
    # Read the zoom windows off the real stack rather than hardcoding them —
    # they moved when the estimated thicknesses were replaced with SKY130's
    # real ones (the whole stack got ~2x taller), and move again per-profile
    # for the BSPDN study (`stack=techmap.BSPDN_STACK`). Use each band's
    # cumulative z1 (its absolute top), not its own thickness_um — those
    # coincide only when Si_substrate happens to be the first band (true for
    # the frontside stack, false for BSPDN, where backside bands sit below it).
    from gds import techmap
    bounds = techmap.z_bounds(stack)
    if substrate_z0 is None:
        substrate_z0 = next(ze for _zb, ze, b in bounds if b.name == "Si_substrate")
    if feol_top is None:
        feol_top = next(zb for zb, _ze, b in bounds if b.name == "li1")
    materials_present = sorted({r.material for r in tri_region})
    color_of = {m: MATERIAL_COLORS.get(m, "#999999") for m in materials_present}
    verts = xz[tri_nodes]
    colors = [color_of[r.material] for r in tri_region]

    fig, (ax_full, ax_beol, ax_feol) = plt.subplots(
        1, 3, figsize=(22, 7), gridspec_kw={"width_ratios": [1, 1.4, 1.4]}
    )

    ax_full.add_collection(PolyCollection(verts, facecolors=colors, edgecolors="black", linewidths=0.05))
    ax_full.set_xlim(x0, x1)
    ax_full.set_ylim(0, z1)
    ax_full.set_aspect("equal")
    ax_full.set_title(f"Full stack — cut y={cut_y_um}um")
    ax_full.set_xlabel("x (um)")
    ax_full.set_ylabel("z (um)")

    # BEOL zoom: from the FEOL top up through the metal stack.
    beol_z0 = substrate_z0
    ax_beol.add_collection(PolyCollection(verts, facecolors=colors, edgecolors="none"))
    ax_beol.set_xlim(x0, x1)
    ax_beol.set_ylim(beol_z0, z1)
    ax_beol.set_aspect("auto")
    ax_beol.set_title(f"BEOL zoom, z=[{beol_z0:.2f}, {z1:.2f}] um")
    ax_beol.set_xlabel("x (um)")
    ax_beol.set_ylabel("z (um)")

    # FEOL zoom: substrate surface through licon1 (diff/channel/poly/licon1),
    # where the real transistor structure lives and is otherwise invisible
    # against the ~50um substrate + multi-um metal stack.
    feol_z0, feol_z1 = substrate_z0 - 0.01, feol_top
    ax_feol.add_collection(PolyCollection(verts, facecolors=colors, edgecolors="none"))
    ax_feol.set_xlim(x0, x1)
    ax_feol.set_ylim(feol_z0, feol_z1)
    ax_feol.set_aspect("auto")
    ax_feol.set_title(f"FEOL zoom, z=[{feol_z0:.2f}, {feol_z1:.2f}] um\n(diff -> channel -> poly + licon1)")
    ax_feol.set_xlabel("x (um)")
    ax_feol.set_ylabel("z (um)")

    handles = [Patch(facecolor=color_of[m], edgecolor="black", label=m) for m in materials_present]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(materials_present), 8), fontsize=8)
    fig.suptitle("GDS cross-section — material regions")
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def render_sources(xz, tri_nodes, tri_region, x_bounds, out_path):
    """Channel (hot-carrier dissipation, in the top-10nm inversion layer of
    diff) and contact (I^2R heating, in licon1) sources are two physically
    distinct mechanisms in two distinct z-bands (see SRAM_THERMAL_REPORT.md
    section 3) — colored with two separate hues and two separate power
    scales, since they're never meant to be compared on the same axis (a
    "high" contact power and a "high" channel power aren't the same kind of
    number).

    Drawn as two stacked panels, not one shared z-axis: the channel band is
    only 0.01um thick vs. licon1's 0.43um (SKY130's real heights — see
    gds/techmap.py), a 40x difference. One shared y-axis would make the
    channel band a sub-pixel line regardless of color; each panel instead
    gets its own z-window sized to its own band, so both are equally legible.
    """
    x0, x1 = x_bounds
    is_channel = lambda r: r.source is not None and r.source.kind == "channel"
    is_contact = lambda r: r.source is not None and r.source.kind == "contact"
    ch_powers = [r.source.power_uw for r in tri_region if is_channel(r)]
    co_powers = [r.source.power_uw for r in tri_region if is_contact(r)]
    if not ch_powers and not co_powers:
        return

    cmap_ch, cmap_co = plt.get_cmap("Reds"), plt.get_cmap("Blues")
    norm_ch = plt.Normalize(0, max(ch_powers) if ch_powers else 1.0)
    norm_co = plt.Normalize(0, max(co_powers) if co_powers else 1.0)

    def color_of(r):
        if is_channel(r):
            return cmap_ch(0.25 + 0.75 * norm_ch(r.source.power_uw))
        if is_contact(r):
            return cmap_co(0.25 + 0.75 * norm_co(r.source.power_uw))
        return "#e8e8e8"

    verts = xz[tri_nodes]
    colors = [color_of(r) for r in tri_region]
    n_ch, n_co = len(ch_powers), len(co_powers)

    def z_window(kind_check, pad_factor=1.5):
        zs = [xz[tri_nodes[i]][:, 1] for i, r in enumerate(tri_region) if kind_check(r)]
        if not zs:
            return None
        lo, hi = min(z.min() for z in zs), max(z.max() for z in zs)
        pad = max((hi - lo) * pad_factor, 0.002)
        return lo - pad, hi + pad

    ch_window = z_window(is_channel) if ch_powers else None
    co_window = z_window(is_contact) if co_powers else None
    n_rows = (1 if co_window else 0) + (1 if ch_window else 0)

    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 3.2 * n_rows + 1), squeeze=False)
    axes = axes[:, 0]
    row = 0

    if co_window:
        ax = axes[row]; row += 1
        ax.add_collection(PolyCollection(verts, facecolors=colors, edgecolors="none"))
        ax.set_xlim(x0, x1)
        ax.set_ylim(*co_window)
        ax.set_aspect("auto")
        ax.set_ylabel("z (um)")
        ax.set_title(f"{n_co} contact sources (I²R heating, blue, licon1 band, 0.43um thick)")
        fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_co, norm=norm_co), ax=ax,
                     label="contact power (uW)", fraction=0.02, pad=0.01)

    if ch_window:
        ax = axes[row]; row += 1
        ax.add_collection(PolyCollection(verts, facecolors=colors, edgecolors="none"))
        ax.set_xlim(x0, x1)
        ax.set_ylim(*ch_window)
        ax.set_aspect("auto")
        ax.set_xlabel("x (um)")
        ax.set_ylabel("z (um)")
        ax.set_title(f"{n_ch} channel sources (hot-carrier dissipation, red, inversion layer, 0.01um thick)")
        fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_ch, norm=norm_ch), ax=ax,
                     label="channel power (uW)", fraction=0.02, pad=0.01)

    fig.suptitle("Heat sources on this cut — two mechanisms, two z-bands, note the different z-scales")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
