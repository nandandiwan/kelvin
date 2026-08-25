"""GATE 0 visual-verification renders: material regions + heat sources,
straight off the raw gmsh mesh (before dolfinx touches it). Matplotlib, Agg
backend — these are files on disk, not an interactive session.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch

MATERIAL_COLORS = {
    "Si_bulk":     "#B0B0B0",
    "Si_SD_doped": "#8899AA",
    "Si_channel":  "#6677AA",
    "SiO2":        "#CFE8F3",
    "low_k":       "#D8C9EA",
    "HfO2":        "#F4A6A6",
    "TiN":         "#C2A14D",
    "W":           "#7A7A7A",
    "NiSi":        "#E8C36B",
    "Cu_fine":     "#D98C4A",
    "Cu_thick":    "#B5652D",
    "TaN":         "#5B4636",
    "SiN":         "#8FBF8F",
}

# z-window that contains the transistor stack + first two metal layers,
# where all the interesting fine structure lives (see PLAN.md's note that
# these layers are invisible at full-domain aspect ratio).
BEOL_ZOOM_Z = (49.90, 52.10)

# Tighter window on just the transistor's own vertical anatomy (STI_SD ->
# MOL_contacts, z=50.00-50.26) — the z-direction gradient through channel /
# gate stack / silicide / contacts is invisible even in the BEOL zoom above.
DEVICE_ZOOM_Z = (49.985, 50.275)


def _poly(xz, tri_nodes):
    return xz[tri_nodes]


def render_regions(xz, tri_nodes, tri_region, chip, row, out_path):
    materials_present = sorted({r.material for r in tri_region})
    color_of = {m: MATERIAL_COLORS.get(m, "#999999") for m in materials_present}
    verts = _poly(xz, tri_nodes)
    colors = [color_of[r.material] for r in tri_region]

    fig, (ax_full, ax_beol, ax_dev) = plt.subplots(
        1, 3, figsize=(20, 7), gridspec_kw={"width_ratios": [1, 1.4, 1.4]}
    )

    ax_full.add_collection(PolyCollection(verts, facecolors=colors, edgecolors="black", linewidths=0.05))
    ax_full.set_xlim(0, chip.layout.tile_um)
    ax_full.set_ylim(0, chip.stack.total_thickness_um)
    ax_full.set_aspect("equal")
    ax_full.set_title(f"Full stack — row {row}")
    ax_full.set_xlabel("x (um)")
    ax_full.set_ylabel("z (um)")

    z0, z1 = BEOL_ZOOM_Z
    ax_beol.add_collection(PolyCollection(verts, facecolors=colors, edgecolors="black", linewidths=0.15))
    ax_beol.set_xlim(0, chip.layout.tile_um)
    ax_beol.set_ylim(z0, z1)
    ax_beol.set_aspect("auto")
    ax_beol.set_title(f"BEOL/transistor zoom, z=[{z0:.2f}, {z1:.2f}] um")
    ax_beol.set_xlabel("x (um)")
    ax_beol.set_ylabel("z (um)")

    dz0, dz1 = DEVICE_ZOOM_Z
    # no mesh edges here: at this zoom the triangle count per device is high
    # enough that overlapping edge lines render as solid black and hide the
    # material fill colors, which are the actual signal at this scale.
    ax_dev.add_collection(PolyCollection(verts, facecolors=colors, edgecolors="none"))
    ax_dev.set_xlim(4.5, 17.5)  # the row-0 device span, cols 0-5
    ax_dev.set_ylim(dz0, dz1)
    ax_dev.set_aspect("auto")
    ax_dev.set_title(f"Transistor anatomy, z=[{dz0:.3f}, {dz1:.3f}] um\n(STI_SD -> channel -> gate -> silicide -> MOL)")
    ax_dev.set_xlabel("x (um)")
    ax_dev.set_ylabel("z (um)")

    handles = [Patch(facecolor=color_of[m], edgecolor="black", label=m) for m in materials_present]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(materials_present), 7), fontsize=8)
    fig.suptitle("Chip cross-section — material regions")
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def render_sources(xz, tri_nodes, tri_region, chip, row, out_path):
    devices = sorted((d for d in chip.layout.devices if d.row == row), key=lambda d: d.x_um)
    powers = [r.source.power_uw for r in tri_region if r.source is not None]
    max_p = max(powers, default=1.0)
    cmap = plt.get_cmap("YlOrRd")
    norm = plt.Normalize(0, max_p)

    verts = _poly(xz, tri_nodes)
    colors = [cmap(norm(r.source.power_uw)) if r.source is not None else "#e5e5e5" for r in tri_region]

    fig, ax = plt.subplots(figsize=(15, 4.5))
    ax.add_collection(PolyCollection(verts, facecolors=colors, edgecolors="none"))
    ax.set_xlim(0, chip.layout.tile_um)
    ax.set_ylim(50.0, 50.15)  # STI_channel + silicide band
    ax.set_aspect("auto")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("z (um)")
    ax.set_title(f"Device heat sources — row {row} (channel + contact, color = power)")

    for d in devices:
        via = "via" if d.has_via_stack else "no-via"
        ax.text(d.x_um, 50.148, f"{d.name}\n{d.power_uw:.0f}uW  {via}",
                ha="center", va="top", fontsize=6.5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=ax, label="source power (uW)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
