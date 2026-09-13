"""Temperature-field render. Builds the matplotlib triangulation entirely
from the P1 FunctionSpace's own dofmap + dof coordinates (not from
mesh.geometry) — self-consistent by construction, no risk of a mismatched
geometry/dof ordering between T.x.array and the plotted connectivity.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from mesh.viz import BEOL_ZOOM_Z


def render_temperature_generic(T, x_bounds, z_bounds, zoom_z_bounds, full_title, zoom_title, out_path,
                                zoom_x_bounds=None):
    """Domain-agnostic core: takes explicit (x0,x1)/(z0,z1) bounds instead of
    a synthetic ChipSpec, so it works for both the synthetic chip and a
    GDS-derived cross-section.

    `zoom_x_bounds` adds a third panel zoomed in *x* as well as z, with its
    own color scale. Without it, every panel spans the full die width, where
    a sub-micron device is far under one pixel and any per-device temperature
    structure is invisible regardless of how fine the mesh is — the full-width
    panels only ever show the large-scale gradient.
    """
    V = T.function_space
    coords_um = V.tabulate_dof_coordinates() * 1e6
    cells = V.dofmap.list
    triang = mtri.Triangulation(coords_um[:, 0], coords_um[:, 1], cells)
    values = T.x.array

    n_panels = 3 if zoom_x_bounds is not None else 2
    ratios = [1, 1.4, 1.4][:n_panels]
    fig, axes = plt.subplots(1, n_panels, figsize=(7.5 * n_panels, 7),
                              gridspec_kw={"width_ratios": ratios})
    ax_full, ax_dev = axes[0], axes[1]

    x0, x1 = x_bounds
    tpc = ax_full.tripcolor(triang, values, shading="gouraud", cmap="inferno")
    ax_full.set_xlim(x0, x1)
    ax_full.set_ylim(*z_bounds)
    ax_full.set_aspect("equal")
    ax_full.set_title(full_title)
    ax_full.set_xlabel("x (um)")
    ax_full.set_ylabel("z (um)")

    zz0, zz1 = zoom_z_bounds
    ax_dev.tripcolor(triang, values, shading="gouraud", cmap="inferno")
    ax_dev.set_xlim(x0, x1)
    ax_dev.set_ylim(zz0, zz1)
    ax_dev.set_aspect("auto")
    ax_dev.set_title(zoom_title)
    ax_dev.set_xlabel("x (um)")
    ax_dev.set_ylabel("z (um)")

    fig.colorbar(tpc, ax=[ax_full, ax_dev], label="T (K)", shrink=0.8)

    if zoom_x_bounds is not None:
        zx0, zx1 = zoom_x_bounds
        ax_x = axes[2]
        # Local color scale: the die-wide gradient is much larger than the
        # per-device structure, so a shared scale would flatten this panel.
        in_win = ((coords_um[:, 0] >= zx0) & (coords_um[:, 0] <= zx1)
                   & (coords_um[:, 1] >= zz0) & (coords_um[:, 1] <= zz1))
        vmin, vmax = (values[in_win].min(), values[in_win].max()) if in_win.any() else (None, None)
        tpc_x = ax_x.tripcolor(triang, values, shading="gouraud", cmap="inferno",
                                vmin=vmin, vmax=vmax)
        ax_x.set_xlim(zx0, zx1)
        ax_x.set_ylim(zz0, zz1)
        ax_x.set_aspect("auto")
        ax_x.set_title(f"Device-scale zoom, x=[{zx0:.1f}, {zx1:.1f}] um\n(own color scale)")
        ax_x.set_xlabel("x (um)")
        ax_x.set_ylabel("z (um)")
        fig.colorbar(tpc_x, ax=ax_x, label="T (K)", shrink=0.8)

    fig.suptitle("Steady-state temperature")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def render_temperature(T, chip, row, out_path):
    render_temperature_generic(
        T, (0, chip.layout.tile_um), (0, chip.stack.total_thickness_um), BEOL_ZOOM_Z,
        f"Full stack — row {row}",
        f"BEOL/transistor zoom, z=[{BEOL_ZOOM_Z[0]:.2f}, {BEOL_ZOOM_Z[1]:.2f}] um",
        out_path,
    )
