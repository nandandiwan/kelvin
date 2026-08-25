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


def render_temperature(T, chip, row, out_path):
    V = T.function_space
    coords_um = V.tabulate_dof_coordinates() * 1e6
    cells = V.dofmap.list
    triang = mtri.Triangulation(coords_um[:, 0], coords_um[:, 1], cells)
    values = T.x.array

    fig, (ax_full, ax_dev) = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [1, 1.4]})

    tpc = ax_full.tripcolor(triang, values, shading="gouraud", cmap="inferno")
    ax_full.set_xlim(0, chip.layout.tile_um)
    ax_full.set_ylim(0, chip.stack.total_thickness_um)
    ax_full.set_aspect("equal")
    ax_full.set_title(f"Full stack — row {row}")
    ax_full.set_xlabel("x (um)")
    ax_full.set_ylabel("z (um)")

    z0, z1 = BEOL_ZOOM_Z
    ax_dev.tripcolor(triang, values, shading="gouraud", cmap="inferno")
    ax_dev.set_xlim(0, chip.layout.tile_um)
    ax_dev.set_ylim(z0, z1)
    ax_dev.set_aspect("auto")
    ax_dev.set_title(f"BEOL/transistor zoom, z=[{z0:.2f}, {z1:.2f}] um")
    ax_dev.set_xlabel("x (um)")
    ax_dev.set_ylabel("z (um)")

    fig.colorbar(tpc, ax=[ax_full, ax_dev], label="T (K)", shrink=0.8)
    fig.suptitle("Steady-state temperature")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
