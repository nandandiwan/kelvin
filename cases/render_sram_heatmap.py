"""Presentation figures: whole-die SRAM heatmaps at several z-levels, plus
vertical cross-sections and a through-stack temperature profile, from the
memory-saving coarse 3D solve (mesh/gds_coarse.py + gds/upscale.py — see
cases/run_gds_coarse_solve.py's docstring for the method and its validated
tile-size range).

Power is real, not assumed: gds.power parses the macro's own Liberty (.lib)
characterization; --freq/--activity pick the operating point (default 1GHz,
50% activity -> ~8.5mW, a "moderately active" scenario, not a worst case).

Every panel free-scales its own colorbar (min/max of that slice) rather than
sharing one fixed range across all panels — the physically real dT here is
sub-Kelvin, and a shared absolute range would wash out the in-plane pattern
this whole exercise is trying to show. Each colorbar is labeled with its own
range so nothing is misleadingly implied about absolute magnitude.

    python cases/render_sram_heatmap.py [--tile 1.0] [--freq 1e9] [--activity 0.5]
"""

import argparse
import sys
import types
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dolfinx.fem import functionspace

from gds import techmap
from gds.power import parse_lib
from gds.read import load_top_cell, flatten_by_layer
from gds.sources import build_heat_sources
from gds.upscale import build_tile_grid
from mesh.gds_coarse import build_gds_coarse_mesh
from mesh.gds_section import die_bounds
from solve.steady import solve_steady_from_fields
from spec.chip import BoundaryConditions

GDS_PATH = "data/sram22_2048x8m8w1.gds"
LIB_PATH = "data/sram22_2048x8m8w1_tt_025C_1v80.lib"
OUT_DIR = Path("out/sram_heatmap")

CMAP = "inferno"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "figure.dpi": 140,
})


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tile", type=float, default=1.0, help="lateral tile size, um")
    p.add_argument("--freq", type=float, default=1e9, help="clock frequency, Hz")
    p.add_argument("--activity", type=float, default=0.5, help="fraction of cycles that are real accesses")
    p.add_argument("--stack", choices=["frontside", "bspdn"], default="frontside")
    p.add_argument("--gds", default=GDS_PATH)
    p.add_argument("--lib", default=LIB_PATH)
    p.add_argument("--resolve", action="store_true", help="ignore any cached fields and re-run the solve")
    return p.parse_args()


def solve_whole_die(args):
    lib, top = load_top_cell(args.gds)
    by_layer = flatten_by_layer(top)
    x0, x1, y0, y1 = die_bounds(by_layer)

    macro = parse_lib(args.lib)
    power_w = macro.total_power_w(args.freq, args.activity, write=True)
    area_cm2 = (x1 - x0) * (y1 - y0) * 1e-8
    print(f"die: {x1-x0:.1f} x {y1-y0:.1f} um  ({area_cm2:.4f} cm^2)")
    print(f"power: {power_w*1e3:.4f} mW  ({power_w/area_cm2:.2f} W/cm^2) "
          f"@ {args.freq:.2e} Hz, {args.activity*100:.0f}% activity")

    channel_sources, contact_sources = build_heat_sources(by_layer, power_w)
    stack = techmap.STACKS[args.stack]
    tile_grid = build_tile_grid(by_layer, channel_sources, contact_sources, stack,
                                 (x0, x1, y0, y1), args.tile)
    print(f"tile grid: {tile_grid.nx}x{tile_grid.ny} tiles, slabs: {tile_grid.slab_order}")

    mesh_data, k, rho_cp, q = build_gds_coarse_mesh(tile_grid)
    n_cells = mesh_data.mesh.topology.index_map(mesh_data.mesh.topology.dim).size_local
    print(f"mesh: {n_cells:,} hex cells")

    bcs = BoundaryConditions(ambient_t_k=300.0, backside_h_eff=20000.0, top_h_eff=None)
    chip = types.SimpleNamespace(bcs=bcs)
    T = solve_steady_from_fields(mesh_data, k, q, chip)

    coords = T.function_space.tabulate_dof_coordinates()
    return coords, T.x.array.copy(), tile_grid, power_w, args.freq, args.activity


def slab_at_z(tile_grid, z_m):
    z_um = z_m * 1e6
    for name in tile_grid.slab_order:
        z0, z1 = tile_grid.slabs[name][0], tile_grid.slabs[name][1]
        if z0 - 1e-6 <= z_um <= z1 + 1e-6:
            return name
    return "?"


def xy_slice(coords, T, z_level, atol=1e-13):
    mask = np.isclose(coords[:, 2], z_level, atol=atol)
    return coords[mask, 0], coords[mask, 1], T[mask]


def to_grid(x, y, t, x_edges, y_edges):
    x0, dx = x_edges[0] * 1e-6, (x_edges[1] - x_edges[0]) * 1e-6
    y0, dy = y_edges[0] * 1e-6, (y_edges[1] - y_edges[0]) * 1e-6
    nx, ny = len(x_edges) - 1, len(y_edges) - 1
    xi = np.clip(np.round((x - x0) / dx).astype(int), 0, nx)
    yi = np.clip(np.round((y - y0) / dy).astype(int), 0, ny)
    img = np.full((ny + 1, nx + 1), np.nan)
    img[yi, xi] = t
    return img


def plot_xy(img, x_edges, y_edges, title, fname, cmap=CMAP):
    dT = img - 300.0
    fig, ax = plt.subplots(figsize=(9, 5.2))
    im = ax.imshow(dT, origin="lower", cmap=cmap, aspect="equal",
                    extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
    ax.set_xlabel("x (μm)")
    ax.set_ylabel("y (μm)")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(f"ΔT (K)   [{np.nanmin(dT):.3g} → {np.nanmax(dT):.3g} K]")
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {fname}")


CACHE_PATH = OUT_DIR / "fields_cache.npz"


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if CACHE_PATH.exists() and not getattr(args, "resolve", False):
        print(f"loading cached fields from {CACHE_PATH} (delete it, or pass --resolve, to re-solve)")
        d = np.load(CACHE_PATH, allow_pickle=True)
        coords, T = d["coords"], d["T"]
        x_edges, y_edges = d["x_edges"], d["y_edges"]
        slab_order = list(d["slab_order"])
        slab_z0, slab_z1 = d["slab_z0"], d["slab_z1"]
        power_w, freq, activity = float(d["power_w"]), float(d["freq"]), float(d["activity"])
        tile_grid = types.SimpleNamespace(
            x_edges=x_edges, y_edges=y_edges, slab_order=slab_order,
            slabs={n: (slab_z0[i], slab_z1[i]) for i, n in enumerate(slab_order)},
        )
    else:
        coords, T, tile_grid, power_w, freq, activity = solve_whole_die(args)
        slab_z0 = np.array([tile_grid.slabs[n][0] for n in tile_grid.slab_order])
        slab_z1 = np.array([tile_grid.slabs[n][1] for n in tile_grid.slab_order])
        np.savez(CACHE_PATH, coords=coords, T=T, x_edges=tile_grid.x_edges, y_edges=tile_grid.y_edges,
                 slab_order=np.array(tile_grid.slab_order), slab_z0=slab_z0, slab_z1=slab_z1,
                 power_w=power_w, freq=freq, activity=activity)
        print(f"cached fields to {CACHE_PATH}")

    z_vals = coords[:, 2]
    z_levels = np.unique(np.round(z_vals, 12))
    print(f"z-levels in mesh (um): {[round(z*1e6,4) for z in z_levels]}")

    x_edges, y_edges = tile_grid.x_edges, tile_grid.y_edges
    peak_idx = np.argmax(T)
    x_h, y_h, z_h = coords[peak_idx]
    print(f"peak T={T[peak_idx]:.6f} K at x={x_h*1e6:.2f} y={y_h*1e6:.2f} z={z_h*1e6:.3f} um")

    # --- Figures 1-4: XY heatmaps at the bottom, top-of-substrate,
    # mid-device-layer (peak), and top-of-chip z-levels.
    device_z = z_h  # the level the global peak actually sits on
    picks = [
        (z_levels[0], "bottom — heat-sink interface"),
        (z_levels[len(z_levels) // 3], "lower substrate"),
        (device_z, "device layer (FEOL/BEOL) — peak plane"),
        (z_levels[-1], "top — chip surface"),
    ]
    for i, (z, label) in enumerate(picks, start=1):
        x, y, t = xy_slice(coords, T, z)
        img = to_grid(x, y, t, x_edges, y_edges)
        slab = slab_at_z(tile_grid, z)
        plot_xy(img, x_edges, y_edges,
                f"Top view — z = {z*1e6:.2f} μm ({label}, slab: {slab})",
                f"fig{i}_xy_z{z*1e6:.2f}um.png")

    # --- Figure 5: zoomed hot region at the device layer. Crop BEFORE
    # imshow (not just set_xlim/set_ylim after) so the colorbar autoscales
    # to this window's own local contrast, not the whole die's range —
    # otherwise a near-uniform peak plateau renders as a flat, washed-out
    # panel even though it has real local structure at its own scale.
    x, y, t = xy_slice(coords, T, device_z)
    img = to_grid(x, y, t, x_edges, y_edges)
    half = 40.0  # um
    x0_um, x1_um = x_h * 1e6 - half, x_h * 1e6 + half
    y0_um, y1_um = y_h * 1e6 - half, y_h * 1e6 + half
    nx_full, ny_full = len(x_edges) - 1, len(y_edges) - 1
    ix0 = max(0, int((x0_um - x_edges[0]) / (x_edges[1] - x_edges[0])))
    ix1 = min(nx_full + 1, int(np.ceil((x1_um - x_edges[0]) / (x_edges[1] - x_edges[0]))))
    iy0 = max(0, int((y0_um - y_edges[0]) / (y_edges[1] - y_edges[0])))
    iy1 = min(ny_full + 1, int(np.ceil((y1_um - y_edges[0]) / (y_edges[1] - y_edges[0]))))
    dT_crop = img[iy0:iy1, ix0:ix1] - 300.0
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(dT_crop, origin="lower", cmap=CMAP, aspect="equal",
                    extent=[x0_um, x1_um, y0_um, y1_um])
    ax.set_xlabel("x (μm)"); ax.set_ylabel("y (μm)")
    ax.set_title(f"Zoomed hotspot — device layer, {2*half:.0f}×{2*half:.0f} μm window")
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(f"ΔT (K)   [{np.nanmin(dT_crop):.4g} → {np.nanmax(dT_crop):.4g} K]")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5_xy_zoom_hotspot.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig5_xy_zoom_hotspot.png")

    # --- Figures 6-7: vertical XZ and YZ cross-sections through the peak.
    def vertical_slice(fixed_axis, fixed_val, varying_edges, label, fname, xlabel):
        mask = np.isclose(coords[:, fixed_axis], fixed_val, atol=1e-13)
        v = coords[mask, 1 - fixed_axis]  # the other lateral coordinate
        z = coords[mask, 2]
        t = T[mask]
        nv = len(varying_edges) - 1
        v0, dv = varying_edges[0] * 1e-6, (varying_edges[1] - varying_edges[0]) * 1e-6
        vi = np.clip(np.round((v - v0) / dv).astype(int), 0, nv)
        zi_map = {round(zz, 12): i for i, zz in enumerate(z_levels)}
        zi = np.array([zi_map[round(zz, 12)] for zz in z])
        img = np.full((len(z_levels), nv + 1), np.nan)
        img[zi, vi] = t
        dT = img - 300.0
        fig, ax = plt.subplots(figsize=(10, 4.2))
        # "gouraud" treats X,Y as the true vertex positions with smooth
        # interpolation between them and no extrapolation beyond the given
        # range — "nearest"/"flat" treat X,Y as cell CENTERS and silently
        # extend the plotted range by half a cell spacing past each edge
        # (visible as z running below 0 / above the chip surface).
        im = ax.pcolormesh(np.linspace(varying_edges[0], varying_edges[-1], nv + 1),
                            z_levels * 1e6, dT, cmap=CMAP, shading="gouraud")
        ax.set_xlabel(xlabel); ax.set_ylabel("z (μm)")
        ax.set_title(label)
        cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
        cb.set_label(f"ΔT (K)   [{np.nanmin(dT):.3g} → {np.nanmax(dT):.3g} K]")
        fig.tight_layout()
        fig.savefig(OUT_DIR / fname, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {fname}")

    # snap peak x,y onto the actual node grid before slicing
    x_node = coords[np.isclose(coords[:, 0], x_h, atol=1e-13)][0, 0]
    y_node = coords[np.isclose(coords[:, 1], y_h, atol=1e-13)][0, 1]
    vertical_slice(1, y_node, x_edges,
                    f"Vertical slice (X-Z) through hotspot — y = {y_node*1e6:.2f} μm",
                    "fig6_xz_through_hotspot.png", "x (μm)")
    vertical_slice(0, x_node, y_edges,
                    f"Vertical slice (Y-Z) through hotspot — x = {x_node*1e6:.2f} μm",
                    "fig7_yz_through_hotspot.png", "y (μm)")

    # --- Figure 8: 1D temperature-vs-depth profile at the hotspot (x,y).
    mask = np.isclose(coords[:, 0], x_node, atol=1e-13) & np.isclose(coords[:, 1], y_node, atol=1e-13)
    z_prof = coords[mask, 2] * 1e6
    t_prof = T[mask]
    order = np.argsort(z_prof)
    z_prof, t_prof = z_prof[order], t_prof[order]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(t_prof - 300.0, z_prof, "o-", color="#B7410E", lw=2, ms=5)
    ax.set_xlabel("ΔT (K)")
    ax.set_ylabel("z (μm)  —  0 = heat sink, top = chip surface")
    ax.set_title("Through-stack temperature profile at hotspot")
    ax.grid(alpha=0.3)
    for name in tile_grid.slab_order:
        z0 = tile_grid.slabs[name][0]
        ax.axhline(z0, color="gray", ls="--", lw=0.8, alpha=0.6)
        ax.text(ax.get_xlim()[1], z0, f" {name}", va="bottom", ha="right", fontsize=8, color="gray")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig8_vertical_profile.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig8_vertical_profile.png")

    # --- Figure 9: composite 2x2 layer overview.
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (z, label) in zip(axes.flat, picks):
        x, y, t = xy_slice(coords, T, z)
        img = to_grid(x, y, t, x_edges, y_edges)
        dT = img - 300.0
        im = ax.imshow(dT, origin="lower", cmap=CMAP, aspect="equal",
                        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
        ax.set_title(f"z={z*1e6:.2f}μm: {label}", fontsize=10)
        ax.set_xlabel("x (μm)", fontsize=9); ax.set_ylabel("y (μm)", fontsize=9)
        cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cb.ax.tick_params(labelsize=8)
    fig.suptitle(f"SRAM macro thermal map — {power_w*1e3:.2f} mW @ {freq:.1e} Hz, "
                 f"{activity*100:.0f}% activity", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_DIR / "fig9_layer_overview.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig9_layer_overview.png")

    print(f"\nAll figures written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
