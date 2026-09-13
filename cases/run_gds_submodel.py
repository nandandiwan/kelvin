"""Submodeling (global-local analysis): take the whole-die coarse solve
already cached by cases/render_sram_heatmap.py, interpolate its temperature
field onto the four LATERAL cut faces (FACET_X0/X1/Y0/Y1) of a small
full-resolution real-geometry window, and solve that window with those as
Dirichlet BCs instead of leaving the cut artificial and adiabatic. The
window's real bottom (heat sink, Robin) and top (chip surface) BCs are left
untouched — only the lateral cuts are artificial, and only they need a BC
borrowed from the surrounding die.

This is what run_gds_3d.py + run_gds_coarse_solve.py's earlier "validate"
step was NOT doing: that comparison ran the fine window with the lateral
cuts adiabatic (an implicit "this window is thermally isolated" assumption)
and only checked mean dT against the coarse mean, never claiming an accurate
LOCAL peak. This script exists to get an accurate local peak, at multiple
locations across the die, by giving each fine window the correct far-field
thermal environment from the one coarse global solve.

Requires out/sram_heatmap/fields_cache.npz to exist (run
cases/render_sram_heatmap.py once first).

    python cases/run_gds_submodel.py
"""

import sys
import types
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dolfinx.fem import Function, dirichletbc, locate_dofs_topological

from gds import techmap
from gds.power import parse_lib
from gds.read import load_top_cell, flatten_by_layer
from mesh.build import FACET_X0, FACET_X1, FACET_Y0, FACET_Y1
from mesh.gds_build import build_gds_3d_mesh
from mesh.viz3d import render_temperature_xy_slice, render_temperature_xz_slice
from post.metrics import tmax
from solve.steady import solve_steady
from spec.chip import BoundaryConditions

GDS_PATH = "data/sram22_2048x8m8w1.gds"
LIB_PATH = "data/sram22_2048x8m8w1_tt_025C_1v80.lib"
CACHE_PATH = Path("out/sram_heatmap/fields_cache.npz")
FREQ, ACTIVITY = 1e9, 0.5  # must match the cached coarse run (render_sram_heatmap.py's defaults)

# Different parts of the die, chosen from the coarse heatmap in
# out/sram_heatmap/fig9_layer_overview.png: a cold left-margin region, the
# hot bitcell-array band, and the banded periphery/decoder region.
WINDOWS = [
    ("cold_margin", 50.0, 400.0, 2.0),
    ("hot_array", 250.0, 400.0, 2.0),
    ("periphery_band", 250.0, 190.0, 2.0),
]
REFINE = 1.4  # scales element size up -> coarser/faster; see mesh/gds_sizing.py


def build_coarse_interpolator():
    d = np.load(CACHE_PATH, allow_pickle=True)
    coords, T = d["coords"], d["T"]
    x_edges, y_edges = d["x_edges"], d["y_edges"]
    z_levels = np.unique(np.round(coords[:, 2], 12))

    nx, ny = len(x_edges) - 1, len(y_edges) - 1
    x0, dx = x_edges[0] * 1e-6, (x_edges[1] - x_edges[0]) * 1e-6
    y0, dy = y_edges[0] * 1e-6, (y_edges[1] - y_edges[0]) * 1e-6

    value_grid = np.full((nx + 1, ny + 1, len(z_levels)), 300.0)
    for k, z in enumerate(z_levels):
        mask = np.isclose(coords[:, 2], z, atol=1e-13)
        x, y, t = coords[mask, 0], coords[mask, 1], T[mask]
        xi = np.clip(np.round((x - x0) / dx).astype(int), 0, nx)
        yi = np.clip(np.round((y - y0) / dy).astype(int), 0, ny)
        value_grid[xi, yi, k] = t

    x_grid = x_edges * 1e-6
    y_grid = y_edges * 1e-6
    return RegularGridInterpolator((x_grid, y_grid, z_levels), value_grid,
                                    bounds_error=False, fill_value=None)


def run_submodel(by_layer, power_w, interp, label, cx, cy, half):
    window = (cx - half, cx + half, cy - half, cy + half)
    out_dir = f"out/gds3d_submodel_{label}"
    print(f"\n=== {label}: window {window} ===")

    mesh_data, registry = build_gds_3d_mesh(
        by_layer, window, power_w, out_dir=out_dir, refine=REFINE, renders=False,
        stack=techmap.FRONTSIDE_STACK,
    )

    fdim = mesh_data.mesh.topology.dim - 1
    lateral_facets = np.concatenate([mesh_data.facet_tags.find(tag)
                                      for tag in (FACET_X0, FACET_X1, FACET_Y0, FACET_Y1)])

    def bc_builder(V):
        # Built against steady_form's OWN V (passed in here), not a
        # separately-created functionspace() — see solve_steady's
        # docstring for why that distinction is load-bearing.
        g = Function(V)
        g.x.array[:] = interp(V.tabulate_dof_coordinates())
        dofs = locate_dofs_topological(V, fdim, lateral_facets)
        print(f"  lateral cut BC: {len(dofs)} dofs pinned to the coarse whole-die solve "
              f"(range {g.x.array.min():.6f}-{g.x.array.max():.6f} K)")
        return [dirichletbc(g, dofs)]

    bcs = BoundaryConditions(ambient_t_k=300.0, backside_h_eff=20000.0, top_h_eff=None)
    chip = types.SimpleNamespace(bcs=bcs)
    T, k, q = solve_steady(mesh_data, registry, chip, source_depth_m=None, bc_builder=bc_builder)

    tmax_k, coords = tmax(T)
    coarse_here = float(interp(np.array([[cx * 1e-6, cy * 1e-6, coords[2]]]))[0])
    print(f"  submodel peak: {tmax_k:.6f} K (dT={tmax_k-300.0:.6e} K) "
          f"at x={coords[0]*1e6:.2f} y={coords[1]*1e6:.2f} z={coords[2]*1e6:.3f} um")
    print(f"  coarse model at that (x,y,z): {coarse_here:.6f} K (dT={coarse_here-300.0:.6e} K)")

    z_mid = coords[2] * 1e6
    render_temperature_xy_slice(T, z_mid, f"{label}: XY @ z={z_mid:.2f}um (submodel, coarse-BC lateral cuts)",
                                 f"{out_dir}/T_xy_slice.png")
    render_temperature_xz_slice(T, cy, f"{label}: XZ @ y={cy:.1f}um",
                                 f"{out_dir}/T_xz_slice.png")
    print(f"  wrote {out_dir}/T_xy_slice.png, {out_dir}/T_xz_slice.png")
    return tmax_k, coarse_here


def main():
    if not CACHE_PATH.exists():
        raise SystemExit(f"{CACHE_PATH} not found — run cases/render_sram_heatmap.py first")

    lib, top = load_top_cell(GDS_PATH)
    by_layer = flatten_by_layer(top)
    macro = parse_lib(LIB_PATH)
    power_w = macro.total_power_w(FREQ, ACTIVITY, write=True)
    print(f"macro power: {power_w*1e3:.4f} mW @ {FREQ:.1e} Hz, {ACTIVITY*100:.0f}% activity "
          f"(same operating point as the cached coarse solve)")

    interp = build_coarse_interpolator()

    results = []
    for label, cx, cy, half in WINDOWS:
        tmax_k, coarse_k = run_submodel(by_layer, power_w, interp, label, cx, cy, half)
        results.append((label, tmax_k, coarse_k))

    print("\n=== summary ===")
    for label, tmax_k, coarse_k in results:
        print(f"{label:16s}  submodel peak dT={tmax_k-300.0:.4e} K   "
              f"coarse-only dT={coarse_k-300.0:.4e} K")


if __name__ == "__main__":
    main()
