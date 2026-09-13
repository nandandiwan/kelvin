"""Whole-die (or any window) coarse 3D solve via gds.upscale + mesh.gds_coarse
— no gmsh, no OCC, no removeAllDuplicates. Cell count is `tile_um` and
`nz_per_slab`, chosen by you, not a consequence of real polygon density (see
cases/run_gds_3d.py's probe: full resolution extrapolates to ~27 billion
tets / ~0.87TB of mesh connectivity alone on this machine's 502GB).

--validate runs on the SAME window as an existing cases/run_gds_3d.py fine
solve and reports both the peak AND the volume-weighted mean dT — mean is the
metric that actually matters here (see run()'s docstring note: a coarse
tile-averaged model recovers the BULK/regional field, not one real device's
peak junction temperature in a sparse window with only a handful of real
sources, so peak-vs-peak is the wrong comparison and will look far worse than
the model actually is).

Recommended tile size: ~0.5-2um, matching real feature/cluster spacing, NOT
sub-100nm. Measured at this window: 0.5um tile agrees with the fine reference
to 13.4% on mean dT. Below ~0.2um the z-cell count needed to keep hex cells
near-cubic for the heterogeneous feol_beol slab is capped (_MAX_NZ_PER_SLAB
in mesh/gds_coarse.py) to bound cell count, so cells get needle-shaped again
and small discrete-maximum-principle violations reappear (verified: mean dT
went negative — physically impossible for a q>=0 problem — at 0.1um and
finer). That sub-100nm regime is what the fine windowed solver
(cases/run_gds_3d.py) is for; this coarse path is for the whole die.

Confirmed on the real full SRAM die at --tile 2.0: 294,500 hex cells, 2.79GB
peak RSS, ~2 minutes wall time — vs. ~27 billion tets / ~0.87TB of mesh
connectivity alone that full-resolution 3D would need (cases/run_gds_3d.py's
--probe). This is the memory-saving win the whole architecture is for.

    python cases/run_gds_coarse_solve.py --validate [--tile 0.5]
    python cases/run_gds_coarse_solve.py --whole-die [--tile 2.0] [--mw 1.0] [--stack bspdn]
"""

import argparse
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ufl
from dolfinx.fem import assemble_scalar, form

from gds import techmap
from gds.read import load_top_cell, flatten_by_layer
from gds.sources import build_heat_sources
from gds.upscale import build_tile_grid
from mesh.gds_coarse import build_gds_coarse_mesh
from mesh.gds_section import die_bounds
from post.metrics import tmax
from solve.steady import solve_steady_from_fields
from spec.chip import BoundaryConditions

GDS_PATH = "data/sram22_2048x8m8w1.gds"
# Matches cases/run_gds_3d.py's already-built --window 6 --cx 216 --cy 255.5.
VALIDATE_WINDOW = (213.0, 219.0, 252.5, 258.5)
VALIDATE_FINE_DT = {"frontside": 3.114773e-06, "bspdn": 1.070397e-06}  # K, at --mw 1e-6
# Volume-weighted mean dT over the SAME fine solve — read directly from its
# saved solution.h5, since a global peak-vs-peak comparison is not meaningful
# here (see run()'s docstring note below): 715,794-node nodal mean.
VALIDATE_FINE_MEAN_DT = {"frontside": 3.8954675574132125e-07}  # K, at --mw 1e-6


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--validate", action="store_true")
    p.add_argument("--whole-die", action="store_true")
    p.add_argument("--tile", type=float, default=0.5, help="lateral tile size, um")
    p.add_argument("--mw", type=float, default=1e-6, help="total DIE power, mW")
    p.add_argument("--stack", choices=["frontside", "bspdn"], default="frontside")
    p.add_argument("--top-h-eff", type=float, default=None)
    p.add_argument("--gds", default=GDS_PATH)
    return p.parse_args()


def run(by_layer, channel_sources, contact_sources, stack, window, tile_um, power_w, top_h_eff, out_tag):
    tile_grid = build_tile_grid(by_layer, channel_sources, contact_sources, stack, window, tile_um)
    print(f"[{out_tag}] tile grid: {tile_grid.nx}x{tile_grid.ny} tiles ({tile_um}um), "
          f"{len(tile_grid.slab_order)} coarse z-slabs: {tile_grid.slab_order}")

    mesh_data, k, rho_cp, q = build_gds_coarse_mesh(tile_grid)
    n_cells = mesh_data.mesh.topology.index_map(mesh_data.mesh.topology.dim).size_local
    print(f"[{out_tag}] mesh: {n_cells:,} hex cells")

    bcs = BoundaryConditions(ambient_t_k=300.0, backside_h_eff=20000.0, top_h_eff=top_h_eff)
    chip = types.SimpleNamespace(bcs=bcs)
    T = solve_steady_from_fields(mesh_data, k, q, chip)

    tmax_k, coords = tmax(T)
    dt = tmax_k - 300.0
    print(f"[{out_tag}] Tmax = {tmax_k:.6f} K (dT = {dt:.6e} K) "
          f"at x={coords[0]*1e6:.2f}um y={coords[1]*1e6:.2f}um z={coords[2]*1e6:.3f}um")

    # Volume-weighted mean, not just the global peak: a coarse tile-averaged
    # model is expected to recover the BULK/regional field, not a single
    # real device's peak junction temperature inside one tile — especially
    # in a small, sparse window with only a handful of real devices, where
    # "global max" in the coarse model is really "whichever tile happened to
    # average in the most power," a different physical quantity than the
    # fine model's true per-device peak. Mean dT is the metric this
    # upscaling is actually meant to get right.
    mesh = mesh_data.mesh
    dx = ufl.Measure("dx", domain=mesh)
    vol = assemble_scalar(form(1 * dx))
    mean_dt = assemble_scalar(form((T - 300.0) * dx)) / vol
    print(f"[{out_tag}] mean dT = {mean_dt:.6e} K (volume-weighted)")
    return dt, mean_dt, n_cells


def main():
    args = parse_args()
    lib, top = load_top_cell(args.gds)
    by_layer = flatten_by_layer(top)

    power_w = args.mw * 1e-3
    channel_sources, contact_sources = build_heat_sources(by_layer, power_w)
    stack = techmap.STACKS[args.stack]

    if args.validate:
        dt_coarse, mean_dt_coarse, n_cells = run(
            by_layer, channel_sources, contact_sources, stack,
            VALIDATE_WINDOW, args.tile, power_w, args.top_h_eff, "validate")
        dt_fine = VALIDATE_FINE_DT[args.stack]
        rel_err = abs(dt_coarse - dt_fine) / abs(dt_fine) if dt_fine else float("nan")
        print(f"\n=== validation vs. direct fine 3D solve (cases/run_gds_3d.py) ===")
        print(f"fine (real geometry, {VALIDATE_WINDOW}, --mw 1e-6): peak dT = {dt_fine:.6e} K")
        print(f"coarse (upscaled, tile={args.tile}um):              peak dT = {dt_coarse:.6e} K")
        print(f"peak relative difference: {rel_err*100:.1f}% "
              f"(NOT the right metric for a small/sparse window — see run()'s docstring note)")
        mean_dt_fine = VALIDATE_FINE_MEAN_DT.get(args.stack)
        if mean_dt_fine is not None:
            mean_rel_err = abs(mean_dt_coarse - mean_dt_fine) / abs(mean_dt_fine)
            print(f"fine mean dT (volume-weighted):   {mean_dt_fine:.6e} K")
            print(f"coarse mean dT (volume-weighted): {mean_dt_coarse:.6e} K")
            print(f"mean relative difference: {mean_rel_err*100:.1f}%  <-- the meaningful check")
        return

    if args.whole_die:
        x0, x1, y0, y1 = die_bounds(by_layer)
        print(f"whole die: {x1-x0:.1f} x {y1-y0:.1f} um")
        run(by_layer, channel_sources, contact_sources, stack, (x0, x1, y0, y1),
            args.tile, power_w, args.top_h_eff, "whole_die")
        return

    print("pass --validate or --whole-die")


if __name__ == "__main__":
    main()
