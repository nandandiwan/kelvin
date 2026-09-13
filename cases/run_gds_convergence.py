"""Mesh-convergence check for the GDS 2D cross-section: solve the same
problem at several uniformly-scaled mesh resolutions and report how Tmax
moves. This is the gate that energy conservation can NOT provide — energy
conservation is an exact algebraic identity of the discrete weak form and
holds at any resolution, so it says nothing about whether the mesh is fine
enough to resolve the actual temperature field (see SRAM_THERMAL_REPORT.md).

    python cases/run_gds_convergence.py [cut_y_um] [total_power_mw] [gds_path] [--stack bspdn]

`refine` scales every band's target element size: 2.0 = twice as coarse,
0.5 = twice as fine (~4x the cells in 2D). Convergence means dT stops moving
as refine shrinks.

`--stack bspdn` re-runs this against gds.techmap.BSPDN_STACK instead of the
default frontside stack — the frontside stack converging to 0.002% does NOT
imply BSPDN's sub-micron backside bands are equally converged; they introduce
genuinely different length scales (a 0.4um Si layer, 90nm nTSV features) and
must be checked independently (see SRAM_THERMAL_REPORT.md's BSPDN section).
"""

import argparse
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gds import techmap
from gds.read import load_top_cell, flatten_by_layer, named_cell_bbox_um
from mesh.gds_build import build_gds_2d_mesh
from mesh.gds_section import die_bounds
from post.metrics import tmax
from solve.steady import solve_steady
from spec.chip import BoundaryConditions

GDS_PATH = "data/sram22_2048x8m8w1.gds"
BITCELL_NAME = "sram_sp_cell"
REFINEMENTS = [2.0, 1.4, 1.0, 0.7, 0.5]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("cut_y_um", type=float, nargs="?", default=255.5)
    p.add_argument("total_power_mw", type=float, nargs="?", default=1.0)
    p.add_argument("gds_path", type=str, nargs="?", default=GDS_PATH)
    p.add_argument("--stack", choices=["frontside", "bspdn"], default="frontside")
    return p.parse_args()


def main():
    args = parse_args()
    cut_y_um = args.cut_y_um
    total_power_mw = args.total_power_mw
    gds_path = args.gds_path
    stack = techmap.STACKS[args.stack]

    lib, top = load_top_cell(gds_path)
    by_layer = flatten_by_layer(top)

    try:
        _, y0c, _, y1c = named_cell_bbox_um(lib, BITCELL_NAME)
        depth_um = y1c - y0c
    except KeyError:
        _x0, _x1, y0, y1 = die_bounds(by_layer)
        depth_um = y1 - y0

    bcs = BoundaryConditions(ambient_t_k=300.0, backside_h_eff=20000.0, top_face="adiabatic")
    chip = types.SimpleNamespace(bcs=bcs)

    print(f"stack={args.stack}, cut y={cut_y_um}um, {total_power_mw}mW, homogenization depth {depth_um:.3f}um")
    print(f"{'refine':>8}{'cells':>12}{'dT (K)':>14}{'build+solve (s)':>18}")

    rows = []
    for refine in REFINEMENTS:
        t0 = time.time()
        mesh_data, registry = build_gds_2d_mesh(
            by_layer, cut_y_um, total_power_mw * 1e-3,
            out_dir=f"out/gds/conv_{args.stack}_{refine}", refine=refine, renders=False, stack=stack,
        )
        T, k, q = solve_steady(mesh_data, registry, chip, source_depth_m=depth_um * 1e-6)
        tmax_k, _coords = tmax(T)
        dt = tmax_k - bcs.ambient_t_k
        mesh = mesh_data.mesh
        n_cells = mesh.topology.index_map(mesh.topology.dim).size_local
        elapsed = time.time() - t0
        rows.append((refine, n_cells, dt))
        print(f"{refine:>8.2f}{n_cells:>12,}{dt:>14.6f}{elapsed:>18.1f}", flush=True)

    print("\nrelative change in dT vs the next-coarser mesh:")
    for (r_prev, _c_prev, dt_prev), (r, _c, dt) in zip(rows, rows[1:]):
        rel = abs(dt - dt_prev) / abs(dt_prev) if dt_prev else float("nan")
        print(f"  refine {r_prev:.2f} -> {r:.2f}: {rel*100:.3f}%")


if __name__ == "__main__":
    main()
