"""3D GDS build+solve: real device geometry, genuinely resolved in x, y, AND
z — no 2D-cut homogenization trick needed (source_depth_m=None; a true 3D
mesh resolves each source's real depth directly, per PLAN.md).

NOT the whole die: at real polygon density even the 2D path's per-cutline
polygon counts run into the hundreds of thousands (see GDS_PLAN.md); in 3D,
backside grids and per-source geometry stop being a 1D line of features
along a cutline and become a full 2D array across the die, which raises
entity count sharply, not marginally. This builds a bounded lateral WINDOW
of real geometry instead — same principle as the original synthetic 3D
chip's 24x24um tile, now sourced from real polygons instead of synthetic
device boxes. Window size is a direct memory/time knob: it was measured
empirically before committing to a build (see SRAM_THERMAL_REPORT.md) —
run --probe first on a new window size before spending minutes on a mesh.

    python cases/run_gds_3d.py --probe [--window 6] [--cx 216] [--cy 255.5]
    python cases/run_gds_3d.py [--window 6] [--cx 216] [--cy 255.5] [--mw 1e-6] [--stack bspdn]
"""

import argparse
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gdstk

from gds import techmap
from gds.read import load_top_cell, flatten_by_layer
from gds.sources import build_heat_sources
from gds.techmap import RELEVANT_LAYERS
from mesh.gds_build import build_gds_3d_mesh
from post.metrics import tmax
from solve.steady import solve_steady
from spec.chip import BoundaryConditions

GDS_PATH = "data/sram22_2048x8m8w1.gds"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cx", type=float, default=216.0)
    p.add_argument("--cy", type=float, default=255.5)
    p.add_argument("--window", type=float, default=6.0, help="window side length, um")
    p.add_argument("--mw", type=float, default=1e-6, help="total DIE power in mW (only the window's real share lands inside it)")
    p.add_argument("--stack", choices=["frontside", "bspdn"], default="frontside")
    p.add_argument("--refine", type=float, default=1.0)
    p.add_argument("--top-h-eff", type=float, default=None)
    p.add_argument("--gds", default=GDS_PATH)
    p.add_argument("--probe", action="store_true", help="just count real polygons in the window and exit — run this before a real build")
    return p.parse_args()


def window_bounds(args):
    h = args.window / 2
    return (args.cx - h, args.cx + h, args.cy - h, args.cy + h)


def probe(by_layer, window):
    x0, x1, y0, y1 = window
    win = gdstk.rectangle((x0, y0), (x1, y1))
    total = 0
    for key in RELEVANT_LAYERS:
        polys = by_layer.get(key, [])
        if not polys:
            continue
        n = len(gdstk.boolean(polys, [win], "and"))
        total += n
        print(f"  {key}: {n}")
    print(f"window {x1-x0:.1f}x{y1-y0:.1f}um: {total} real polygons "
          f"(the original synthetic 3D chip's 939 boxes took ~9.5min; "
          f"z-band duplication (e.g. licon1 spans 2 bands) pushes actual "
          f"3D entity count somewhat above this)")


def main():
    args = parse_args()
    window = window_bounds(args)

    lib, top = load_top_cell(args.gds)
    by_layer = flatten_by_layer(top)

    if args.probe:
        probe(by_layer, window)
        return

    stack = techmap.STACKS[args.stack]
    power_w = args.mw * 1e-3

    mesh_data, registry = build_gds_3d_mesh(
        by_layer, window, power_w, out_dir=f"out/gds3d_{args.stack}",
        refine=args.refine, stack=stack,
    )

    bcs = BoundaryConditions(ambient_t_k=300.0, backside_h_eff=20000.0, top_h_eff=args.top_h_eff)
    chip = types.SimpleNamespace(bcs=bcs)
    # source_depth_m=None: true 3D, no 2D-cut homogenization needed.
    T, k, q = solve_steady(mesh_data, registry, chip, source_depth_m=None)

    tmax_k, coords = tmax(T)
    print(f"Tmax = {tmax_k:.6f} K (dT = {tmax_k-300.0:.6e} K) "
          f"at x={coords[0]*1e6:.3f}um y={coords[1]*1e6:.3f}um z={coords[2]*1e6:.3f}um")

    from post import io as post_io
    post_io.write_solution(mesh_data, T, k, q, out_dir=f"out/gds3d_{args.stack}")
    print(f"wrote out/gds3d_{args.stack}/{{gds_volume.msh, mesh_tags.xdmf, stack_table.txt, "
          f"slice_*.png, solution.xdmf}}")


if __name__ == "__main__":
    main()
