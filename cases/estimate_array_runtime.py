"""Extrapolate SRAM-array runtimes from the measured per-cell costs in
out/bench/runtime.json (cases/bench_runtime.py).

Two model families, because they answer different questions and differ by
~4 orders of magnitude in cost:

  FULL RESOLUTION -- every device channel, contact and metal line meshed, as
  cases/run_bitcell_compact.py does. Cell density is taken from the pad=0
  bitcell run, which is real geometry edge to edge. (The padded runs have a
  much LOWER density because the pad is plain background material that meshes
  coarsely, so they must not be used to set this number.)

  HOMOGENIZED -- the array as effective per-tile properties on a structured
  mesh (gds/upscale.py + mesh/gds_coarse.py). Physically justified here rather
  than a compromise: the bitcell pitch is 1.2um and the thermal healing length
  sqrt(k t / h) is 629um, so the array is isothermal across ~500 cells and
  resolving individual cells inside it cannot change the answer. See
  cases/fig_healing_length.py.

    python cases/estimate_array_runtime.py [--cores 16]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

BENCH = Path("out/bench/runtime.json")
CELL_AREA_UM2 = 1.2 * 1.58
# Peak RSS per cell. Measured for the structured-hex whole-die path
# (294,500 cells -> 2.79GB); used here as an ORDER-OF-MAGNITUDE guide for the
# tet path, not a measurement of it. Flagged in the output as an estimate.
BYTES_PER_CELL = 9.5e3

ARRAYS = [(1, 1), (2, 2), (4, 4), (8, 8), (16, 16), (32, 32), (128, 128)]


def fit_per_cell(rows, key):
    """Cost per cell, from the largest run (least dominated by fixed overhead)."""
    r = max(rows, key=lambda r: r["n_cells"])
    return r[key] / r["n_cells"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cores", type=int, default=16)
    p.add_argument("--steps", type=int, default=60,
                   help="transient steps for a duty-cycle-averaged run "
                        "(cases/run_bitcell_transient.py uses 30 active + 30 idle)")
    args = p.parse_args()

    if not BENCH.exists():
        raise SystemExit(f"{BENCH} missing -- run cases/bench_runtime.py first")
    rows = json.loads(BENCH.read_text())

    base = min(rows, key=lambda r: r["n_cells"])          # pad=0, real geometry throughout
    density = base["n_cells"] / CELL_AREA_UM2
    c_mesh = fit_per_cell(rows, "t_mesh_s")
    c_steady = fit_per_cell(rows, "t_steady_s")
    c_fixed = fit_per_cell(rows, "t_step_fixed_dt_s")
    c_vary = fit_per_cell(rows, "t_step_varying_dt_s")

    print(f"measured on {len(rows)} meshes, {args.cores} cores")
    for r in rows:
        print(f"  {r['n_cells']:>10,} cells   mesh {r['t_mesh_s']:7.2f}s  "
              f"steady {r['t_steady_s']:6.2f}s  step(fixed) {r['t_step_fixed_dt_s']:6.3f}s  "
              f"step(vary) {r['t_step_varying_dt_s']:6.3f}s")
    print(f"\nper cell: mesh {c_mesh*1e6:.2f}us  steady {c_steady*1e6:.2f}us  "
          f"step-fixed {c_fixed*1e6:.2f}us  step-vary {c_vary*1e6:.2f}us")
    print(f"full-resolution cell density: {density:,.0f} mesh cells per um^2 "
          f"of real layout\n")

    def fmt(sec):
        if sec < 90:
            return f"{sec:.0f}s"
        if sec < 5400:
            return f"{sec/60:.1f}min"
        if sec < 86400 * 2:
            return f"{sec/3600:.1f}h"
        return f"{sec/86400:.1f}d"

    print("FULL RESOLUTION (every device meshed)")
    print(f"{'array':>10} {'bits':>8} {'area um^2':>10} {'cells':>14} {'RAM~':>8} "
          f"{'mesh':>9} {'steady':>9} {'+transient':>11}")
    for nx, ny in ARRAYS:
        bits = nx * ny
        area = bits * CELL_AREA_UM2
        n = density * area
        ram = n * BYTES_PER_CELL / 1e9
        t_m, t_s = n * c_mesh, n * c_steady
        t_t = args.steps * n * c_vary
        flag = "" if ram < 400 else "   << exceeds this machine"
        print(f"{nx}x{ny:<7} {bits:>8,} {area:>10.1f} {n:>14,.0f} {ram:>7.0f}G "
              f"{fmt(t_m):>9} {fmt(t_m+t_s):>9} {fmt(t_m+t_s+t_t):>11}{flag}")

    # A real macro, for scale.
    bits = 2048 * 8
    area = bits * CELL_AREA_UM2
    n = density * area
    print(f"\n  a real 16k-bit macro would be {n:,.0f} cells "
          f"({n*BYTES_PER_CELL/1e12:.1f} TB) -- not reachable, and not needed")

    print("\nHOMOGENIZED (effective per-tile properties, structured mesh)")
    print("  whole-die model measured previously at 294,500 cells / 2.79GB.")
    for n in (3e5, 1e6, 3e6):
        t_s = n * c_steady
        print(f"  {n:>10,.0f} cells   steady {fmt(t_s):>8}   "
              f"+{args.steps} transient steps {fmt(t_s + args.steps*n*c_vary):>8}")

    print(f"\nNOTE: mesh generation is {c_mesh/c_steady:.1f}x the steady solve, and gmsh is "
          f"single-threaded\n      per process -- so extra cores help by running "
          f"CONFIGURATIONS in parallel\n      (cases/run_oprins_fig12.py's subprocess pattern), "
          f"not one big mesh faster.")
    print(f"      RAM figures are order-of-magnitude (scaled from the hex path), not measured "
          f"for tets.")


if __name__ == "__main__":
    main()
