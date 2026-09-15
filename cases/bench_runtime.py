"""Measured cost of the 3D path: mesh generation, one steady solve, and one
transient step, as a function of cell count -- so array-scale runtimes can be
extrapolated from data instead of guessed.

Run single-job on all cores. The isolation sweep's wall-clock numbers are NOT
a valid basis for this: it ran 4 jobs x 4 threads concurrently, so every timing
there is contended.

Transient steps are timed in two regimes, because they differ by an order of
magnitude and which one you get depends on the time-stepping schedule:
  - FIXED dt: the system matrix and the AMG hierarchy are built once and
    reused; each step is one CG solve.
  - CHANGING dt: the mass term rho_cp/dt changes, so the matrix is reassembled
    and the preconditioner re-set up every step. A geometrically growing dt
    schedule (cases/run_bitcell_transient.py) pays this on every step.

    python cases/bench_runtime.py [--pads 0 1 2] [--steps 5]
"""

import argparse
import json
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gdstk
import numpy as np

from gds import techmap
from gds.read import flatten_by_layer
from gds.sources import _dims_um, extract_channels, extract_contacts
from gds.spice_power import named_bias_point_power_w
from mesh.gds_build import build_gds_3d_mesh
from physics.coeffs import build_coeffs
from post.metrics import tmax
from solve.steady import solve_steady_from_fields
from solve.transient import TransientHeatSolver
from spec.chip import BoundaryConditions, SourceBox

OUT = Path("out/bench")
GDS_PATH = "data/sram22_64x22m4w22.gds"
CELL_NAME = "sram_sp_cell"
_NWELL_SPLIT_X_UM = -0.72


def _classify(x_ext, y_ext, is_left):
    if abs(y_ext - 0.025) < 0.01:
        return "parasitic"
    if abs(x_ext - 0.21) < 0.01:
        return "latch"
    return "pullup" if is_left else "access"


def bench_one(pad_um, refine, n_steps):
    lib = gdstk.read_gds(GDS_PATH)
    cell = next(c for c in lib.cells if c.name == CELL_NAME)
    by_layer = flatten_by_layer(cell)
    (x0, y0), (x1, y1) = cell.bounding_box()
    window = (x0 - pad_um, x1 + pad_um, y0 - pad_um, y1 + pad_um)

    dp = named_bias_point_power_w("crowbar", row_hit_rate=1.0)
    cls_w = {"access": dp["X0"], "latch": dp["X1"], "pullup": dp["X5"],
             "parasitic": dp["X3"]}
    ch_src = []
    for i, poly in enumerate(extract_channels(by_layer)):
        cx, cy, xe, ye = _dims_um(poly)
        cls = _classify(xe, ye, is_left=cx < _NWELL_SPLIT_X_UM)
        ch_src.append((SourceBox(device=f"{cls}{i}", kind="channel", x_um=cx, y_um=cy,
                                 z0_um=0.0, w_um=xe, l_um=ye,
                                 t_um=techmap.CHANNEL_THICKNESS_UM,
                                 power_uw=cls_w[cls] * 1e6), poly))
    co_src = [(SourceBox(device=f"co{i}", kind="contact", x_um=_dims_um(p)[0],
                         y_um=_dims_um(p)[1], z0_um=0.0, w_um=_dims_um(p)[2],
                         l_um=_dims_um(p)[3], t_um=techmap.LICON1_THICKNESS_UM,
                         power_uw=0.0), p)
              for i, p in enumerate(extract_contacts(by_layer))]
    total_w = sum(b.power_uw for b, _ in ch_src) * 1e-6

    t0 = time.perf_counter()
    mesh_data, registry = build_gds_3d_mesh(
        by_layer, window, total_w, out_dir=f"{OUT}/pad_{pad_um:g}", refine=refine,
        renders=False, stack=techmap.FRONTSIDE_STACK,
        channel_sources=ch_src, contact_sources=co_src)
    t_mesh = time.perf_counter() - t0

    n_cells = mesh_data.mesh.topology.index_map(mesh_data.mesh.topology.dim).size_local
    k, rho_cp, q = build_coeffs(mesh_data.mesh, mesh_data.cell_tags, registry,
                                source_depth_m=None)
    chip = types.SimpleNamespace(
        bcs=BoundaryConditions(ambient_t_k=300.0, backside_h_eff=20000.0, top_h_eff=None))

    t0 = time.perf_counter()
    T = solve_steady_from_fields(mesh_data, k, q, chip)
    t_steady = time.perf_counter() - t0

    # Transient. The first step also pays matrix assembly + AMG setup, so it is
    # timed separately from the steady-state per-step cost.
    solver = TransientHeatSolver(mesh_data, k, rho_cp, chip, dt=1e-6, T0=300.0)
    q_arr = q.x.array.copy()
    t0 = time.perf_counter()
    solver.step(q_arr, dt=1e-6)
    t_first = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n_steps):
        solver.step(q_arr, dt=1e-6)          # fixed dt -> matrix reused
    t_fixed = (time.perf_counter() - t0) / n_steps

    dt = 1e-6
    t0 = time.perf_counter()
    for _ in range(n_steps):
        dt *= 1.4                             # changing dt -> reassemble each step
        solver.step(q_arr, dt=dt)
    t_vary = (time.perf_counter() - t0) / n_steps

    tk, _ = tmax(T)
    return {"pad_um": pad_um, "refine": refine, "n_cells": n_cells,
            "t_mesh_s": t_mesh, "t_steady_s": t_steady, "t_first_step_s": t_first,
            "t_step_fixed_dt_s": t_fixed, "t_step_varying_dt_s": t_vary,
            "dT_K": tk - 300.0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pads", type=float, nargs="+", default=[0.0, 1.0, 2.0])
    p.add_argument("--refine", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=5)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for pad in args.pads:
        r = bench_one(pad, args.refine, args.steps)
        rows.append(r)
        print(f"  pad={pad:>4.1f}  {r['n_cells']:>10,} cells  mesh {r['t_mesh_s']:7.2f}s  "
              f"steady {r['t_steady_s']:7.2f}s  step(fixed dt) {r['t_step_fixed_dt_s']:6.3f}s  "
              f"step(varying dt) {r['t_step_varying_dt_s']:6.3f}s", flush=True)
    (OUT / "runtime.json").write_text(json.dumps(rows, indent=2))

    print(f"\n{'cells':>12} {'mesh us/cell':>13} {'steady us/cell':>15} "
          f"{'step-fixed us/cell':>19} {'step-vary us/cell':>18}")
    for r in rows:
        n = r["n_cells"]
        print(f"{n:>12,} {r['t_mesh_s']/n*1e6:>13.2f} {r['t_steady_s']/n*1e6:>15.2f} "
              f"{r['t_step_fixed_dt_s']/n*1e6:>19.2f} {r['t_step_varying_dt_s']/n*1e6:>18.2f}")
    print(f"\nwrote {OUT}/runtime.json")


if __name__ == "__main__":
    main()
