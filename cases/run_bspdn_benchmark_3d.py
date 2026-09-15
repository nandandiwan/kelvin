"""BSPDN validation benchmark in TRUE 3D — the 3D counterpart of
cases/run_bspdn_benchmark.py, which runs the same four configs (A-D) on a 2D
cross-section.

Why this exists: the 2D cut resolves one lateral axis and homogenizes the
other (`source_depth_m`), which turns [Oprins]' genuinely small 400x1000nm
heat source into an infinite LINE source. A line source has far lower
spreading resistance than a real 3D point source — and the backside metal's
published 27-30% benefit comes precisely from relieving that 3D spreading
bottleneck. With little bottleneck left to relieve, the 2D model
under-reports the benefit (measured: 5.4% vs 27-30%). Resolving both lateral
axes is the fix.

Two geometry corrections ride along, both in gds/techmap.py +
mesh/gds_volume.py (see GdsLayerBand.synthetic_shape):
  - backside metal is now emitted as continuous LINES (a rail conducts along
    its length — that lateral path IS the mechanism being measured), not the
    disconnected posts the 3D emitter previously drew for every SYNTHETIC
    band;
  - nTSVs stay discrete posts, one per metal line ([Oprins]: nTSV pitch ==
    M1 pitch).

    python cases/run_bspdn_benchmark_3d.py [--window-um 3.0] [--refine 1.0]
"""

import argparse
import json
import os
import subprocess
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gdstk

import numpy as np
from dolfinx.fem import Function, dirichletbc, locate_dofs_topological
from scipy.interpolate import RegularGridInterpolator

from cases.run_bspdn_benchmark import (
    BACKSIDE_H_EFF, SOURCE_L_UM, SOURCE_W_UM,
)
from gds import techmap
from gds.sources import CHANNEL_POWER_FRAC, build_heat_sources
from gds.techmap import DIFF, POLY
from gds.upscale import build_tile_grid
from mesh.build import FACET_X0, FACET_X1, FACET_Y0, FACET_Y1
from mesh.gds_build import build_gds_3d_mesh
from mesh.gds_coarse import build_gds_coarse_mesh
from post.budget import power_balance, print_power_balance
from post.metrics import tmax
from solve.steady import solve_steady, solve_steady_from_fields
from spec.chip import BoundaryConditions

# The fine 3D window is centred on the source. It must be big enough that the
# adiabatic lateral walls don't clamp the lateral spreading being measured --
# checked directly by --window-um sweep, not assumed.
DEFAULT_WINDOW_UM = 3.0
CENTER_UM = 10.0  # matches cases/run_bspdn_benchmark.py's DIE_UM/2 source placement

# Source footprint, overridable so the ORIENTATION of the rectangular source
# relative to the backside metal lines can be tested. [Oprins] Table 1 gives
# 400 x 1000 nm but their Fig. 6 defines the area as 4*P_BPR*P_M1, which does
# not by itself say which axis is which -- and it matters: the long axis
# lying ACROSS the 500nm-pitch lines spans two of them instead of one.
SOURCE_WL_UM = [SOURCE_W_UM, SOURCE_L_UM]


def synthetic_by_layer(window_um: float):
    """One DIFF == one POLY rectangle at the window centre, so
    gds.sources.extract_channels' DIFF∩POLY returns exactly this footprint as
    the sole heat source (no LICON1 layer -> extract_contacts returns []).
    Same construction as cases/run_bspdn_benchmark.py's, re-centred for a
    window that is no longer the whole 20x20um cell."""
    sw, sl = SOURCE_WL_UM
    x0, x1 = CENTER_UM - sw / 2, CENTER_UM + sw / 2
    y0, y1 = CENTER_UM - sl / 2, CENTER_UM + sl / 2
    rect = gdstk.rectangle((x0, y0), (x1, y1))
    half = window_um / 2
    die = gdstk.rectangle((CENTER_UM - half, CENTER_UM - half),
                           (CENTER_UM + half, CENTER_UM + half))
    return {DIFF: [rect], POLY: [rect], (0, 0): [die]}


def _interpolator_from_solution(T):
    """RegularGridInterpolator over a solved field on a STRUCTURED box mesh
    (mesh/gds_coarse.py's create_box + z-warp), so the fine stage can read the
    coarse far-field temperature at arbitrary (x, y, z). Same role as
    cases/run_gds_submodel.py::build_coarse_interpolator, but sourced from a
    solve we just ran rather than a cached whole-die .npz -- this benchmark's
    outer domain is self-contained, so there is nothing to cache."""
    V = T.function_space
    coords = np.round(V.tabulate_dof_coordinates(), 12)
    xs, ys, zs = (np.unique(coords[:, i]) for i in range(3))
    grid = np.full((len(xs), len(ys), len(zs)), np.nan)
    grid[np.searchsorted(xs, coords[:, 0]),
         np.searchsorted(ys, coords[:, 1]),
         np.searchsorted(zs, coords[:, 2])] = T.x.array
    if np.isnan(grid).any():  # structured box: every node should be filled
        raise ValueError(f"coarse grid has {int(np.isnan(grid).sum())} unfilled nodes")
    return RegularGridInterpolator((xs, ys, zs), grid, bounds_error=False, fill_value=None)


def coarse_stage(stack, top_h_eff, power_w, outer_um, tile_um):
    """STAGE 1 of the submodel: the full outer cell, solved cheaply with the
    via/metal grid HOMOGENIZED to its area-fraction effective properties
    (gds/upscale.py). That homogenization is the right physics out here --
    far from the source the field varies over many pitches, so the discrete
    pattern doesn't matter -- and it buys the true 3D far-field lateral
    spreading that a small adiabatic-walled window cannot produce."""
    by_layer = synthetic_by_layer(outer_um)
    half = outer_um / 2
    window = (CENTER_UM - half, CENTER_UM + half, CENTER_UM - half, CENTER_UM + half)
    channel_sources, contact_sources = build_heat_sources(by_layer, power_w / CHANNEL_POWER_FRAC)

    tile_grid = build_tile_grid(by_layer, channel_sources, contact_sources, stack, window, tile_um)
    mesh_data, k, rho_cp, q = build_gds_coarse_mesh(tile_grid)
    bcs = BoundaryConditions(ambient_t_k=300.0, backside_h_eff=BACKSIDE_H_EFF, top_h_eff=top_h_eff)
    chip = types.SimpleNamespace(bcs=bcs)
    T = solve_steady_from_fields(mesh_data, k, q, chip)
    n_cells = mesh_data.mesh.topology.index_map(mesh_data.mesh.topology.dim).size_local
    tk, _ = tmax(T)
    print(f"    stage1 (coarse {outer_um}um, tile={tile_um}um): Tmax={tk:.6f}K "
          f"dT={tk-300.0:.6e}K cells={n_cells}")
    return _interpolator_from_solution(T)


def run_config(label, stack, top_h_eff, power_w, out_dir, window_um, refine,
                submodel=False, outer_um=20.0, tile_um=0.5, radiation=False):
    by_layer = synthetic_by_layer(window_um)
    half = window_um / 2
    window = (CENTER_UM - half, CENTER_UM + half, CENTER_UM - half, CENTER_UM + half)

    # Same CHANNEL_POWER_FRAC compensation as the 2D benchmark: with no
    # LICON1 layer the contact 30% would otherwise be silently dropped.
    channel_sources, contact_sources = build_heat_sources(by_layer, power_w / CHANNEL_POWER_FRAC)

    mesh_data, registry = build_gds_3d_mesh(
        by_layer, window, power_w / CHANNEL_POWER_FRAC, out_dir=out_dir,
        refine=refine, renders=False, stack=stack,
        channel_sources=channel_sources, contact_sources=contact_sources,
        # Configs A-D MUST be meshed equivalently or the ratios between them
        # measure resolution, not physics -- see apply_gds_3d_sizing's docstring
        # (measured: B came out 4.33M cells vs C's 2.04M without this).
        uniform_background=True,
        # Phase the backside metal/uTSV grid so a feature sits centred on the
        # heat source, as in [Oprins] Fig. 6 (their small source IS "the area
        # contacted by a single uTSV"). Without this the via lands under the
        # heater only by luck of where the window edge fell.
        grid_anchor_um=(CENTER_UM, CENTER_UM),
    )
    # Asymptotic radiation BC on the lateral cuts (h = k/r): lets heat leave
    # sideways as if the medium continued outward, instead of the natural
    # zero-flux "box" -- see physics/bcs.py::lateral_radiation_terms. r is the
    # source-to-cut distance, i.e. half the window.
    bcs = BoundaryConditions(ambient_t_k=300.0, backside_h_eff=BACKSIDE_H_EFF, top_h_eff=top_h_eff,
                              lateral_radiation_r_m=(window_um / 2 * 1e-6) if radiation else None)
    chip = types.SimpleNamespace(bcs=bcs)

    bc_builder = None
    if submodel:
        interp = coarse_stage(stack, top_h_eff, power_w, outer_um, tile_um)
        fdim = mesh_data.mesh.topology.dim - 1
        lateral_facets = np.concatenate([mesh_data.facet_tags.find(tag)
                                          for tag in (FACET_X0, FACET_X1, FACET_Y0, FACET_Y1)])

        def bc_builder(V):
            # Built against steady_form's OWN V (passed in here), never a
            # separately-created functionspace() -- see solve_steady's
            # docstring: that mistake silently yields a garbage solution
            # (~11 orders of magnitude off) rather than an error.
            g = Function(V)
            g.x.array[:] = interp(V.tabulate_dof_coordinates())
            dofs = locate_dofs_topological(V, fdim, lateral_facets)
            # Report the values actually PINNED, not the interpolant over the
            # whole window -- the latter includes the hot centre and makes the
            # cut look far hotter than it is.
            pinned = g.x.array[dofs]
            print(f"    stage2 lateral BC: {len(dofs)} dofs pinned to stage1 "
                  f"({pinned.min():.4f}-{pinned.max():.4f} K)")
            return [dirichletbc(g, dofs)]

    # source_depth_m=None: a true 3D mesh resolves all three dimensions, so
    # the 2D path's depth homogenization must NOT be applied here.
    T, k, q = solve_steady(mesh_data, registry, chip, source_depth_m=None, bc_builder=bc_builder)

    n_cells = mesh_data.mesh.topology.index_map(mesh_data.mesh.topology.dim).size_local
    tmax_k, _ = tmax(T)
    print(f"  [{label}] Tmax={tmax_k:.6f}K dT={tmax_k-300.0:.6e}K "
          f"cells={n_cells} stack_top={techmap.total_thickness_um(stack):.3f}um")
    bal = power_balance(mesh_data, registry, chip, T, k, q, source_depth_m=None)
    # With Dirichlet lateral cuts, the window is a carved-out piece of a larger
    # domain and heat legitimately leaves through its SIDES -- so the
    # `p_gen == Robin outflow` identity does not apply (measured: only 23uW of
    # 1000uW leaves the bottom sink; the rest exits laterally). The fraction
    # leaving laterally is itself the useful diagnostic: it says how strongly
    # the BC ring, rather than the source, is setting the answer.
    if submodel:
        lateral_frac = 1.0 - bal["p_out_robin_w"] / bal["p_gen_w"]
        print(f"    stage2 lateral outflow: {100*lateral_frac:.1f}% of injected power "
              f"(Robin identity N/A for a Dirichlet-cut window)")
    print_power_balance(bal, check_robin=not submodel)
    return tmax_k - 300.0, n_cells


# Each config is an independent mesh build + solve, so they run as separate
# PROCESSES, not threads: gmsh is a global singleton within a process
# (gmsh.initialize()/finalize() in mesh/gds_build.py), so two configs sharing
# one interpreter would corrupt each other's model. Subprocesses also give
# each solve its own PETSc state.
CONFIGS = {
    # name: (stack_factory, uses_top_h_eff)
    "A":      (lambda: techmap.FRONTSIDE_STACK, False),
    "B":      (lambda: techmap.bspdn_stack("bench3d_b", 0.5, "Si_thin_500nm", with_backside_metal=False), False),
    "C":      (lambda: techmap.bspdn_stack("bench3d_c", 0.5, "Si_thin_500nm", with_backside_metal=True), False),
    "D":      (lambda: techmap.bspdn_stack("bench3d_c", 0.5, "Si_thin_500nm", with_backside_metal=True), True),
    "C_bulk": (lambda: techmap.bspdn_stack("bench3d_c_bulk", 0.5, "Si_bulk", with_backside_metal=True), False),
}
_LABELS = {
    "A": "FSPDN baseline (50um Si, bulk k)",
    "B": "BSPDN, 500nm Si, no backside metal (Oprins' 2.2x case)",
    "C": "BSPDN, 500nm Si, with backside metal (Oprins' 60% net penalty case)",
    "D": "C + dual-sided cooling (ITherm's 22% recovery case)",
    "C_bulk": "C with bulk Si k (thin-film-k ablation)",
}


def run_one(name, power_w, window_um, refine, top_h_eff, tag,
             submodel=False, outer_um=20.0, tile_um=0.5):
    stack_factory, uses_top = CONFIGS[name]
    return run_config(name, stack_factory(), top_h_eff if uses_top else None, power_w,
                       f"out/gds/bench3d_{tag}_{name.lower()}", window_um, refine,
                       submodel=submodel, outer_um=outer_um, tile_um=tile_um)


def _parallel(args, tag, power_w, max_workers, max_cores):
    """Fan the 5 configs out as concurrent subprocesses, each re-entering this
    same script with --config. Capped at `max_workers` (this is a SHARED
    machine -- 112 cores exist but heavy jobs from other users share them)."""
    results_dir = Path(f"out/gds/bench3d_{tag}_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    procs, logs = {}, {}
    pending = list(CONFIGS)
    running = {}

    # Hard total-core budget: without this each worker's PETSc/BLAS/OpenMP
    # runtime grabs as many threads as it likes (measured: 5 workers ran at
    # ~11 cores average, 66 CPU-minutes in 6 wall-minutes). This is a SHARED
    # machine, so the cap is on TOTAL cores across all workers, not per
    # worker. OMP_NUM_THREADS also covers gmsh, whose meshing parallelism is
    # OpenMP-based -- no separate gmsh knob needed.
    threads = max(1, max_cores // max(1, max_workers))
    thread_env = {var: str(threads) for var in
                  ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}
    print(f"  core budget: {max_workers} workers x {threads} threads = "
          f"{max_workers*threads} of {max_cores} allowed", flush=True)

    def launch(name):
        out_json = results_dir / f"{name}.json"
        log = open(results_dir / f"{name}.log", "w")
        cmd = [sys.executable, __file__, "--config", name, "--mw", str(args.mw),
               "--window-um", str(args.window_um), "--refine", str(args.refine),
               "--top-h-eff", str(args.top_h_eff), "--out-json", str(out_json),
               "--outer-um", str(args.outer_um), "--tile-um", str(args.tile_um)]
        if args.submodel:
            cmd.append("--submodel")
        logs[name] = log
        return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                 cwd=str(Path(__file__).resolve().parents[1]),
                                 env={**os.environ, **thread_env})

    while pending or running:
        while pending and len(running) < max_workers:
            name = pending.pop(0)
            running[name] = launch(name)
            print(f"  launched {name}: {_LABELS[name]}", flush=True)
        done = [n for n, p in running.items() if p.poll() is not None]
        if not done:
            import time
            time.sleep(2)
            continue
        for name in done:
            rc = running.pop(name).returncode
            logs[name].close()
            if rc != 0:
                raise SystemExit(f"config {name} failed (rc={rc}); see {results_dir}/{name}.log")
            procs[name] = json.loads((results_dir / f"{name}.json").read_text())
            print(f"  [{name}] dT={procs[name]['dt_k']:.6e}K "
                  f"cells={procs[name]['n_cells']}", flush=True)
    return {n: r["dt_k"] for n, r in procs.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mw", type=float, default=1.0)
    p.add_argument("--window-um", type=float, default=DEFAULT_WINDOW_UM)
    p.add_argument("--refine", type=float, default=1.0)
    p.add_argument("--top-h-eff", type=float, default=BACKSIDE_H_EFF)
    p.add_argument("--config", choices=sorted(CONFIGS),
                    help="run ONE config and write --out-json (used by the parallel driver)")
    p.add_argument("--out-json", help="where --config writes its result")
    p.add_argument("--jobs", type=int, default=5,
                    help="max concurrent config subprocesses (shared machine: keep well under nproc)")
    p.add_argument("--max-cores", type=int, default=16,
                    help="TOTAL core budget across all workers (shared machine)")
    p.add_argument("--serial", action="store_true", help="run configs one at a time")
    p.add_argument("--submodel", action="store_true",
                    help="two-stage: homogenized outer cell supplies Dirichlet BCs to the fine window")
    p.add_argument("--outer-um", type=float, default=20.0, help="--submodel outer domain size")
    p.add_argument("--tile-um", type=float, default=0.5, help="--submodel outer tile size")
    args = p.parse_args()
    power_w = args.mw * 1e-3
    tag = f"w{args.window_um:g}_r{args.refine:g}" + (f"_sub{args.outer_um:g}" if args.submodel else "")

    if args.config:  # single-config worker
        dt, n_cells = run_one(args.config, power_w, args.window_um, args.refine, args.top_h_eff, tag,
                               submodel=args.submodel, outer_um=args.outer_um, tile_um=args.tile_um)
        if args.out_json:
            Path(args.out_json).write_text(json.dumps({"config": args.config, "dt_k": dt,
                                                        "n_cells": n_cells}))
        return

    print(f"TRUE 3D: {args.window_um}x{args.window_um}um window, "
          f"{SOURCE_W_UM*1000:.0f}x{SOURCE_L_UM*1000:.0f}nm source, {args.mw}mW, "
          f"h_eff={BACKSIDE_H_EFF:.0f} W/m2K, refine={args.refine}\n")

    if args.serial:
        dt = {n: run_one(n, power_w, args.window_um, args.refine, args.top_h_eff, tag,
                          submodel=args.submodel, outer_um=args.outer_um, tile_um=args.tile_um)[0]
              for n in CONFIGS}
    else:
        dt = _parallel(args, tag, power_w, max_workers=args.jobs, max_cores=args.max_cores)

    dt_a, dt_b, dt_c, dt_d, dt_c_bulk = (dt["A"], dt["B"], dt["C"], dt["D"], dt["C_bulk"])
    print("\n=== ratios vs. published numbers (TRUE 3D) ===")
    print(f"B/A, Si thinning alone (Oprins: 2.20x):              {dt_b/dt_a:.3f}x")
    print(f"(B-C)/B, backside metal's own reduction (Oprins: 27-30%): {100*(dt_b-dt_c)/dt_b:.1f}%")
    print(f"C/A, net BSPDN penalty (Oprins: 1.60x, i.e. +60%):   {dt_c/dt_a:.3f}x  (+{100*(dt_c/dt_a-1):.1f}%)")
    print(f"(C-D)/C, dual-sided cooling's own reduction (ITherm: 22%): {100*(dt_c-dt_d)/dt_c:.1f}%")

    print("\n=== thin-film-k ablation (Oprins: k reduction alone -> ~50% additional) ===")
    geometric = dt_c_bulk - dt_a
    print(f"thin-film-k additional contribution, (C - C_bulk)/(C_bulk - A): "
          f"{100*(dt_c-dt_c_bulk)/geometric:.1f}% (Oprins: ~50%)" if geometric else "n/a")


if __name__ == "__main__":
    main()
