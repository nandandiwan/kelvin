"""Backside power delivery (BSPDN) thermal comparison: three configurations,
identical layout and identical total power, quantifying the thermal cost of
removing the frontside Si spreader and how much dual-sided cooling wins back.

    A. Baseline (frontside PDN)    50.2um Si,  Robin bottom only
    B. BSPDN                       ~0.4um Si,  Robin bottom only (backside sink)
    C. BSPDN + dual-sided cooling  ~0.4um Si,  Robin bottom AND top

Devices, sources, and all frontside BEOL are byte-for-byte identical across
configs — the entire delta between A and B is a `gds.techmap.StackProfile`
swap (backside geometry, all placeholder/synthesized numbers pending real
reference data — see gds/techmap.py's BSPDN block and SRAM_THERMAL_REPORT.md).

    python cases/run_bspdn_study.py [--cut-y 255.5] [--freq 1e8] [--mw MW]
                                     [--top-h-eff 20000] [--gds PATH]
"""

import argparse
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gds import techmap
from gds.power import parse_lib
from gds.read import load_top_cell, flatten_by_layer, named_cell_bbox_um
from mesh.gds_build import build_gds_2d_mesh
from mesh.gds_section import die_bounds
from post.metrics import tmax
from solve.steady import solve_steady
from spec.chip import BoundaryConditions

GDS_PATH = "data/sram22_2048x8m8w1.gds"
LIB_PATH = "data/sram22_2048x8m8w1_tt_025C_1v80.lib"
BITCELL_NAME = "sram_sp_cell"
BACKSIDE_H_EFF = 20000.0  # same lumped-sink value as the frontside baseline


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cut-y", type=float, default=255.5)
    p.add_argument("--freq", type=float, default=1e8)
    p.add_argument("--activity", type=float, default=1.0)
    p.add_argument("--mw", type=float, default=None, help="override total die power, bypasses .lib")
    p.add_argument("--top-h-eff", type=float, default=BACKSIDE_H_EFF,
                    help="dual-sided cooling's top-sink h_eff (config C only)")
    p.add_argument("--gds", default=GDS_PATH)
    p.add_argument("--lib", default=LIB_PATH)
    return p.parse_args()


def total_power_w(args) -> float:
    if args.mw is not None:
        return args.mw * 1e-3
    mp = parse_lib(args.lib)
    return mp.total_power_w(args.freq, activity=args.activity, write=True)


def homogenization_depth_m(lib, by_layer) -> float:
    try:
        _, y0c, _, y1c = named_cell_bbox_um(lib, BITCELL_NAME)
        return (y1c - y0c) * 1e-6
    except KeyError:
        x0, x1, y0, y1 = die_bounds(by_layer)
        return (y1 - y0) * 1e-6


def run_config(label, by_layer, stack, cut_y_um, power_w, depth_m, top_h_eff, out_dir):
    mesh_data, registry = build_gds_2d_mesh(
        by_layer, cut_y_um, power_w, out_dir=out_dir, stack=stack, renders=True,
    )
    bcs = BoundaryConditions(ambient_t_k=300.0, backside_h_eff=BACKSIDE_H_EFF, top_h_eff=top_h_eff)
    chip = types.SimpleNamespace(bcs=bcs)
    T, k, q = solve_steady(mesh_data, registry, chip, source_depth_m=depth_m)
    tmax_k, coords = tmax(T)
    mesh = mesh_data.mesh
    n_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    stack_top_um = techmap.total_thickness_um(stack)
    return {
        "label": label,
        "n_cells": n_cells,
        "stack_top_um": stack_top_um,
        "tmax_k": tmax_k,
        "dT_k": tmax_k - bcs.ambient_t_k,
        "hot_x_um": coords[0] * 1e6,
        "hot_z_um": coords[1] * 1e6,
    }


def main():
    args = parse_args()
    power_w = total_power_w(args)
    print(f"total die power: {power_w*1e3:.4f} mW")

    lib, top = load_top_cell(args.gds)
    by_layer = flatten_by_layer(top)
    depth_m = homogenization_depth_m(lib, by_layer)
    print(f"homogenization depth: {depth_m*1e6:.3f} um\n")

    configs = [
        ("A: frontside PDN",       techmap.FRONTSIDE_STACK, None,          "out/gds/bspdn_a"),
        ("B: BSPDN",               techmap.BSPDN_STACK,     None,          "out/gds/bspdn_b"),
        ("C: BSPDN + dual-sided",  techmap.BSPDN_STACK,     args.top_h_eff, "out/gds/bspdn_c"),
    ]

    results = []
    for label, stack, top_h_eff, out_dir in configs:
        print(f"--- {label} ---")
        r = run_config(label, by_layer, stack, args.cut_y, power_w, depth_m, top_h_eff, out_dir)
        print(f"  stack top: {r['stack_top_um']:.3f} um, cells: {r['n_cells']:,}")
        print(f"  Tmax = {r['tmax_k']:.4f} K (dT = {r['dT_k']:.4f} K) "
              f"at x={r['hot_x_um']:.2f}um z={r['hot_z_um']:.3f}um")
        results.append(r)

    a, b, c = results
    penalty = b["dT_k"] - a["dT_k"]
    recovery = b["dT_k"] - c["dT_k"]
    vs_a = {a["label"]: 0.0, b["label"]: penalty, c["label"]: c["dT_k"] - a["dT_k"]}

    print("\n=== summary ===")
    print(f"{'config':<28}{'dT (K)':>10}{'vs A (K)':>12}")
    for r in results:
        print(f"{r['label']:<28}{r['dT_k']:>10.4f}{vs_a[r['label']]:>+12.4f}")

    print(f"\nBSPDN penalty (B vs A):      {penalty:+.4f} K")
    # ITherm's own definition: dual-sided cooling reduction is relative to
    # BSPDN alone (C), not relative to the A-vs-B penalty (see
    # cases/run_bspdn_benchmark.py's same fix — that script's docstring has
    # the full explanation).
    if b["dT_k"]:
        print(f"dual-sided cooling recovers: {recovery:+.4f} K "
              f"({100*recovery/b['dT_k']:.1f}% of BSPDN's own dT, ITherm's definition)")
    print("\nNote: absolute dT here is small (this macro's power density is far below a logic core's,"
          " and this project's usual h_eff=20,000 W/m2K is a much more heavily backside-sink-dominated"
          " regime than the near-isothermal 'high-performance heat sink' cases/run_bspdn_benchmark.py"
          " validates against — see SRAM_THERMAL_REPORT.md section 7 for both, together.")


if __name__ == "__main__":
    main()
