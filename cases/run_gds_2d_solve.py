"""Solve the GDS 2D cross-section. Same physics/solve/post code as the
synthetic chip, completely unchanged — only the geometry+sources came from
gds/. source_depth_m is required here (not None): a 2D cut only resolves x
and z, so treating each source's real 3D volume as literally injected into
this y-uniform/infinitely-extruded model overstates power by roughly
1/(device width) — exactly PLAN.md's Tmax=4135K bug, encountered before.

Homogenization depth: this macro's bitcell array genuinely repeats in Y (a
real pitch to measure, unlike the old 2-row ring-oscillator layout, where
"die Y-extent" was the only defensible choice absent a repeating structure)
— using the full 193.7um die Y-extent here instead would dilute injected
power ~64x (the number of rows), understating Tmax by roughly that factor.
`gds.read.named_cell_bbox_um` measures the real per-row pitch directly from
the GDS hierarchy (the `sram_sp_cell` cell's own bounding box, 1.2 x 1.58um)
rather than guessing it, and falls back to the full die Y-extent if that
cell isn't found (e.g. a different macro/layout entirely).

Total die power now comes from the macro's own Liberty characterization
(gds/power.py) rather than an assumed round number: P = P_leakage + f_clk *
E_per_cycle, at a chosen access frequency and activity factor. Pass --mw to
override with a raw number directly (e.g. for comparing against the old
default, or for the BSPDN study's controlled A/B/C power-matched runs).

    python cases/run_gds_2d_solve.py [--cut-y 255.5] [--freq 1e8] [--mw MW] [--gds PATH]
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
from post import io as post_io
from post.metrics import tmax
from solve.steady import solve_steady
from spec.chip import BoundaryConditions

# A real, placed SKY130 SRAM macro (BSD-3, github.com/ucb-substrate/
# sram22_sky130_macros) — 2048 words x 8 bits, 310.7 x 760.2um die,
# 16,384 bitcells (6T each) + periphery.
GDS_PATH = "data/sram22_2048x8m8w1.gds"
LIB_PATH = "data/sram22_2048x8m8w1_tt_025C_1v80.lib"
BITCELL_NAME = "sram_sp_cell"


def parse_args():
    p = argparse.ArgumentParser()
    # Default cutline found by histogramming diff∩poly channel y-centers
    # across the whole die and picking the densest 1um row (see GDS_PLAN.md).
    p.add_argument("--cut-y", type=float, default=255.5)
    p.add_argument("--freq", type=float, default=1e8, help="access frequency, Hz")
    p.add_argument("--activity", type=float, default=1.0, help="fraction of cycles that are real accesses")
    p.add_argument("--write", action="store_true", default=True)
    p.add_argument("--mw", type=float, default=None, help="override: total die power in mW, bypasses .lib")
    p.add_argument("--gds", default=GDS_PATH)
    p.add_argument("--lib", default=LIB_PATH)
    return p.parse_args()


def total_power_w(args) -> float:
    if args.mw is not None:
        print(f"total power: {args.mw} mW (--mw override, .lib not read)")
        return args.mw * 1e-3
    mp = parse_lib(args.lib)
    p = mp.total_power_w(args.freq, activity=args.activity, write=args.write)
    e_access = mp.energy_per_access_j(args.write)
    print(f"total power: {p*1e3:.4f} mW  (from {Path(args.lib).name}: "
          f"leakage={mp.leakage_w*1e6:.3f}uW + {args.freq:.3e}Hz * {e_access*1e12:.3f}pJ/access "
          f"at activity={args.activity:.2f})")
    return p


def main():
    args = parse_args()
    cut_y_um = args.cut_y
    total_power_mw = total_power_w(args) * 1e3
    gds_path = args.gds

    lib, top = load_top_cell(gds_path)
    # No union_merge: see gds/read.py's docstring — it doesn't scale and
    # isn't needed for the cutline-intersection path's correctness.
    by_layer = flatten_by_layer(top)

    try:
        _, y0c, _, y1c = named_cell_bbox_um(lib, BITCELL_NAME)
        source_depth_um = y1c - y0c
        print(f"homogenization depth (measured {BITCELL_NAME!r} row pitch): {source_depth_um:.3f} um")
    except KeyError:
        x0, x1, y0, y1 = die_bounds(by_layer)
        source_depth_um = y1 - y0
        print(f"'{BITCELL_NAME}' not found; falling back to die Y-extent: {source_depth_um:.3f} um")

    mesh_data, registry = build_gds_2d_mesh(by_layer, cut_y_um, total_power_mw * 1e-3)

    bcs = BoundaryConditions(ambient_t_k=300.0, backside_h_eff=20000.0, top_face="adiabatic")
    chip = types.SimpleNamespace(bcs=bcs)

    T, k, q = solve_steady(mesh_data, registry, chip, source_depth_m=source_depth_um * 1e-6)

    tmax_k, coords = tmax(T)
    hot_x = coords[0] * 1e6
    print(f"Tmax = {tmax_k:.3f} K at x={hot_x:.3f}um z={coords[1]*1e6:.3f}um "
          f"(ambient {bcs.ambient_t_k:.1f} K)")
    print(f"dT = {tmax_k - bcs.ambient_t_k:.4f} K")
    print(f"total power on this cut: {total_power_mw} mW")

    post_io.write_solution(mesh_data, T, k, q, out_dir="out/gds")

    from post.viz import render_temperature_generic
    x0, x1, _, _ = die_bounds(by_layer)
    z_top = techmap.total_thickness_um()
    zoom_z0 = techmap.find("Si_substrate").thickness_um - 0.02
    render_temperature_generic(
        T, (x0, x1), (0, z_top), (zoom_z0, z_top),
        f"Full stack — cut y={cut_y_um}um",
        f"FEOL/BEOL zoom, z=[{zoom_z0:.2f}, {z_top:.2f}] um",
        "out/gds/temperature_2d.png",
        zoom_x_bounds=(hot_x - 7.5, hot_x + 7.5),
    )

    print("wrote out/gds/{solution.xdmf, temperature_2d.png}")


if __name__ == "__main__":
    main()
