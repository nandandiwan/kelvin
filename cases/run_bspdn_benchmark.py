"""BSPDN validation benchmark: reproduce the *ratios* reported by real
published BSPDN thermal studies, on a small synthetic localized heat source
(not the SRAM layout — see cases/run_bspdn_study.py for that), so the tool's
predictions can be checked against numbers this project didn't produce.

Primary target — H. Oprins et al. (imec), "Package level thermal analysis of
backside power delivery network (BS-PDN) configurations," iTherm 2022:
  - Si thinning 200um -> 500nm, NO backside metal: **2.2x** temperature increase
  - Adding a single backside metal layer + dense nTSV array: **27-30%**
    temperature *reduction* vs. the no-metal-thinned case
  - Net: **60%** thermal penalty for BSPDN (500nm Si + backside metal) vs.
    conventional FSPDN
  - Thin-film Si conductivity reduction ALONE (isolated from the pure
    geometric/spreading effect) contributes **~50%** of the total increase

Secondary — Xie, Lyu, Wei (Purdue), ITherm 2024: dual-sided cooling recovers
**22%** of the BSPDN penalty (our config D).

Four configs, identical synthetic layout and power throughout:
  A: FSPDN baseline           — thick (50um) Si, bulk k
  B: BSPDN, no backside metal — 500nm Si, thin-film k, nothing else changed
  C: BSPDN, with backside metal — 500nm Si, thin-film k, Cu M1 + W nTSVs
  D: BSPDN + dual-sided cooling — C + a second Robin sink on top

A ratio, not an absolute-temperature, comparison — see the docstring in
gds/techmap.py's bspdn_stack for why that's the right level to validate at
without reproducing Oprins' full 3D unit-cell/normalization scheme exactly.

    python cases/run_bspdn_benchmark.py [--mw 1.0] [--top-h-eff 20000]
"""

import argparse
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gdstk

from gds import techmap
from gds.sources import CHANNEL_POWER_FRAC
from gds.techmap import DIFF, POLY
from mesh.gds_build import build_gds_2d_mesh
from post.metrics import tmax
from solve.steady import solve_steady
from spec.chip import BoundaryConditions

# Synthetic unit-cell die: big enough to see lateral spreading away from the
# source, small enough to mesh fast. Not the SRAM layout.
DIE_UM = 20.0
# Oprins' "small heat source": 400nm x 1000nm. w_um (first) is the axis our
# 2D cut resolves; l_um (second) only affects the trivial single-source
# power split (see gds/sources.py) and is otherwise replaced by
# source_depth_m below, per physics/coeffs.py's homogenization convention.
SOURCE_W_UM, SOURCE_L_UM = 0.4, 1.0
CUT_Y_UM = DIE_UM / 2
SOURCE_DEPTH_M = SOURCE_L_UM * 1e-6
# NOT this project's usual h_eff=20000 (a "well-cooled real chip" convention
# used elsewhere, e.g. cases/run_bspdn_study.py). Oprins' benchmark assumes
# "a high-performance heat sink" for their unit-cell model — verified
# empirically to mean *near-isothermal*: B/A converges to 2.36x (their
# published 2.2x) as h_eff -> large and has already saturated by 1e8 (1e10
# gives 2.378x, +1% more) — i.e. this is the Dirichlet/perfect-heat-sink
# limit, not a specific finite h_eff value chosen to fit their number.
BACKSIDE_H_EFF = 1.0e8


def synthetic_by_layer():
    """One DIFF rectangle == one POLY rectangle, centered on the die, so
    gds.sources.extract_channels' DIFF∩POLY returns exactly this footprint
    as the sole heat source — no LICON1 layer, so extract_contacts returns
    [] and 100% of the (compensated, see main()) power lands on the channel
    term. This is the "make our own rectangles" case: no real GDS involved.
    """
    cx, cy = DIE_UM / 2, DIE_UM / 2
    x0, x1 = cx - SOURCE_W_UM / 2, cx + SOURCE_W_UM / 2
    y0, y1 = cy - SOURCE_L_UM / 2, cy + SOURCE_L_UM / 2
    rect = gdstk.rectangle((x0, y0), (x1, y1))
    die = gdstk.rectangle((0.0, 0.0), (DIE_UM, DIE_UM))
    # A die-bounds-only background rectangle on an unused layer keeps
    # die_bounds() correct without it accidentally becoming a heat source.
    return {DIFF: [rect], POLY: [rect], (0, 0): [die]}


def run_config(label, stack, top_h_eff, power_w, out_dir):
    by_layer = synthetic_by_layer()
    # Compensate CHANNEL_POWER_FRAC so the full requested power lands on the
    # channel term (no LICON1 layer means the contact 30% would otherwise be
    # silently lost, not injected anywhere).
    mesh_data, registry = build_gds_2d_mesh(
        by_layer, CUT_Y_UM, power_w / CHANNEL_POWER_FRAC, out_dir=out_dir, stack=stack,
    )
    bcs = BoundaryConditions(ambient_t_k=300.0, backside_h_eff=BACKSIDE_H_EFF, top_h_eff=top_h_eff)
    chip = types.SimpleNamespace(bcs=bcs)
    T, k, q = solve_steady(mesh_data, registry, chip, source_depth_m=SOURCE_DEPTH_M)
    tmax_k, coords = tmax(T)
    print(f"  [{label}] Tmax={tmax_k:.6f}K dT={tmax_k-300.0:.6e}K "
          f"stack_top={techmap.total_thickness_um(stack):.3f}um")
    return tmax_k - 300.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mw", type=float, default=1.0)
    p.add_argument("--top-h-eff", type=float, default=BACKSIDE_H_EFF,
                    help="default matches the bottom sink's near-isothermal regime (see BACKSIDE_H_EFF)")
    args = p.parse_args()
    power_w = args.mw * 1e-3

    print(f"synthetic {DIE_UM}x{DIE_UM}um unit cell, {SOURCE_W_UM*1000:.0f}x{SOURCE_L_UM*1000:.0f}nm "
          f"heat source, {args.mw}mW, h_eff={BACKSIDE_H_EFF:.0f} W/m2K\n")

    print("A: FSPDN baseline (50um Si, bulk k)")
    dt_a = run_config("A", techmap.FRONTSIDE_STACK, None, power_w, "out/gds/bench_a")

    print("B: BSPDN, 500nm Si, no backside metal (Oprins' 2.2x case)")
    stack_b = techmap.bspdn_stack("bench_b", 0.5, "Si_thin_500nm", with_backside_metal=False)
    dt_b = run_config("B", stack_b, None, power_w, "out/gds/bench_b")

    print("C: BSPDN, 500nm Si, with backside metal (Oprins' 60% net penalty case)")
    stack_c = techmap.bspdn_stack("bench_c", 0.5, "Si_thin_500nm", with_backside_metal=True)
    dt_c = run_config("C", stack_c, None, power_w, "out/gds/bench_c")

    print("D: C + dual-sided cooling (ITherm's 22% recovery case)")
    dt_d = run_config("D", stack_c, args.top_h_eff, power_w, "out/gds/bench_d")

    print("\n=== ratios vs. published numbers ===")
    print(f"B/A, Si thinning alone (Oprins: 2.20x):              {dt_b/dt_a:.3f}x")
    # Oprins' own definition: "temperature reduction of 27%/30% for the
    # considered heat source area" — relative to B (the no-metal case this
    # reduction is measured against), not relative to the A-vs-B penalty.
    print(f"(B-C)/B, backside metal's own reduction (Oprins: 27-30%): "
          f"{100*(dt_b-dt_c)/dt_b:.1f}%")
    print(f"C/A, net BSPDN penalty (Oprins: 1.60x, i.e. +60%):   {dt_c/dt_a:.3f}x  (+{100*(dt_c/dt_a-1):.1f}%)")
    # ITherm's own definition: "decrease the logic maximum temperatures by
    # 22%" — relative to C (BSPDN alone), not relative to the A-vs-C penalty.
    print(f"(C-D)/C, dual-sided cooling's own reduction (ITherm: 22%): "
          f"{100*(dt_c-dt_d)/dt_c:.1f}%")

    print("\n=== thin-film-k ablation (Oprins: k reduction alone -> ~50% additional) ===")
    stack_c_bulk = techmap.bspdn_stack("bench_c_bulk", 0.5, "Si_bulk", with_backside_metal=True)
    dt_c_bulk = run_config("C (bulk Si k)", stack_c_bulk, None, power_w, "out/gds/bench_c_bulk")
    # Oprins: thin-film k contributes ~50% *on top of* the pure geometric/
    # spreading effect already captured by (C_bulk - A).
    geometric = dt_c_bulk - dt_a
    print(f"thin-film-k additional contribution, (C - C_bulk)/(C_bulk - A): "
          f"{100*(dt_c-dt_c_bulk)/geometric:.1f}% (Oprins: ~50%)" if geometric else "n/a")


if __name__ == "__main__":
    main()
