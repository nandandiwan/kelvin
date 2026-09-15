"""Reproduce [Oprins] iTherm 2022's published BS-PDN result with THAT PAPER'S
OWN inputs -- the point being to test the SOLVER (same inputs, same answer?)
rather than this project's independent material choices.

Published targets ([Oprins] Section IV.A, Fig. 12, and Conclusions):
    Si thinning 200um -> 500nm, NO backside metal ........... 2.2x
    + single backside metal layer, 16.7% uTSV density ....... -27% (small source)
    net BS-PDN penalty vs conventional FS-PDN ............... 1.6x (+60%)

Their setup, read from the paper rather than inferred (Section III, Table 1):
    reference           200um Si, full thickness, FS-PDN
    thinned             500nm Si
    backside metal M1   Cu, 250nm wide, 210nm high, 500nm pitch, CONTINUOUS LINES
    uTSV                W, 180 x 250 nm, one per M1 line (~16.7% density)
    heat source (small) 400 x 1000 nm, 1 mW
    effective k         Si 84, Cu 330 W/m-K (their own Monte Carlo BTE, Fig. 9);
                        SiO2 1.4 (matches this project's own entry exactly)
    package unit cell   200 x 200 x 300 um, high-performance heat sink
    detail model        5 x 5 um around the source (their Fig. 7b)

Two-scale approach, mirroring theirs: a large HOMOGENIZED outer domain
supplies the far field, and a 5x5um window resolves the discrete metal lines
and uTSVs -- see cases/run_bspdn_benchmark_3d.py for the machinery and its
window-independence check.

    python cases/run_oprins_reproduction.py [--outer-um 100] [--window-um 5]
"""

import argparse
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cases.run_bspdn_benchmark_3d as b3d
from cases.run_bspdn_benchmark_3d import run_config
from gds import techmap
from gds.techmap import GdsLayerBand, StackProfile, _DEVICE_AND_BEOL_BANDS

# [Oprins]' reference: full-thickness 200um Si, conventional frontside PDN --
# no carrier, no bonding oxide, no backside metal. Built directly rather than
# via bspdn_stack (which always prepends a carrier + bonding interface).
def reference_stack(si_um: float = 200.0) -> StackProfile:
    return StackProfile("oprins_ref", (
        GdsLayerBand("Si_substrate", si_um, "Si_bulk", (), 5.0),
    ) + _DEVICE_AND_BEOL_BANDS)


def thinned_stack(si_um: float, with_metal: bool, si_material: str,
                   total_um: float = 200.0, with_bpr: bool = True) -> StackProfile:
    """Thinned device wafer bonded to a carrier. [Oprins] keep the TOTAL
    package thickness the same as the reference (their Fig. 4), so the
    carrier takes up whatever the thinned Si gave away -- otherwise the
    comparison would conflate thinning with a shorter path to the sink."""
    carrier_um = max(total_um - si_um - 0.5, 1.0)
    return techmap.oprins_stack(
        f"oprins_{si_um:g}_{'m' if with_metal else 'nom'}", si_um, si_material,
        with_backside_metal=with_metal, carrier_um=carrier_um, with_bpr=with_bpr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mw", type=float, default=1.0)
    p.add_argument("--outer-um", type=float, default=100.0)
    p.add_argument("--window-um", type=float, default=5.0)
    p.add_argument("--tile-um", type=float, default=1.0)
    p.add_argument("--refine", type=float, default=1.0)
    p.add_argument("--si-nm", type=float, default=500.0)
    p.add_argument("--source-rot90", action="store_true",
                    help="rotate the rectangular heat source 90deg vs the metal lines")
    p.add_argument("--radiation", action="store_true",
                    help="asymptotic radiation BC (h=k/r) on lateral cuts instead of the two-stage submodel")
    p.add_argument("--bpr", dest="bpr", action="store_true",
                    help="add Ru buried power rails -- see thinned_stack (off by default, over-blocks the heat path)")
    args = p.parse_args()
    power_w = args.mw * 1e-3
    si_um = args.si_nm / 1000.0
    if args.source_rot90:
        b3d.SOURCE_WL_UM = [b3d.SOURCE_WL_UM[1], b3d.SOURCE_WL_UM[0]]
    tag = f"oprins_si{args.si_nm:g}_w{args.window_um:g}_o{args.outer_um:g}"
    kw = dict(window_um=args.window_um, refine=args.refine,
              submodel=not args.radiation, radiation=args.radiation,
              outer_um=args.outer_um, tile_um=args.tile_um)

    print(f"[Oprins] reproduction: {args.si_nm:g}nm Si vs 200um reference, {args.mw}mW, "
          f"400x1000nm source\n  outer={args.outer_um}um (homogenized, tile={args.tile_um}um), "
          f"window={args.window_um}um (discrete metal lines + uTSVs)\n")

    print("REF: 200um Si, conventional FS-PDN")
    dt_ref, _ = run_config("REF", reference_stack(200.0), None, power_w,
                            f"out/gds/{tag}_ref", **kw)

    print(f"THIN: {args.si_nm:g}nm Si, no backside metal")
    dt_nom, _ = run_config("THIN", thinned_stack(si_um, False, "Si_oprins_500nm", with_bpr=args.bpr), None, power_w,
                            f"out/gds/{tag}_nom", **kw)

    print(f"THIN+BSM: {args.si_nm:g}nm Si, Cu backside metal + uTSVs")
    dt_m, _ = run_config("THIN+BSM", thinned_stack(si_um, True, "Si_oprins_500nm", with_bpr=args.bpr), None, power_w,
                          f"out/gds/{tag}_metal", **kw)

    print("\n=== vs [Oprins] iTherm 2022 published ===")
    print(f"  thinning alone      {dt_nom/dt_ref:6.3f}x   published 2.2x")
    print(f"  backside metal      {100*(dt_nom-dt_m)/dt_nom:6.1f}%   published 27% (small source)")
    print(f"  net BS-PDN penalty  {dt_m/dt_ref:6.3f}x   published 1.6x (+60%)")


if __name__ == "__main__":
    main()
