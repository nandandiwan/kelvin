"""How much a bitcell heats depends on how much heat sink it is allowed to
use -- i.e. on whether its neighbours are hot too.

This historical padding study retains ADIABATIC side walls explicitly,
even though the compact case now defaults to periodic faces. At zero pad
all heat must leave through one cell's backside footprint. Padding the
window with background silicon allows spreading before heat reaches the
insulating outer wall and increases the available backside sink area.
Insulation is not generally equivalent to a periodic array; isolation
requires convergence with increasing padding.

Sweeping the pad traces the crossover between the two regimes, and separates
the two resistances that set the answer:

    R_sink   = 1 / (h * A)        falls as the padded area grows
    R_spread ~ 1 / (2 * k * L)    the 3D spreading resistance out of a
                                  source of size L, which does NOT fall

Small pad: R_sink dominates and dT ~ 1/A. Large pad: extra sink area is too
far away to reach and dT flattens onto the spreading limit. Where the knee
sits is the length scale over which a single cell's heat actually spreads.

    python cases/run_bitcell_isolation.py [--jobs 4] [--max-cores 16]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path("out/bitcell_isolation")
PADS_UM = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
CELL_X_UM, CELL_Y_UM = 1.2, 1.58
H_EFF = 20000.0


def _worker(pad_um, refine, out_json):
    """One pad value in its own process -- gmsh is a per-process singleton."""
    code = f"""
import sys, json; sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})
from cases.run_bitcell_compact import run
tmax, n = run(pad_um={pad_um!r}, refine={refine!r},
              out_dir="out/bitcell_isolation/pad_{pad_um:g}", renders=False,
              lateral_bc="insulating", power_model="dc-surrogate", point="crowbar")
json.dump({{"pad_um": {pad_um!r}, "tmax_k": tmax, "n_cells": n}}, open({out_json!r}, "w"))
"""
    return subprocess.Popen([sys.executable, "-c", code],
                            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                            cwd=str(Path(__file__).resolve().parents[1]), env={**os.environ})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--max-cores", type=int, default=16)
    p.add_argument("--refine", type=float, default=1.0)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    threads = max(1, args.max_cores // max(1, args.jobs))
    os.environ.update({v: str(threads) for v in
                       ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")})
    print(f"core budget: {args.jobs} x {threads} = {args.jobs*threads} of {args.max_cores}")

    pending, running, results = [], {}, {}
    for pad in PADS_UM:
        f = OUT / f"pad_{pad:g}.json"
        if args.resume and f.exists():
            results[pad] = json.loads(f.read_text())
        else:
            pending.append((pad, f))

    t0 = time.time()
    while pending or running:
        while pending and len(running) < args.jobs:
            pad, f = pending.pop(0)
            running[pad] = (_worker(pad, args.refine, str(f)), f)
        done = [k for k, (proc, _) in running.items() if proc.poll() is not None]
        if not done:
            time.sleep(2); continue
        for pad in done:
            proc, f = running.pop(pad)
            if proc.returncode != 0 or not f.exists():
                print(f"  FAILED pad={pad} rc={proc.returncode}"); continue
            results[pad] = json.loads(f.read_text())
            d = results[pad]
            print(f"  pad={pad:>5.2f}um  dT={d['tmax_k']-300.0:9.4f} K  "
                  f"[{d['n_cells']:,} cells, {time.time()-t0:.0f}s]", flush=True)

    print(f"\n{'pad(um)':>8} {'window(um)':>16} {'area(um^2)':>11} {'dT(K)':>9} "
          f"{'1/(hA) dT':>10} {'ratio':>7}")
    for pad in PADS_UM:
        if pad not in results:
            continue
        d = results[pad]
        wx, wy = CELL_X_UM + 2 * pad, CELL_Y_UM + 2 * pad
        area = wx * wy
        p_w = 1.1842e-6   # crowbar, worst case -- same power at every pad
        dt_sink = p_w / (H_EFF * area * 1e-12)
        dt = d["tmax_k"] - 300.0
        print(f"{pad:>8.2f} {wx:>7.2f} x{wy:>7.2f} {area:>11.2f} {dt:>9.4f} "
              f"{dt_sink:>10.4f} {dt/dt_sink:>7.3f}")


if __name__ == "__main__":
    main()
