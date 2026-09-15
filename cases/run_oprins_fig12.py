"""Reproduce [Oprins] iTherm 2022 Fig. 12 / Fig. 10: normalized temperature
increase vs Si substrate thickness, with and without the backside metal.

A curve across a decade+ of thickness is far stronger evidence than a single
ratio -- it tests the whole thickness dependence, not one operating point.
Published anchors on that curve (their Section IV.A and Conclusions):
    500nm Si, no backside metal ....... 2.2x
    500nm Si, with backside metal ..... ~1.6x
    thick Si (>=100um) ................ 1.0 by definition

Also decomposes the effect the way their Fig. 10 does, by running each
thickness twice:
  - size-dependent k  (this project's Liu & Asheghi MEASURED fit, or
    [Oprins]' own 84 W/m-K at their 500nm reference point)
  - bulk k held fixed, isolating the pure geometry/spreading effect
The gap between the two curves is "reduced Si conductivity impact"; the
bulk-k curve alone is "reduced Si thickness impact".

Uses the asymptotic radiation BC (physics/bcs.py::lateral_radiation_terms)
so each point is a small, cheap domain rather than a 200um one -- verified
to reproduce the big-domain answer to ~0.2%.

    python cases/run_oprins_fig12.py [--jobs 5] [--max-cores 16]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("out/oprins_fig12")
# [Oprins] Fig. 12 spans 100nm -> 1e6 nm; sample it geometrically.
SI_NM = [100, 200, 300, 500, 1000, 2500, 10000, 50000]
REF_NM = 200000.0   # their 200um full-thickness reference
PUBLISHED = {500: {"no_metal": 2.2, "metal": 1.6}}


def _cache_name(si_nm, metal, k_mode):
    """Filename for one point. "fit"/"bulk" keep the original 0/1 bulk_k
    naming so the 25 points already solved (hours of 3D solves) stay valid;
    "oprins" is a later addition and gets its own suffix."""
    if k_mode in ("fit", "bulk"):
        return f"pt_{si_nm:g}_{int(metal)}_{int(k_mode == 'bulk')}.json"
    return f"pt_{si_nm:g}_{int(metal)}_{k_mode}.json"


def _load_cached():
    results = {}
    for f in sorted(OUT.glob("pt_*.json")):
        d = json.loads(f.read_text())
        k_mode = d.get("k_mode") or ("bulk" if d.get("bulk_k") else "fit")
        results[(float(d["si_nm"]), bool(d["metal"]), k_mode)] = d["dt_k"]
    return results


def _worker(si_nm, metal, k_mode, window_um, out_json):
    """One (thickness, metal, k-model) point, in its own process: gmsh is a
    per-process singleton (see cases/run_bspdn_benchmark_3d.py)."""
    code = f"""
import sys, json; sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})
from cases.run_oprins_reproduction import reference_stack, thinned_stack
from cases.run_bspdn_benchmark_3d import run_config
si_nm, metal, k_mode = {si_nm!r}, {metal!r}, {k_mode!r}
si_um = si_nm / 1000.0
kw = dict(window_um={window_um!r}, refine=1.0, submodel=False, radiation=True)
from spec.materials import MATERIALS, Material, k_si_thin_film_w_mk
if si_nm >= {REF_NM!r}:
    stack = reference_stack(si_um)
else:
    if k_mode == "bulk":
        mat = "Si_bulk"          # size effect switched OFF: pure geometry
    elif k_mode == "oprins":
        # [Oprins]' OWN k at their own 500nm reference point (their Fig. 9
        # Monte-Carlo BTE, 84 W/m-K). Only meaningful AT 500nm -- this is the
        # same-inputs-same-answer solver check, not a k model of our own.
        mat = "Si_oprins_500nm"
    else:
        # size-dependent k at THIS thickness, from the Liu & Asheghi MEASURED
        # fit (spec/materials.py). Registered per point because the sweep
        # needs k(d), not one fixed 500nm value.
        mat = "Si_sweep_%g" % si_nm
        MATERIALS[mat] = Material(mat, k=k_si_thin_film_w_mk(si_um), rho=2330.0, cp=712.0)
    stack = thinned_stack(si_um, metal, mat)
dt, n = run_config("pt", stack, None, 1e-3, "out/gds/fig12_tmp_%s_%s_%s" % (si_nm, metal, k_mode),
                   **kw)
json.dump({{"si_nm": si_nm, "metal": metal, "k_mode": k_mode, "dt_k": dt, "n_cells": n}},
          open({out_json!r}, "w"))
"""
    return subprocess.Popen([sys.executable, "-c", code],
                             stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                             cwd=str(Path(__file__).resolve().parents[1]),
                             env={**os.environ})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=int, default=5)
    p.add_argument("--max-cores", type=int, default=16)
    p.add_argument("--window-um", type=float, default=5.0)
    p.add_argument("--resume", action="store_true",
                   help="skip points whose JSON is already in OUT")
    p.add_argument("--plot-only", action="store_true",
                   help="re-draw the figure from the per-point JSON already in OUT, "
                        "without re-solving anything (each point is a multi-million-cell "
                        "3D solve; the JSON is the expensive artifact, the PNG is not)")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        print(f"loaded {len(_load_cached())} cached points from {OUT}")
        return _plot(_load_cached())

    threads = max(1, args.max_cores // max(1, args.jobs))
    os.environ.update({v: str(threads) for v in
                       ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")})
    print(f"core budget: {args.jobs} workers x {threads} threads = {args.jobs*threads} "
          f"of {args.max_cores}")

    jobs = [(REF_NM, False, "fit")]
    for si in SI_NM:
        jobs += [(si, False, "fit"), (si, True, "fit"), (si, False, "bulk")]
    # [Oprins]' own 84 W/m-K, only at their own 500nm reference point.
    jobs += [(500, False, "oprins"), (500, True, "oprins")]

    results = _load_cached() if args.resume else {}
    running = {}
    pending = [j for j in jobs if (float(j[0]), j[1], j[2]) not in results]
    if args.resume:
        print(f"resuming: {len(results)} cached, {len(pending)} to run")
    t0 = time.time()
    while pending or running:
        while pending and len(running) < args.jobs:
            si, metal, k_mode = pending.pop(0)
            f = OUT / _cache_name(si, metal, k_mode)
            running[(float(si), metal, k_mode)] = (
                _worker(si, metal, k_mode, args.window_um, str(f)), f)
        done = [k for k, (proc, _) in running.items() if proc.poll() is not None]
        if not done:
            time.sleep(2); continue
        for key in done:
            proc, f = running.pop(key)
            if proc.returncode != 0 or not f.exists():
                print(f"  FAILED {key} (rc={proc.returncode})"); continue
            results[key] = json.loads(f.read_text())["dt_k"]
            print(f"  si={key[0]:>8.0f}nm metal={int(key[1])} k={key[2]:<6s}: "
                  f"dT={results[key]:.4f}K   [{time.time()-t0:.0f}s]", flush=True)

    return _plot(results)


def _plot(results):
    # Keys arrive as float si_nm from JSON but SI_NM is int -- normalize so
    # lookups can't silently miss (a missed key here reads as "that point
    # failed", which would be a quietly wrong figure, not an error).
    results = {(float(s), m, b): v for (s, m, b), v in results.items()}
    dt_ref = results.get((float(REF_NM), False, "fit"))
    if not dt_ref:
        raise SystemExit("reference point failed; cannot normalize")

    def curve(metal, k_mode):
        xs = [s for s in SI_NM if (float(s), metal, k_mode) in results]
        return xs, [results[(float(s), metal, k_mode)] / dt_ref for s in xs]

    # Two curves only, as [Oprins] Fig. 12 shows them. The bulk-k decomposition
    # curve and the k-model discussion belong in the talk, not on the axes.
    x_nom, y_nom = curve(False, "fit")
    x_met, y_met = curve(True, "fit")
    x_opr, y_opr = curve(False, "oprins")
    x_opm, y_opm = curve(True, "oprins")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_nom, y_nom, "o-", color="#d62728", lw=2.2, label="No backside metal")
    ax.plot(x_met, y_met, "o-", color="#1f77b4", lw=2.2, label="With backside metal + µTSVs")
    # Same geometry, same solver, [Oprins]' OWN k (84 W/m-K at 500nm) instead
    # of ours -- the apples-to-apples "same inputs -> same answer?" check.
    for xs, ys, col in ((x_opr, y_opr, "#d62728"), (x_opm, y_opm, "#1f77b4")):
        if xs:
            ax.plot(xs, ys, "D", ms=10, color=col, mfc="none", mew=2, zorder=5,
                    label="_nolegend_")
    ax.plot([], [], "D", ms=9, color="k", mfc="none", mew=2, label="same, with Oprins' own k")
    for si, pub in PUBLISHED.items():
        ax.plot([si], [pub["no_metal"]], "*", ms=19, color="#d62728", mec="k", zorder=6)
        ax.plot([si], [pub["metal"]], "*", ms=19, color="#1f77b4", mec="k", zorder=6)
    ax.plot([], [], "*", ms=15, color="w", mec="k", label="Oprins, published")
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("Si substrate thickness (nm)")
    ax.set_ylabel(f"Normalized temperature increase  (/ {REF_NM/1000:.0f} µm reference)")
    ax.set_title("Self-heating vs Si thickness, with and without backside metal",
                 fontsize=13)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig12_reproduction.png", dpi=150)
    print(f"\nwrote {OUT}/fig12_reproduction.png")

    print(f"\n{'Si (nm)':>9} {'no metal':>10} {'with metal':>11} {'bulk-k':>9}  published")
    for s in SI_NM:
        n = results.get((float(s), False, "fit")); m = results.get((float(s), True, "fit"))
        b = results.get((float(s), False, "bulk"))
        pub = PUBLISHED.get(s)
        pub_s = f"  {pub['no_metal']}x / {pub['metal']}x" if pub else ""
        print(f"{s:>9.0f} {n/dt_ref if n else float('nan'):>10.3f} "
              f"{m/dt_ref if m else float('nan'):>11.3f} "
              f"{b/dt_ref if b else float('nan'):>9.3f}{pub_s}")


if __name__ == "__main__":
    main()
