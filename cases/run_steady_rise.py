"""Watch a cell (or an array) heat from 300 K all the way to steady state,
with the substrate it actually dumps heat into included in the tracking.

Three differences from the nanosecond pulsed runs
(run_bitcell_access_transient.py / run_array_full.py):

1. SUSTAINED power, not pulsed. Those runs resolve individual word-line
   pulses at fixed 25 ps steps, which is the right tool for "how hot does a
   transistor get during one switching event" but can never reach steady
   state: the thermal time constant here is tau = rho c t / h ~ 4.4 ms, so
   spanning it at 25 ps would take ~200 million steps. Holding the power on
   continuously and asking "where does it end up" is the complementary
   question, and the one that shows the full 300 K -> steady rise.

2. GEOMETRICALLY GROWING dt (the schedule cases/run_bitcell_transient.py
   already uses): dt0=1us growing 1.4x per step spans microseconds to tens
   of milliseconds in ~35 steps. Fixed dt cannot cover 8 decades of time.

3. SUBSTRATE IS TRACKED. The drawn GDS layers are only the top 3.2um of a
   53.4um model; the 50.2um of silicon underneath is where the heat actually
   goes, and is the only path out (the sole sink is the backside Robin BC at
   z=0; the top is adiabatic and the lateral faces only spread, they do not
   remove). Rendering without it is what made the earlier animations look
   like the heat was stuck in the bottom drawn layer.

Far-field lateral BC in both modes, so single-cell and array stay comparable.

    python cases/run_steady_rise.py --mode cell
    python cases/run_steady_rise.py --mode array [--rows 3] [--cols 5]
"""

import argparse
import dataclasses
import json
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from cases.run_bitcell_gallery import build_bitcell_geometry, channel_sources_for
from gds import techmap
from gds.sources import _dims_um
from gds.spice_power import NAMED_BIAS_POINTS, bias_device_power_w
from gds.tile_array import tile_array
from mesh.gds_build import build_gds_3d_mesh
from mesh.gds_svg_viz import build_layer_regions, dof_indices_per_volume
from physics.coeffs import build_coeffs
from solve.transient import TransientHeatSolver
from spec.chip import BoundaryConditions, SourceBox

DT0_S = 1e-9        # start well below the 0.25ns local equilibration time
DT_RATIO = 1.4
N_STEPS = 50        # 1e-9*(1.4^50-1)/0.4 ~ 1.7s of simulated time >> tau=4.4ms


def dt_schedule(n_steps, dt0=DT0_S, ratio=DT_RATIO):
    dt = dt0
    for _ in range(n_steps):
        yield dt
        dt *= ratio


def build_cell(device_power_w):
    by_layer, window, channel_info, contacts = build_bitcell_geometry()
    contact_sources = [
        (SourceBox(device=f"co{i}", kind="contact", x_um=_dims_um(p_)[0],
                   y_um=_dims_um(p_)[1], z0_um=0.0, w_um=_dims_um(p_)[2],
                   l_um=_dims_um(p_)[3], t_um=techmap.LICON1_THICKNESS_UM,
                   power_uw=0.0), p_)
        for i, p_ in enumerate(contacts)]
    channel_sources = channel_sources_for(channel_info, device_power_w)
    return by_layer, window, channel_sources, contact_sources


def build_array(device_power_w, rows, cols, active_row):
    by_layer, window, ch_src, co_src = tile_array(rows, cols)
    out = []
    for box, poly in ch_src:
        row = int(box.device.split("c")[0][1:])
        instance = box.device.rsplit("_", 1)[1]
        p_uw = (device_power_w[instance] * 1e6) if row == active_row else 0.0
        out.append((dataclasses.replace(box, power_uw=p_uw), poly))
    return by_layer, window, out, co_src


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["cell", "array"], default="cell")
    p.add_argument("--rows", type=int, default=3)
    p.add_argument("--cols", type=int, default=5)
    p.add_argument("--active-row", type=int, default=None)
    p.add_argument("--refine", type=float, default=None,
                   help="default: 1.0 for a cell, 1.6 for an array")
    p.add_argument("--steps", type=int, default=N_STEPS)
    p.add_argument("--freq-mhz", type=float, default=None,
                   help="clock frequency to duty-scale the power to. The cell only "
                        "draws crowbar current during the ~0.24ns access, so the "
                        "sustained average is raw x (access / period): 258.6 MHz "
                        "(this macro's real max) gives +31 K, 10 MHz gives +1.2 K. "
                        "Omit for RAW sustained power, which settles near +500 K -- "
                        "the 'crowbarring forever' bound, not an operating point.")
    p.add_argument("--tag", default=None, help="output dir suffix")
    args = p.parse_args()

    refine = args.refine if args.refine is not None else (1.0 if args.mode == "cell" else 1.6)
    tag = args.tag or (f"{args.freq_mhz:g}mhz" if args.freq_mhz else "raw")
    out = Path(f"out/steady_rise_{args.mode}_{tag}")
    out.mkdir(parents=True, exist_ok=True)

    wl, bl, br, q, qb, _ = NAMED_BIAS_POINTS["crowbar"]
    # Use each instance's own power rather than duplicating one device per class.
    device_power_w = bias_device_power_w(wl, bl, br, q, qb)

    # Duty-scale to a clock: the cell draws crowbar current only during the
    # real .lib-mined access window, so the sustained average power is
    # raw x (access / period). This is the same duty factor
    # gds.spice_power.named_bias_point_power_w applies, just parameterised by
    # frequency instead of pinned to this macro's own max clock.
    duty = 1.0
    if args.freq_mhz:
        from gds.power import mine_access_timing
        timing = mine_access_timing("data/sram22_2048x8m8w1_tt_025C_1v80.lib")
        period_ns = 1000.0 / args.freq_mhz
        duty = min(1.0, timing.min_pulse_width_high_ns / period_ns)
        device_power_w = {k: v * duty for k, v in device_power_w.items()}
        print(f"clock {args.freq_mhz:g} MHz -> period {period_ns:.3f} ns, "
              f"access {timing.min_pulse_width_high_ns:.3f} ns -> duty {duty:.5f}")

    active_row = args.active_row if args.active_row is not None else args.rows // 2
    if args.mode == "cell":
        by_layer, window, ch_src, co_src = build_cell(device_power_w)
        label = "single bitcell"
    else:
        by_layer, window, ch_src, co_src = build_array(device_power_w, args.rows, args.cols, active_row)
        label = f"{args.rows}x{args.cols} array, row {active_row} active"
    x0, x1, y0, y1 = window
    total_w = sum(b.power_uw for b, _ in ch_src) * 1e-6
    print(f"{label}: window {x1-x0:.2f} x {y1-y0:.2f} um, sustained power "
          f"{total_w*1e6:.3f} uW")

    t0 = time.time()
    mesh_data, registry = build_gds_3d_mesh(
        by_layer, window, max(total_w, 1e-18), out_dir=str(out), refine=refine,
        renders=False, stack=techmap.FRONTSIDE_STACK, channel_sources=ch_src,
        contact_sources=co_src)
    n_cells = mesh_data.mesh.topology.index_map(mesh_data.mesh.topology.dim).size_local
    print(f"mesh: {n_cells:,} cells ({time.time()-t0:.0f}s)")

    k, rho_cp, q_f = build_coeffs(mesh_data.mesh, mesh_data.cell_tags, registry,
                                  source_depth_m=None)
    r_m = (min(x1 - x0, y1 - y0) / 2) * 1e-6
    chip = types.SimpleNamespace(
        bcs=BoundaryConditions(ambient_t_k=300.0, backside_h_eff=20000.0, top_h_eff=None,
                               lateral_radiation_r_m=r_m))
    print(f"far-field lateral BC r={r_m*1e6:.3f} um; backside sink h=2e4 is the ONLY "
          f"heat-removal path (top adiabatic)")

    # Substrate included: it is 50.2um of the 53.4um model and the whole
    # downward path out. Without it the animation shows heat pooling in the
    # lowest DRAWN layer with nowhere to go.
    regions = build_layer_regions(by_layer, techmap.FRONTSIDE_STACK, window,
                                  include_substrate=True)
    n_vol = sum(len(r["volumes"]) for r in regions)

    solver = TransientHeatSolver(mesh_data, k, rho_cp, chip, dt=DT0_S, T0=300.0)
    dof = solver.V.tabulate_dof_coordinates() * 1e6
    per_volume_idx = dof_indices_per_volume(regions, dof)
    print(f"tracking {n_vol} shapes ({len(regions)} regions, substrate included)")

    q_arr = q_f.x.array.copy()
    layer_hist = np.zeros((args.steps, n_vol), dtype=np.float32)
    hist = []
    t_s = 0.0
    for i, dt in enumerate(dt_schedule(args.steps)):
        T = solver.step(q_arr, dt=dt)
        t_s += dt
        dT_mK = (T.x.array - 300.0) * 1e3
        for j, idx in enumerate(per_volume_idx):
            layer_hist[i, j] = dT_mK[idx].max()
        hist.append({"t_s": t_s, "dt_s": dt, "tmax_mK": float(dT_mK.max())})
        if (i + 1) % max(1, args.steps // 12) == 0:
            print(f"  step {i+1:3d}  t={t_s*1e3:11.5f} ms  dt={dt*1e9:10.1f} ns  "
                  f"Tmax=300+{dT_mK.max()/1e3:9.4f} K   [{time.time()-t0:.0f}s]", flush=True)

    np.save(out / "layer_history_mK.npy", layer_hist)
    meta = {"mode": args.mode, "label": label, "freq_mhz": args.freq_mhz, "duty": duty,
            "n_cells": n_cells, "n_volumes": n_vol,
            "n_regions": len(regions), "steps": args.steps, "dt0_s": DT0_S,
            "dt_ratio": DT_RATIO, "total_power_w": total_w, "r_m": r_m,
            "window_um": [x0, x1, y0, y1], "rows": args.rows, "cols": args.cols,
            "active_row": active_row, "times_s": [h["t_s"] for h in hist],
            "tmax_mK": [h["tmax_mK"] for h in hist]}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nfinal: Tmax = 300 + {hist[-1]['tmax_mK']/1e3:.4f} K at "
          f"t = {hist[-1]['t_s']*1e3:.2f} ms")
    print(f"wrote {out}/layer_history_mK.npy {layer_hist.shape}, {out}/meta.json")


if __name__ == "__main__":
    main()
