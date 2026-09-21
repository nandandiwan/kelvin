"""Slow-clock thermal swing, with the substrate that actually carries the heat.

Three fixes over the earlier array/cell animations:

1. CORRECT BOUNDARY CONDITIONS. The earlier runs put the asymptotic radiation
   BC (h=k/r) on the lateral faces with r = half the window. That is right
   where it was validated (the BSPDN case: 200um-tall domain, h_backside=1e8,
   lateral/backside conductance ratio 263 -- a correction). It is badly wrong
   here: this domain is a TALL THIN COLUMN (1.2x1.58um footprint, 53.4um
   tall), so the lateral faces carry 156x the backside's area, and r=0.6um
   makes h_lat 12000x larger too -- lateral/backside conductance ratio
   1.9e6, i.e. the "spreading" BC became a Dirichlet clamp to ambient and
   drained essentially all the heat out the sides. Measured directly: a
   sustained-power run plateaued at 0.27 K instead of the analytic 504 K.
   Lateral faces are ADIABATIC here (array-periodic), which is what
   cases/run_bitcell_compact.py always used and which reproduces the
   analytic P/(hA) to 4 digits.

2. SUBSTRATE TRACKED AND DRAWN. The drawn GDS layers span 3.16um; the
   silicon beneath them is 50.21um -- 94% of the model and the only path to
   the sink. Slabbed over the FULL depth here (not just the top few um)
   because at millisecond timescales sqrt(alpha*t) is ~600um, so the whole
   substrate participates.

3. A CLOCK SLOW ENOUGH TO SEE. At the real 258.6 MHz the period is 1e-6 of
   the thermal time constant tau = rho*c*t/h = 4.165 ms, so the silicon
   integrates over ~10^6 cycles and the temperature is a flat line at the
   duty-averaged value. Stretching the period to ~tau while KEEPING the real
   duty (0.06194, the .lib access/period ratio) leaves the mean untouched at
   ~31 K but opens a ~29 K peak-to-peak swing (18.8 -> 47.9 K predicted),
   which is what makes the rise and fall visible at all.

    python cases/run_slow_clock.py --mode cell
    python cases/run_slow_clock.py --mode array
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
from gds.power import mine_access_timing
from gds.sources import _dims_um
from gds.spice_power import NAMED_BIAS_POINTS, bias_device_power_w
from gds.tile_array import tile_array
from mesh.gds_build import build_gds_3d_mesh
from mesh.gds_svg_viz import build_layer_regions, dof_indices_per_volume
from physics.coeffs import build_coeffs
from solve.transient import TransientHeatSolver
from spec.chip import BoundaryConditions, SourceBox

LIB = "data/sram22_2048x8m8w1_tt_025C_1v80.lib"
RHO_CP, H_BACK = 2330.0 * 712.0, 2e4
# Full substrate depth: at ms timescales the thermal front reaches all of it.
SUBSTRATE_DEPTHS_UM = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 35.0, 50.3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["cell", "array"], default="cell")
    p.add_argument("--rows", type=int, default=3)
    p.add_argument("--cols", type=int, default=5)
    p.add_argument("--active-row", type=int, default=None)
    p.add_argument("--refine", type=float, default=None)
    p.add_argument("--periods", type=float, default=2.0)
    p.add_argument("--period-over-tau", type=float, default=1.0,
                   help="clock period as a multiple of tau. 1.0 gives the biggest "
                        "readable swing; >>1 approaches full raw-power excursions")
    p.add_argument("--dt-us", type=float, default=12.0)
    args = p.parse_args()

    refine = args.refine if args.refine is not None else (1.0 if args.mode == "cell" else 1.6)
    active_row = args.active_row if args.active_row is not None else args.rows // 2
    out = Path(f"out/slow_clock_{args.mode}")
    out.mkdir(parents=True, exist_ok=True)

    wl, bl, br, q, qb, _ = NAMED_BIAS_POINTS["crowbar"]
    # Keep the SPICE instance identity through mirrored geometry and row selection.
    device_power_w = bias_device_power_w(wl, bl, br, q, qb)

    if args.mode == "cell":
        by_layer, window, channel_info, contacts = build_bitcell_geometry()
        co_src = [(SourceBox(device=f"co{i}", kind="contact", x_um=_dims_um(p_)[0],
                             y_um=_dims_um(p_)[1], z0_um=0.0, w_um=_dims_um(p_)[2],
                             l_um=_dims_um(p_)[3], t_um=techmap.LICON1_THICKNESS_UM,
                             power_uw=0.0), p_) for i, p_ in enumerate(contacts)]
        ch_src = channel_sources_for(channel_info, device_power_w)
        label = "single bitcell"
    else:
        by_layer, window, ch_all, co_src = tile_array(args.rows, args.cols)
        ch_src = []
        for box, poly in ch_all:
            row = int(box.device.split("c")[0][1:])
            instance = box.device.rsplit("_", 1)[1]
            p_uw = (device_power_w[instance] * 1e6) if row == active_row else 0.0
            ch_src.append((dataclasses.replace(box, power_uw=p_uw), poly))
        label = f"{args.rows}x{args.cols} array, row {active_row} active"

    x0, x1, y0, y1 = window
    total_w = sum(b.power_uw for b, _ in ch_src) * 1e-6

    # tau from the real substrate thickness in this stack
    z_sub_top = next(z1 for z0, z1, b in techmap.z_bounds(techmap.FRONTSIDE_STACK)
                     if b.name == "Si_substrate")
    tau_s = RHO_CP * (z_sub_top * 1e-6) / H_BACK
    timing = mine_access_timing(LIB)
    duty = timing.min_pulse_width_high_ns / timing.min_period_ns
    period_s = args.period_over_tau * tau_s
    pulse_s = duty * period_s
    dt = args.dt_us * 1e-6
    n_steps = int(round(args.periods * period_s / dt))

    print(f"{label}: window {x1-x0:.2f} x {y1-y0:.2f} um, ON-phase power {total_w*1e6:.3f} uW")
    print(f"tau = {tau_s*1e3:.3f} ms; clock period {period_s*1e3:.3f} ms "
          f"({args.period_over_tau:g} tau, {1/period_s:.1f} Hz), duty {duty:.5f} "
          f"-> ON {pulse_s*1e6:.1f} us")
    print(f"dt {args.dt_us:g} us, {n_steps} steps for {args.periods:g} periods")

    t0 = time.time()
    mesh_data, registry = build_gds_3d_mesh(
        by_layer, window, max(total_w, 1e-18), out_dir=str(out), refine=refine,
        renders=False, stack=techmap.FRONTSIDE_STACK, channel_sources=ch_src,
        contact_sources=co_src)
    n_cells = mesh_data.mesh.topology.index_map(mesh_data.mesh.topology.dim).size_local
    print(f"mesh: {n_cells:,} cells ({time.time()-t0:.0f}s)")

    k, rho_cp, q_f = build_coeffs(mesh_data.mesh, mesh_data.cell_tags, registry,
                                  source_depth_m=None)
    # ADIABATIC lateral walls -- see this module's docstring for why the
    # far-field BC used earlier was wrong for this aspect ratio.
    chip = types.SimpleNamespace(
        bcs=BoundaryConditions(ambient_t_k=300.0, backside_h_eff=H_BACK, top_h_eff=None,
                               lateral_radiation_r_m=None))
    print("BCs: backside Robin h=2e4 (only sink), lateral ADIABATIC, top adiabatic")

    regions = build_layer_regions(by_layer, techmap.FRONTSIDE_STACK, window,
                                  include_substrate=True,
                                  substrate_depths_um=SUBSTRATE_DEPTHS_UM)
    n_vol = sum(len(r["volumes"]) for r in regions)

    solver = TransientHeatSolver(mesh_data, k, rho_cp, chip, dt=dt, T0=300.0)
    dof = solver.V.tabulate_dof_coordinates() * 1e6
    per_volume_idx = dof_indices_per_volume(regions, dof)
    print(f"tracking {n_vol} shapes across {len(regions)} regions "
          f"({len(regions) - 8} substrate slabs to full depth)")

    # Vertical probe through the hottest device, plus the bottom-face heat
    # flux. The 3D colour map cannot answer "where does the heat leave"
    # because at Bi=0.0068 the block is isothermal -- everything is the same
    # colour BECAUSE the physics says so. The depth profile and the face flux
    # can answer it: the profile shows how little of the drop happens inside
    # the silicon, and the flux shows the backside carrying all of it.
    z_probe_um = np.concatenate([
        np.linspace(0.0, z_sub_top, 40),                 # through the substrate
        np.linspace(z_sub_top, dof[:, 2].max(), 12)[1:],  # through the device stack
    ])
    xc, yc = (x0 + x1) / 2, (y0 + y1) / 2
    probe_idx = []
    for zq in z_probe_um:
        d2 = (dof[:, 0] - xc) ** 2 + (dof[:, 1] - yc) ** 2 + (dof[:, 2] - zq) ** 2
        probe_idx.append(int(np.argmin(d2)))
    probe_idx = np.array(probe_idx)
    z_probe_actual = dof[probe_idx, 2]

    import ufl
    from dolfinx.fem import assemble_scalar, form
    from mesh.build import FACET_BOTTOM
    ds = ufl.Measure("ds", domain=mesh_data.mesh, subdomain_data=mesh_data.facet_tags)

    q_on = q_f.x.array.copy()
    q_off = np.zeros_like(q_on)
    layer_hist = np.zeros((n_steps, n_vol), dtype=np.float32)
    depth_hist = np.zeros((n_steps, len(probe_idx)), dtype=np.float32)
    flux_hist = np.zeros(n_steps, dtype=np.float64)
    hist = []
    for i in range(n_steps):
        t_s = i * dt
        on = (t_s % period_s) < pulse_s
        T = solver.step(q_on if on else q_off, dt=dt)
        dT_mK = (T.x.array - 300.0) * 1e3
        for j, idx in enumerate(per_volume_idx):
            layer_hist[i, j] = dT_mK[idx].max()
        depth_hist[i, :] = dT_mK[probe_idx]
        # Heat actually leaving through the backside Robin face this instant.
        flux_hist[i] = assemble_scalar(form(H_BACK * (T - 300.0) * ds(FACET_BOTTOM)))
        hist.append({"t_ms": (t_s + dt) * 1e3, "on": bool(on),
                     "tmax_K": float(dT_mK.max() / 1e3),
                     "p_out_bottom_uW": float(flux_hist[i] * 1e6)})
        if (i + 1) % max(1, n_steps // 14) == 0:
            print(f"  t={hist[-1]['t_ms']:8.3f} ms  {'ON ' if on else 'off'}  "
                  f"Tmax=300+{hist[-1]['tmax_K']:8.3f} K   [{time.time()-t0:.0f}s]", flush=True)

    np.save(out / "layer_history_mK.npy", layer_hist)
    np.save(out / "depth_history_mK.npy", depth_hist)
    np.save(out / "z_probe_um.npy", z_probe_actual)
    np.save(out / "flux_bottom_W.npy", flux_hist)
    tmax = [h["tmax_K"] for h in hist]
    meta = {"mode": args.mode, "label": label, "n_cells": n_cells, "n_volumes": n_vol,
            "n_regions": len(regions), "n_steps": n_steps, "dt_us": args.dt_us,
            "period_s": period_s, "pulse_s": pulse_s, "duty": duty, "tau_s": tau_s,
            "period_over_tau": args.period_over_tau, "total_power_w_on": total_w,
            "window_um": [x0, x1, y0, y1], "rows": args.rows, "cols": args.cols,
            "active_row": active_row, "lateral_bc": "adiabatic",
            "times_ms": [h["t_ms"] for h in hist], "on": [h["on"] for h in hist],
            "tmax_K": tmax}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nswing over the last period: {min(tmax[n_steps//2:]):.2f} -> "
          f"{max(tmax[n_steps//2:]):.2f} K above ambient")
    print(f"wrote {out}/layer_history_mK.npy {layer_hist.shape}, {out}/meta.json")


if __name__ == "__main__":
    main()
