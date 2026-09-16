"""Real-clock thermal run on a domain big enough that the far field is
genuinely far: real device geometry in the middle, a large plain-silicon
surround, and a substrate deep enough that the bottom sits essentially at
ambient.

Why the domain has to change. Every earlier run put the cell (or a 3x5 tile)
in a window barely wider than the geometry itself, on the stack's stock
50.2um substrate. That forces a choice between two wrong lateral BCs --
adiabatic (array-periodic: every neighbour equally hot) or the asymptotic
radiation BC (which, on a tall thin column, drains everything out the sides;
measured lateral/backside conductance ratio 1.9e6). With a wide enough
surround the question goes away: put the walls where the field has already
decayed to ambient and it no longer matters what you write there. That is
also the only way to SEE the temperature fall back to 300 K, which needs the
decay to happen inside the modelled domain rather than across a boundary
condition.

Clock: the real 258.6 MHz. At that rate the period is ~1e-6 of the thermal
time constant, so the silicon integrates over ~10^6 cycles and only the
duty-averaged power matters -- resolving individual pulses is both
impossible (10^6 steps to reach steady state) and pointless. The duty factor
is the real .lib access/period ratio.

Saves full field snapshots (float32) alongside the reduced histories, so the
far-field cross-sections and any later post-processing need no re-solve.

    python cases/run_deep.py --mode cell --mesh-only      # cost probe first
    python cases/run_deep.py --mode cell
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
from gds.tile_array import CELL_H_UM, CELL_W_UM, tile_array
from gds.techmap import GdsLayerBand, StackProfile, _DEVICE_AND_BEOL_BANDS
from mesh.gds_build import build_gds_3d_mesh
from mesh.gds_svg_viz import build_layer_regions, dof_indices_per_volume
from physics.coeffs import build_coeffs
from solve.transient import TransientHeatSolver
from spec.chip import BoundaryConditions, SourceBox

LIB = "data/sram22_2048x8m8w1_tt_025C_1v80.lib"
RHO_CP, K_SI = 2330.0 * 712.0, 148.0

# Backside sink. 1e8 (a die on a good heat spreader), NOT the 2e4 used in the
# small-window runs, and the choice is forced by what is being asked of the
# domain:
#
#   h=2e4 is so weak that the whole block floats up at P/(h*A_window) -- a
#   pedestal that scales as 1/A and therefore NEVER converges as the domain
#   grows (40um window: 37.0 mK, 60um: 16.4 mK, 100um: 5.9 mK). "Big enough
#   that more doesn't change the middle" is unachievable with it, and the
#   bottom face never approaches ambient either.
#
#   h=1e8 pins the bottom at ambient (7 uK for a 40um window), so the rise is
#   set by SPREADING from the source (P * 0.475/(k*a) ~ 2.7 mK), which
#   saturates once the window greatly exceeds the source -- i.e. it converges,
#   and the decay to 300 K happens inside the silicon where it can be seen.
#   It also drops the time constant from 8.3 ms to ~112 us of depth diffusion.
H_BACK = 1e8


def deep_stack(substrate_um: float, substrate_mesh_um: float = 5.0) -> StackProfile:
    """The frontside stack with a deeper substrate. The substrate band carries
    no drawn features, so mesh/gds_sizing.py sizes it from its own
    `mesh_size_um` across the whole window -- which is what keeps a large
    surround affordable (coarse silicon far from the devices)."""
    return StackProfile(f"deep{substrate_um:g}", (
        GdsLayerBand("Si_substrate", substrate_um, "Si_bulk", (), substrate_mesh_um),
    ) + _DEVICE_AND_BEOL_BANDS)


def centred_window(geom_window, span_um):
    x0, x1, y0, y1 = geom_window
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    h = span_um / 2
    return (cx - h, cx + h, cy - h, cy + h)


def dt_schedule(dt0, ratio, n):
    dt = dt0
    for _ in range(n):
        yield dt
        dt *= ratio


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["cell", "array"], default="cell")
    p.add_argument("--rows", type=int, default=6)
    p.add_argument("--cols", type=int, default=6)
    p.add_argument("--window-um", type=float, default=60.0)
    p.add_argument("--substrate-um", type=float, default=100.0)
    p.add_argument("--h-back", type=float, default=H_BACK,
                   help="backside sink coefficient; see H_BACK for why 1e8")
    p.add_argument("--substrate-mesh-um", type=float, default=5.0)
    p.add_argument("--refine", type=float, default=1.0)
    p.add_argument("--freq-mhz", type=float, default=None,
                   help="default: this macro's real max clock from the .lib")
    p.add_argument("--steps", type=int, default=46)
    p.add_argument("--dt0-s", type=float, default=1e-9)
    p.add_argument("--dt-ratio", type=float, default=1.55)
    p.add_argument("--snapshots", type=int, default=46,
                   help="full-field snapshots saved (float32) for post-processing")
    p.add_argument("--pattern", default="row-walk",
                   choices=["row-walk", "single-row", "all"],
                   help="array only: which cells are active. row-walk steps the "
                        "active word line through the rows as the run proceeds")
    p.add_argument("--mesh-only", action="store_true")
    p.add_argument("--tag", default=None)
    args = p.parse_args()

    tag = args.tag or args.mode
    out = Path(f"out/deep_{tag}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "fields").mkdir(exist_ok=True)

    timing = mine_access_timing(LIB)
    f_mhz = args.freq_mhz or (1000.0 / timing.min_period_ns)
    period_ns = 1000.0 / f_mhz
    duty = min(1.0, timing.min_pulse_width_high_ns / period_ns)

    wl, bl, br, q, qb, _ = NAMED_BIAS_POINTS["crowbar"]
    raw = bias_device_power_w(wl, bl, br, q, qb)
    class_w = {k: v * duty for k, v in
               {"access": raw["X0"], "latch": raw["X1"], "pullup": raw["X5"],
                "parasitic": raw["X3"]}.items()}
    print(f"clock {f_mhz:.1f} MHz (period {period_ns:.3f} ns), access "
          f"{timing.min_pulse_width_high_ns:.3f} ns -> duty {duty:.5f}")
    print("duty-averaged per-device power (W):",
          {k: f"{v:.3e}" for k, v in class_w.items()})

    if args.mode == "cell":
        by_layer, geom_window, channel_info, contacts = build_bitcell_geometry()
        co_src = [(SourceBox(device=f"co{i}", kind="contact", x_um=_dims_um(pp)[0],
                             y_um=_dims_um(pp)[1], z0_um=0.0, w_um=_dims_um(pp)[2],
                             l_um=_dims_um(pp)[3], t_um=techmap.LICON1_THICKNESS_UM,
                             power_uw=0.0), pp) for i, pp in enumerate(contacts)]
        ch_src = channel_sources_for(channel_info, class_w)
        label = "single bitcell"
        active_rows = None
    else:
        by_layer, geom_window, ch_all, co_src = tile_array(args.rows, args.cols)
        # Recentre the tiled array on the origin so the big window brackets it.
        active_rows = ({args.rows // 2} if args.pattern == "single-row"
                       else set(range(args.rows)) if args.pattern == "all"
                       else {args.rows // 2})   # row-walk starts mid, steps in time
        ch_src = []
        for box, poly in ch_all:
            row = int(box.device.split("c")[0][1:])
            cls = "".join(c for c in box.device.split("_")[1] if not c.isdigit())
            on = (row in active_rows) if args.pattern != "row-walk" else True
            ch_src.append((dataclasses.replace(
                box, power_uw=(class_w[cls] * 1e6) if on else 0.0), poly))
        label = f"{args.rows}x{args.cols} array ({args.pattern})"

    window = centred_window(geom_window, args.window_um)
    stack = deep_stack(args.substrate_um, args.substrate_mesh_um)
    z_sub_top = args.substrate_um
    z_top = max(z1 for _, z1, _ in techmap.z_bounds(stack))
    total_w = sum(b.power_uw for b, _ in ch_src) * 1e-6
    tau_s = RHO_CP * (args.substrate_um * 1e-6) / args.h_back

    print(f"{label}: geometry {geom_window[1]-geom_window[0]:.2f} x "
          f"{geom_window[3]-geom_window[2]:.2f} um inside a "
          f"{args.window_um:g} x {args.window_um:g} um window")
    print(f"substrate {args.substrate_um:g} um (stack top z={z_top:.2f} um), "
          f"tau ~ {tau_s*1e3:.1f} ms, sustained power {total_w*1e6:.4f} uW")

    t0 = time.time()
    mesh_data, registry = build_gds_3d_mesh(
        by_layer, window, max(total_w, 1e-18), out_dir=str(out), refine=args.refine,
        renders=False, stack=stack, channel_sources=ch_src, contact_sources=co_src)
    n_cells = mesh_data.mesh.topology.index_map(mesh_data.mesh.topology.dim).size_local
    n_dofs_est = mesh_data.mesh.geometry.x.shape[0]
    print(f"mesh: {n_cells:,} cells, {n_dofs_est:,} geometry nodes "
          f"({time.time()-t0:.0f}s)")
    if args.mesh_only:
        return

    k, rho_cp, q_f = build_coeffs(mesh_data.mesh, mesh_data.cell_tags, registry,
                                  source_depth_m=None)
    chip = types.SimpleNamespace(
        bcs=BoundaryConditions(ambient_t_k=300.0, backside_h_eff=args.h_back, top_h_eff=None,
                               lateral_radiation_r_m=None))
    print(f"BCs: backside Robin h={args.h_back:.0e} (only sink); lateral + top adiabatic, placed "
          "far enough out that the field is already ~ambient there")

    solver = TransientHeatSolver(mesh_data, k, rho_cp, chip, dt=args.dt0_s, T0=300.0)
    dof = solver.V.tabulate_dof_coordinates() * 1e6
    n_dof = dof.shape[0]
    np.save(out / "fields" / "dof_um.npy", dof.astype(np.float32))
    print(f"{n_dof:,} P1 dofs; snapshot set = "
          f"{n_dof*args.snapshots*4/1e9:.2f} GB at float32")

    regions = build_layer_regions(by_layer, stack, geom_window, include_substrate=True,
                                  substrate_depths_um=(0.5, 2.0, 10.0, 40.0,
                                                       args.substrate_um))
    per_volume_idx = dof_indices_per_volume(regions, dof)
    n_vol = sum(len(r["volumes"]) for r in regions)

    # Vertical probe through the geometry centre, and a lateral probe along
    # the surface -- the two cuts the far-field figures need.
    cx, cy = (geom_window[0] + geom_window[1]) / 2, (geom_window[2] + geom_window[3]) / 2
    z_probe = np.concatenate([np.linspace(0, z_sub_top, 60), np.linspace(z_sub_top, z_top, 10)[1:]])
    vidx = [int(np.argmin((dof[:, 0]-cx)**2 + (dof[:, 1]-cy)**2 + (dof[:, 2]-zq)**2))
            for zq in z_probe]
    r_probe = np.linspace(0, args.window_um / 2 * 0.98, 60)
    hidx = [int(np.argmin((dof[:, 0]-(cx+rq))**2 + (dof[:, 1]-cy)**2
                          + (dof[:, 2]-z_sub_top)**2)) for rq in r_probe]
    np.save(out / "z_probe_um.npy", dof[vidx, 2])
    np.save(out / "r_probe_um.npy", dof[hidx, 0] - cx)

    import ufl
    from dolfinx.fem import assemble_scalar, form
    from mesh.build import FACET_BOTTOM
    ds = ufl.Measure("ds", domain=mesh_data.mesh, subdomain_data=mesh_data.facet_tags)

    q_arr = q_f.x.array.copy()
    snap_at = set(np.linspace(0, args.steps - 1, min(args.snapshots, args.steps)).astype(int))
    layer_hist = np.zeros((args.steps, n_vol), dtype=np.float32)
    vprof = np.zeros((args.steps, len(vidx)), dtype=np.float32)
    hprof = np.zeros((args.steps, len(hidx)), dtype=np.float32)
    flux = np.zeros(args.steps)
    hist = []
    t_s = 0.0
    for i, dt in enumerate(dt_schedule(args.dt0_s, args.dt_ratio, args.steps)):
        T = solver.step(q_arr, dt=dt)
        t_s += dt
        dT = T.x.array - 300.0
        for j, idx in enumerate(per_volume_idx):
            layer_hist[i, j] = dT[idx].max() * 1e3
        vprof[i] = dT[vidx]
        hprof[i] = dT[hidx]
        flux[i] = assemble_scalar(form(args.h_back * (T - 300.0) * ds(FACET_BOTTOM)))
        hist.append({"t_s": t_s, "dt_s": dt, "dTmax_K": float(dT.max()),
                     "p_out_W": float(flux[i])})
        if i in snap_at:
            np.save(out / "fields" / f"dT_{i:03d}.npy", dT.astype(np.float32))
        if (i + 1) % max(1, args.steps // 12) == 0:
            print(f"  step {i+1:3d}  t={t_s*1e3:11.4f} ms  dTmax={dT.max():9.5f} K  "
                  f"out={flux[i]*1e6:8.4f} uW / in {total_w*1e6:.4f}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    np.save(out / "layer_history_mK.npy", layer_hist)
    np.save(out / "vprof_K.npy", vprof)
    np.save(out / "hprof_K.npy", hprof)
    np.save(out / "flux_bottom_W.npy", flux)
    meta = {"mode": args.mode, "label": label, "n_cells": n_cells, "n_dof": n_dof,
            "n_volumes": n_vol, "steps": args.steps, "dt0_s": args.dt0_s,
            "dt_ratio": args.dt_ratio, "freq_mhz": f_mhz, "duty": duty,
            "total_power_w": total_w, "window_um": list(window),
            "geom_window_um": list(geom_window), "substrate_um": args.substrate_um,
            "z_sub_top_um": z_sub_top, "z_top_um": z_top, "tau_s": tau_s,
            "rows": args.rows, "cols": args.cols, "pattern": args.pattern,
            "refine": args.refine, "snapshot_steps": sorted(int(s) for s in snap_at),
            "times_s": [h["t_s"] for h in hist], "dTmax_K": [h["dTmax_K"] for h in hist],
            "p_out_W": [h["p_out_W"] for h in hist]}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nfinal dTmax {hist[-1]['dTmax_K']:.5f} K at t={hist[-1]['t_s']*1e3:.1f} ms; "
          f"outflow {flux[-1]*1e6:.4f} uW vs {total_w*1e6:.4f} uW in "
          f"({flux[-1]/total_w*100:.1f}% -- 100% means steady)")
    print(f"wrote {out}/ (fields/, layer_history, vprof, hprof, flux, meta)")


if __name__ == "__main__":
    main()
