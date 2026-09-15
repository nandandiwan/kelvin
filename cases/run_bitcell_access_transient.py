"""What one access actually does, at nanosecond resolution.

Every steady-state number in this project uses DUTY-SCALED power: the peak
instantaneous per-device power times (access duration / clock period) times
the row-hit rate (gds.spice_power.named_bias_point_power_w). That is correct
for the steady field -- over 4.4 ms of thermal time constant, only the average
can matter -- but it is exactly the wrong quantity for asking "how hot does a
transistor get while it is switching".

This run applies the RAW, un-duty-scaled crowbar power as a step from ambient
and resolves the first few nanoseconds with a fixed small dt. Two reasons that
is the right experiment:

  - Local equilibration is fast. A channel of side a reaches its constriction-
    limited local temperature in ~a^2/alpha; for a = 0.15um in silicon that is
    0.25 ns. So the local peak is fully developed within one access.
  - The problem is linear, so this spike is additive on top of whatever
    pedestal the array average produces. Starting from a uniform 300 K
    therefore measures the spike itself, cleanly.

CROWBAR is used rather than READ because it is symmetric (X0==X2, X1==X7,
X5==X6 at that bias, verified directly), so the per-class power mapping
cases/run_bitcell_gallery.py uses is exact for it. READ is strongly
asymmetric and the class mapping would misstate it.

    python cases/run_bitcell_access_transient.py [--ns 4] [--dt-ps 50]
"""

import argparse
import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from cases.run_bitcell_gallery import build_bitcell_geometry, channel_sources_for
from gds import techmap
from gds.sources import _dims_um
from gds.spice_power import NAMED_BIAS_POINTS, bias_device_power_w
from mesh.gds_build import build_gds_3d_mesh
from mesh.gds_svg_viz import build_layer_regions, dof_indices_per_volume
from physics.coeffs import build_coeffs
from solve.transient import TransientHeatSolver
from spec.chip import BoundaryConditions, SourceBox

OUT = Path("out/bitcell_access")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ns", type=float, default=4.0, help="how long to simulate")
    p.add_argument("--dt-ps", type=float, default=50.0, help="fixed step, picoseconds")
    p.add_argument("--refine", type=float, default=1.0)
    p.add_argument("--far-field", action="store_true",
                   help="use the same asymptotic radiation lateral BC (h=k/r, "
                        "r = half the shorter window dimension) the array run "
                        "uses, instead of this script's original adiabatic "
                        "(zero-flux) lateral walls -- for an apples-to-apples "
                        "comparison against cases/run_array_full.py. Off by "
                        "default so the original fig6i result is unchanged.")
    p.add_argument("--save-layers", action="store_true",
                   help="also track the hottest dof inside every real GDS shape "
                        "(gate stripes, contacts, metal segments) at every step, "
                        "for cases/fig_bitcell_access_layers.py's per-device "
                        "animated GIF -- off by default since it isn't free "
                        "(one extra reduction per shape per step)")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # RAW power -- no duty scaling, no row-hit rate. This is the instantaneous
    # condition during the switching event itself.
    wl, bl, br, q, qb, _kind = NAMED_BIAS_POINTS["crowbar"]
    raw = bias_device_power_w(wl, bl, br, q, qb)
    class_w = {"access": raw["X0"], "latch": raw["X1"],
               "pullup": raw["X5"], "parasitic": raw["X3"]}
    print("raw per-device power (W):", {k: f"{v:.4e}" for k, v in class_w.items()})

    by_layer, window, channel_info, contacts = build_bitcell_geometry()
    contact_sources = [
        (SourceBox(device=f"co{i}", kind="contact", x_um=_dims_um(p_)[0],
                   y_um=_dims_um(p_)[1], z0_um=0.0, w_um=_dims_um(p_)[2],
                   l_um=_dims_um(p_)[3], t_um=techmap.LICON1_THICKNESS_UM,
                   power_uw=0.0), p_)
        for i, p_ in enumerate(contacts)]
    channel_sources = channel_sources_for(channel_info, class_w)
    total_w = sum(b.power_uw for b, _ in channel_sources) * 1e-6
    print(f"total instantaneous power into the cell: {total_w*1e6:.4f} uW")

    mesh_data, registry = build_gds_3d_mesh(
        by_layer, window, total_w, out_dir=str(OUT), refine=args.refine, renders=False,
        stack=techmap.FRONTSIDE_STACK, channel_sources=channel_sources,
        contact_sources=contact_sources)
    k, rho_cp, q_f = build_coeffs(mesh_data.mesh, mesh_data.cell_tags, registry,
                                  source_depth_m=None)
    r_m = None
    if args.far_field:
        x0, x1, y0, y1 = window
        r_m = (min(x1 - x0, y1 - y0) / 2) * 1e-6
        print(f"far-field lateral radiation BC: r = {r_m*1e6:.3f} um "
              f"(same convention as cases/run_array_full.py)")
    chip = types.SimpleNamespace(
        bcs=BoundaryConditions(ambient_t_k=300.0, backside_h_eff=20000.0, top_h_eff=None,
                               lateral_radiation_r_m=r_m))

    dt = args.dt_ps * 1e-12
    n_steps = int(round(args.ns * 1e-9 / dt))
    solver = TransientHeatSolver(mesh_data, k, rho_cp, chip, dt=dt, T0=300.0)
    q_arr = q_f.x.array.copy()

    V = solver.V
    dof = V.tabulate_dof_coordinates() * 1e6
    # Sample the channel plane: the z where the sources live.
    z_ch = techmap.z_bounds(techmap.FRONTSIDE_STACK)[2][0]
    plane = np.isclose(dof[:, 2], z_ch, atol=2e-3)
    if plane.sum() < 10:
        plane = np.isclose(dof[:, 2], dof[:, 2][np.argmin(np.abs(dof[:, 2] - z_ch))], atol=1e-7)
    print(f"channel plane z={z_ch:.4f}um: {plane.sum()} dofs")

    per_volume_idx = None
    layer_hist = None
    if args.save_layers:
        regions = build_layer_regions(by_layer, techmap.FRONTSIDE_STACK, window)
        per_volume_idx = dof_indices_per_volume(regions, dof)
        layer_hist = np.zeros((n_steps, len(per_volume_idx)), dtype=np.float32)
        print(f"tracking {len(per_volume_idx)} real GDS shapes across all layers")

    hist = []
    for i in range(n_steps):
        T = solver.step(q_arr, dt=dt)
        t_s = (i + 1) * dt
        dT = T.x.array - 300.0
        pl = dT[plane]
        hist.append({"t_ns": t_s * 1e9, "tmax_k": float(dT.max()),
                     "plane_max": float(pl.max()), "plane_min": float(pl.min()),
                     "spread_mK": float((pl.max() - pl.min()) * 1e3)})
        if layer_hist is not None:
            dT_mK = dT * 1e3
            for j, idx in enumerate(per_volume_idx):
                layer_hist[i, j] = dT_mK[idx].max()
        if (i + 1) % max(1, n_steps // 8) == 0:
            h = hist[-1]
            print(f"  t={h['t_ns']:6.3f} ns   dTmax={h['tmax_k']*1e3:9.3f} mK   "
                  f"device spread={h['spread_mK']:9.3f} mK", flush=True)

    tag = "farfield" if args.far_field else "adiabatic"
    out_name = "access_transient_farfield.json" if args.far_field else "access_transient.json"
    (OUT / out_name).write_text(json.dumps(hist, indent=2))
    if layer_hist is not None:
        np.save(OUT / f"layer_history_mK_{tag}.npy", layer_hist)
        (OUT / f"layer_meta_{tag}.json").write_text(json.dumps({
            "n_steps": n_steps, "n_volumes": len(per_volume_idx), "dt_ps": args.dt_ps,
            "far_field": args.far_field, "window_um": list(window)}, indent=2))
        print(f"wrote {OUT}/layer_history_mK_{tag}.npy {layer_hist.shape}, "
              f"{OUT}/layer_meta_{tag}.json")
    final = hist[-1]
    print(f"\nat t={final['t_ns']:.2f} ns: peak {final['tmax_k']*1e3:.2f} mK above ambient, "
          f"device-to-device spread {final['spread_mK']:.2f} mK")
    print(f"steady-state duty-averaged crowbar spread was 16.71 mK "
          f"-> ratio {final['spread_mK']/16.71:.1f}x")
    print(f"wrote {OUT}/{out_name}")


if __name__ == "__main__":
    main()
