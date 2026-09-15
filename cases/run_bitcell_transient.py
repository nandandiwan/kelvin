"""Transient thermal response of the real bitcell to a compact-model-driven
workload turning on, then off: sustained duty-cycle-averaged active power
(same magnitude as cases/run_bitcell_compact.py's steady case) until Tmax
approaches steady state, then leakage-only idle until it relaxes back near
ambient. Saves every step's full temperature field (not just Tmax) so
figures/animations can be re-rendered later without re-solving -- see
cases/render_transient_layers.py, which does exactly that for a
fixed-viewpoint isometric layer-stack GIF of the hot spot forming.

Why sustained power, not individual ~3.87ns electrical pulses: verified
directly (not assumed) that this is required, not a simplification of
convenience. A first attempt used a uniform ~5ns timestep sized off the
LOCAL device-layer thermal time constant (tau~L^2/alpha, L~1.5um bitcell
lateral scale) = ~25ns, and after 15ns of sustained active power the
computed rise was ~0.02K -- 1000x smaller than expected from that tau. The
reason: the mesh's actual heat sink is a Robin BC at the WAFER BACKSIDE, 50um
below the device layer, not at the bitcell's own lateral edges. The
GOVERNING time constant is set by that 50um path (tau ~ (50um)^2/alpha ~
28ms), not the 1.5um lateral one -- six orders of magnitude slower. At that
timescale, ~3.87ns electrical cycles are not "fast compared to thermal
response," they are INFINITELY fast -- millions of them happen within a
single relevant thermal timestep, so their time-averaged (duty-cycle-scaled)
power is not an approximation of the real forcing, it IS the real forcing
as far as this GLOBAL/BULK solve can ever resolve.

That said, the LOCAL channel-scale response is a different, much faster
timescale (tau ~ (100nm)^2/alpha ~ 0.1ns -- faster than even one real
access pulse), so a duty-cycle-averaged power UNDERSTATES the true local
peak during an actual access. `--instantaneous` runs one real (unscaled)
pulse at the point's raw peak power for the point's real access duration,
to measure that local overshoot directly instead of assuming the sustained
run's answer also describes the local instant -- see
optimized-singing-creek.md Part 1(e).

    python cases/run_bitcell_transient.py [--point crowbar] [--row-hit-rate 1.0]
    python cases/run_bitcell_transient.py --point read_1 --instantaneous
"""

import argparse
import json
import sys
import types
from pathlib import Path

import gdstk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gds import techmap
from gds.power import mine_access_timing
from gds.read import flatten_by_layer
from gds.sources import _dims_um, extract_channels, extract_contacts
from gds.spice_power import LIB_PATH, bias_device_power_w, named_bias_point_power_w, selfconsistent_device_power_w
from mesh.gds_build import build_gds_3d_mesh
from mesh.viz3d import render_temperature_xy_slice
from physics.coeffs import build_coeffs
from post.budget import power_balance, print_power_balance
from post.metrics import tmax
from solve.transient import TransientHeatSolver
from spec.chip import BoundaryConditions, SourceBox

GDS_PATH = "data/sram22_64x22m4w22.gds"
CELL_NAME = "sram_sp_cell"
_NWELL_SPLIT_X_UM = -0.72

# Governing time constant tau ~ (50um)^2 / alpha ~ 28ms -- the path DOWN TO
# THE BACKSIDE HEAT SINK, not the bitcell's own ~1.5um lateral scale (see
# module docstring for how this was found: a lateral-scale timestep gave a
# rise 1000x too small). Geometrically growing dt covers ns-to-~100ms in a
# few dozen steps per phase instead of millions of uniform ones.
DT0_S = 1e-6      # 1us initial step
DT_RATIO = 1.4    # growth factor per step
N_ACTIVE_STEPS = 30   # reaches ~1e-6*(1.4^30-1)/0.4 ~ 60ms of active time -> near steady state
N_IDLE_STEPS = 30     # same schedule again for the relaxation back toward ambient
N_RENDER_FRAMES_PER_PHASE = 20  # frames picked at uniform Tmax increments, not uniform step index


def _dt_schedule(n_steps, dt0=DT0_S, ratio=DT_RATIO):
    dt = dt0
    for _ in range(n_steps):
        yield dt
        dt *= ratio


def _classify(x_ext, y_ext, is_left):
    if abs(y_ext - 0.025) < 0.01:
        return "parasitic"
    if abs(x_ext - 0.21) < 0.01:
        return "latch"
    return "pullup" if is_left else "access"


def build_bitcell_mesh(class_power_w, out_dir):
    lib = gdstk.read_gds(GDS_PATH)
    cell = next(c for c in lib.cells if c.name == CELL_NAME)
    by_layer = flatten_by_layer(cell)
    (x0, y0), (x1, y1) = cell.bounding_box()
    window = (x0, x1, y0, y1)

    channels = extract_channels(by_layer)
    channel_sources = []
    for i, poly in enumerate(channels):
        cx, cy, x_ext, y_ext = _dims_um(poly)
        cls = _classify(x_ext, y_ext, is_left=cx < _NWELL_SPLIT_X_UM)
        box = SourceBox(device=f"{cls}{i}", kind="channel", x_um=cx, y_um=cy, z0_um=0.0,
                         w_um=x_ext, l_um=y_ext, t_um=techmap.CHANNEL_THICKNESS_UM,
                         power_uw=class_power_w[cls] * 1e6)
        channel_sources.append((box, poly))

    contacts = extract_contacts(by_layer)
    contact_sources = [
        (SourceBox(device=f"co{i}", kind="contact", x_um=_dims_um(poly)[0], y_um=_dims_um(poly)[1],
                    z0_um=0.0, w_um=_dims_um(poly)[2], l_um=_dims_um(poly)[3],
                    t_um=techmap.LICON1_THICKNESS_UM, power_uw=0.0), poly)
        for i, poly in enumerate(contacts)
    ]

    total_power_w = max(sum(b.power_uw for b, _ in channel_sources) * 1e-6, 1e-18)
    mesh_data, registry = build_gds_3d_mesh(
        by_layer, window, total_power_w, out_dir=str(out_dir), refine=1.0, renders=False,
        stack=techmap.FRONTSIDE_STACK, channel_sources=channel_sources, contact_sources=contact_sources,
    )
    return mesh_data, registry, window


def _class_power(device_power_w):
    return {"access": device_power_w["X0"], "latch": device_power_w["X1"],
            "pullup": device_power_w["X5"], "parasitic": device_power_w["X3"]}


def _select_uniform_dt_frames(tmax_hist, n_frames):
    """Indices into tmax_hist spaced at uniform TEMPERATURE increments, not
    uniform step index -- fixes the GIF looking like it "suddenly changes":
    under a geometrically growing dt, uniform step index is uniform in
    LOG-time, so most steps sit in the flat early window and all visible
    change lands in the last few frames of each phase."""
    t = np.asarray(tmax_hist)
    targets = np.linspace(t[0], t[-1], n_frames)
    idx = [int(np.argmin(np.abs(t - target))) for target in targets]
    # de-duplicate while preserving order
    seen, out = set(), []
    for i in idx:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--point", default="crowbar", choices=["hold_1", "write_1_settled", "read_1", "crowbar"])
    p.add_argument("--row-hit-rate", type=float, default=1.0, help="1.0=worst case, 1/N_ROWS=average case")
    p.add_argument("--instantaneous", action="store_true",
                   help="one real (unscaled) access pulse at raw peak power, instead of a sustained "
                        "duty-cycle-averaged workload -- measures the LOCAL overshoot a sustained run can't see")
    args = p.parse_args()

    out_dir = Path(f"out/bitcell_transient_{args.point}" + ("_instantaneous" if args.instantaneous else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    fields_dir = out_dir / "fields"
    fields_dir.mkdir(exist_ok=True)

    timing = mine_access_timing(LIB_PATH)

    if args.instantaneous:
        # RAW (unscaled) peak power for one real pulse -- the point's own
        # bias, not duty-cycle-averaged. Pulse duration: the real clk->Q
        # access time (see gds.power.AccessTiming's docstring for why that,
        # not the clock's own min_pulse_width, is the better access-duration
        # proxy), capped at one period since it can exceed it here.
        from gds.spice_power import NAMED_BIAS_POINTS
        wl, bl, br, q, qb, kind = NAMED_BIAS_POINTS[args.point]
        raw = bias_device_power_w(wl, bl, br, q, qb) if kind == "forced" \
            else selfconsistent_device_power_w(wl, bl, br, q, qb)
        class_power_active = _class_power(raw)
        pulse_s = min(timing.clk_to_q_access_ns, timing.min_period_ns) * 1e-9
        print(f"instantaneous mode: raw peak power, one {pulse_s*1e9:.4f}ns pulse")
    else:
        device_power_w = named_bias_point_power_w(args.point, row_hit_rate=args.row_hit_rate)
        class_power_active = _class_power(device_power_w)
        pulse_s = None
        print(f"sustained mode: row_hit_rate={args.row_hit_rate:.4f}")

    print(f"active per-class power (W): {({k: f'{v:.3e}' for k, v in class_power_active.items()})}")

    mesh_data, registry, window = build_bitcell_mesh(class_power_active, out_dir)
    k, rho_cp, q_active = build_coeffs(mesh_data.mesh, mesh_data.cell_tags, registry, source_depth_m=None)
    q_idle = q_active.x.array * 0.0  # leakage is ~1e6x smaller than active power here -- see
                                       # cases/run_bitcell_gallery.py's hold_1 numbers.

    bcs = BoundaryConditions(ambient_t_k=300.0, backside_h_eff=20000.0, top_h_eff=None)
    chip = types.SimpleNamespace(bcs=bcs)

    if args.instantaneous:
        # One real pulse at LOCAL-scale resolution (tau_local ~ 0.1ns, see
        # module docstring), then relax with the usual growing schedule.
        dt_pulse = pulse_s / 8
        solver = TransientHeatSolver(mesh_data, k, rho_cp, chip, dt=dt_pulse, T0=300.0)
        np.save(fields_dir / "dof_coords.npy", solver.V.tabulate_dof_coordinates())
        np.save(fields_dir / "connectivity.npy", solver.V.dofmap.list)
        times_s, tmax_hist = [0.0], [300.0]
        np.save(fields_dir / "T_000.npy", solver.T_prev.x.array.copy())
        t_s = 0.0
        step_i = 1
        n_pulse_steps = 8
        for _ in range(n_pulse_steps):
            T = solver.step(q_active.x.array, dt=dt_pulse)
            t_s += dt_pulse
            times_s.append(t_s); tmax_hist.append(tmax(T)[0])
            np.save(fields_dir / f"T_{step_i:03d}.npy", T.x.array.copy()); step_i += 1
        active_end_s = t_s
        for dt in _dt_schedule(N_IDLE_STEPS, dt0=dt_pulse * 2):
            T = solver.step(q_idle, dt=dt)
            t_s += dt
            times_s.append(t_s); tmax_hist.append(tmax(T)[0])
            np.save(fields_dir / f"T_{step_i:03d}.npy", T.x.array.copy()); step_i += 1
    else:
        solver = TransientHeatSolver(mesh_data, k, rho_cp, chip, dt=DT0_S, T0=300.0)
        np.save(fields_dir / "dof_coords.npy", solver.V.tabulate_dof_coordinates())
        np.save(fields_dir / "connectivity.npy", solver.V.dofmap.list)
        times_s, tmax_hist = [0.0], [300.0]
        np.save(fields_dir / "T_000.npy", solver.T_prev.x.array.copy())
        t_s = 0.0
        step_i = 1
        for dt in _dt_schedule(N_ACTIVE_STEPS):
            T = solver.step(q_active.x.array, dt=dt)
            t_s += dt
            times_s.append(t_s); tmax_hist.append(tmax(T)[0])
            np.save(fields_dir / f"T_{step_i:03d}.npy", T.x.array.copy()); step_i += 1
        active_end_s = t_s

        # P0.1 audit check (optimized-singing-creek.md): does q_active
        # deliver the intended per-device power on THIS mesh -- unverified
        # for the transient path until now.
        print_power_balance(power_balance(mesh_data, registry, chip, T, k, q_active, source_depth_m=None),
                             check_robin=False)  # not yet at steady state -- see post/budget.py's docstring
        for dt in _dt_schedule(N_IDLE_STEPS):
            T = solver.step(q_idle, dt=dt)
            t_s += dt
            times_s.append(t_s); tmax_hist.append(tmax(T)[0])
            np.save(fields_dir / f"T_{step_i:03d}.npy", T.x.array.copy()); step_i += 1

    times_s = np.array(times_s)
    tmax_hist = np.array(tmax_hist)
    np.save(fields_dir / "times_s.npy", times_s)
    np.save(fields_dir / "tmax_hist.npy", tmax_hist)
    meta = {"active_end_s": active_end_s, "n_steps": len(times_s), "point": args.point,
            "instantaneous": args.instantaneous}
    (fields_dir / "meta.json").write_text(json.dumps(meta))
    print(f"saved {len(times_s)} field snapshots to {fields_dir}")

    # Two-panel PHASE-RELATIVE plot: rise vs t, decay vs (t - active_end_s).
    # Fixes the "sudden jump" look -- a single absolute-log-time axis
    # compresses an entire ~60ms decay into a sliver next to a ~60ms rise
    # that ends at the SAME absolute time, even though both phases are
    # individually gradual.
    active_mask = times_s <= active_end_s
    idle_mask = ~active_mask
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    dT = tmax_hist - 300.0
    ax1.plot(np.maximum(times_s[active_mask], times_s[1] if len(times_s) > 1 else 1e-9) * 1e3,
             dT[active_mask], "o-", color="#B7410E", lw=2, ms=4)
    ax1.set_xscale("log"); ax1.set_xlabel("time since workload start (ms)"); ax1.set_ylabel("peak ΔT (K)")
    ax1.set_title("Active: rise toward steady state"); ax1.grid(alpha=0.3, which="both")

    idle_t_rel = times_s[idle_mask] - active_end_s
    idle_t_rel = np.maximum(idle_t_rel, idle_t_rel[idle_t_rel > 0].min() if (idle_t_rel > 0).any() else 1e-9)
    ax2.plot(idle_t_rel * 1e3, dT[idle_mask], "o-", color="#2E5A88", lw=2, ms=4)
    ax2.set_xscale("log"); ax2.set_xlabel("time since workload stop (ms)"); ax2.set_ylabel("peak ΔT (K)")
    ax2.set_title("Idle: relaxation toward ambient"); ax2.grid(alpha=0.3, which="both")
    fig.suptitle(f"Bitcell hotspot transient ({args.point}"
                 f"{', instantaneous pulse' if args.instantaneous else f', row_hit_rate={args.row_hit_rate:.3f}'})")
    fig.tight_layout()
    fig.savefig(out_dir / "tmax_vs_time.png", dpi=140)
    plt.close(fig)
    print(f"wrote {out_dir}/tmax_vs_time.png")

    # GIF: frames at uniform Tmax increments (not uniform step index) within
    # each phase, so the animation progresses smoothly instead of sitting
    # still then jumping.
    active_idx = [i for i in range(len(times_s)) if active_mask[i]]
    idle_idx = [i for i in range(len(times_s)) if idle_mask[i]]
    frame_idx = (
        [active_idx[i] for i in _select_uniform_dt_frames(tmax_hist[active_idx], N_RENDER_FRAMES_PER_PHASE)]
        + [idle_idx[i] for i in _select_uniform_dt_frames(tmax_hist[idle_idx], N_RENDER_FRAMES_PER_PHASE)]
    ) if len(idle_idx) > 1 else active_idx

    # Fixed slice location + fixed color scale, both computed ONCE from the
    # GLOBAL peak across the whole run, not per frame: verified directly
    # that the previous per-frame choices caused the "sudden change halfway
    # through" look reported -- (1) render_temperature_xy_slice's default
    # clim=None auto-scales each frame's colormap to THAT frame's own
    # min/max dT, so a barely-warmed idle frame gets stretched to the full
    # colormap just like the true peak frame, reading as an abrupt,
    # meaningless jump in color right at the active/idle switch even though
    # the physical temperature is changing smoothly; (2) z_um was
    # recomputed per frame from that frame's own argmax location, which can
    # shift between layers as the peak cools/relaxes, jumping the slice
    # plane itself out from under the animation.
    dof_coords_arr = np.load(fields_dir / "dof_coords.npy")
    global_peak_i = int(np.argmax(tmax_hist))
    T_peak_arr = np.load(fields_dir / f"T_{global_peak_i:03d}.npy")
    z_um_fixed = dof_coords_arr[int(np.argmax(T_peak_arr)), 2] * 1e6
    # NOTE: the plotted scalar is dT = T - 300 (see mesh/viz3d.py's
    # _temperature_grid), not absolute T -- clim must be in THAT frame
    # (0..peak dT), not (300..peak T), or every frame's data (0-31K) sits
    # far below a (300, 331)-style clim and clamps to solid black.
    clim = (0.0, float(tmax_hist.max() - 300.0))

    from dolfinx.fem import Function
    frame_paths = []
    for i in frame_idx:
        T_arr = np.load(fields_dir / f"T_{i:03d}.npy")
        tk = float(T_arr.max())
        # Reuse render_temperature_xy_slice by wrapping the raw array back
        # into a Function on the solver's own V (cheap, same mesh throughout).
        T_fn = Function(solver.V)
        T_fn.x.array[:] = T_arr
        label = "active" if i in active_idx else "idle"
        frame_path = out_dir / f"frame_{len(frame_paths):03d}.png"
        render_temperature_xy_slice(T_fn, z_um_fixed, f"t={times_s[i]*1e3:.4f}ms ({label}) Tmax={tk:.3f}K",
                                     str(frame_path), clim=clim)
        frame_paths.append(frame_path)

    frames = [Image.open(p_).convert("RGB") for p_ in frame_paths]
    gif_path = out_dir / "hotspot_evolution.gif"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=150, loop=0)
    print(f"wrote {gif_path} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
