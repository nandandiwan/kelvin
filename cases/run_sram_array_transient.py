"""Slide 7: what a read/write access actually does across an SRAM array, with
NO duty-cycle averaging -- the raw switching power, applied as a real pulsed
waveform, across several REAL abutted bitcells.

Three things this does differently from every other transient in the project:

1. REAL TILED CELLS, not one cell replicated blindly nor the GDS's own
   pre-built array cells. `gds/tile_array.py` tiles the single validated
   `sram_sp_cell` with the standard checkerboard mirror real SRAM arrays use
   at column/row boundaries. The GDS's own `sp_cell_array_center` (and
   siblings) were checked and REJECTED: that cell places `sram_sp_cell` and
   `sram_sp_cell_opt1a` at the IDENTICAL origin for every site -- two full
   transistor sets on top of each other, which is not real, buildable
   silicon. Cross-checked directly: the macro's actual top-level cell
   (`sram22_64x22m4w22`) has 1622 references, every one uniquely named, and
   references neither of those array cells at all -- they are unused
   generator-staging artifacts. See gds/tile_array.py's docstring.

2. RAW INSTANTANEOUS POWER, pulsed as a waveform. Everywhere else in this
   project, per-device power is multiplied by (access duration / clock
   period) x row-hit rate before it reaches the solver -- correct for the
   steady field, but it averages away exactly the variation asked for here.
   Here the word-line pulse is a real waveform: full raw power for the access
   window on the selected row, zero elsewhere and on every other row,
   repeated every clock period.

3. FAR-FIELD LATERAL BOUNDARIES (not adiabatic). Adiabatic side walls mean
   "this tile is the whole universe, and every tile around it does the same
   thing" -- array-periodic, the opposite of what's needed when the question
   is how an active row differs from its idle neighbours. The asymptotic
   radiation BC (physics/bcs.py::lateral_radiation_terms, h = k/r, already
   verified elsewhere in this project to reproduce a much larger domain to
   ~0.2%) lets heat leave sideways as it would into the rest of a real die.

Sizing: 3 rows x 5 columns (15 real cells). The middle row is driven, so it
has idle neighbours on both sides; 5 columns gives room to see lateral spread
along a row, not just across rows. Chosen by measuring, not guessing -- see
the mesh-only calibration this was run against before committing to the full
transient (out/sram_array_transient/calibration.md).

    python cases/run_sram_array_transient.py [--rows 3] [--cols 5] [--cycles 3]
"""

import argparse
import dataclasses
import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from gds import techmap
from gds.bitcell_mapping import validate_device_powers
from gds.power import mine_access_timing
from gds.spice_power import NAMED_BIAS_POINTS, bias_device_power_w
from gds.tile_array import CELL_H_UM, CELL_W_UM, tile_array
from mesh.gds_build import build_gds_3d_mesh
from physics.coeffs import build_coeffs
from post.budget import verify_device_source_powers
from solve.transient import TransientHeatSolver
from spec.chip import BoundaryConditions

OUT = Path("out/sram_array_transient")
LIB = "data/sram22_2048x8m8w1_tt_025C_1v80.lib"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=3)
    p.add_argument("--cols", type=int, default=5)
    p.add_argument("--active-row", type=int, default=None,
                   help="default: the middle row")
    p.add_argument("--cycles", type=float, default=3.0, help="clock cycles to simulate")
    p.add_argument("--dt-ps", type=float, default=50.0)
    p.add_argument("--refine", type=float, default=1.6)
    p.add_argument("--mesh-only", action="store_true")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    active_row = args.active_row if args.active_row is not None else args.rows // 2

    by_layer, window, ch_src, co_src = tile_array(args.rows, args.cols)
    x0, x1, y0, y1 = window
    print(f"{args.rows} x {args.cols} real bitcells, {x1-x0:.2f} x {y1-y0:.2f} um, "
          f"active row = {active_row}")

    wl, bl, br, q, qb, _ = NAMED_BIAS_POINTS["crowbar"]
    raw = validate_device_powers(bias_device_power_w(wl, bl, br, q, qb))
    print("raw per-device power (W):", {k: f"{v:.3e}" for k, v in raw.items()})
    # Build the electrical budget independently of the mapped source list, so
    # the mesh audit catches omitted/duplicated instances as well as bad powers.
    electrical_power_w = {
        f"r{row}c{col}_{instance}": power if row == active_row else 0.0
        for row in range(args.rows) for col in range(args.cols)
        for instance, power in raw.items()
    }

    # SourceBox is frozen -- rebuild each one with its real power rather than
    # mutating. Parse "r{row}c{col}_{instance}" back out and zero every device
    # outside the active row: HOLD power (~1e-13 W) is eight orders below
    # crowbar and only adds solver noise for no visible effect.
    n_active = 0
    new_ch_src = []
    for box, poly in ch_src:
        cell_id, _ = box.device.rsplit("_", 1)
        row = int(cell_id.split("c", 1)[0][1:])
        p_uw = electrical_power_w[box.device] * 1e6
        new_ch_src.append((dataclasses.replace(box, power_uw=p_uw), poly))
        n_active += row == active_row
    ch_src = new_ch_src
    total_w = sum(b.power_uw for b, _ in ch_src) * 1e-6
    print(f"{len(ch_src)} channels total, {n_active} in the active row, "
          f"{total_w*1e6:.3f} uW while the pulse is high")

    mesh_data, registry = build_gds_3d_mesh(
        by_layer, window, max(total_w, 1e-18), out_dir=str(OUT), refine=args.refine,
        renders=False, stack=techmap.FRONTSIDE_STACK, channel_sources=ch_src,
        contact_sources=co_src)
    n_cells = mesh_data.mesh.topology.index_map(mesh_data.mesh.topology.dim).size_local
    print(f"mesh: {n_cells:,} cells")
    if args.mesh_only:
        return

    k, rho_cp, q_f = build_coeffs(mesh_data.mesh, mesh_data.cell_tags, registry,
                                  source_depth_m=None)
    source_audit = verify_device_source_powers(mesh_data, registry, q_f, electrical_power_w)
    if mesh_data.mesh.comm.rank == 0:
        (OUT / "source_power_audit.json").write_text(json.dumps(source_audit, indent=2))
    # Far-field lateral faces: r = half the SHORTER window dimension, the same
    # convention cases/run_bspdn_benchmark_3d.py uses (verified there to
    # reproduce a 100um domain to ~0.2% from a few-um window).
    r_m = (min(x1 - x0, y1 - y0) / 2) * 1e-6
    chip = types.SimpleNamespace(
        bcs=BoundaryConditions(ambient_t_k=300.0, backside_h_eff=20000.0, top_h_eff=None,
                               lateral_radiation_r_m=r_m))
    print(f"far-field lateral radiation BC: r = {r_m*1e6:.3f} um")

    timing = mine_access_timing(LIB)
    period_s = timing.min_period_ns * 1e-9
    pulse_s = timing.min_pulse_width_high_ns * 1e-9
    dt = args.dt_ps * 1e-12
    n_steps = int(round(args.cycles * period_s / dt))
    print(f"clock {timing.min_period_ns:.3f} ns, access {timing.min_pulse_width_high_ns:.3f} ns, "
          f"dt {args.dt_ps:g} ps, {n_steps} steps for {args.cycles:g} cycles")

    solver = TransientHeatSolver(mesh_data, k, rho_cp, chip, dt=dt, T0=300.0)
    q_on = q_f.x.array.copy()
    q_off = np.zeros_like(q_on)

    # Per-cell probes on the channel plane: every dof inside cell (row, col)'s
    # own footprint, so "how hot is this cell" is a real spatial max.
    V = solver.V
    dof = V.tabulate_dof_coordinates() * 1e6
    z_ch = techmap.z_bounds(techmap.FRONTSIDE_STACK)[2][0]
    plane = np.isclose(dof[:, 2], z_ch, atol=2e-3)
    probes = {}
    for row in range(args.rows):
        for col in range(args.cols):
            m = (plane
                 & (dof[:, 0] >= col * CELL_W_UM) & (dof[:, 0] < (col + 1) * CELL_W_UM)
                 & (dof[:, 1] >= row * CELL_H_UM) & (dof[:, 1] < (row + 1) * CELL_H_UM))
            if m.sum():
                probes[f"r{row}c{col}"] = np.flatnonzero(m)
    print(f"{len(probes)} cell probes on the channel plane ({plane.sum()} dofs)")

    hist = []
    for i in range(n_steps):
        t_s = i * dt
        on = (t_s % period_s) < pulse_s
        T = solver.step(q_on if on else q_off, dt=dt)
        dT = T.x.array - 300.0
        rec = {"t_ns": (t_s + dt) * 1e9, "wl_high": bool(on), "tmax_mK": float(dT.max() * 1e3)}
        for name, idx in probes.items():
            rec[name] = float(dT[idx].max() * 1e3)
        hist.append(rec)
        if (i + 1) % max(1, n_steps // 12) == 0:
            act = [v for nm, v in rec.items() if nm.startswith(f"r{active_row}c")]
            idle = [v for nm, v in rec.items()
                    if nm.startswith("r") and not nm.startswith(f"r{active_row}c")]
            print(f"  t={rec['t_ns']:7.3f} ns  WL={'H' if on else 'L'}  "
                  f"active row max={max(act):8.3f} mK   idle max={max(idle):8.3f} mK",
                  flush=True)

    meta = {"n_cells": n_cells, "n_rows": args.rows, "n_cols": args.cols,
            "active_row": active_row, "period_ns": timing.min_period_ns,
            "pulse_ns": timing.min_pulse_width_high_ns, "dt_ps": args.dt_ps,
            "total_power_w_on": total_w, "r_m": r_m, "window_um": [x0, x1, y0, y1]}
    (OUT / "array_transient.json").write_text(json.dumps({"meta": meta, "hist": hist}, indent=2))
    np.save(OUT / "dof_um.npy", dof)
    np.save(OUT / "dT_final_mK.npy", (T.x.array - 300.0) * 1e3)
    print(f"\nwrote {OUT}/array_transient.json")


if __name__ == "__main__":
    main()
