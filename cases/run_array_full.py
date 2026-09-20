"""One array mesh build, reused for everything expensive: the 2D cross-section
figure and a per-SHAPE-resolved (not per-cell-block) transient, at finer dt and
more cycles than the first pass.

The first array GIF (fig7_array_evolution.gif) only tracked one max-temperature
number per CELL, so every frame was a single flat-colored 3x5 block grid --
correct, but it reads as "a flash in the middle" because it throws away all
sub-cell structure. This script tracks the same quantity `render_transient_layers.py`
already proved out for a single cell: the hottest dof inside EVERY drawn GDS
shape (every gate stripe, every contact, every metal segment) across the whole
tiled array, at every step. That is what cases/fig_array_transient_layers.py
needs to animate individual transistors lighting up and heat creeping into
neighbouring cells, in both the 2D plan and the 3D isometric.

Same mesh (rebuilt once here) is also used, before any solve, for the array's
2D cross-section (cases/fig_array_cross_section.py-equivalent, inlined below
since it needs the same mesh_data this script already has in hand).

    python cases/run_array_full.py [--rows 3] [--cols 5] [--cycles 5] [--dt-ps 25]
"""

import argparse
import json
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from gds import techmap
from gds.bitcell_mapping import validate_device_powers
from gds.power import mine_access_timing
from gds.spice_power import NAMED_BIAS_POINTS, bias_device_power_w
from gds.tile_array import CELL_H_UM, CELL_W_UM, tile_array
from mesh.gds_build import build_gds_3d_mesh
from mesh.gds_svg_viz import build_layer_regions, dof_indices_per_volume
from mesh.viz import MATERIAL_COLORS
from physics.coeffs import build_coeffs
from post.budget import verify_device_source_powers
from solve.transient import TransientHeatSolver
from spec.chip import BoundaryConditions

OUT = Path("out/sram_array_transient")
LIB = "data/sram22_2048x8m8w1_tt_025C_1v80.lib"
Z_PAD_UM = 0.05


def fig_cross_section(mesh_data, registry, window, active_row):
    import pyvista as pv
    from matplotlib.collections import PolyCollection

    pv.OFF_SCREEN = True
    mesh = mesh_data.mesh
    n_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    conn = mesh.geometry.dofmap.reshape(n_cells, 4)
    cells = np.hstack([np.full((n_cells, 1), 4, dtype=np.int64), conn]).ravel()
    grid = pv.UnstructuredGrid(cells, np.full(n_cells, 10, dtype=np.uint8), mesh.geometry.x * 1e6)

    mat_by_tag = {r.tag_id: r.material for r in registry.all()}
    palette = sorted(set(mat_by_tag.values()))
    idx_of = {m: i for i, m in enumerate(palette)}
    ct = mesh_data.cell_tags
    in_range = ct.indices < n_cells
    mat_idx = np.full(n_cells, -1, dtype=np.int32)
    mat_idx[ct.indices[in_range]] = [idx_of[mat_by_tag[v]] for v in ct.values[in_range]]
    grid.cell_data["material_idx"] = mat_idx
    grid = grid.threshold(-0.5, scalars="material_idx")

    x0, x1, y0, y1 = window
    y_cut = active_row * CELL_H_UM + CELL_H_UM / 2   # through the active row's devices
    sliced = grid.slice(normal="y", origin=(0, y_cut, 0))
    z_bot = techmap.z_bounds(techmap.FRONTSIDE_STACK)[1][0]
    z_feol = techmap.z_bounds(techmap.FRONTSIDE_STACK)[3][1]
    z_top = max(z1 for _, z1, _ in techmap.z_bounds(techmap.FRONTSIDE_STACK))

    tri = sliced.triangulate()
    faces = tri.faces.reshape(-1, 4)[:, 1:]
    pts = tri.points
    verts = pts[faces][:, :, [0, 2]]
    idxv = np.asarray(tri.cell_data["material_idx"], dtype=int)
    colors = [MATERIAL_COLORS.get(palette[i], "#999999") for i in idxv]

    fig, (ax, axz) = plt.subplots(1, 2, figsize=(17.0, 6.6),
                                  gridspec_kw={"width_ratios": [2.0, 1.0]})
    for a, (zlo, zhi), title in (
            (ax, (z_bot - 0.12, z_top + 0.03), "Full stack, cut through the active row"),
            (axz, (z_bot - 0.06, z_feol + 0.05), "FEOL zoom")):
        a.add_collection(PolyCollection(verts, facecolors=colors,
                                        edgecolors="#33333355", linewidths=0.2))
        a.set_xlim(x0, x1)
        a.set_ylim(zlo, zhi)
        a.set_xlabel("x (µm)")
        a.set_ylabel("z (µm)")
        a.set_title(title, fontsize=12)
        for c in range(1, 5):
            a.axvline(c * CELL_W_UM, color="#ffffff", lw=0.6, ls="--", alpha=0.5)

    used = sorted({mat_by_tag[v] for v in ct.values[in_range]})
    fig.legend(handles=[mpatches.Patch(facecolor=MATERIAL_COLORS.get(m, "#999999"),
                                       edgecolor="k", label=m) for m in used],
               fontsize=9.5, ncol=min(len(used), 9), loc="lower center",
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"3x5 real bitcell array — cross section at y = {y_cut:.2f} µm "
                 f"(row {active_row}, the active row)", fontsize=13)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(Path("out/presentation") / "fig7d_array_cross_section.png", dpi=160)
    plt.close(fig)
    print(f"wrote out/presentation/fig7d_array_cross_section.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=3)
    p.add_argument("--cols", type=int, default=5)
    p.add_argument("--active-row", type=int, default=None)
    p.add_argument("--cycles", type=float, default=5.0)
    p.add_argument("--dt-ps", type=float, default=25.0)
    p.add_argument("--refine", type=float, default=1.6)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    active_row = args.active_row if args.active_row is not None else args.rows // 2

    by_layer, window, ch_src, co_src = tile_array(args.rows, args.cols)
    x0, x1, y0, y1 = window
    print(f"{args.rows} x {args.cols} real bitcells, {x1-x0:.2f} x {y1-y0:.2f} um, "
          f"active row = {active_row}")

    wl, bl, br, q, qb, _ = NAMED_BIAS_POINTS["crowbar"]
    raw = validate_device_powers(bias_device_power_w(wl, bl, br, q, qb))
    # Independent electrical inventory: do not reconstruct expected powers
    # from the generated SourceBoxes being audited.
    electrical_power_w = {
        f"r{row}c{col}_{instance}": power if row == active_row else 0.0
        for row in range(args.rows) for col in range(args.cols)
        for instance, power in raw.items()
    }

    import dataclasses
    new_ch_src = []
    n_active = 0
    for box, poly in ch_src:
        cell_id, _ = box.device.rsplit("_", 1)
        row = int(cell_id.split("c", 1)[0][1:])
        p_uw = electrical_power_w[box.device] * 1e6
        new_ch_src.append((dataclasses.replace(box, power_uw=p_uw), poly))
        n_active += row == active_row
    ch_src = new_ch_src
    total_w = sum(b.power_uw for b, _ in ch_src) * 1e-6
    print(f"{len(ch_src)} channels, {n_active} in the active row, "
          f"{total_w*1e6:.3f} uW while the pulse is high")

    t0 = time.time()
    mesh_data, registry = build_gds_3d_mesh(
        by_layer, window, max(total_w, 1e-18), out_dir=str(OUT), refine=args.refine,
        renders=False, stack=techmap.FRONTSIDE_STACK, channel_sources=ch_src,
        contact_sources=co_src)
    n_cells = mesh_data.mesh.topology.index_map(mesh_data.mesh.topology.dim).size_local
    print(f"mesh: {n_cells:,} cells ({time.time()-t0:.0f}s)")

    fig_cross_section(mesh_data, registry, window, active_row)

    k, rho_cp, q_f = build_coeffs(mesh_data.mesh, mesh_data.cell_tags, registry,
                                  source_depth_m=None)
    source_audit = verify_device_source_powers(mesh_data, registry, q_f, electrical_power_w)
    if mesh_data.mesh.comm.rank == 0:
        (OUT / "source_power_audit.json").write_text(json.dumps(source_audit, indent=2))
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

    V = solver.V
    dof = V.tabulate_dof_coordinates() * 1e6
    z_ch = techmap.z_bounds(techmap.FRONTSIDE_STACK)[2][0]
    plane = np.isclose(dof[:, 2], z_ch, atol=2e-3)
    cell_probes = {}
    for row in range(args.rows):
        for col in range(args.cols):
            m = (plane & (dof[:, 0] >= col * CELL_W_UM) & (dof[:, 0] < (col + 1) * CELL_W_UM)
                 & (dof[:, 1] >= row * CELL_H_UM) & (dof[:, 1] < (row + 1) * CELL_H_UM))
            if m.sum():
                cell_probes[f"r{row}c{col}"] = np.flatnonzero(m)

    regions = build_layer_regions(by_layer, techmap.FRONTSIDE_STACK, window)
    per_volume_idx = dof_indices_per_volume(regions, dof)
    n_vol = len(per_volume_idx)
    print(f"{len(cell_probes)} cell probes, {n_vol} tracked shapes across all layers")

    layer_hist = np.zeros((n_steps, n_vol), dtype=np.float32)
    hist = []
    t_run0 = time.time()
    for i in range(n_steps):
        t_s = i * dt
        on = (t_s % period_s) < pulse_s
        T = solver.step(q_on if on else q_off, dt=dt)
        dT_mK = (T.x.array - 300.0) * 1e3
        for j, idx in enumerate(per_volume_idx):
            layer_hist[i, j] = dT_mK[idx].max()
        rec = {"t_ns": (t_s + dt) * 1e9, "wl_high": bool(on)}
        for name, idx in cell_probes.items():
            rec[name] = float(dT_mK[idx].max())
        hist.append(rec)
        if (i + 1) % max(1, n_steps // 15) == 0:
            act = [v for nm, v in rec.items() if nm.startswith(f"r{active_row}c")]
            idle = [v for nm, v in rec.items()
                    if nm.startswith("r") and not nm.startswith(f"r{active_row}c")]
            print(f"  t={rec['t_ns']:7.3f} ns  WL={'H' if on else 'L'}  "
                  f"active={max(act):8.3f} mK  idle={max(idle):8.3f} mK  "
                  f"[{time.time()-t_run0:.0f}s elapsed]", flush=True)

    np.save(OUT / "layer_history_mK.npy", layer_hist)
    meta = {"n_cells": n_cells, "n_rows": args.rows, "n_cols": args.cols,
            "active_row": active_row, "period_ns": timing.min_period_ns,
            "pulse_ns": timing.min_pulse_width_high_ns, "dt_ps": args.dt_ps,
            "n_steps": n_steps, "n_volumes": n_vol, "total_power_w_on": total_w,
            "r_m": r_m, "window_um": [x0, x1, y0, y1]}
    (OUT / "array_transient.json").write_text(json.dumps({"meta": meta, "hist": hist}, indent=2))
    (OUT / "layer_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {OUT}/layer_history_mK.npy ({layer_hist.shape}), "
          f"{OUT}/array_transient.json, {OUT}/layer_meta.json")


if __name__ == "__main__":
    main()
