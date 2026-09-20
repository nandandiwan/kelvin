"""Multiple operating-point hotspot gallery: real per-device compact-model
power at several named bias points (idle/HOLD, WRITE settled, READ, the
worst-case mid-switching CROWBAR instant — see gds.spice_power's
NAMED_BIAS_POINTS), each mapped onto the real 3D bitcell geometry and solved
to steady state. Same channel/contact geometry and instance-to-channel
mapping as cases/run_bitcell_compact.py, with each of X0..X7 receiving its own
power even in asymmetric operating points; one mesh build per operating
point (k/q must stay registry-consistent with the mesh that produced them),
which is cheap at this scale (~15s each, already established).

Two access-pattern scenarios, both shown (see optimized-singing-creek.md
Part 1(a)): a cell is only active while ITS row is selected -- 1 of
gds.spice_netlist.N_ROWS rows in this macro -- and every non-HOLD point's
power already accounts for that (gds.spice_power.named_bias_point_power_w's
`row_hit_rate`), so this is not a duty-cycle knob layered on afterward here.
  - worst case:   row_hit_rate=1.0    -- same row hammered every cycle
  - average case: row_hit_rate=1/N_ROWS -- uniform random access across rows

    python cases/run_bitcell_gallery.py
"""

import json
import sys
import types
from pathlib import Path

import gdstk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gds import techmap
from gds.bitcell_mapping import channel_sources_for, map_bitcell_channels
from gds.read import flatten_by_layer
from gds.sources import _dims_um, extract_contacts
from gds.spice_netlist import N_ROWS
from gds.spice_power import NAMED_BIAS_POINTS, named_bias_point_power_w
from mesh.gds_build import build_gds_3d_mesh
from physics.coeffs import build_coeffs
from post.budget import power_balance, print_power_balance, verify_device_source_powers
from post.metrics import tmax
from solve.steady import solve_steady_from_fields
from spec.chip import BoundaryConditions, SourceBox

GDS_PATH = "data/sram22_64x22m4w22.gds"
CELL_NAME = "sram_sp_cell"
OUT_DIR = Path("out/bitcell_gallery")

SCENARIOS = [("worst_case", 1.0), ("average_case", 1.0 / N_ROWS)]


def build_bitcell_geometry():
    lib = gdstk.read_gds(GDS_PATH)
    cell = next(c for c in lib.cells if c.name == CELL_NAME)
    by_layer = flatten_by_layer(cell)
    (x0, y0), (x1, y1) = cell.bounding_box()
    channel_info = map_bitcell_channels(by_layer)
    contacts = extract_contacts(by_layer)
    return by_layer, (x0, x1, y0, y1), channel_info, contacts


def solve_operating_point(by_layer, window, channel_info, contact_sources, chip, name, row_hit_rate, out_tag):
    device_power_w = named_bias_point_power_w(name, row_hit_rate=row_hit_rate)
    print(f"{out_tag}: per-instance power (W) = {({k: f'{v:.3e}' for k, v in device_power_w.items()})}")

    my_channel_sources = channel_sources_for(channel_info, device_power_w)
    total_power_w = max(sum(b.power_uw for b, _ in my_channel_sources) * 1e-6, 1e-18)
    mesh_data, registry = build_gds_3d_mesh(
        by_layer, window, total_power_w, out_dir=f"{OUT_DIR}/{out_tag}", refine=1.0, renders=False,
        stack=techmap.FRONTSIDE_STACK, channel_sources=my_channel_sources, contact_sources=contact_sources,
    )
    k_f, rho_cp_f, q_f = build_coeffs(mesh_data.mesh, mesh_data.cell_tags, registry, source_depth_m=None)
    source_audit = verify_device_source_powers(mesh_data, registry, q_f, device_power_w)
    if mesh_data.mesh.comm.rank == 0:
        (OUT_DIR / out_tag / "source_power_audit.json").write_text(json.dumps(source_audit, indent=2))
    T = solve_steady_from_fields(mesh_data, k_f, q_f, chip)
    tk, coords = tmax(T)
    print(f"  Tmax = {tk:.6f} K (dT = {tk-300.0:.6e} K)")
    print_power_balance(power_balance(mesh_data, registry, chip, T, k_f, q_f, source_depth_m=None))
    return T, tk, coords


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_layer, window, channel_info, contacts = build_bitcell_geometry()

    contact_sources = [
        (SourceBox(device=f"co{i}", kind="contact", x_um=_dims_um(poly)[0], y_um=_dims_um(poly)[1],
                    z0_um=0.0, w_um=_dims_um(poly)[2], l_um=_dims_um(poly)[3],
                    t_um=techmap.LICON1_THICKNESS_UM, power_uw=0.0), poly)
        for i, poly in enumerate(contacts)
    ]

    bcs = BoundaryConditions(ambient_t_k=300.0, backside_h_eff=20000.0, top_h_eff=None)
    chip = types.SimpleNamespace(bcs=bcs)

    # Geometry (channel/contact positions, real W/L) is identical across
    # every point AND scenario -- only per-device POWER differs -- but k/q
    # must stay registry-consistent with the mesh that produced them, so
    # each (point, scenario) pair gets its own mesh build (cheap at this
    # scale, ~15s each, already established). HOLD is scenario-independent
    # (SUSTAINED_POINTS in gds/spice_power.py), so it's solved once and
    # reused for both rows of the gallery rather than solved twice.
    results = {}
    for scenario_name, row_hit_rate in SCENARIOS:
        for name in NAMED_BIAS_POINTS:
            if name == "hold_1" and "hold_1" in results:
                results[(scenario_name, name)] = results[("worst_case", "hold_1")]
                continue
            out_tag = f"{scenario_name}_{name}"
            results[(scenario_name, name)] = solve_operating_point(
                by_layer, window, channel_info, contact_sources, chip, name, row_hit_rate, out_tag)

    # Composite gallery figure: one row per scenario, one column per
    # operating point, matplotlib tripcolor on the P1 dof coordinates (same
    # self-consistent construction post/viz.py uses) at each point's peak z.
    import matplotlib.tri as mtri
    import numpy as np

    point_names = list(NAMED_BIAS_POINTS)
    fig, axes = plt.subplots(len(SCENARIOS), len(point_names),
                              figsize=(5.2 * len(point_names), 5.2 * len(SCENARIOS)))
    for row, (scenario_name, row_hit_rate) in enumerate(SCENARIOS):
        for col, name in enumerate(point_names):
            ax = axes[row, col]
            T, tk, coords = results[(scenario_name, name)]
            V = T.function_space
            dof_coords = V.tabulate_dof_coordinates()
            mask = np.isclose(dof_coords[:, 2], coords[2], atol=1e-13)
            x, y, t = dof_coords[mask, 0] * 1e6, dof_coords[mask, 1] * 1e6, T.x.array[mask]
            if len(x) < 3:
                continue
            triang = mtri.Triangulation(x, y)
            tpc = ax.tripcolor(triang, t - 300.0, shading="gouraud", cmap="inferno")
            ax.set_title(f"{name} [{scenario_name}]\nTmax={tk:.4f}K (dT={tk-300:.3e}K)", fontsize=10)
            ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
            ax.set_aspect("equal")
            fig.colorbar(tpc, ax=ax, shrink=0.8, label="dT (K)")
    fig.suptitle(f"Bitcell hotspot map: operating point x access pattern "
                 f"(real compact-model power, {N_ROWS} real rows)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gallery.png", dpi=140)
    plt.close(fig)
    print(f"wrote {OUT_DIR}/gallery.png")


if __name__ == "__main__":
    main()
