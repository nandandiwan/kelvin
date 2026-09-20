"""P1.5 audit: dT vs backside_h_eff for the crowbar headline case. Builds the
mesh + k/q ONCE (h_eff doesn't affect either), then re-solves
(solve_steady_from_fields) at each h_eff -- cheap, since only the Robin
term changes. See optimized-singing-creek.md's F2: h_eff sets 99.3% of the
as-built (pad_um=0) thermal resistance, so the headline dT is a stated
boundary-condition assumption, not a prediction, and its sensitivity is the
honest thing to show alongside a single number.

    python cases/run_bitcell_heff_sensitivity.py
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

from cases.run_bitcell_compact import CELL_NAME, GDS_PATH
from gds import techmap
from gds.bitcell_mapping import channel_sources_for, map_bitcell_channels
from gds.read import flatten_by_layer
from gds.sources import _dims_um, extract_contacts
from gds.spice_power import named_bias_point_power_w
from mesh.gds_build import build_gds_3d_mesh
from physics.coeffs import build_coeffs
from post.budget import verify_device_source_powers
from post.metrics import tmax
from solve.steady import solve_steady_from_fields
from spec.chip import BoundaryConditions, SourceBox

OUT_DIR = "out/bitcell_heff_sensitivity"
H_EFF_SWEEP = [2000.0, 5000.0, 10000.0, 20000.0, 40000.0, 80000.0, 160000.0]


def main():
    lib = gdstk.read_gds(GDS_PATH)
    cell = next(c for c in lib.cells if c.name == CELL_NAME)
    by_layer = flatten_by_layer(cell)
    (x0, y0), (x1, y1) = cell.bounding_box()
    window = (x0, x1, y0, y1)

    device_power_w = named_bias_point_power_w("crowbar", row_hit_rate=1.0)
    channel_sources = channel_sources_for(map_bitcell_channels(by_layer), device_power_w)

    contacts = extract_contacts(by_layer)
    contact_sources = [
        (SourceBox(device=f"co{i}", kind="contact", x_um=_dims_um(p)[0], y_um=_dims_um(p)[1],
                    z0_um=0.0, w_um=_dims_um(p)[2], l_um=_dims_um(p)[3],
                    t_um=techmap.LICON1_THICKNESS_UM, power_uw=0.0), p)
        for i, p in enumerate(contacts)
    ]

    total_power_w = sum(b.power_uw for b, _ in channel_sources) * 1e-6
    mesh_data, registry = build_gds_3d_mesh(
        by_layer, window, total_power_w, out_dir=OUT_DIR, refine=1.0, renders=False,
        stack=techmap.FRONTSIDE_STACK, channel_sources=channel_sources, contact_sources=contact_sources,
    )
    k, rho_cp, q = build_coeffs(mesh_data.mesh, mesh_data.cell_tags, registry, source_depth_m=None)
    source_audit = verify_device_source_powers(mesh_data, registry, q, device_power_w)
    if mesh_data.mesh.comm.rank == 0:
        (Path(OUT_DIR) / "source_power_audit.json").write_text(json.dumps(source_audit, indent=2))

    dT_by_h = {}
    for h_eff in H_EFF_SWEEP:
        bcs = BoundaryConditions(ambient_t_k=300.0, backside_h_eff=h_eff, top_h_eff=None)
        chip = types.SimpleNamespace(bcs=bcs)
        T = solve_steady_from_fields(mesh_data, k, q, chip)
        tk, _ = tmax(T)
        dT_by_h[h_eff] = tk - 300.0
        print(f"h_eff={h_eff:9.1f} W/m2/K  ->  Tmax={tk:.4f} K  (dT={tk-300.0:.4f} K)")

    fig, ax = plt.subplots(figsize=(6, 5))
    hs = sorted(dT_by_h)
    ax.plot(hs, [dT_by_h[h] for h in hs], "o-", color="#B7410E")
    ax.axvline(20000.0, color="gray", ls="--", lw=1, label="stated value (20000 W/m2/K)")
    ax.set_xscale("log"); ax.set_xlabel("backside h_eff (W/m2/K)"); ax.set_ylabel("crowbar peak dT (K)")
    ax.set_title("Crowbar dT sensitivity to backside h_eff\n(stated boundary condition, not a prediction)")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/dT_vs_heff.png", dpi=140)
    print(f"wrote {OUT_DIR}/dT_vs_heff.png")


if __name__ == "__main__":
    main()
