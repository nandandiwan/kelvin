"""Single real SRAM bitcell (sram_sp_cell, from data/sram22_64x22m4w22.gds),
full 3D thermal solve, driven by REAL per-device power from a SKY130 compact
model (BSIM3, via ngspice) instead of the flat/proportional-to-width split
gds.sources.build_heat_sources otherwise assumes.

No transient run needed (per PLAN.md's [[optimized-singing-creek]] Phase 1
notes on ngspice's unresolved transient convergence issue on this circuit):
gds.spice_power.named_bias_point_power_w("crowbar", ...) gets real per-device
power from a DC-only "mid-switching crowbar" snapshot -- Q=QB forced to
0.9V, the worst-case moment both pull-down NMOS are simultaneously partially
on during an actual bit-flip -- already scaled by real access duration AND
real row-selection frequency (gds.spice_power's row_hit_rate), not just the
peak instantaneous value.

Device -> real channel mapping (data/sram_sp_cell.spice's real W/L, cross-
checked against data/sram22_64x22m4w22.gds's real NWELL position placing
PMOS at x<-0.72um, NMOS at x>-0.72um within the cell's own local (-1.2..0,
-1.58..0)um bbox):
  - X0, X2 (nfet_pass,  W=0.14 L=0.150, x>-0.72): 2 access transistors
  - X1, X7 (nfet_latch, W=0.21 L=0.150, x>-0.72): 2 pull-down NMOS
  - X5, X6 (pfet_pass,  W=0.14 L=0.150, x<-0.72): 2 pull-up PMOS
  - X3, X4 (pfet_pass,  W=0.14 L=0.025, x<-0.72): 2 parasitic D=S devices
    from real LVS-style extraction (P=0 always -- Vds=0 by construction)
The crowbar bias is symmetric (Q bias == QB bias, BL == BR) so X0==X2,
X1==X7, X5==X6 exactly -- verified directly, not assumed -- so each pair
maps to the same power regardless of which specific instance is which.

    python cases/run_bitcell_compact.py
"""

import argparse
import sys
import types
from pathlib import Path

import gdstk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gds import techmap
from gds.read import flatten_by_layer
from gds.sources import _dims_um, extract_channels, extract_contacts
from gds.spice_power import named_bias_point_power_w
from mesh.gds_build import build_gds_3d_mesh
from mesh.viz3d import render_temperature_xy_slice, render_temperature_xz_slice
from post.budget import power_balance, print_power_balance
from post.metrics import tmax
from solve.steady import solve_steady
from spec.chip import BoundaryConditions, SourceBox

GDS_PATH = "data/sram22_64x22m4w22.gds"
CELL_NAME = "sram_sp_cell"
OUT_DIR = "out/bitcell_compact"

# (x_ext_um, y_ext_um, is_left) -> device power lookup key. y_ext is real L
# (gate length), x_ext is real W (device width) -- see gds/sources.py's
# _dims_um docstring for why the overlap-box axes map this way in this
# layout's orientation.
_NWELL_SPLIT_X_UM = -0.72  # PMOS (nwell) at x < this, NMOS at x > this, in local cell coords


def _classify(x_ext, y_ext, is_left):
    if abs(y_ext - 0.025) < 0.01:
        return "parasitic"  # X3/X4, P=0 always
    if abs(x_ext - 0.21) < 0.01:
        return "latch"      # X1/X7
    return "pullup" if is_left else "access"  # X5/X6 : X0/X2


def run(pad_um: float = 0.0, refine: float = 1.0, out_dir: str = OUT_DIR, renders: bool = True):
    """`pad_um`: widen the lateral window by this much on every side beyond
    the real cell footprint, filled with plain background material (no real
    neighboring-device geometry exists to put there -- see
    mesh/gds_volume.py::emit_gds_3d_geometry, which fills the whole window
    with `band.background` before punching in real polygons). pad_um=0.0
    (default) reproduces the as-built array-periodic case (adiabatic
    sidewalls sitting right at the cell edge, i.e. "every cell in the array
    is equally hot"). pad_um>0 gives the heat room to spread laterally
    before hitting the adiabatic cut, approximating an increasingly
    isolated single hot cell in quiet surrounding silicon -- see
    optimized-singing-creek.md's F2 finding for why this distinction
    matters (99.3% of the as-built thermal resistance is the Robin BC over
    the bare 1.9um^2 cell footprint, not conduction)."""
    lib = gdstk.read_gds(GDS_PATH)
    cell = next(c for c in lib.cells if c.name == CELL_NAME)
    by_layer = flatten_by_layer(cell)
    (x0, y0), (x1, y1) = cell.bounding_box()
    window = (x0 - pad_um, x1 + pad_um, y0 - pad_um, y1 + pad_um)
    print(f"bitcell window: {window}, size {x1-x0+2*pad_um:.3f} x {y1-y0+2*pad_um:.3f} um "
          f"(pad_um={pad_um}, refine={refine})")

    # The crowbar snapshot is a PEAK instantaneous condition (both pull-down
    # NMOS partially on simultaneously) -- real crowbar current only flows
    # while THIS row is selected (1-in-N_ROWS on average, see gds.spice_netlist.N_ROWS)
    # and only for the real access duration within that cycle, not
    # continuously, so feeding it into a STEADY-STATE thermal solve as-is
    # overstates the temperature by 1/duty (confirmed directly: gave
    # Tmax=807.8K, dT=507.8K, before any scaling was added). WORST CASE
    # here: row_hit_rate=1.0, i.e. this row hammered every cycle -- see
    # cases/run_bitcell_gallery.py for the average-case (1/N_ROWS) comparison.
    class_power_w_named = named_bias_point_power_w("crowbar", row_hit_rate=1.0)
    class_power_w = {
        "access": class_power_w_named["X0"],
        "latch": class_power_w_named["X1"],
        "pullup": class_power_w_named["X5"],
        "parasitic": class_power_w_named["X3"],
    }
    print("real per-class compact-model power (worst-case duty-cycle-scaled):",
          {k: f"{v*1e6:.4f} uW" for k, v in class_power_w.items()})

    channels = extract_channels(by_layer)
    channel_sources = []
    for i, poly in enumerate(channels):
        cx, cy, x_ext, y_ext = _dims_um(poly)
        cls = _classify(x_ext, y_ext, is_left=cx < _NWELL_SPLIT_X_UM)
        power_w = class_power_w[cls]
        box = SourceBox(
            device=f"{cls}{i}", kind="channel", x_um=cx, y_um=cy, z0_um=0.0,
            w_um=x_ext, l_um=y_ext, t_um=techmap.CHANNEL_THICKNESS_UM,
            power_uw=power_w * 1e6,
        )
        channel_sources.append((box, poly))
        print(f"  ch{i}: cls={cls:9s} cx={cx:.3f} cy={cy:.3f} -> {power_w*1e6:.4f} uW")

    # Contact (S/D resistance) heating has no direct compact-model analogue
    # without parasitic-resistance extraction (out of scope here) -- zero,
    # documented, not silently assumed. Channel power alone is the point of
    # this run: real per-device current from a real compact model.
    contacts = extract_contacts(by_layer)
    contact_sources = []
    for i, poly in enumerate(contacts):
        cx, cy, x_ext, y_ext = _dims_um(poly)
        box = SourceBox(
            device=f"co{i}", kind="contact", x_um=cx, y_um=cy, z0_um=0.0,
            w_um=x_ext, l_um=y_ext, t_um=techmap.LICON1_THICKNESS_UM, power_uw=0.0,
        )
        contact_sources.append((box, poly))

    total_power_w = sum(b.power_uw for b, _ in channel_sources) * 1e-6
    mesh_data, registry = build_gds_3d_mesh(
        by_layer, window, total_power_w, out_dir=out_dir, refine=refine, renders=renders,
        stack=techmap.FRONTSIDE_STACK, channel_sources=channel_sources, contact_sources=contact_sources,
    )

    bcs = BoundaryConditions(ambient_t_k=300.0, backside_h_eff=20000.0, top_h_eff=None)
    chip = types.SimpleNamespace(bcs=bcs)
    T, k, q = solve_steady(mesh_data, registry, chip, source_depth_m=None)

    balance = power_balance(mesh_data, registry, chip, T, k, q, source_depth_m=None)
    for tag, d in sorted(balance["per_tag"].items()):
        print(f"[budget]   tag {tag} ({d['label']}): meshed/nominal volume ratio = {d['ratio']:.6f}")
    print_power_balance(balance)

    tmax_k, coords = tmax(T)
    num_cells = mesh_data.mesh.topology.index_map(mesh_data.mesh.topology.dim).size_local
    print(f"Tmax = {tmax_k:.6f} K (dT = {tmax_k-300.0:.6e} K) "
          f"at x={coords[0]*1e6:.3f} y={coords[1]*1e6:.3f} z={coords[2]*1e6:.3f} um "
          f"[{num_cells} cells]")

    if renders:
        z_mid = coords[2] * 1e6
        y_mid = (y0 + y1) / 2
        render_temperature_xy_slice(T, z_mid, f"bitcell hotspot (compact-model power), z={z_mid:.2f}um",
                                     f"{out_dir}/T_xy_slice.png")
        render_temperature_xz_slice(T, y_mid, f"bitcell hotspot (compact-model power), y={y_mid:.2f}um",
                                     f"{out_dir}/T_xz_slice.png", zoom_z_um=(z_mid - 3.0, z_mid + 1.0))
        print(f"wrote {out_dir}/T_xy_slice.png, {out_dir}/T_xz_slice.png")

    return tmax_k, num_cells


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pad-um", type=float, default=0.0)
    p.add_argument("--refine", type=float, default=1.0)
    p.add_argument("--out-dir", default=OUT_DIR)
    args = p.parse_args()
    run(pad_um=args.pad_um, refine=args.refine, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
