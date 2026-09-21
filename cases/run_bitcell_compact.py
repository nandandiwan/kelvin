"""Single real SRAM bitcell (sram_sp_cell, from data/sram22_64x22m4w22.gds),
full 3D thermal solve, driven by REAL per-device power from a SKY130 compact
model (BSIM3, via ngspice) instead of the flat/proportional-to-width split
gds.sources.build_heat_sources otherwise assumes.

By default, a validated transient-SPICE READ supplies the integrated
per-device event energies. The steady source is E_device * row_hit_rate /
period, exactly the same energy used by the pulsed thermal driver. Explicit
layout-derived resistor losses are included by default with the notebook mesh;
--interconnect none retains the former ideal-wire electrical model. Explicit
--power-model dc-surrogate retains the older DC/timing approximations,
including the artificial crowbar snapshot; those are not switching events.

gds.bitcell_mapping associates each channel polygon with its extracted
instance X0..X7. Every channel receives that instance's own compact-model
power; access/latch/pullup/parasitic classes are descriptive labels, not
power-sharing rules. The source mapping does not rely on paired devices
having equal power.

The lateral faces are periodic in x/y by default. This repeats the cell by
translation, with identical activity in every copy; it is not a claim that
the real mirrored SRAM macro has this exact one-cell repeat unit. Use
--lateral-bc insulating for the former zero-flux sidewalls.

    python cases/run_bitcell_compact.py
"""

import argparse
import json
import math
import sys
import types
from pathlib import Path

import gdstk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gds import techmap
from gds.bitcell_mapping import channel_sources_for, map_bitcell_channels
from gds.read import flatten_by_layer
from gds.sources import _dims_um, extract_contacts
from gds.spice_power import (
    ACCESS_ENERGY_MODEL_REVISION, LIB_PATH, NAMED_BIAS_POINTS,
    SPICE_MODEL_REVISION, named_bias_point_power_w,
)
from gds.thermal_workload import (
    SPICE_ACCESS_POINTS, THERMAL_POWER_REVISION, build_spice_workload,
)
from mesh.gds_build import build_gds_3d_mesh
from mesh.sram import build_sram_mesh
from mesh.viz3d import render_temperature_xy_slice, render_temperature_xz_slice
from post.budget import power_balance, print_power_balance, verify_device_source_powers
from post.metrics import tmax
from solve.steady import solve_steady, solve_steady_from_fields
from spec.chip import BoundaryConditions, SourceBox

GDS_PATH = "data/sram22_64x22m4w22.gds"
CELL_NAME = "sram_sp_cell"
OUT_DIR = "out/bitcell_compact"


def _check_output_directory(out_dir):
    """Preserve prior electrical/thermal results; geometry-only reuse is safe."""
    destination = Path(out_dir)
    artifacts = [destination / name for name in
                 ("simulation_summary.json", "power_workload.json")]
    spice_dir = destination / "spice"
    if any(path.exists() for path in artifacts) or (
            spice_dir.exists() and (not spice_dir.is_dir() or any(spice_dir.iterdir()))):
        raise ValueError(f"{destination} already contains simulation results; "
                         "choose a new --out-dir to preserve them")


def _build_power_workload(*, point, power_model, row_hit_rate, period_ns,
                          pulse_width_ns, spice_step_ps, out_dir, lib_path=LIB_PATH,
                          interconnect="none", interconnect_step_um=0.05):
    """Resolve the electrical workload without redistributing instance energy."""
    if power_model == "spice-transient":
        if point not in SPICE_ACCESS_POINTS:
            raise ValueError(f"{point!r} is not a transient-SPICE access; choose one of "
                             f"{', '.join(SPICE_ACCESS_POINTS)}, or explicitly use "
                             "--power-model dc-surrogate for a legacy bias point")
        network_options = ({"interconnect": interconnect, "interconnect_step_um": interconnect_step_um}
                           if interconnect != "none" or interconnect_step_um != 0.05 else {})
        workload = build_spice_workload(
            point=point, row_hit_rate=row_hit_rate, period_ns=period_ns,
            pulse_width_ns=pulse_width_ns, spice_max_step_ps=spice_step_ps,
            out_dir=Path(out_dir) / "spice", lib_path=lib_path, **network_options,
        )
        powers = dict(workload.device_average_power_w)
        metadata = dict(workload.metadata)
        metadata.update({
            "power_model": power_model, "point": point,
            "thermal_solver_connected": True, "application": "steady-average",
            "thermal_power_revision": THERMAL_POWER_REVISION,
            "spice_model_revision": SPICE_MODEL_REVISION,
            "period_s": workload.period_s, "row_hit_rate": workload.row_hit_rate,
            "device_energy_j": dict(workload.device_energy_j),
            "device_average_power_w": powers,
        })
        return powers, metadata
    if power_model != "dc-surrogate":
        raise ValueError("power_model must be 'spice-transient' or 'dc-surrogate'")
    if interconnect != "none" or interconnect_step_um != 0.05:
        raise ValueError("Layout interconnect heat is only supported by --power-model spice-transient")
    if point not in NAMED_BIAS_POINTS:
        raise ValueError(f"{point!r} is not a legacy DC bias point; "
                         f"choose one of {', '.join(NAMED_BIAS_POINTS)}")
    if period_ns is not None or pulse_width_ns is not None or spice_step_ps != 1.0:
        raise ValueError("--period-ns, --pulse-width-ns and --spice-step-ps "
                         "are only supported by --power-model spice-transient")
    powers = named_bias_point_power_w(point, row_hit_rate=row_hit_rate, lib_path=lib_path)
    return powers, {
        "power_model": power_model, "point": point, "row_hit_rate": row_hit_rate,
        "spice_model_revision": SPICE_MODEL_REVISION,
        "access_energy_model_revision": ACCESS_ENERGY_MODEL_REVISION,
        "device_average_power_w": powers,
        "description": "Legacy DC/timing surrogate, not a transient-SPICE access",
    }


def _audit_averaged_energy(source_audit, metadata):
    """Check the final meshed watts against independently integrated SPICE joules."""
    if metadata["power_model"] != "spice-transient":
        return None
    period_s = metadata["period_s"]
    rate = metadata["row_hit_rate"]
    records = {}
    for device, energy in metadata["device_energy_j"].items():
        expected = energy * rate / period_s
        meshed = source_audit["per_device"][device]["meshed_w"]
        if not math.isclose(meshed, expected, rel_tol=1e-6, abs_tol=1e-30):
            raise ValueError(f"Averaged access-energy audit failed for {device}: "
                             f"meshed {meshed:.9e} W != E*rate/period {expected:.9e} W")
        records[device] = {"event_energy_j": energy, "expected_average_w": expected,
                           "meshed_average_w": meshed}
    event_energy = sum(metadata["device_energy_j"].values()) + sum(metadata.get("resistor_energy_j", {}).values())
    if not math.isclose(source_audit["meshed_total_w"], event_energy * rate / period_s,
                        rel_tol=1e-6, abs_tol=1e-30):
        raise ValueError("Combined averaged source does not match transistor plus resistor event energies")
    return {"period_s": period_s, "row_hit_rate": rate, "per_device": records,
            "event_energy_j": event_energy,
            "interconnect_energy_j": sum(metadata.get("resistor_energy_j", {}).values()),
            "meshed_average_w": source_audit["meshed_total_w"]}


def run(pad_um: float = 0.0, refine: float = 1.0, out_dir: str = OUT_DIR,
        renders: bool = True, lateral_bc: str = "periodic", mesh_method: str = "notebook",
        mesh_manifest=None, point: str = "read_1", power_model: str = "spice-transient",
        row_hit_rate: float = 1.0, period_ns=None, pulse_width_ns=None,
        spice_step_ps: float = 1.0, interconnect=None, interconnect_step_um=0.05):
    """`pad_um`: widen the lateral window by this much on every side beyond
    the real cell footprint, filled with plain background material (no real
    neighboring-device geometry exists to put there -- see
    mesh/gds_volume.py::emit_gds_3d_geometry, which fills the whole window
    with `band.background` before punching in real polygons). pad_um=0.0
    (default) with lateral_bc="periodic" models an idealized translational
    array with one equally powered cell per period. pad_um>0 increases the
    repeat distance by adding background, not real neighboring cells.
    lateral_bc="insulating" instead prohibits all sidewall heat flux;
    this is not generally equivalent to periodicity. In either case the
    only net heat sink is the unchanged backside Robin boundary."""
    from mpi4py import MPI
    if MPI.COMM_WORLD.size != 1:
        raise ValueError("This SPICE/output driver currently requires a single MPI rank")
    if lateral_bc not in ("periodic", "insulating"):
        raise ValueError("lateral_bc must be 'periodic' or 'insulating'")
    if mesh_method not in ("notebook", "legacy"):
        raise ValueError("mesh_method must be 'notebook' or 'legacy'")
    if mesh_manifest is not None and mesh_method != "notebook":
        raise ValueError("--mesh-manifest requires --mesh-method notebook")
    interconnect = interconnect or ("layout" if power_model == "spice-transient" else "none")
    if interconnect not in {"layout", "none"}:
        raise ValueError("interconnect must be 'layout' or 'none'")
    if interconnect == "layout" and (mesh_method != "notebook" or power_model != "spice-transient"):
        raise ValueError("--interconnect layout requires the notebook mesh and transient SPICE")
    _check_output_directory(out_dir)
    device_power_w, workload_metadata = _build_power_workload(
        point=point, power_model=power_model, row_hit_rate=row_hit_rate,
        period_ns=period_ns, pulse_width_ns=pulse_width_ns,
        spice_step_ps=spice_step_ps, out_dir=out_dir,
        interconnect=interconnect, interconnect_step_um=interconnect_step_um,
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "power_workload.json").write_text(
        json.dumps(workload_metadata, indent=2) + "\n")
    lib = gdstk.read_gds(GDS_PATH)
    cell = next(c for c in lib.cells if c.name == CELL_NAME)
    by_layer = flatten_by_layer(cell)
    (x0, y0), (x1, y1) = cell.bounding_box()
    window = (x0 - pad_um, x1 + pad_um, y0 - pad_um, y1 + pad_um)
    print(f"bitcell window: {window}, size {x1-x0+2*pad_um:.3f} x {y1-y0+2*pad_um:.3f} um "
          f"(pad_um={pad_um}, refine={refine})")
    print(f"lateral boundary conditions: {lateral_bc} (x and y)")

    print(f"electrical power model: {power_model}, point={point}, row_hit_rate={row_hit_rate}")
    print("per-instance average power:",
          {k: f"{v*1e6:.4f} uW" for k, v in device_power_w.items()})

    mapped_channels = map_bitcell_channels(by_layer)
    channel_sources = channel_sources_for(mapped_channels, device_power_w)
    for mapped, (box, _) in zip(mapped_channels, channel_sources):
        print(f"  {mapped.instance}: cls={mapped.device_class:9s} "
              f"cx={box.x_um:.3f} cy={box.y_um:.3f} -> {box.power_uw:.4f} uW")

    # Zero-power geometry placeholders for the legacy mesher. The notebook
    # path deposits the SPICE resistor losses separately by exact spatial
    # projection; it must not assign all contacts one uniform layer power.
    contacts = extract_contacts(by_layer)
    contact_sources = []
    for i, poly in enumerate(contacts):
        cx, cy, x_ext, y_ext = _dims_um(poly)
        box = SourceBox(
            device=f"co{i}", kind="contact", x_um=cx, y_um=cy, z0_um=0.0,
            w_um=x_ext, l_um=y_ext, t_um=techmap.LICON1_THICKNESS_UM, power_uw=0.0,
        )
        contact_sources.append((box, poly))

    bcs = BoundaryConditions(
        ambient_t_k=300.0, backside_h_eff=20000.0, top_h_eff=None,
        periodic_x=lateral_bc == "periodic", periodic_y=lateral_bc == "periodic",
    )
    chip = types.SimpleNamespace(bcs=bcs)
    print(f"mesh method: {mesh_method}")
    if mesh_method == "notebook":
        case = build_sram_mesh(device_power_w, out_dir, pad_um=pad_um, refine=refine,
                               renders=renders, bcs=bcs, manifest_path=mesh_manifest)
        mesh_data, registry = case.mesh_data, case.registry
    else:
        total_power_w = sum(b.power_uw for b, _ in channel_sources) * 1e-6
        mesh_data, registry = build_gds_3d_mesh(
            by_layer, window, total_power_w, out_dir=out_dir, refine=refine, renders=renders,
            stack=techmap.FRONTSIDE_STACK, channel_sources=channel_sources, contact_sources=contact_sources,
        )
    network = workload_metadata.get("interconnect_network")
    if network:
        from physics.coeffs import build_coeffs
        from physics.interconnect_sources import CombinedHeatSources

        k, _, q_channels = build_coeffs(mesh_data.mesh, mesh_data.cell_tags, registry)
        verify_device_source_powers(mesh_data, registry, q_channels, device_power_w)
        print("Mapping layout resistor losses into intersecting thermal tetrahedra", flush=True)
        sources = CombinedHeatSources(mesh_data, registry, case.manifest, network)
        wire_powers = workload_metadata["interconnect_average_power_w"]
        q = sources.set_powers(device_power_w, wire_powers)
        source_audit = sources.audit_powers(q, device_power_w, wire_powers)
        (Path(out_dir) / "interconnect_projection.json").write_text(
            json.dumps(sources.projection_report, indent=2) + "\n")
        T = solve_steady_from_fields(mesh_data, k, q, chip)
    else:
        T, k, q = solve_steady(mesh_data, registry, chip, source_depth_m=None)
        source_audit = verify_device_source_powers(mesh_data, registry, q, device_power_w)
    energy_audit = _audit_averaged_energy(source_audit, workload_metadata)
    if mesh_data.mesh.comm.rank == 0:
        (Path(out_dir) / "source_power_audit.json").write_text(json.dumps(source_audit, indent=2))
        if energy_audit is not None:
            (Path(out_dir) / "source_energy_audit.json").write_text(
                json.dumps(energy_audit, indent=2) + "\n")

    balance_options = ({"intended_total_w": workload_metadata["average_total_local_power_w"]}
                       if network else {})
    balance = power_balance(mesh_data, registry, chip, T, k, q, source_depth_m=None, **balance_options)
    for tag, d in sorted(balance["per_tag"].items()):
        print(f"[budget]   tag {tag} ({d['label']}): meshed/nominal volume ratio = {d['ratio']:.6f}")
    print_power_balance(balance)

    tmax_k, coords = tmax(T)
    num_cells = mesh_data.mesh.topology.index_map(mesh_data.mesh.topology.dim).size_local
    print(f"Tmax = {tmax_k:.6f} K (dT = {tmax_k-300.0:.6e} K) "
          f"at x={coords[0]*1e6:.3f} y={coords[1]*1e6:.3f} z={coords[2]*1e6:.3f} um "
          f"[{num_cells} cells]")
    (Path(out_dir) / "simulation_summary.json").write_text(json.dumps({
        "mesh_method": mesh_method, "point": point, "lateral_bc": lateral_bc,
        "power_model": power_model, "power_workload": workload_metadata,
        "spice_model_revision": SPICE_MODEL_REVISION,
        "tmax_k": tmax_k, "cell_count": num_cells,
        "source_power_w": source_audit["meshed_total_w"],
        "mesh_manifest": case.report["source_manifest"] if mesh_method == "notebook" else None,
        "robin_heat_out_w": balance["p_out_robin_w"],
        "heat_balance_relative_error": balance["robin_rel_err"],
        "backside_h_eff_w_m2_k": bcs.backside_h_eff,
    }, indent=2) + "\n")

    if renders:
        z_mid = coords[2] * 1e6
        y_mid = (y0 + y1) / 2
        render_temperature_xy_slice(T, z_mid, f"bitcell hotspot (compact-model power), z={z_mid:.2f}um",
                                     f"{out_dir}/T_xy_slice.png")
        render_temperature_xz_slice(T, y_mid, f"bitcell hotspot (compact-model power), y={y_mid:.2f}um",
                                     f"{out_dir}/T_xz_slice.png", zoom_z_um=(z_mid - 3.0, z_mid + 1.0))
        print(f"wrote {out_dir}/T_xy_slice.png, {out_dir}/T_xz_slice.png")

    return tmax_k, num_cells


def _argument_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--pad-um", type=float, default=0.0)
    p.add_argument("--refine", type=float, default=1.0)
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--mesh-method", choices=["notebook", "legacy"], default="notebook",
                   help="exact notebook polygons (default), or former bounding-box mesher")
    p.add_argument("--mesh-manifest", help="reuse a sealed notebook material_regions.json without remeshing")
    p.add_argument("--power-model", choices=["spice-transient", "dc-surrogate"],
                   default="spice-transient", help="measured SPICE event energies (default), or legacy DC approximation")
    p.add_argument("--point", choices=sorted(set(SPICE_ACCESS_POINTS) | set(NAMED_BIAS_POINTS)),
                   default="read_1", help="READ/WRITE access (default: read_1); crowbar/hold/settled-write require dc-surrogate")
    p.add_argument("--row-hit-rate", type=float, default=1.0,
                   help="fraction of cycles selecting this row; scales event energy / period")
    p.add_argument("--period-ns", type=float, help="access period (default: Liberty minimum period)")
    p.add_argument("--pulse-width-ns", type=float, help="SPICE wordline high time (default: Liberty minimum high time)")
    p.add_argument("--spice-step-ps", type=float, default=1.0, help="maximum electrical timestep in ps")
    p.add_argument("--interconnect", choices=["layout", "none"], default=None,
                   help="layout-linked Joule heat (default for SPICE); none uses ideal wires")
    p.add_argument("--interconnect-step-um", type=float, default=0.05,
                   help="electrical sheet-network tile size in micrometres")
    p.add_argument("--no-renders", action="store_true")
    p.add_argument("--lateral-bc", choices=["periodic", "insulating"], default="periodic",
                   help="x/y boundary condition (default: periodic, same cell/activity repeated by translation)")
    return p


def main():
    args = _argument_parser().parse_args()
    run(pad_um=args.pad_um, refine=args.refine, out_dir=args.out_dir, lateral_bc=args.lateral_bc,
        mesh_method=args.mesh_method, mesh_manifest=args.mesh_manifest,
        point=args.point, renders=not args.no_renders, power_model=args.power_model,
        row_hit_rate=args.row_hit_rate, period_ns=args.period_ns,
        pulse_width_ns=args.pulse_width_ns, spice_step_ps=args.spice_step_ps,
        interconnect=args.interconnect, interconnect_step_um=args.interconnect_step_um)


if __name__ == "__main__":
    main()
