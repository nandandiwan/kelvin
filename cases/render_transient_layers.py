"""Fixed-viewpoint GIF of a bitcell transient run showing PER-TRANSISTOR
detail -- 2D mask plan + painter-sorted isometric stack, same style as
mesh/gds_svg_viz.py's geometry render, but each real GDS polygon is colored
by the local temperature EXCESS at its own location. Built PURELY from the
fields cases/run_bitcell_transient.py already saved -- no re-solving.

Why excess and not absolute dT (measured on this run's own fields, not
assumed): the device-layer spatial spread is a ~0.0167K ripple that stays
essentially CONSTANT through the whole active phase while the bulk baseline
climbs from 0.0003K to 31.4K -- the bulk rise is set by the 50um path down
to the backside sink, the ripple by the per-device power split. On an
absolute 0..31K scale the ripple is ~1/2000 of the range, so every
transistor maps to the same color and the animation degenerates into one
flat tint changing over time. Coloring each frame's excess above its own
coolest element keeps the devices legible at every instant, and still fades
honestly to dark once power stops, because the excess itself then collapses
by ~19x (0.0167K -> 0.0009K) -- the per-device structure IS the signature
of active power.

    python cases/render_transient_layers.py [--dir out/bitcell_transient_crowbar]
"""

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import gdstk
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cases.run_bitcell_transient import CELL_NAME, GDS_PATH, N_RENDER_FRAMES_PER_PHASE, _select_uniform_dt_frames
from gds import techmap
from gds.bitcell_mapping import mapping_revision
from gds.spice_power import ACCESS_ENERGY_MODEL_REVISION, SPICE_MODEL_REVISION
from gds.thermal_workload import THERMAL_POWER_REVISION
from gds.read import flatten_by_layer
from mesh.gds_svg_viz import build_layer_regions, render_layer_regions_svg

Z_PAD_UM = 1e-9  # numerical coordinate tolerance, not a physical layer padding


def _validate_mapping_metadata(meta):
    is_spice = meta.get("power_model") == "spice-transient"
    if (meta.get("instantaneous") and not is_spice
            and meta.get("access_energy_model_revision") != ACCESS_ENERGY_MODEL_REVISION):
        raise ValueError("Saved pulse fields use an obsolete or unverified access-energy model; "
                         "rerun cases/run_bitcell_transient.py before rendering these fields")
    if meta.get("spice_model_revision") != SPICE_MODEL_REVISION:
        raise ValueError("Saved fields predate the verified SPICE geometry-unit correction; "
                         "rerun cases/run_bitcell_transient.py before rendering these fields")
    if meta.get("source_mapping_sha256") != mapping_revision():
        raise ValueError(
            "Saved transient fields use an obsolete or unverified transistor-power mapping; "
            "rerun cases/run_bitcell_transient.py with the same workload and --out-dir "
            "before rendering these fields."
        )
    if is_spice:
        if meta.get("thermal_power_revision") != THERMAL_POWER_REVISION:
            raise ValueError("Saved fields use an obsolete SPICE-to-thermal transfer; "
                             "rerun cases/run_bitcell_transient.py")
        path = Path(meta.get("power_workload_path") or "")
        if (not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != meta.get("power_workload_sha256")):
            raise ValueError("Saved SPICE workload is missing or changed; "
                             "rerun cases/run_bitcell_transient.py")
        workload = json.loads(path.read_text())
        if (workload.get("kind") != "spice-transient"
                or workload.get("thermal_power_revision") != THERMAL_POWER_REVISION):
            raise ValueError("Saved workload does not identify the SPICE-to-thermal model; "
                             "rerun cases/run_bitcell_transient.py")
        artifact_dir = Path(workload.get("spice_artifact_dir") or "")
        artifacts = workload.get("spice_artifact_sha256")
        if not artifacts:
            raise ValueError("Saved workload has no verified electrical artifacts; "
                             "rerun cases/run_bitcell_transient.py")
        for filename, expected in artifacts.items():
            artifact = artifact_dir / filename
            if not artifact.is_file() or hashlib.sha256(artifact.read_bytes()).hexdigest() != expected:
                raise ValueError("Saved SPICE electrical artifact is missing or changed; "
                                 "rerun cases/run_bitcell_transient.py")
    if meta.get("mesh_method") == "notebook":
        path = Path(meta.get("mesh_manifest") or "")
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != meta.get("mesh_manifest_sha256"):
            raise ValueError("Saved notebook mesh manifest is missing or changed; rerun "
                             "cases/run_bitcell_transient.py before rendering")


def _dof_indices_per_volume(regions, dof_coords_um):
    """Sample only nodes in each actual polygon prism, including its boundary.

    A tiny coordinate tolerance includes floating-point perturbations of
    conformal boundary nodes; it must not reach another physical layer or
    fill a concave polygon's gaps. Missing samples fail explicitly rather
    than borrowing an unrelated globally nearest node.
    """
    x, y, z = dof_coords_um[:, 0], dof_coords_um[:, 1], dof_coords_um[:, 2]
    per_volume = []
    for region in regions:
        in_z = (z >= region["z_min_um"] - Z_PAD_UM) & (z <= region["z_max_um"] + Z_PAD_UM)
        for volume in region["volumes"]:
            pts = volume["footprint_xy_um"]
            x0 = min(p[0] for p in pts); x1 = max(p[0] for p in pts)
            y0 = min(p[1] for p in pts); y1 = max(p[1] for p in pts)
            mask = (in_z & (x >= x0 - Z_PAD_UM) & (x <= x1 + Z_PAD_UM)
                    & (y >= y0 - Z_PAD_UM) & (y <= y1 + Z_PAD_UM))
            candidates = np.flatnonzero(mask)
            # Offset only at numerical scale (1e-9 um), with a still finer
            # clipping grid so gdstk's default 1e-3 um precision cannot
            # collapse this tolerance or alter nanometre-sized features.
            footprint = gdstk.offset([gdstk.Polygon(pts)], Z_PAD_UM, precision=1e-12)
            inside = np.asarray(gdstk.inside(dof_coords_um[candidates, :2], footprint), dtype=bool)
            idx = candidates[inside]
            if idx.size == 0:
                raise ValueError(
                    f"No temperature nodes lie in polygon prism {region.get('name', '<unnamed>')!r} "
                    f"at z={region['z_min_um']}..{region['z_max_um']} um; "
                    "use a conformal mesh or interpolate the field onto that geometry"
                )
            per_volume.append((volume, idx))
    return per_volume


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="out/bitcell_transient_spice-transient_read_1",
                    help="a cases/run_bitcell_transient.py output directory (contains fields/)")
    p.add_argument("--frames-per-phase", type=int, default=N_RENDER_FRAMES_PER_PHASE)
    args = p.parse_args()

    fields_dir = Path(args.dir) / "fields"
    meta = json.loads((fields_dir / "meta.json").read_text())
    _validate_mapping_metadata(meta)
    import cairosvg

    dof_coords_um = np.load(fields_dir / "dof_coords.npy") * 1e6
    times_s = np.load(fields_dir / "times_s.npy")
    tmax_hist = np.load(fields_dir / "tmax_hist.npy")
    active_end_s = meta["active_end_s"]

    active_mask = times_s <= active_end_s
    active_idx = [i for i in range(len(times_s)) if active_mask[i]]
    idle_idx = [i for i in range(len(times_s)) if not active_mask[i]]
    frame_idx = [active_idx[i] for i in _select_uniform_dt_frames(tmax_hist[active_idx], args.frames_per_phase)]
    if len(idle_idx) > 1:
        frame_idx += [idle_idx[i] for i in _select_uniform_dt_frames(tmax_hist[idle_idx], args.frames_per_phase)]

    # Layer geometry, cheap to re-derive (no mesh rebuild) -- same real GDS
    # polygons and stack z-bounds the solve itself used.
    lib = gdstk.read_gds(GDS_PATH)
    cell = next(c for c in lib.cells if c.name == CELL_NAME)
    by_layer = flatten_by_layer(cell)
    (x0, y0), (x1, y1) = cell.bounding_box()
    window = (x0, x1, y0, y1)

    if meta.get("mesh_method") == "notebook":
        manifest = json.loads(Path(meta["mesh_manifest"]).read_text())
        # Use the solved polygons AND their z origin, not the old box-stack heights.
        regions = [r for r in manifest["regions"]
                   if r.get("role") in ("conductor", "semiconductor", "source")]
    else:
        regions = build_layer_regions(by_layer, techmap.FRONTSIDE_STACK, window)
    per_volume = _dof_indices_per_volume(regions, dof_coords_um)

    # Per-frame excess, then one GLOBAL scale over all frames so the
    # structure's intensity is comparable frame to frame (it really does
    # fade when the power stops -- that must show, not be normalized away).
    excess_by_frame = []
    for i in frame_idx:
        T = np.load(fields_dir / f"T_{i:03d}.npy")
        vals = np.array([T[idx].max() for _, idx in per_volume])
        excess_by_frame.append(vals - vals.min())
    clim = (0.0, float(max(e.max() for e in excess_by_frame)))
    print(f"per-device excess scale: 0 .. {clim[1]*1e3:.3f} mK "
          f"(vs bulk rise of {tmax_hist.max()-300.0:.2f} K)")

    frame_paths = []
    with tempfile.TemporaryDirectory() as tmp:
        for n, (i, excess) in enumerate(zip(frame_idx, excess_by_frame)):
            for (volume, _), value in zip(per_volume, excess):
                volume["value"] = float(value)
            label = "active" if i in active_idx else "idle"
            rel_t = times_s[i] if label == "active" else times_s[i] - active_end_s
            time_scale, time_unit = (1e9, "ns") if meta.get("instantaneous") else (1e3, "ms")
            title = (f"{meta['point']} [{label}]  t={rel_t*time_scale:.4f}{time_unit}  "
                     f"Tmax={tmax_hist[i]:.4f}K (peak +{tmax_hist[i]-300.0:.3f}K)  "
                     f"device spread={excess.max()*1e3:.2f}mK")
            svg = render_layer_regions_svg(regions, clim, title,
                                            scale_label="local excess above coolest element (K)")
            frame_path = f"{tmp}/f{n:03d}.png"
            cairosvg.svg2png(bytestring=svg.encode(), write_to=frame_path, scale=1.6)
            frame_paths.append(frame_path)

        frames = [Image.open(pth).convert("RGB") for pth in frame_paths]
        out_path = Path(args.dir) / "hotspot_evolution_layers.gif"
        frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=180, loop=0)
    print(f"wrote {out_path} ({len(frames)} frames, purely from saved fields, no re-solve)")


if __name__ == "__main__":
    main()
