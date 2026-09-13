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
import json
import sys
import tempfile
from pathlib import Path

import cairosvg
import gdstk
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cases.run_bitcell_transient import CELL_NAME, GDS_PATH, N_RENDER_FRAMES_PER_PHASE, _select_uniform_dt_frames
from gds import techmap
from gds.read import flatten_by_layer
from mesh.gds_svg_viz import build_layer_regions, render_layer_regions_svg

Z_PAD_UM = 0.05  # tolerance when matching dofs to a layer's own z band


def _dof_indices_per_volume(regions, dof_coords_um):
    """For each polygon, the dofs sitting inside its xy bounding box and
    within its layer's z band -- computed ONCE (geometry doesn't move) and
    reused for every frame. Sampling a polygon's HOTTEST dof rather than
    just its centroid matters: a gate stripe's centroid can fall between
    two devices and miss the hot channel entirely."""
    x, y, z = dof_coords_um[:, 0], dof_coords_um[:, 1], dof_coords_um[:, 2]
    per_volume = []
    for region in regions:
        in_z = (z >= region["z_min_um"] - Z_PAD_UM) & (z <= region["z_max_um"] + Z_PAD_UM)
        for volume in region["volumes"]:
            pts = volume["footprint_xy_um"]
            x0 = min(p[0] for p in pts); x1 = max(p[0] for p in pts)
            y0 = min(p[1] for p in pts); y1 = max(p[1] for p in pts)
            mask = in_z & (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
            idx = np.flatnonzero(mask)
            if idx.size == 0:  # sliver thinner than the mesh -- fall back to nearest dof
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                cz = (region["z_min_um"] + region["z_max_um"]) / 2
                idx = np.array([int(np.argmin((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2))])
            per_volume.append((volume, idx))
    return per_volume


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="out/bitcell_transient_crowbar",
                    help="a cases/run_bitcell_transient.py output directory (contains fields/)")
    p.add_argument("--frames-per-phase", type=int, default=N_RENDER_FRAMES_PER_PHASE)
    args = p.parse_args()

    fields_dir = Path(args.dir) / "fields"
    dof_coords_um = np.load(fields_dir / "dof_coords.npy") * 1e6
    times_s = np.load(fields_dir / "times_s.npy")
    tmax_hist = np.load(fields_dir / "tmax_hist.npy")
    meta = json.loads((fields_dir / "meta.json").read_text())
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
            title = (f"{meta['point']} [{label}]  t={rel_t*1e3:.4f}ms  "
                     f"Tmax={tmax_hist[i]:.4f}K (bulk +{tmax_hist[i]-300.0:.3f}K)  "
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
