"""2D + 3D animated views of the slow-clock runs, with the substrate visible
so the downward heat path is actually on screen.

Two rendering choices worth stating:

`--max-depth-um` limits how much substrate is DRAWN (all of it is tracked).
The full model is 53.4um deep over a 1.2-6um footprint -- an aspect ratio of
9:1 to 34:1 -- which renders isometrically as an unreadable needle. Drawing
the top slice of substrate keeps the device stack legible while still showing
the heat crossing into silicon.

`--z-exaggeration` is set explicitly for the same reason: the renderer's
automatic rule is tuned for a stack whose height is comparable to its width,
and returns 1.0 here, which puts the whole 53um on the same scale as a 1.6um
cell. Compressing z is what keeps both the thin BEOL layers and the thick
substrate slabs visible in one frame.

    python cases/fig_slow_clock_layers.py --mode cell
    python cases/fig_slow_clock_layers.py --mode array
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cases.run_bitcell_gallery import build_bitcell_geometry
from cases.run_slow_clock import SUBSTRATE_DEPTHS_UM
from gds import techmap
from gds.tile_array import tile_array
from mesh.gds_svg_viz import build_layer_regions, render_layer_regions_svg

FIG = Path("out/presentation")
SVG_WIDTH = 1180
PANEL_CUT_SVG_X = 570


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["cell", "array"], default="cell")
    p.add_argument("--frames", type=int, default=150)
    p.add_argument("--duration-ms", type=int, default=140)
    p.add_argument("--scale", type=float, default=1.5)
    p.add_argument("--max-depth-um", type=float, default=None,
                   help="substrate depth to DRAW (default 10um for a cell, 20um "
                        "for an array); all depths stay tracked regardless")
    p.add_argument("--z-exaggeration", type=float, default=None)
    args = p.parse_args()

    src = Path(f"out/slow_clock_{args.mode}")
    meta = json.loads((src / "meta.json").read_text())
    layer_hist = np.load(src / "layer_history_mK.npy")
    n_steps, n_vol = layer_hist.shape
    print(f"{meta['label']}: {n_steps} steps x {n_vol} shapes, "
          f"period {meta['period_s']*1e3:.3f} ms, tau {meta['tau_s']*1e3:.3f} ms")

    if args.mode == "cell":
        by_layer, window, _ci, _co = build_bitcell_geometry()
    else:
        by_layer, window, _ch, _co = tile_array(meta["rows"], meta["cols"])

    regions = build_layer_regions(by_layer, techmap.FRONTSIDE_STACK, window,
                                  include_substrate=True,
                                  substrate_depths_um=SUBSTRATE_DEPTHS_UM)
    if sum(len(r["volumes"]) for r in regions) != n_vol:
        raise SystemExit("geometry mismatch vs the recorded history")

    # Values are recorded in region/volume order, so trim AFTER pairing them.
    flat = []
    for r in regions:
        for v in r["volumes"]:
            flat.append((r, v))

    max_depth = args.max_depth_um or (10.0 if args.mode == "cell" else 20.0)
    z_sub_top = next(z1 for z0, z1, b in techmap.z_bounds(techmap.FRONTSIDE_STACK)
                     if b.name == "Si_substrate")
    draw_regions = [r for r in regions
                    if r.get("role") != "substrate" or r["z_max_um"] >= z_sub_top - max_depth]
    kept = {id(r) for r in draw_regions}
    print(f"drawing {len(draw_regions)}/{len(regions)} regions "
          f"(substrate to {max_depth:g}um of {z_sub_top:.1f}um)")

    lateral = max(window[1] - window[0], window[3] - window[2])
    z_lo = min(r["z_min_um"] for r in draw_regions)
    z_hi = max(r["z_max_um"] for r in draw_regions)
    z_exag = args.z_exaggeration or (1.5 * lateral / (z_hi - z_lo))
    print(f"z span {z_hi-z_lo:.2f}um over {lateral:.2f}um lateral -> z_exaggeration {z_exag:.3f}")

    idx = np.linspace(0, n_steps - 1, min(args.frames, n_steps)).astype(int)
    clim = (0.0, float(layer_hist[idx].max()))
    print(f"colour scale 0 .. {clim[1]/1e3:.2f} K above ambient, {len(idx)} frames")

    frames_2d, frames_3d = [], []
    for n, i in enumerate(idx):
        vals = layer_hist[i]
        for (r, v), val in zip(flat, vals):
            v["value"] = float(val)
        t_ms = meta["times_ms"][i]
        on = meta["on"][i]
        tmax_K = meta["tmax_K"][i]
        title = (f"{meta['label']} — {1/meta['period_s']:.0f} Hz clock "
                 f"(period = {meta['period_over_tau']:g}·tau)   "
                 f"t = {t_ms:.2f} ms   {'ON' if on else 'off'}   "
                 f"Tmax = 300 + {tmax_K:.1f} K")
        svg = render_layer_regions_svg(draw_regions, clim, title,
                                       scale_label="temperature above ambient (mK)",
                                       width=SVG_WIDTH, z_exaggeration=z_exag)
        png = FIG / f"_slow_{args.mode}_{n:04d}.png"
        import cairosvg
        cairosvg.svg2png(bytestring=svg.encode(), write_to=str(png), scale=args.scale)
        img = Image.open(png).convert("RGB")
        cut = round(PANEL_CUT_SVG_X * args.scale)
        frames_2d.append(img.crop((0, 0, cut, img.height)))
        c3 = img.crop((cut, 0, img.width, img.height))
        d = ImageDraw.Draw(c3)
        # Cover the bled-through tail of the shared SVG title (it sits at SVG
        # y=42, i.e. ~63px at scale 1.5) before drawing our own caption -- a
        # 34px bar left the big fragment visible underneath.
        d.rectangle([0, 0, c3.width, round(80 * args.scale / 1.5)], fill=(255, 255, 255))
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/dejavu/DejaVuSans.ttf", round(15 * args.scale / 1.5))
        except OSError:
            font = None
        d.text((14, 14), title, fill=(38, 50, 56), font=font)
        frames_3d.append(c3)
        png.unlink()
        if (n + 1) % 25 == 0:
            print(f"  rendered {n+1}/{len(idx)}", flush=True)

    o2 = FIG / f"fig8_slow_clock_{args.mode}_2d.gif"
    o3 = FIG / f"fig8_slow_clock_{args.mode}_3d.gif"
    frames_2d[0].save(o2, save_all=True, append_images=frames_2d[1:],
                      duration=args.duration_ms, loop=0)
    frames_3d[0].save(o3, save_all=True, append_images=frames_3d[1:],
                      duration=args.duration_ms, loop=0)
    print(f"wrote {o2} and {o3} ({len(idx)} frames, "
          f"{len(idx)*args.duration_ms/1000:.1f}s playback)")


if __name__ == "__main__":
    main()
