"""Slide 7: animated 2D and 3D views of the tiled SRAM array, coloured per
real GDS shape (every gate stripe, contact, and metal segment individually),
built from cases/run_array_full.py's `layer_history_mK.npy`.

Replaces fig7_array_evolution.gif, which tracked one max temperature per
CELL -- correct, but every frame was a single flat 3x5 block grid, which
reads as "a flash in the middle" because it throws away all sub-cell detail.
This uses the same per-shape sampling `render_transient_layers.py` already
proved out for a single cell, generalized to the whole tiled array, so
individual transistors are visible lighting up and heat visibly creeping
(or not) into neighbouring cells.

`render_layer_regions_svg` draws BOTH a top-down plan and a painter-sorted
isometric in one image already -- this crops that into two separate frame
sequences (2D and 3D) rather than rendering twice, since the geometry and
values are identical either way.

    python cases/fig_array_transient_layers.py [--frames 160] [--duration-ms 180]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gds import techmap
from gds.tile_array import tile_array
from mesh.gds_svg_viz import build_layer_regions, render_layer_regions_svg

OUT = Path("out/sram_array_transient")
FIG = Path("out/presentation")
SVG_WIDTH = 1180
# render_layer_regions_svg lays out panels at FIXED pixel x -- left panel
# spans SVG x in [35, 555], right panel starts at x=585 -- regardless of the
# `width` parameter (which only pads blank space on the right). The crop
# point is therefore a fixed SVG-unit x, not a fraction of image width: using
# a fraction here first (595/1180 against a 1400-wide render) put the cut
# inside the right panel and let the isometric bleed into the "2D" gif.
PANEL_CUT_SVG_X = 570


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=160)
    p.add_argument("--duration-ms", type=int, default=180,
                   help="per-frame GIF duration -- deliberately slow, per feedback "
                        "that the first version read as a single flash")
    p.add_argument("--scale", type=float, default=1.5, help="cairosvg rasterization scale")
    args = p.parse_args()

    meta = json.loads((OUT / "layer_meta.json").read_text())
    layer_hist = np.load(OUT / "layer_history_mK.npy")
    n_steps, n_vol = layer_hist.shape
    print(f"{n_steps} steps x {n_vol} tracked shapes, "
          f"{meta['n_rows']}x{meta['n_cols']} array, active row {meta['active_row']}")

    by_layer, window, _ch, _co = tile_array(meta["n_rows"], meta["n_cols"])
    regions = build_layer_regions(by_layer, techmap.FRONTSIDE_STACK, window)
    n_regions_vol = sum(len(r["volumes"]) for r in regions)
    if n_regions_vol != n_vol:
        raise SystemExit(f"geometry mismatch: {n_regions_vol} volumes now vs "
                         f"{n_vol} tracked -- tile_array or techmap changed since the run")

    idx = np.linspace(0, n_steps - 1, min(args.frames, n_steps)).astype(int)
    clim = (0.0, float(layer_hist[idx].max()))
    print(f"colour scale: 0 .. {clim[1]:.1f} mK above ambient, {len(idx)} frames")

    period_ns, pulse_ns = meta["period_ns"], meta["pulse_ns"]
    frames_2d, frames_3d = [], []
    for n, i in enumerate(idx):
        vals = layer_hist[i]
        k = 0
        for region in regions:
            for volume in region["volumes"]:
                volume["value"] = float(vals[k]); k += 1

        t_ns = (i + 1) * meta["dt_ps"] * 1e-3
        wl_high = (t_ns % period_ns) < pulse_ns
        cyc = int(t_ns // period_ns) + 1
        title = (f"3x5 real bitcell array, row {meta['active_row']} active   "
                 f"t = {t_ns:.3f} ns   cycle {cyc}   WL {'HIGH' if wl_high else 'low'}")
        svg = render_layer_regions_svg(regions, clim, title,
                                       scale_label="temperature above ambient (mK)",
                                       width=SVG_WIDTH)
        png_path = FIG / f"_arrlayer_{n:04d}.png"
        import cairosvg
        cairosvg.svg2png(bytestring=svg.encode(), write_to=str(png_path), scale=args.scale)
        img = Image.open(png_path).convert("RGB")
        cut = round(PANEL_CUT_SVG_X * args.scale)
        frames_2d.append(img.crop((0, 0, cut, img.height)))
        # The shared header text sits at fixed x=35 in the FULL two-panel SVG,
        # so cropping to [cut:width] keeps only whatever fragment of it
        # happened to fall past the cut point (a single long title string,
        # not per-panel text) -- verified directly: the 3D-only frame showed
        # just "WL low", the tail of the real title. Draw a proper caption on
        # the 3D crop instead of relying on that bled-through fragment.
        crop3d = img.crop((cut, 0, img.width, img.height))
        draw = ImageDraw.Draw(crop3d)
        draw.rectangle([0, 0, crop3d.width, 34], fill=(255, 255, 255))
        draw.text((14, 8), title, fill=(38, 50, 56))
        frames_3d.append(crop3d)
        png_path.unlink()
        if (n + 1) % 20 == 0:
            print(f"  rendered {n+1}/{len(idx)} frames", flush=True)

    out2d = FIG / "fig7_array_evolution_2d.gif"
    out3d = FIG / "fig7_array_evolution_3d.gif"
    frames_2d[0].save(out2d, save_all=True, append_images=frames_2d[1:],
                      duration=args.duration_ms, loop=0)
    frames_3d[0].save(out3d, save_all=True, append_images=frames_3d[1:],
                      duration=args.duration_ms, loop=0)
    print(f"wrote {out2d} and {out3d} "
          f"({len(idx)} frames, {args.duration_ms}ms/frame = "
          f"{len(idx)*args.duration_ms/1000:.1f}s playback)")


if __name__ == "__main__":
    main()
