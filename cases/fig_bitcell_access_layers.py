"""Per-device animated 2D + 3D views of the SINGLE bitcell's raw switching
event, at the same far-field BC as the array run -- the single-cell
counterpart to cases/fig_array_transient_layers.py, from
cases/run_bitcell_access_transient.py --far-field --save-layers.

Same rendering path as the array version (render_layer_regions_svg's two-panel
plan+isometric, cropped into separate 2D/3D frame sequences), so the two GIFs
are visually comparable side by side: one bitcell switching in isolation vs.
the same bitcell as the active row of a real tiled array.

    python cases/fig_bitcell_access_layers.py [--frames 160] [--duration-ms 180]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw

from cases.run_bitcell_gallery import build_bitcell_geometry
from gds import techmap
from mesh.gds_svg_viz import build_layer_regions, render_layer_regions_svg

OUT = Path("out/bitcell_access")
FIG = Path("out/presentation")
SVG_WIDTH = 1180
# See cases/fig_array_transient_layers.py's comment: panel x is FIXED in SVG
# units regardless of the `width` parameter, so the crop point must be too.
PANEL_CUT_SVG_X = 570


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="farfield", choices=["farfield", "adiabatic"])
    p.add_argument("--frames", type=int, default=160)
    p.add_argument("--duration-ms", type=int, default=180)
    p.add_argument("--scale", type=float, default=1.5)
    args = p.parse_args()

    meta = json.loads((OUT / f"layer_meta_{args.tag}.json").read_text())
    layer_hist = np.load(OUT / f"layer_history_mK_{args.tag}.npy")
    n_steps, n_vol = layer_hist.shape
    print(f"{n_steps} steps x {n_vol} tracked shapes ({args.tag} BC)")

    by_layer, window, _channel_info, _contacts = build_bitcell_geometry()
    regions = build_layer_regions(by_layer, techmap.FRONTSIDE_STACK, window)
    n_regions_vol = sum(len(r["volumes"]) for r in regions)
    if n_regions_vol != n_vol:
        raise SystemExit(f"geometry mismatch: {n_regions_vol} volumes now vs "
                         f"{n_vol} tracked -- geometry changed since the run")

    idx = np.linspace(0, n_steps - 1, min(args.frames, n_steps)).astype(int)
    clim = (0.0, float(layer_hist[idx].max()))
    print(f"colour scale: 0 .. {clim[1]:.1f} mK above ambient, {len(idx)} frames")

    frames_2d, frames_3d = [], []
    for n, i in enumerate(idx):
        vals = layer_hist[i]
        k = 0
        for region in regions:
            for volume in region["volumes"]:
                volume["value"] = float(vals[k]); k += 1

        t_ns = (i + 1) * meta["dt_ps"] * 1e-3
        bc_label = "far-field radiation BC" if meta["far_field"] else "adiabatic walls"
        title = f"single bitcell, {bc_label}   t = {t_ns:.3f} ns"
        svg = render_layer_regions_svg(regions, clim, title,
                                       scale_label="temperature above ambient (mK)",
                                       width=SVG_WIDTH)
        png_path = FIG / f"_bclayer_{n:04d}.png"
        import cairosvg
        cairosvg.svg2png(bytestring=svg.encode(), write_to=str(png_path), scale=args.scale)
        img = Image.open(png_path).convert("RGB")
        cut = round(PANEL_CUT_SVG_X * args.scale)
        frames_2d.append(img.crop((0, 0, cut, img.height)))
        # Same fix as fig_array_transient_layers.py: the shared header text
        # sits at fixed x=35 in the full SVG, so cropping to the right panel
        # alone keeps only whatever fragment fell past the cut. Draw a real
        # caption on the 3D crop instead.
        crop3d = img.crop((cut, 0, img.width, img.height))
        draw = ImageDraw.Draw(crop3d)
        draw.rectangle([0, 0, crop3d.width, 34], fill=(255, 255, 255))
        draw.text((14, 8), title, fill=(38, 50, 56))
        frames_3d.append(crop3d)
        png_path.unlink()
        if (n + 1) % 20 == 0:
            print(f"  rendered {n+1}/{len(idx)} frames", flush=True)

    out2d = FIG / f"fig6k_bitcell_evolution_2d_{args.tag}.gif"
    out3d = FIG / f"fig6k_bitcell_evolution_3d_{args.tag}.gif"
    frames_2d[0].save(out2d, save_all=True, append_images=frames_2d[1:],
                      duration=args.duration_ms, loop=0)
    frames_3d[0].save(out3d, save_all=True, append_images=frames_3d[1:],
                      duration=args.duration_ms, loop=0)
    print(f"wrote {out2d} and {out3d} ({len(idx)} frames, {args.duration_ms}ms/frame)")


if __name__ == "__main__":
    main()
