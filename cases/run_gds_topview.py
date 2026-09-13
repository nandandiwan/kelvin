"""Render the top-down (X-Y plan) layout view of a GDS macro — nwell/diff/poly,
the "standard" chip-layout view, plus a zoomed inset showing individual
transistors. No mesh/solve involved.

    conda run -n thermals python cases/run_gds_topview.py [highlight_y_um] [gds_path]

Writes out/gds/layout_top_view.png.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gds.read import load_top_cell, flatten_by_layer
from gds.topview import render_top_view
from mesh.gds_section import die_bounds

GDS_PATH = "data/sram22_2048x8m8w1.gds"


def main():
    highlight_y_um = float(sys.argv[1]) if len(sys.argv) > 1 else 255.5
    gds_path = sys.argv[2] if len(sys.argv) > 2 else GDS_PATH

    lib, top = load_top_cell(gds_path)
    by_layer = flatten_by_layer(top)
    bounds = die_bounds(by_layer)

    render_top_view(by_layer, bounds, "out/gds/layout_top_view.png", highlight_y_um=highlight_y_um)
    print(f"die bounds: {bounds}")
    print("wrote out/gds/layout_top_view.png")


if __name__ == "__main__":
    main()
