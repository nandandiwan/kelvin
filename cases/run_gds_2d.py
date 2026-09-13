"""Build the GDS 2D cross-section mesh (GATE 0) and stop — no physics yet.

    conda run -n thermals python cases/run_gds_2d.py [cut_y_um] [total_power_mw] [gds_path]

Writes out/gds/{gds_section.msh, mesh_tags.xdmf/.h5, regions_2d.png,
sources.png, stack_table.txt} and prints a short summary.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gds.read import load_top_cell, flatten_by_layer
from mesh.gds_build import build_gds_2d_mesh

# A real, placed SKY130 SRAM macro (BSD-3, github.com/ucb-substrate/
# sram22_sky130_macros) — 2048 words x 8 bits, 310.7 x 760.2um die,
# 16,384 bitcells (6T each) + periphery.
GDS_PATH = "data/sram22_2048x8m8w1.gds"


def main():
    # Default cutline found by histogramming diff∩poly channel y-centers
    # across the whole die and picking the densest 1um row (see GDS_PLAN.md).
    cut_y_um = float(sys.argv[1]) if len(sys.argv) > 1 else 255.5
    total_power_mw = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    gds_path = sys.argv[3] if len(sys.argv) > 3 else GDS_PATH

    lib, top = load_top_cell(gds_path)
    # No union_merge: the cutline-intersection path only ever does a boolean
    # AND against a thin strip, which is cheap even on raw unmerged polygons
    # (union_merge itself does NOT scale — see gds/read.py's docstring).
    by_layer = flatten_by_layer(top)

    mesh_data, registry = build_gds_2d_mesh(by_layer, cut_y_um, total_power_mw * 1e-3)

    mesh = mesh_data.mesh
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    print(f"cut y={cut_y_um}um: {num_cells} cells, {len(registry.all())} tagged regions")
    print(f"total power: {total_power_mw} mW")
    print("wrote out/gds/{gds_section.msh, mesh_tags.xdmf, regions_2d.png, "
          "sources.png, stack_table.txt}")


if __name__ == "__main__":
    main()
