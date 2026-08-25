"""Build the LOD-1 2D cross-section mesh (GATE 0) and stop — no physics yet.

    conda run -n thermals python cases/run_2d.py [row]

Writes out/mesh/{chip_section.msh, mesh_tags.xdmf/.h5, regions_2d.png,
sources.png, stack_table.txt} and prints a short summary.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spec import default_chip
from mesh.build import build_2d_mesh


def main():
    row = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    chip = default_chip()
    mesh_data, registry = build_2d_mesh(chip, row=row)

    mesh = mesh_data.mesh
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    print(f"row {row}: {num_cells} cells, {len(registry.all())} tagged regions")
    print(f"total device power: {chip.total_power_uw:.1f} uW")
    print("wrote out/mesh/{chip_section.msh, mesh_tags.xdmf, regions_2d.png, "
          "sources.png, stack_table.txt}")


if __name__ == "__main__":
    main()
