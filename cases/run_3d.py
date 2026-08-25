"""Build the LOD-1 3D volume mesh (GATE 0, 3D) — geometry + mesh only,
no physics yet.

    conda run -n thermals python cases/run_3d.py

Writes out/mesh3d/{chip_volume.msh, mesh_tags.xdmf/.h5, stack_table.txt}.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spec import default_chip
from mesh.build import build_3d_mesh


def main():
    t0 = time.time()
    chip = default_chip()
    mesh_data, registry = build_3d_mesh(chip)
    dt = time.time() - t0

    mesh = mesh_data.mesh
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    print(f"3D build done in {dt:.1f}s: {num_cells} cells, {len(registry.all())} tagged regions")


if __name__ == "__main__":
    main()
