"""Create PyVista/VTK inspection artifacts from a notebook-generated MSH."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mesh.gds_viz import render_gds_mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("msh", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--png", type=Path)
    parser.add_argument("--vtu", type=Path)
    args = parser.parse_args()
    stem = args.msh.with_suffix("")
    png = args.png or stem.with_name(stem.name + "_pyvista.png")
    vtu = args.vtu or stem.with_name(stem.name + "_cells.vtu")
    print(json.dumps(render_gds_mesh(args.msh, args.manifest, png, vtu), indent=2))


if __name__ == "__main__":
    main()
