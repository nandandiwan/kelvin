"""Generate a solver-ready SRAM mesh using the shared read_gds.ipynb backend."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mesh.sram import generate_sram_mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="out/sram_notebook_mesh")
    parser.add_argument("--pad-um", type=float, default=0.0)
    parser.add_argument("--refine", type=float, default=1.0,
                        help="target mesh length multiplier; smaller means finer")
    parser.add_argument("--no-renders", action="store_true")
    args = parser.parse_args()
    generate_sram_mesh(args.out_dir, pad_um=args.pad_um, refine=args.refine,
                       renders=not args.no_renders)


if __name__ == "__main__":
    main()
