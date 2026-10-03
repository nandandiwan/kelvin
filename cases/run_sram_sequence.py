"""Run a continuous small-array electrical sequence from an explicit JSON file."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gds.spice_sequence import run_sequence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("schedule", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--temperature-c", type=float, default=26.85)
    parser.add_argument("--spice-step-ps", type=float, default=1.0)
    args = parser.parse_args()
    result = run_sequence(args.out_dir, json.loads(args.schedule.read_text()),
                          temperature_c=args.temperature_c, max_step_ps=args.spice_step_ps)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
