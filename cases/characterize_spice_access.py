"""Run validated electrical accesses and save per-device waveforms/heat.

    python cases/characterize_spice_access.py --check-convergence

This does not run the thermal solver. Outputs retain the SPICE deck, complete
waveform, log, metadata, per-device channel-power NPZ, and verification report.
Use a fresh output directory for each experiment.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from gds.spice_transient import run_access_transient


def plot_access(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (voltage, power) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    time_ns = result.time_s * 1e9
    for name, color in (("q", "#d62f27"), ("qb", "#4274d9"), ("bl", "#f0ae38"),
                        ("br", "#8dbb8c"), ("wl", "#a866c8")):
        voltage.plot(time_ns, result.curves[f"v_{name}"], label=name.upper(), color=color)
    voltage.set_ylabel("voltage (V)")
    voltage.legend(ncol=5)
    colors = ["#d62f27", "#4274d9", "#f0ae38", "#aaaaaa", "#555555", "#a866c8", "#55a76d", "#05a5b5"]
    for (name, values), color in zip(result.device_channel_power_w.items(), colors):
        power.plot(time_ns, values * 1e6, label=name, color=color)
    if result.interconnect_power_w:
        power.plot(time_ns, sum(result.interconnect_power_w.values()) * 1e6,
                   label="wires + contacts", color="#151515", linestyle="--")
    power.set_ylabel("local dissipated power (µW)")
    power.set_xlabel("time (ns)")
    power.legend(ncol=8)
    for axis in (voltage, power):
        axis.grid(alpha=0.2)
        for boundary in (result.metadata["access_start_s"], result.metadata["access_end_s"]):
            axis.axvline(boundary * 1e9, color="gray", alpha=0.4, linestyle="--")
    initial, final = (int(result.metadata[k]) for k in ("initial_stored_one", "final_stored_one"))
    fig.suptitle(f"SPICE {result.metadata['operation'].upper()}: {initial} → {final} · "
                 f"channel heat {result.report['channel_energy_j'] * 1e15:.4f} fJ\n"
                 "isolated loaded cell · fixed-width WL pulse · 25 °C")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", choices=["read", "write", "all"], default="all")
    parser.add_argument("--initial-state", choices=["0", "1", "both"], default="both")
    parser.add_argument("--pulse-width-ns", type=float, default=None,
                        help="WL high plateau; default borrowed Liberty clock-high proxy")
    parser.add_argument("--max-step-ps", type=float, default=1.0)
    parser.add_argument("--interconnect", choices=["layout", "none"], default="layout")
    parser.add_argument("--interconnect-step-um", type=float, default=0.05)
    parser.add_argument("--out-dir", default="out/spice_access")
    parser.add_argument("--check-convergence", action="store_true",
                        help="rerun at half and quarter max timestep; check integrated channel energy")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    out = Path(args.out_dir).resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("--out-dir must be empty or new to preserve previous simulation evidence")
    operations = ["read", "write"] if args.operation == "all" else [args.operation]
    states = [False, True] if args.initial_state == "both" else [args.initial_state == "1"]
    results = {}
    for operation in operations:
        for state in states:
            name = f"{operation}_{int(state)}" + (f"_to_{int(not state)}" if operation == "write" else "")
            result = run_access_transient(operation, stored_one=state,
                                          pulse_width_ns=args.pulse_width_ns,
                                          max_step_ps=args.max_step_ps, out_dir=out / name,
                                          interconnect=args.interconnect,
                                          interconnect_step_um=args.interconnect_step_um)
            if not args.no_plots:
                plot_access(result, out / name / "waveforms.png")
            report = dict(result.report)
            if args.check_convergence:
                refinements = [run_access_transient(
                    operation, stored_one=state, pulse_width_ns=args.pulse_width_ns,
                    max_step_ps=args.max_step_ps / factor,
                    out_dir=out / f"{name}_dt_div_{factor}",
                    interconnect=args.interconnect,
                    interconnect_step_um=args.interconnect_step_um) for factor in (2, 4)]
                reference = refinements[-1].report["channel_energy_j"]
                report["timestep_convergence"] = {
                    "max_step_ps": [args.max_step_ps / factor for factor in (1, 2, 4)],
                    "channel_energy_j": [r.report["channel_energy_j"] for r in (result, *refinements)],
                    "coarse_vs_fine_relative_energy_error": abs(report["channel_energy_j"] / reference - 1),
                    "coarse_vs_fine_relative_peak_error": abs(
                        np.max(sum(result.device_channel_power_w.values())) /
                        np.max(sum(refinements[-1].device_channel_power_w.values())) - 1),
                }
                wire_reference = sum(refinements[-1].interconnect_energy_j().values())
                if wire_reference:
                    report["timestep_convergence"].update({
                        "interconnect_energy_j": [sum(r.interconnect_energy_j().values())
                                                  for r in (result, *refinements)],
                        "coarse_vs_fine_relative_interconnect_energy_error": abs(
                            sum(result.interconnect_energy_j().values()) / wire_reference - 1),
                    })
                    if report["timestep_convergence"]["coarse_vs_fine_relative_interconnect_energy_error"] > 0.01:
                        raise RuntimeError(f"{name}: wire/contact energy convergence worse than 1%")
                if report["timestep_convergence"]["coarse_vs_fine_relative_energy_error"] > 0.01:
                    raise RuntimeError(f"{name}: timestep energy convergence worse than 1%; refine further")
            results[name] = report
            print(f"{name}: verified; channel heat = {report['channel_energy_j'] * 1e15:.6f} fJ; "
                  f"wires/contacts = {sum(result.interconnect_energy_j().values()) * 1e15:.6f} fJ; "
                  f"{report['sample_count']} samples", flush=True)
    (out / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"Saved electrical traces to {out}; thermal solver has not been run.")


if __name__ == "__main__":
    main()
