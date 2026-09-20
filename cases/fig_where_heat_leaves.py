"""Where the heat actually leaves, and why the 3D colour map cannot show it.

The slow-clock 3D animation renders the whole block one uniform colour. That
is not a rendering failure -- it is the answer. Two independent checks:

    Biot number  h*t/k = 0.0068          147x below the lumped-capacitance
                                          threshold: the body heats and cools
                                          as ONE lump, by definition
    diffusion length over one ON phase   152 um = 3.0x the 50.2um substrate
                                          thickness: heat crosses the whole
                                          substrate ~3 times per ON phase

So at 240 Hz there IS no spatial gradient to see. What there is instead:
essentially the entire temperature drop happens at the backside boundary, not
inside the silicon. This figure shows that directly.

  left   T(z) from the backside sink up through the substrate and the device
         stack, at several instants -- nearly flat across 50um of silicon,
         with the drop concentrated at the sink face
  right  heat leaving through the backside face vs time, against the power
         going in -- they match at steady state, and the ONLY exit is the
         bottom (lateral and top faces are adiabatic)

    python cases/fig_where_heat_leaves.py [--mode cell]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = Path("out/presentation")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["cell", "array"], default="cell")
    args = p.parse_args()

    src = Path(f"out/slow_clock_{args.mode}")
    meta = json.loads((src / "meta.json").read_text())
    depth = np.load(src / "depth_history_mK.npy") / 1e3      # K
    z_um = np.load(src / "z_probe_um.npy")
    flux = np.load(src / "flux_bottom_W.npy")
    t_ms = np.array(meta["times_ms"])
    on = np.array(meta["on"])
    order = np.argsort(z_um)
    z_um, depth = z_um[order], depth[:, order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.8))

    # ---- left: T(z) at several instants ----
    pulse_ms = meta["pulse_s"] * 1e3
    picks = [(np.argmin(np.abs(t_ms - t)), f"t = {t:.2f} ms") for t in
             (pulse_ms * 0.25, pulse_ms, pulse_ms * 4, meta["period_s"] * 1e3 * 0.98)]
    colors = plt.cm.inferno(np.linspace(0.25, 0.8, len(picks)))
    for (i, lab), c in zip(picks, colors):
        state = "ON" if on[i] else "off"
        ax1.plot(z_um, depth[i], lw=2.3, color=c, label=f"{lab} ({state})")

    z_sub_top = 50.2062
    ax1.axvspan(0, z_sub_top, color="#9fb3c8", alpha=0.18)
    ax1.axvspan(z_sub_top, z_um.max(), color="#e6a700", alpha=0.18)
    ax1.text(z_sub_top / 2, ax1.get_ylim()[1] * 0.94, "silicon substrate (50.2 µm)",
             ha="center", fontsize=9.5, color="#33506b")
    ax1.text(z_sub_top + 1.6, ax1.get_ylim()[1] * 0.94, "device\nstack",
             ha="center", fontsize=9, color="#8a6500")
    ax1.annotate("backside sink\n(the only exit)", xy=(0, depth[picks[1][0], 0]),
                 xytext=(6, depth[picks[1][0], 0] * 0.45), fontsize=9.5, color="#c0392b",
                 arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.6))
    ax1.set_xlabel("height above the backside sink, z (µm)")
    ax1.set_ylabel("temperature above ambient (K)")
    ax1.set_title("Almost no gradient inside the silicon\n"
                  "Bi = 0.0068 -> the block is isothermal by construction", fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9)

    # How flat, quantitatively.
    i_peak = int(np.argmax(depth.max(axis=1)))
    prof = depth[i_peak]
    in_si = z_um <= z_sub_top
    drop_si = prof[in_si].max() - prof[in_si].min()
    ax1.text(0.03, 0.05, f"at peak: {drop_si:.3f} K across all 50.2 µm of silicon,\n"
                         f"vs {prof.max():.1f} K total above ambient "
                         f"({drop_si/prof.max()*100:.1f}%)",
             transform=ax1.transAxes, fontsize=9, color="#2e7d32",
             bbox=dict(fc="w", ec="#2e7d32", alpha=0.9))

    # ---- right: power in vs power out the backside ----
    p_in = np.where(on, meta["total_power_w_on"], 0.0) * 1e6
    ax2.plot(t_ms, p_in, lw=1.8, color="#7b2fbe", label="power in (word line)")
    ax2.plot(t_ms, flux * 1e6, lw=2.4, color="#c0392b",
             label="power out through the backside")
    ax2.axhline(meta["total_power_w_on"] * meta["duty"] * 1e6, color="#2e7d32",
                lw=1.4, ls="--", label="duty-averaged input")
    ax2.set_xlabel("time (ms)")
    ax2.set_ylabel("power (µW)")
    ax2.set_yscale("log")
    ax2.set_title("All of it leaves through the bottom\n"
                  "lateral and top faces are adiabatic — zero flux", fontsize=12)
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=9, loc="lower right")

    fig.suptitle(f"{meta['label']} at {1/meta['period_s']:.0f} Hz — where the heat goes",
                 fontsize=13)
    fig.tight_layout()
    out = FIG / f"fig8_where_heat_leaves_{args.mode}.png"
    fig.savefig(out, dpi=165)
    plt.close(fig)

    print(f"peak profile: {prof.max():.2f} K at the devices, "
          f"{prof[in_si].max():.2f} K at the top of the silicon, "
          f"{prof[0]:.2f} K at the sink face")
    print(f"drop across 50.2um of silicon: {drop_si:.4f} K "
          f"({drop_si/prof.max()*100:.2f}% of the total)")
    print(f"peak backside outflow {flux.max()*1e6:.3f} uW vs "
          f"{meta['total_power_w_on']*1e6:.3f} uW in during ON")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
