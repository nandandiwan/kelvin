"""Per-transistor temperature during one switching event, vs what the
steady-state figures show.

Every steady number in this project uses duty-scaled power: peak instantaneous
power x (access duration / clock period) x row-hit rate. For CROWBAR that
factor is min(1, 0.2395ns / 3.867ns) x 1.0 = 0.0619, i.e. the steady field is
driven by 1/16.1 of the instantaneous power. The steady device-to-device
spread of 16.7 mK is therefore NOT a statement about how hot a transistor
gets -- it is that number already divided by 16.1, and then smeared over the
4.4 ms thermal time constant.

cases/run_bitcell_access_transient.py applies the RAW power and resolves the
first nanoseconds. This plots it against two reference lines:

  - the duty-averaged steady spread (16.71 mK), what fig6c shows
  - 16.14 x that, the spread the SAME field would reach at steady state if the
    raw power were sustained -- an independent prediction of where this curve
    is heading, from the duty factor alone

Agreement between the rising curve and that asymptote is the cross-check: the
transient solve and the duty-factor arithmetic are separate calculations.

    python cases/fig_access_transient.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gds.power import mine_access_timing

OUT = Path("out/presentation")
SRC = Path("out/bitcell_access/access_transient.json")
SRC_FF = Path("out/bitcell_access/access_transient_farfield.json")
STEADY_SPREAD_MK = 16.71      # cases/fig_bitcell_operating_points.py, CROWBAR
LIB = "data/sram22_2048x8m8w1_tt_025C_1v80.lib"


def fig_bc_comparison():
    """Adiabatic lateral walls vs the far-field radiation BC (the one
    cases/run_array_full.py uses), same cell, same raw power. The point: at
    nanosecond timescales heat has only diffused ~1um, so which lateral BC is
    used barely matters to the LOCAL device spread -- it matters for what the
    array's IDLE NEIGHBOURS see, not for the driven cell itself. This is the
    apples-to-apples check the single-cell and array runs need to share a BC.
    """
    if not (SRC.exists() and SRC_FF.exists()):
        print(f"  (skipping BC comparison -- need both {SRC} and {SRC_FF})")
        return
    hist_a = json.loads(SRC.read_text())
    hist_f = json.loads(SRC_FF.read_text())
    ta = np.array([h["t_ns"] for h in hist_a]); sa = np.array([h["spread_mK"] for h in hist_a])
    tf = np.array([h["t_ns"] for h in hist_f]); sf = np.array([h["spread_mK"] for h in hist_f])

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    ax.plot(ta, sa, lw=2.4, color="#d62728", label="adiabatic lateral walls (zero flux)")
    ax.plot(tf, sf, lw=2.4, color="#1f77b4", ls="--",
            label="far-field radiation BC, r=0.6µm (same convention as the array run)")
    ax.set_xlabel("time since the switching event began (ns)")
    ax.set_ylabel("device-to-device spread (mK)")
    ax.set_title("Lateral boundary choice barely matters to the local device\n"
                 "(it matters to what idle NEIGHBOURS see, not the driven cell)", fontsize=12.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig6j_bc_comparison.png", dpi=160)
    plt.close(fig)
    at_common = min(ta[-1], tf[-1])
    ia, iff = int(np.argmin(np.abs(ta - at_common))), int(np.argmin(np.abs(tf - at_common)))
    print(f"  at t={at_common:.2f}ns: adiabatic {sa[ia]:.1f} mK vs far-field {sf[iff]:.1f} mK "
          f"({abs(sa[ia]-sf[iff])/sa[ia]*100:.1f}% apart)")
    print(f"wrote {OUT}/fig6j_bc_comparison.png")


def main():
    if not SRC.exists():
        raise SystemExit(f"{SRC} missing -- run cases/run_bitcell_access_transient.py")
    hist = json.loads(SRC.read_text())
    t = np.array([h["t_ns"] for h in hist])
    peak = np.array([h["tmax_k"] for h in hist]) * 1e3
    spread = np.array([h["spread_mK"] for h in hist])

    timing = mine_access_timing(LIB)
    duty = min(1.0, timing.min_pulse_width_high_ns / timing.min_period_ns)
    asymptote = STEADY_SPREAD_MK / duty

    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    ax.plot(t, peak, lw=2.6, color="#d62728", label="peak above ambient")
    ax.plot(t, spread, lw=2.6, color="#1f77b4", label="device-to-device spread")

    ax.axhline(STEADY_SPREAD_MK, color="#1f77b4", lw=1.5, ls="--")
    ax.text(0.06, STEADY_SPREAD_MK * 1.15,
            f"what the steady-state figure shows: {STEADY_SPREAD_MK:.1f} mK\n"
            f"(same power, divided by the {1/duty:.1f}× duty factor)",
            fontsize=9, color="#15507f")
    ax.axhline(asymptote, color="#2e7d32", lw=1.5, ls=":")
    ax.text(0.06, asymptote * 1.06,
            f"predicted steady spread at RAW power: {asymptote:.0f} mK",
            fontsize=9, color="#2e7d32")

    ax.axvspan(0, timing.min_pulse_width_high_ns, color="#e6a700", alpha=0.16)
    ax.text(timing.min_pulse_width_high_ns / 2, 4.5, "access\n%.2f ns" % timing.min_pulse_width_high_ns,
            fontsize=8.5, ha="center", color="#8a6500")
    ax.axvline(timing.min_period_ns, color="#666", lw=1.2, ls=":")
    ax.text(timing.min_period_ns * 0.99, 430, "one clock period ", fontsize=9, ha="right",
            color="#444")

    ax.set_yscale("log")
    ax.set_ylim(3, 600)
    ax.set_xlim(0, t.max())
    ax.set_xlabel("time since the switching event began (ns)")
    ax.set_ylabel("temperature (mK above ambient)")
    ax.set_title("One switching event, resolved\n"
                 "per-transistor structure is a nanosecond phenomenon, not a "
                 "millikelvin one", fontsize=13)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=10, loc="center right")
    fig.tight_layout()
    fig.savefig(OUT / "fig6i_access_transient.png", dpi=165)
    plt.close(fig)

    print(f"duty factor {duty:.4f} (1/{1/duty:.2f})")
    print(f"spread at {t[-1]:.2f} ns: {spread[-1]:.1f} mK = {spread[-1]/STEADY_SPREAD_MK:.1f}x "
          f"the duty-averaged steady value")
    print(f"predicted raw-power steady spread: {asymptote:.1f} mK; "
          f"reached {spread[-1]/asymptote*100:.0f}% of it by {t[-1]:.2f} ns")
    print(f"wrote {OUT}/fig6i_access_transient.png")
    fig_bc_comparison()


if __name__ == "__main__":
    main()
