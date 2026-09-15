"""Slide 7 figures: the raw, non-duty-averaged transient across a real tiled
SRAM array (cases/run_sram_array_transient.py) -- one row driven, its
neighbours idle, far-field lateral boundaries.

  fig7a_array_timeseries.png   active-row vs idle-row peak temperature over
                               several clock cycles -- the direct answer to
                               "how hot does the active row get vs its
                               neighbours, without averaging".
  fig7b_array_snapshots.png    per-cell peak temperature as a small grid
                               (row x col), at several times across one pulse
                               -- shows the hot row appearing and the lateral
                               spread (or lack of it) into idle neighbours.
  fig7_array_evolution.gif     the same grid, animated over many FIXED, SMALL
                               time steps -- smooth by construction, unlike
                               the old duty-cycle-averaged GIF this replaces.

    python cases/fig_sram_array_transient.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

SRC = Path("out/sram_array_transient/array_transient.json")
OUT = Path("out/presentation")


def load():
    if not SRC.exists():
        raise SystemExit(f"{SRC} missing -- run cases/run_sram_array_transient.py first")
    d = json.loads(SRC.read_text())
    return d["meta"], d["hist"]


def grid_at(hist, i, n_rows, n_cols):
    g = np.full((n_rows, n_cols), np.nan)
    rec = hist[i]
    for row in range(n_rows):
        for col in range(n_cols):
            key = f"r{row}c{col}"
            if key in rec:
                g[row, col] = rec[key]
    return g


def fig_timeseries(meta, hist):
    t = np.array([h["t_ns"] for h in hist])
    active_key_prefix = f"r{meta['active_row']}c"
    row_keys = sorted({k for k in hist[0] if k.startswith("r") and "c" in k},
                      key=lambda k: (int(k[1:].split("c")[0]), int(k.split("c")[1])))
    rows = sorted({int(k[1:].split("c")[0]) for k in row_keys})

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    colors = plt.cm.coolwarm(np.linspace(0.15, 0.9, len(rows)))
    for row, col in zip(rows, colors):
        vals = np.array([max(h[k] for k in h if k.startswith(f"r{row}c")) for h in hist])
        lw = 2.6 if row == meta["active_row"] else 1.6
        label = f"row {row} (ACTIVE)" if row == meta["active_row"] else f"row {row} (idle)"
        ax.plot(t, vals, lw=lw, color=col, label=label)

    for k in range(1, int(t[-1] / meta["period_ns"]) + 1):
        ax.axvspan(k * meta["period_ns"], k * meta["period_ns"] + meta["pulse_ns"],
                  color="#7b2fbe", alpha=0.08)
    ax.axvspan(0, meta["pulse_ns"], color="#7b2fbe", alpha=0.08, label="word line high")

    ax.set_xlabel("time (ns)")
    ax.set_ylabel("per-cell peak temperature (mK above ambient)")
    ax.set_title(f"{meta['n_rows']}x{meta['n_cols']} real bitcells, RAW switching power, "
                 f"far-field lateral BC\nrow {meta['active_row']} driven every clock; "
                 f"no duty-cycle averaging", fontsize=12.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig7a_array_timeseries.png", dpi=160)
    plt.close(fig)
    print(f"wrote {OUT}/fig7a_array_timeseries.png")

    final_active = max(hist[-1][k] for k in hist[-1] if k.startswith(f"r{meta['active_row']}c"))
    other_rows = [r for r in rows if r != meta["active_row"]]
    final_idle = max((hist[-1][k] for r in other_rows for k in hist[-1] if k.startswith(f"r{r}c")),
                     default=float("nan"))
    print(f"  final active-row peak {final_active:.2f} mK, idle-row peak {final_idle:.2f} mK, "
          f"ratio {final_active/final_idle:.1f}x" if final_idle > 0 else "  (idle rows still ~0)")


def fig_snapshots(meta, hist):
    t = np.array([h["t_ns"] for h in hist])
    period, pulse = meta["period_ns"], meta["pulse_ns"]
    # Times of interest: start, end of first pulse, mid-idle, end of run.
    targets = [0.0, pulse, period, t[-1]]
    idxs = [int(np.argmin(np.abs(t - tt))) for tt in targets]
    labels = ["t=0", "end of 1st pulse", "1 clock later", f"t={t[-1]:.1f} ns (end)"]

    vmax = max(np.nanmax(grid_at(hist, i, meta["n_rows"], meta["n_cols"])) for i in idxs)
    fig, axes = plt.subplots(1, len(idxs), figsize=(4.0 * len(idxs), 3.6))
    for ax, i, lab in zip(axes, idxs, labels):
        g = grid_at(hist, i, meta["n_rows"], meta["n_cols"])
        im = ax.imshow(g, cmap="inferno", vmin=0, vmax=vmax, origin="lower", aspect="equal")
        ax.set_title(f"{lab}\nt={t[i]:.3f} ns", fontsize=10)
        ax.set_xticks(range(meta["n_cols"]))
        ax.set_yticks(range(meta["n_rows"]))
        ax.axhline(meta["active_row"] - 0.5, color="#39d3ff", lw=1.5)
        ax.axhline(meta["active_row"] + 0.5, color="#39d3ff", lw=1.5)
    fig.colorbar(im, ax=axes, shrink=0.85, label="peak temperature (mK above ambient)")
    fig.suptitle("Per-cell temperature across the pulse (cyan = active row)", fontsize=12.5)
    fig.savefig(OUT / "fig7b_array_snapshots.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/fig7b_array_snapshots.png")


def fig_gif(meta, hist, n_frames=90):
    idx = np.linspace(0, len(hist) - 1, min(n_frames, len(hist))).astype(int)
    vmax = max(np.nanmax(grid_at(hist, i, meta["n_rows"], meta["n_cols"])) for i in idx)
    frames = []
    for i in idx:
        g = grid_at(hist, i, meta["n_rows"], meta["n_cols"])
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        ax.imshow(g, cmap="inferno", vmin=0, vmax=vmax, origin="lower", aspect="equal",
                  interpolation="bicubic")
        ax.axhline(meta["active_row"] - 0.5, color="#39d3ff", lw=1.5)
        ax.axhline(meta["active_row"] + 0.5, color="#39d3ff", lw=1.5)
        wl = "HIGH" if hist[i]["wl_high"] else "low"
        ax.set_title(f"t = {hist[i]['t_ns']:.3f} ns   WL {wl}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        path = OUT / f"_arrframe_{i:04d}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        frames.append(Image.open(path).convert("RGB"))
    out_path = OUT / "fig7_array_evolution.gif"
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=60, loop=0)
    for i in idx:
        (OUT / f"_arrframe_{i:04d}.png").unlink()
    print(f"wrote {out_path} ({len(frames)} frames, uniform time spacing -> smooth by construction)")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta, hist = load()
    print(f"{meta['n_rows']}x{meta['n_cols']} cells, {meta['n_cells']:,} mesh cells, "
          f"active row {meta['active_row']}, {len(hist)} steps")
    fig_timeseries(meta, hist)
    fig_snapshots(meta, hist)
    fig_gif(meta, hist)


if __name__ == "__main__":
    main()
