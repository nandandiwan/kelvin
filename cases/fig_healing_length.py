"""The length scale that separates "transistor detail" from "chip temperature".

A thin slab of silicon sitting on a heat sink has a **thermal healing length**

    L = sqrt(k * t / h)

Below L, lateral conduction is so much easier than escaping through the sink
that the silicon is effectively isothermal: any structure smaller than L gets
smeared out, and only the AVERAGE power density over ~L^2 matters. Above L,
heat escapes downward before it can spread, and real lateral gradients form.

For this project's bitcell substrate (k=148, t=53.4um, h=2e4) that is **629um**
-- 524x the 1.2um cell pitch, 4191x the 0.15um channel. Which is precisely why
the per-transistor structure is tens of mK on top of a ~31 K pedestal: at the
transistor scale you are 4000x inside the isothermal regime.

Validated here, not asserted, in two independent ways:

  1. The slab is thin enough for the reduction to hold at all:
     Biot = h*t/k = 0.0072 << 1.
  2. The 2D "fin" reduction  -k t grad^2 T + h T = q  is solved in dolfinx and
     its far-field decay is fitted. A point source on an infinite fin has the
     closed-form solution  T(r) = P/(2 pi k t) * K0(r/L), so the fitted decay
     length is compared directly against sqrt(k t / h).

    python cases/fig_healing_length.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ufl
from dolfinx import fem, mesh as dmesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from scipy.special import k0, k0e

OUT = Path("out/presentation")
ISO = Path("out/bitcell_isolation")

K_SI, T_M, H_EFF = 148.0, 53.4e-6, 20000.0
L_HEAL = np.sqrt(K_SI * T_M / H_EFF)

# The outer wall is a zero-flux (reflecting) boundary, so it must sit far
# enough out that it does not flatten the decay inside the fit band. A 6mm
# domain (half-width 4.8 L) put the wall INSIDE the 1.5-5 L fit band and
# inflated the fitted decay length by 19%. 16mm puts the wall at 12.7 L.
DOMAIN_M = 16.0e-3
N_ELEM = 800               # 20um cells
SRC_A_M = 140e-6           # source patch side, ~7 cells across
P_W = 1.0                  # linear problem -> dT is directly K/W
FIT_BAND_L = (1.5, 4.0)    # in healing lengths


def solve_fin(src_a_m=SRC_A_M, msh=None):
    """2D fin equation on a square slab, square source of side `src_a_m` at the
    centre. Pass `msh` to reuse one mesh across a source-size sweep."""
    if msh is None:
        msh = dmesh.create_rectangle(
            MPI.COMM_WORLD, [np.array([-DOMAIN_M / 2, -DOMAIN_M / 2]),
                             np.array([DOMAIN_M / 2, DOMAIN_M / 2])],
            [N_ELEM, N_ELEM], dmesh.CellType.triangle)
    SRC_A_M_ = src_a_m
    V = fem.functionspace(msh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    # q as a DG0 field: a source patch defined by an interpolated expression on
    # P1 would be smeared across the elements straddling its edge, which shifts
    # the effective source size -- the very quantity being measured.
    Q = fem.functionspace(msh, ("DG", 0))
    q = fem.Function(Q)
    centers = Q.tabulate_dof_coordinates()
    inside = (np.abs(centers[:, 0]) <= SRC_A_M_ / 2) & (np.abs(centers[:, 1]) <= SRC_A_M_ / 2)
    q.x.array[:] = 0.0
    q.x.array[inside] = 1.0
    # Normalise so the patch carries exactly P_W however many cells it caught.
    area = fem.assemble_scalar(fem.form(q * ufl.dx))
    q.x.array[:] *= P_W / area

    a = (K_SI * T_M * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         + H_EFF * ufl.inner(u, v) * ufl.dx)
    L = ufl.inner(q, v) * ufl.dx
    problem = LinearProblem(a, L, bcs=[], petsc_options_prefix="fin_",
                            petsc_options={"ksp_type": "cg", "pc_type": "hypre",
                                           "ksp_rtol": 1e-12})
    return V, problem.solve(), msh


def radial_profile(V, T):
    """dT vs radius along the +x axis from the source centre."""
    coords = V.tabulate_dof_coordinates()
    on_axis = np.abs(coords[:, 1]) < 1e-9
    r = coords[on_axis, 0]
    keep = r > 0
    order = np.argsort(r[keep])
    return r[keep][order], T.x.array[on_axis][keep][order]


def fit_decay_length(r, dt):
    """K0(r/L) ~ sqrt(pi L / 2r) exp(-r/L), so log(dt * sqrt(r)) is linear in r
    with slope -1/L. Fitted over 1.5..5 healing lengths: closer in the
    asymptotic form is not yet valid, further out the outer wall interferes."""
    band = (r > FIT_BAND_L[0] * L_HEAL) & (r < FIT_BAND_L[1] * L_HEAL)
    slope, _ = np.polyfit(r[band], np.log(dt[band] * np.sqrt(r[band])), 1)
    return -1.0 / slope


RHO_CP = 2330.0 * 712.0
ALPHA = K_SI / RHO_CP            # thermal diffusivity, m^2/s
TAU_S = RHO_CP * T_M / H_EFF     # slab time constant = R*C = (1/hA)(rho cp A t)


def fig_timescales():
    """Per-transistor structure is millikelvin in STEADY STATE. It is not a
    property of the transistors -- it is what is left after 4.4 ms of diffusion
    has smeared everything within 626 um together.

    The transient picture is different: heat diffuses sqrt(alpha t), so on the
    timescale of one access (~1 ns) it has moved 0.30 um -- a quarter of a
    bitcell pitch. At that instant individual devices ARE thermally distinct,
    and the neighbours do not know yet.

    Note the identity, which is a useful self-check rather than a coincidence:
    tau = rho c t / h, so sqrt(alpha * tau) = sqrt(k t / h) = L_heal exactly.
    The healing length IS the distance heat diffuses in one time constant.
    """
    t = np.logspace(-10, -1, 400)
    l_diff = np.sqrt(ALPHA * t) * 1e6

    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    ax.loglog(t, l_diff, lw=2.6, color="#1f77b4")

    for y, lab, col in ((1.2, "bitcell pitch 1.2 µm", "#8a6500"),
                        (176.0, "SRAM macro 176 µm", "#2e7d32"),
                        (L_HEAL * 1e6, f"healing length {L_HEAL*1e6:.0f} µm", "#d62728")):
        ax.axhline(y, color=col, lw=1.3, ls="--")
        ax.text(1.3e-10, y * 1.12, lab, fontsize=9, color=col)

    for x, lab in ((1e-9, "one\naccess"), (3.867e-9, "one\nclock"), (TAU_S, "thermal\n$\\tau$")):
        ax.axvline(x, color="#666", lw=1.0, ls=":")
        ax.text(x * 1.25, 2.5e-2, lab, fontsize=8.5, color="#444")

    # Where the diffusion length crosses the cell pitch: before this, devices
    # are thermally independent; after it, they are not.
    t_cross = (1.2e-6) ** 2 / ALPHA
    ax.plot([t_cross], [1.2], "o", ms=11, color="#8a6500", zorder=6)
    ax.annotate(f"devices stop being independent\nat t ≈ {t_cross*1e9:.0f} ns",
                xy=(t_cross, 1.2), xytext=(t_cross * 6, 0.16), fontsize=9.5,
                color="#8a6500", arrowprops=dict(arrowstyle="->", color="#8a6500", lw=1.5))

    ax.set_xlim(1e-10, 1e-1)
    ax.set_ylim(2e-2, 3e3)
    ax.set_xlabel("time since the power turned on (s)")
    ax.set_ylabel("thermal diffusion length  $\\sqrt{\\alpha t}$  (µm)")
    ax.set_title("Why per-transistor detail is a transient phenomenon\n"
                 "in steady state, 4.4 ms of diffusion has smeared 626 µm together",
                 fontsize=12.5)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "fig6h_timescales.png", dpi=165)
    plt.close(fig)
    print(f"  alpha = {ALPHA:.4e} m^2/s   tau = {TAU_S*1e3:.2f} ms   "
          f"sqrt(alpha*tau) = {np.sqrt(ALPHA*TAU_S)*1e6:.1f} um (== L_heal)")
    print(f"  devices decouple below t = {t_cross*1e9:.0f} ns")
    print(f"wrote {OUT}/fig6h_timescales.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Biot = h*t/k = {H_EFF*T_M/K_SI:.5f}   L_heal = sqrt(k t / h) = {L_HEAL*1e6:.1f} um")
    fig_timescales()

    V, T, msh = solve_fin()
    r, dt = radial_profile(V, T)
    l_fit = fit_decay_length(r, dt)
    print(f"fitted decay length from the 2D fin solve = {l_fit*1e6:.1f} um "
          f"({abs(l_fit-L_HEAL)/L_HEAL*100:.2f}% from the closed form)")

    # Isolated-hot-spot resistance vs source size: one fin solve per size,
    # taking the PEAK. Sizes start at 100um so the coarsest source still spans
    # ~5 elements -- below that the patch is under-resolved and its peak is a
    # mesh artefact, not physics.
    iso_sizes_um, iso_r = [], []
    for a_um in (100, 200, 400, 800, 1600, 3200, 6400):
        _, Ta, _ = solve_fin(a_um * 1e-6, msh=msh)
        iso_sizes_um.append(a_um)
        iso_r.append(float(Ta.x.array.max()))
        print(f"  isolated spot {a_um:>5} um -> {iso_r[-1]:10.2f} K/W   "
              f"(1/(hA) would be {1.0/(H_EFF*(a_um*1e-6)**2):10.2f})")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.4))

    # ---- panel 1: the profile, against the closed form ----
    far = r > SRC_A_M
    ax1.semilogy(r[far] * 1e6, dt[far], lw=2.4, color="#1f77b4", label="2D fin solve (dolfinx)")
    # K0e is exp-scaled, which avoids underflow at large r/L.
    analytic = P_W / (2 * np.pi * K_SI * T_M) * k0e(r[far] / L_HEAL) * np.exp(-r[far] / L_HEAL)
    ax1.semilogy(r[far] * 1e6, analytic, "--", lw=1.8, color="#d62728",
                 label="$\\frac{P}{2\\pi k t}K_0(r/L)$,  $L=\\sqrt{kt/h}$")
    ax1.axvline(L_HEAL * 1e6, color="#2e7d32", lw=1.4, ls=":")
    ax1.text(L_HEAL * 1e6 * 1.1, dt[far].max() * 0.3,
             f"$L$ = {L_HEAL*1e6:.0f} µm", color="#2e7d32", fontsize=11, fontweight="bold")
    ax1.axvline(1.2, color="#8a6500", lw=1.4, ls="--")
    ax1.text(1.35, dt[far].min() * 3, "bitcell pitch\n1.2 µm", color="#8a6500", fontsize=9)
    ax1.annotate("", xy=(1.2, dt[far].min() * 30), xytext=(L_HEAL * 1e6, dt[far].min() * 30),
                 arrowprops=dict(arrowstyle="<->", color="#555", lw=1.3))
    ax1.text(np.sqrt(1.2 * L_HEAL * 1e6), dt[far].min() * 45,
             f"{L_HEAL*1e6/1.2:.0f}× — silicon is isothermal across all of this",
             ha="center", fontsize=9, color="#444")
    ax1.set_xscale("log")
    ax1.set_xlim(1, DOMAIN_M / 2 * 1e6)
    ax1.set_xlabel("distance from the hot spot (µm)")
    ax1.set_ylabel("temperature rise per watt (K/W)")
    ax1.set_title(f"How far heat spreads before it escapes\n"
                  f"fitted decay length {l_fit*1e6:.0f} µm vs "
                  f"$\\sqrt{{kt/h}}$ = {L_HEAL*1e6:.0f} µm", fontsize=12)
    ax1.grid(alpha=0.3, which="both")
    ax1.legend(fontsize=9.5)

    # ---- panel 2: the three scales on one axis ----
    files = sorted(ISO.glob("pad_*.json"), key=lambda f: float(f.stem.split("_")[1]))
    ax2.axvspan(0.05, 1.0, color="#e6a700", alpha=0.16)
    ax2.axvspan(1.0, L_HEAL * 1e6, color="#1f77b4", alpha=0.10)
    ax2.axvspan(L_HEAL * 1e6, 1e4, color="#d62728", alpha=0.12)
    for xc, lab, col in ((0.25, "device\n0.1–1 µm", "#8a6500"),
                         (25.0, "block — isothermal\n1 µm – 0.6 mm", "#15507f"),
                         (2.2e3, "chip\n> 0.6 mm", "#a03030")):
        ax2.text(xc, 4e7, lab, ha="center", fontsize=9.5, color=col, fontweight="bold")

    if files:
        pads = np.array([json.loads(f.read_text())["pad_um"] for f in files])
        dts = np.array([json.loads(f.read_text())["tmax_k"] - 300.0 for f in files])
        side = np.sqrt((1.2 + 2 * pads) * (1.58 + 2 * pads))
        r_meas = dts / 1.1842e-6
        ax2.loglog(side, r_meas, "o-", color="#1f77b4", lw=2.2, ms=7,
                   label="3D GDS bitcell, adiabatic walls\n(= array-periodic: all cells hot)")

    sides = np.logspace(0, 4, 200)
    ax2.loglog(sides, 1.0 / (H_EFF * (sides * 1e-6) ** 2), "--", color="#555555", lw=1.6,
               label="$1/(hA)$ — every cell equally hot")
    ax2.loglog(iso_sizes_um, iso_r, "s-", color="#d62728", lw=2.2, ms=7,
               label="one hot spot in cold silicon (fin solve)")
    ax2.axvline(L_HEAL * 1e6, color="#2e7d32", lw=1.4, ls=":")

    ax2.set_xlim(0.05, 1e4)
    ax2.set_ylim(5e-1, 3e8)
    ax2.set_xlabel("size of the hot region (µm)")
    ax2.set_ylabel("thermal resistance (K/W)")
    ax2.set_title("Same silicon, two different questions\n"
                  "how much it heats depends on how much of it is hot", fontsize=12)
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=8.5, loc="lower left")

    fig.suptitle("Thermal healing length: why transistor detail is millikelvin "
                 "and chip temperature is tens of kelvin", fontsize=13.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig6f_healing_length.png", dpi=165)
    plt.close(fig)
    print(f"wrote {OUT}/fig6f_healing_length.png")


if __name__ == "__main__":
    main()
