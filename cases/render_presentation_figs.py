"""Presentation figures for the opening slides -- the validation ladder that
has to be believed before the SRAM results mean anything.

    slide 2  numerics      FEM vs a closed-form analytic solution, and the
                           order of convergence as the mesh is refined
    slide 3  mesh          grid independence of the real 3D bitcell solve
    slide 4  materials     thin-film Si k(d) against MEASURED data
    slide 5  conservation  delivered power vs intended power

Each figure is generated from the real code path it is claiming to validate
(physics/forms.py, spec/materials.py, ...), not from numbers retyped into a
plotting script -- except where a value is explicitly labelled as measured
output of a previous run, which is cited to VALIDATION.md.

    python cases/render_presentation_figs.py [--only 2]
"""

import argparse
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

import ufl
import dolfinx.mesh as dmesh
from dolfinx.fem import Function, assemble_scalar, form, functionspace
from dolfinx.fem.petsc import LinearProblem

from mesh.build import FACET_BOTTOM
from physics.forms import steady_form
from solve.steady import PETSC_OPTIONS
from spec.chip import BoundaryConditions
from spec.materials import MATERIALS, k_si_thin_film_w_mk

OUT = Path("out/presentation")

# Slab chosen so CONDUCTION dominates the profile, otherwise the figure
# demonstrates nothing about the conduction being validated. The two terms
# are q*t/h (Robin) and q*t^2/2k (conduction), so conduction wins only when
# t*h/2k >> 1. At the project's usual h=2e4 that needs t >> 15mm -- absurd
# for a chip -- so this uses the NEAR-ISOTHERMAL sink (h=1e8, the same
# high-performance-heat-sink regime [Oprins]' benchmark assumes), where
# t*h/2k ~ 170 and the parabola IS the signal: ~0.85 K conduction against
# ~5e-6 K of Robin offset.
T_AMB, H_EFF, K_SLAB, Q_SLAB = 300.0, 1e8, 148.0, 1e9
THICKNESS, WIDTH = 500e-6, 100e-6


def _solve_slab(ny, rtol=None):
    """The SAME steady_form / bcs / solver the chip solves use, on a uniform
    slab -- Robin-cooled bottom, adiabatic elsewhere, uniform q. Reducing the
    real code path to a case with a closed-form answer is the point; a
    bespoke reimplementation would validate nothing."""
    # Refine BOTH directions together so cells stay near-square. Refining y
    # alone drives the aspect ratio to 25:1 by ny=512 and the solve degrades
    # (measured: the L2 error stops falling around ny=128 and then blows up
    # to 4.8e-2 K), which would read as a convergence failure of the method
    # when it is really a mesh-quality artefact of the study itself.
    nx = max(2, int(round(ny * WIDTH / THICKNESS)))
    mesh = dmesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [WIDTH, THICKNESS]], [nx, ny])
    fdim = mesh.topology.dim - 1
    facets = dmesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[1], 0.0))
    facet_tags = dmesh.meshtags(mesh, fdim, facets,
                                 np.full(facets.shape, FACET_BOTTOM, dtype=np.int32))
    dg0 = functionspace(mesh, ("DG", 0))
    k = Function(dg0, name="k"); k.x.array[:] = K_SLAB
    q = Function(dg0, name="q"); q.x.array[:] = Q_SLAB
    chip = types.SimpleNamespace(
        bcs=BoundaryConditions(ambient_t_k=T_AMB, backside_h_eff=H_EFF, top_face="adiabatic"))
    opts = dict(PETSC_OPTIONS)
    if rtol is not None:
        opts["ksp_rtol"] = rtol
    V, a, L = steady_form(mesh, facet_tags, k, q, chip)
    T = LinearProblem(a, L, petsc_options_prefix=f"pres_slab{ny}_{rtol}_",
                       petsc_options=opts).solve()
    y = V.tabulate_dof_coordinates()[:, 1]
    # T(y) = T_amb + q*t/h + q*(t*y - y^2/2)/k   (see tests/test_solve_analytic.py)
    T_exact = T_AMB + Q_SLAB * THICKNESS / H_EFF + Q_SLAB * (THICKNESS * y - y**2 / 2) / K_SLAB

    # L2 error over the DOMAIN, not just at nodes. In 1D with constant
    # coefficients the P1 Galerkin solution is nodally exact, so a nodal
    # max-error measures the linear solver's tolerance rather than the
    # discretisation -- measured directly: it pins to rtol*rise (2.5e-7 K at
    # rtol=1e-10) and does not fall with h, giving a meaningless "order".
    # The L2 norm sees the quadratic the linear basis cannot represent
    # between nodes, and is the quantity that converges at O(h^2).
    xy = ufl.SpatialCoordinate(mesh)
    T_ufl = (T_AMB + Q_SLAB * THICKNESS / H_EFF
             + Q_SLAB * (THICKNESS * xy[1] - xy[1] ** 2 / 2) / K_SLAB)
    l2 = np.sqrt(assemble_scalar(form((T - T_ufl) ** 2 * ufl.dx))
                 / (WIDTH * THICKNESS))
    return y, T.x.array.copy(), T_exact, float(l2)


def slide2_numerics():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    y, T_fem, T_exact, _ = _solve_slab(64)
    order = np.argsort(y)
    ys = y[order] * 1e6
    ax1.plot(ys, T_exact[order] - T_AMB, "-", color="#111", lw=2.5,
             label="analytic $T(y)$", zorder=2)
    ax1.plot(ys[::7], T_fem[order][::7] - T_AMB, "o", ms=7, mfc="none", mec="#d62728",
             mew=1.8, label="FEM solution", zorder=3)
    ax1.set_xlabel("depth y (um)"); ax1.set_ylabel("temperature rise above ambient (K)")
    rise = T_exact.max() - T_AMB
    err = np.max(np.abs(T_fem - T_exact))
    ax1.set_title(f"Uniform slab: FEM vs closed form\nnodal error {err:.1e} K on a "
                  f"{rise:.3f} K rise ({err/rise:.0e} relative)", fontsize=11)
    ax1.legend(); ax1.grid(alpha=0.3)

    # tight solver tolerance so DISCRETISATION, not the KSP, sets the error
    # Range chosen where DISCRETISATION dominates. Past ~h=30nm the L2 error
    # is ~1e-11 of the temperature rise and hits the linear-solver/roundoff
    # floor -- it stops falling and then rises, which is the study bottoming
    # out, not the method failing (measured: 3.2e-8 K at ny=80 -> 1.1e-7 K at
    # ny=160).
    nys = [10, 20, 40, 80]
    hs, errs = [], []
    for ny in nys:
        _y, _tf, _te, l2 = _solve_slab(ny, rtol=1e-12)
        hs.append(THICKNESS / ny)
        errs.append(l2)
    hs, errs = np.array(hs), np.array(errs)
    ax2.loglog(hs * 1e9, errs, "o-", color="#1f77b4", lw=2, ms=7, label="measured $L_2$ error")
    ref = errs[0] * (hs / hs[0]) ** 2
    ax2.loglog(hs * 1e9, ref, "--", color="#7f7f7f", lw=1.6,
               label=r"ideal 2nd order ($\propto h^2$)")
    p = np.polyfit(np.log(hs), np.log(errs), 1)[0]
    ax2.set_xlabel("element size h (nm)"); ax2.set_ylabel("$L_2$ error norm (K)")
    ax2.set_title(f"Convergence under mesh refinement\nmeasured order = {p:.2f} "
                  f"(P1 elements: 2 is the theoretical rate)", fontsize=11)
    ax2.legend(); ax2.grid(alpha=0.3, which="both")

    fig.suptitle("Does the solver solve the equation it claims to?", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "slide2_numerics.png", dpi=150)
    print(f"  slide2: max error {err:.3e} K ({err/rise*100:.4f}% of rise), "
          f"observed convergence order {p:.2f}")


def slide3_mesh():
    # Measured in this project's own 3D convergence sweep (VALIDATION.md §4,
    # cases/run_bitcell_compact.py --refine). Quoted rather than re-run here:
    # the sweep costs ~10 min and is already recorded.
    refine = [2.0, 1.4, 1.0, 0.7]
    cells = [91619, 176143, 346785, 748829]
    dT = [31.4555, 31.4561, 31.4564, 31.4563]

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.plot(cells, dT, "o-", color="#2a9d8f", lw=2, ms=9)
    for c, d, r in zip(cells, dT, refine):
        ax.annotate(f"refine={r}", (c, d), textcoords="offset points", xytext=(0, 11),
                    ha="center", fontsize=8.5, color="#40616b")
    span = max(dT) - min(dT)
    ax.set_ylim(min(dT) - 8 * span, max(dT) + 12 * span)
    ax.set_xscale("log")
    ax.set_xlim(min(cells) * 0.75, max(cells) * 1.5)   # room for the edge labels
    ax.set_xlabel("mesh cells (log)"); ax.set_ylabel("peak $\\Delta T$ (K)")
    ax.set_title(f"3D bitcell: grid independence\n"
                 f"{span/np.mean(dT)*100:.3f}% variation over an "
                 f"{max(cells)/min(cells):.0f}x cell-count range", fontsize=11)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "slide3_mesh_convergence.png", dpi=150)
    print(f"  slide3: dT varies {span:.4f} K = {span/np.mean(dT)*100:.3f}% over "
          f"{max(cells)/min(cells):.0f}x cells")


def slide4_materials():
    """Thin-film Si conductivity. No 500nm marker and no [Oprins] BTE point:
    the thickness sweep comes later in the deck, and the BTE comparison
    belongs with the Oprins reproduction, not here."""
    d_nm = np.logspace(np.log10(10), np.log10(20000), 300)
    k_fit = np.array([k_si_thin_film_w_mk(d / 1000.0) for d in d_nm])
    k_bulk = MATERIALS["Si_bulk"].k

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.semilogx(d_nm, k_fit, "-", color="#1f77b4", lw=2.5,
                label="$k(d)=k_{bulk}\\,/\\,(1+a\\,d^{-p})$,  $a$=72.30, $p$=0.8465")
    ax.axhline(k_bulk, color="#7f7f7f", ls=":", lw=1.6)
    ax.text(14000, k_bulk + 4, f"bulk Si  {k_bulk:.0f} W/m-K", fontsize=9,
            color="#555", ha="right")

    # The two MEASURED anchors the two-parameter fit is solved to pass
    # through exactly -- Liu & Asheghi, J. Heat Transfer 128, 75 (2006):
    # electrical-resistance thermometry on suspended single-crystal SOI
    # bridges. Real measurements, not a curve fitted for convenience.
    ax.plot([20.0, 100.0], [22.0, 60.0], "o", ms=13, mfc="#d62728", mec="k", mew=1.2,
            zorder=5, label="measured: Liu & Asheghi 2006 (SOI bridges)")

    ax.set_xlabel("Si film thickness (nm)")
    ax.set_ylabel("thermal conductivity (W/m-K)")
    ax.set_title("Thin-film Si conductivity", fontsize=13)
    ax.set_ylim(0, k_bulk * 1.18)
    ax.legend(fontsize=9, loc="upper left"); ax.grid(alpha=0.3, which="both")
    fig.text(0.5, 0.008,
             "2-parameter fit solved (not regressed) through both measured points; "
             "$k\\to k_{bulk}$ as $d\\to\\infty$.  spec/materials.py::k_si_thin_film_w_mk",
             ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(OUT / "slide4_materials.png", dpi=150)
    print(f"  slide4: 20nm->{k_si_thin_film_w_mk(0.02):.1f}, 100nm->{k_si_thin_film_w_mk(0.1):.1f}, "
          f"bulk {k_bulk:.0f} W/m-K")


# ---------------------------------------------------------------- slide 2b
# Harder validation than a single uniform slab: a LAYERED stack (tests
# material interfaces, which the real chip has at every band boundary) and a
# TRANSIENT case (tests the time integration, not just the steady operator).

LAYERS = [("Si_bulk", 148.0, 200e-6), ("SiO2", 1.4, 2e-6), ("Si_bulk", 148.0, 50e-6)]
Q_TOP = 2e8          # volumetric source, top layer only
# Near-isothermal sink again, so the profile is CONDUCTION, not a Robin
# offset: at h=2e4 the Robin term is 0.5 K against ~0.03 K of conduction and
# the interfaces being tested would be invisible.
H_ML = 1e8
# Layer boundaries must land on CELL boundaries. The DG0 material lookup is
# by cell midpoint, so with a uniform mesh the 2um oxide's effective
# thickness quantises to the cell size -- at 0.42um cells that is a ~20%
# error in the layer that carries half the resistance, which showed up as a
# 3% temperature error and a fake convergence order of 1.19.
ML_NYS = [126, 252, 504, 1008]      # h = 2um/m exactly divides every boundary


def _solve_multilayer(ny=504):
    """Stack of layers with a source in the TOP layer only, Robin-cooled at
    the base, adiabatic elsewhere. All the generated power must cross every
    layer below, so the exact profile is piecewise-linear through the passive
    layers with a parabola in the source layer -- a direct test that
    interface conditions (continuity of T and of flux through a 100x
    conductivity jump) are handled right."""
    total_t = sum(t for _n, _k, t in LAYERS)
    mesh = dmesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [total_t / 20, total_t]],
                                   [6, ny])
    fdim = mesh.topology.dim - 1
    facets = dmesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[1], 0.0))
    facet_tags = dmesh.meshtags(mesh, fdim, facets,
                                 np.full(facets.shape, FACET_BOTTOM, dtype=np.int32))
    dg0 = functionspace(mesh, ("DG", 0))
    kf, qf = Function(dg0, name="k"), Function(dg0, name="q")
    mid = dg0.tabulate_dof_coordinates()[:, 1]
    edges, z = [], 0.0
    for _n, _k, t in LAYERS:
        edges.append((z, z + t)); z += t
    for (z0, z1), (_n, kv, _t) in zip(edges, LAYERS):
        sel = (mid >= z0 - 1e-15) & (mid < z1 + 1e-15)
        kf.x.array[sel] = kv
    qf.x.array[:] = 0.0
    qf.x.array[mid >= edges[-1][0] - 1e-15] = Q_TOP

    chip = types.SimpleNamespace(
        bcs=BoundaryConditions(ambient_t_k=T_AMB, backside_h_eff=H_ML, top_face="adiabatic"))
    V, a, L = steady_form(mesh, facet_tags, kf, qf, chip)
    T = LinearProblem(a, L, petsc_options_prefix="pres_multi_",
                       petsc_options=PETSC_OPTIONS).solve()
    y = V.tabulate_dof_coordinates()[:, 1]

    P = Q_TOP * LAYERS[-1][2]                 # W/m^2 leaving the top layer
    # Built by CLAMPING into each layer rather than by masking: for a point
    # below a layer the penetration is 0, inside it is partial, above it is
    # the full thickness. A mask of the form (y >= z0) & (y < z1) silently
    # drops the domain's topmost node -- it is excluded from its own layer and
    # then credited with that layer's linear drop instead of its parabolic
    # one, which showed up as a single 1.7e-3 K outlier (5% of the rise) that
    # did not shrink with refinement.
    T_exact = np.full_like(y, T_AMB + P / H_ML)
    for (z0, z1), (_n, kv, t) in zip(edges, LAYERS):
        yy = np.clip(y, z0, z1) - z0
        if (z0, z1) == edges[-1]:             # source layer: parabolic
            T_exact += Q_TOP * (t * yy - yy ** 2 / 2) / kv
        else:                                  # passive layer: full flux, linear
            T_exact += P * yy / kv
    return y, T.x.array.copy(), T_exact, edges


TR_T, TR_K, TR_RHOCP, TR_H, TR_Q = 5e-6, 148.0, 2330.0 * 712.0, 20000.0, 1e9


def _solve_transient(n_steps=300):
    """Slab switched on at t=0, Robin-cooled. Biot number h*L/k = 6.8e-4 << 1,
    so the slab is isothermal through its thickness and the exact answer is
    the lumped response T(t) = T_inf*(1 - exp(-t/tau)), tau = rho*cp*L/h.
    That makes this a clean test of the TIME INTEGRATION specifically, with
    the spatial problem deliberately trivial."""
    from solve.transient import TransientHeatSolver
    mesh = dmesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [TR_T / 4, TR_T]], [3, 24])
    fdim = mesh.topology.dim - 1
    facets = dmesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[1], 0.0))
    facet_tags = dmesh.meshtags(mesh, fdim, facets,
                                 np.full(facets.shape, FACET_BOTTOM, dtype=np.int32))
    dg0 = functionspace(mesh, ("DG", 0))
    kf = Function(dg0); kf.x.array[:] = TR_K
    rcf = Function(dg0); rcf.x.array[:] = TR_RHOCP
    qf = Function(dg0); qf.x.array[:] = TR_Q
    chip = types.SimpleNamespace(
        bcs=BoundaryConditions(ambient_t_k=T_AMB, backside_h_eff=TR_H, top_face="adiabatic"))
    md = types.SimpleNamespace(mesh=mesh, cell_tags=None, facet_tags=facet_tags)

    tau = TR_RHOCP * TR_T / TR_H
    # Backward Euler is FIRST order in time, so the visible error scales with
    # dt: tau/12 leaves ~1.5%, tau/60 leaves ~0.3%. Stated rather than hidden.
    dt = tau / 60
    solver = TransientHeatSolver(md, kf, rcf, chip, dt=dt, T0=T_AMB)
    ts, tmaxs = [0.0], [T_AMB]
    t = 0.0
    for _ in range(n_steps):
        Tn = solver.step(qf.x.array, dt=dt)
        t += dt
        ts.append(t); tmaxs.append(float(Tn.x.array.max()))
    ts, tmaxs = np.array(ts), np.array(tmaxs)
    T_inf = TR_Q * TR_T / TR_H
    exact = T_AMB + T_inf * (1.0 - np.exp(-ts / tau))
    return ts, tmaxs, exact, tau


def slide2_validation():
    fig = plt.figure(figsize=(14.5, 9.2))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.26)

    # (a) what is actually being solved
    ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
    ax.set_title("What the solver solves", fontsize=12, loc="left", fontweight="bold")
    ax.text(0.0, 0.88, "Steady state", fontsize=11, fontweight="bold", color="#1f4e79")
    ax.text(0.03, 0.76, r"$-\nabla\!\cdot\!(k(\mathbf{x})\,\nabla T) = q(\mathbf{x})$",
            fontsize=15)
    ax.text(0.0, 0.60, "Time dependent", fontsize=11, fontweight="bold", color="#1f4e79")
    ax.text(0.03, 0.47, r"$\rho c_p\,\dfrac{\partial T}{\partial t}"
                        r"-\nabla\!\cdot\!(k\nabla T) = q$", fontsize=15)
    ax.text(0.0, 0.31, "Boundary conditions", fontsize=11, fontweight="bold", color="#1f4e79")
    ax.text(0.03, 0.20, r"Robin (sink):   $-k\nabla T\!\cdot\!\mathbf{n}"
                        r" = h_{\mathrm{eff}}\,(T - T_{amb})$", fontsize=11.5)
    ax.text(0.03, 0.10, r"Natural (cut faces):   $-k\nabla T\!\cdot\!\mathbf{n} = 0$",
            fontsize=11.5)
    ax.text(0.03, 0.00, r"Radiating (open far field):  $-k\nabla T\!\cdot\!\mathbf{n}"
                        r" = (k/r)\,(T - T_{amb})$", fontsize=11.5)
    ax.text(0.0, -0.13, "P1 continuous Galerkin;  backward Euler in time",
            fontsize=9.5, style="italic", color="#555")

    # (b) layered stack -- interfaces
    ax = fig.add_subplot(gs[0, 1])
    y, T_fem, T_exact, edges = _solve_multilayer()
    o = np.argsort(y)
    ax.plot(T_exact[o] - T_AMB, y[o] * 1e6, "-", color="#111", lw=2.5, label="analytic")
    ax.plot(T_fem[o][::260] - T_AMB, y[o][::260] * 1e6, "o", ms=6, mfc="none",
            mec="#d62728", mew=1.7, label="FEM")
    for z0, z1 in edges[:-1]:
        ax.axhline(z1 * 1e6, color="#8899a6", lw=1.0, ls="--")
    ax.text(0.03, edges[1][0] * 1e6 * 1.06, "SiO$_2$ interlayer  (k=1.4)",
            transform=ax.get_yaxis_transform(), ha="left", fontsize=9, color="#4a5b66")
    err = np.max(np.abs(T_fem - T_exact)) / (T_exact.max() - T_AMB)
    ax.set_xlabel("temperature rise (K)"); ax.set_ylabel("height (um)")
    ax.set_title(f"Layered stack: 100x conductivity jumps\nmax error "
                 f"{err:.1e} relative", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (c) order of convergence -- measured on the UNIFORM slab, whose exact
    # solution is a parabola the P1 basis genuinely cannot represent. The
    # layered case is a poor convergence test precisely because it is so
    # accurate: on a layer-aligned mesh its exact solution is piecewise
    # LINEAR through the passive layers, which P1 reproduces essentially
    # exactly, leaving almost no discretisation error to converge (measured
    # order 0.95 at an already-1e-5 relative error, i.e. at the floor).
    ax = fig.add_subplot(gs[1, 0])
    nys, errs, hs = [10, 20, 40, 80], [], []
    for ny in nys:
        _y, _tf, _te, l2 = _solve_slab(ny, rtol=1e-12)
        hs.append(THICKNESS / ny)
        errs.append(l2)
    hs, errs = np.array(hs), np.array(errs)
    ax.loglog(hs * 1e6, errs, "o-", color="#1f77b4", lw=2, ms=7, label="measured $L_2$ error")
    ax.loglog(hs * 1e6, errs[0] * (hs / hs[0]) ** 2, "--", color="#7f7f7f", lw=1.5,
              label=r"ideal 2nd order ($\propto h^2$)")
    order = np.polyfit(np.log(hs), np.log(errs), 1)[0]
    ax.set_xlabel("element size (um)"); ax.set_ylabel("$L_2$ error norm (K)")
    ax.set_title(f"Order of convergence (uniform slab)\nmeasured {order:.2f}  "
                 f"(2 = theoretical for P1)", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

    # (d) transient
    ax = fig.add_subplot(gs[1, 1])
    ts, tmaxs, exact, tau = _solve_transient()
    ax.plot(ts * 1e3, exact - T_AMB, "-", color="#111", lw=2.5, label="analytic (lumped)")
    ax.plot(ts[::12] * 1e3, tmaxs[::12] - T_AMB, "o", ms=6, mfc="none", mec="#2a9d8f",
            mew=1.8, label="FEM, backward Euler")
    ax.axvline(tau * 1e3, color="#888", ls=":", lw=1.3)
    ax.text(tau * 1e3 * 1.06, 0.1 * (exact.max() - T_AMB), r"$\tau=\rho c_p L/h$",
            fontsize=9, color="#555")
    terr = np.max(np.abs(tmaxs - exact)) / (exact.max() - T_AMB)
    ax.set_xlabel("time (ms)"); ax.set_ylabel("temperature rise (K)")
    ax.set_title(f"Transient: step response vs exact\nmax error {terr:.1e} relative "
                 f"(Bi={TR_H*TR_T/TR_K:.1e})", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle("Solver verification against closed-form solutions", fontsize=14)
    fig.savefig(OUT / "slide2_validation.png", dpi=150, bbox_inches="tight")
    print(f"  slide2: layered {err:.2e} rel | slab order {order:.2f} | "
          f"transient {terr:.2e} rel")


# ---------------------------------------------------------------- slide 5
# Values measured by this project's own audit (VALIDATION.md sections 2-3),
# via post/budget.py::power_balance on each solved case. Quoted rather than
# re-solved here: the gallery is ~5 min of compute and is already recorded.
CONSERVATION = [
    ("crowbar",        "worst",   1.184203, 1.184203, 1.345e-10),
    ("read",           "worst",   1.295366, 1.295366, 2.163e-11),
    ("crowbar",        "average", 0.074013, 0.074013, 5.825e-10),
    ("read",           "average", 0.080960, 0.080960, 3.825e-09),
]


def slide5_conservation():
    fig = plt.figure(figsize=(12.6, 6.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.22)

    ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
    rows = [["operating point", "case", "intended\n(uW)", "delivered\n(uW)", "ratio"]]
    for name, scen, intended, delivered, _r in CONSERVATION:
        rows.append([name, scen, f"{intended:.6f}", f"{delivered:.6f}",
                     f"{delivered/intended:.6f}"])
    tb = ax.table(cellText=rows[1:], colLabels=rows[0], loc="upper center",
                   cellLoc="center", colWidths=[0.30, 0.20, 0.20, 0.20, 0.18])
    tb.auto_set_font_size(False); tb.set_fontsize(10); tb.scale(1, 2.2)
    for j in range(len(rows[0])):
        tb[0, j].set_facecolor("#1f4e79"); tb[0, j].set_text_props(color="w", weight="bold")
    for i in range(1, len(rows)):
        tb[i, 4].set_facecolor("#e8f4ea"); tb[i, 4].set_text_props(weight="bold")
    ax.set_title("Does the power we intended actually reach the mesh?", fontsize=12,
                 fontweight="bold", pad=26, loc="left")

    ax = fig.add_subplot(gs[0, 1]); ax.axis("off")
    ax.set_title("Two separate identities", fontsize=12, fontweight="bold",
                 loc="left", pad=26)
    ax.text(0.0, 0.80, "1.  Delivered vs intended", fontsize=11, fontweight="bold",
            color="#1f4e79")
    ax.text(0.03, 0.70, r"$\int_\Omega q\,dV \;=\; \sum_i P_i^{\mathrm{SPICE}}$", fontsize=13)
    ax.text(0.03, 0.60, "exact to 1.000000 in every case, and per source\n"
                        "region: meshed volume / nominal volume = 1.000000",
            fontsize=9.5, color="#333")
    ax.text(0.0, 0.44, "2.  Generation vs boundary outflow", fontsize=11,
            fontweight="bold", color="#1f4e79")
    ax.text(0.03, 0.34, r"$\int_\Omega q\,dV \;=\; \oint_{\partial\Omega}"
                        r" h\,(T-T_{amb})\,dS$", fontsize=13)
    ax.text(0.03, 0.24, "residual $10^{-9}$ - $10^{-11}$ relative", fontsize=9.5, color="#333")
    ax.text(0.0, 0.07,
            "Identity 2 holds for ANY q -- it is the Galerkin statement at $v=1$,\n"
            "so it cannot detect injecting the wrong amount of power.\n"
            "Identity 1 is the one that can, and it was missing until this audit.",
            fontsize=9.5, color="#7a3b12",
            bbox=dict(boxstyle="round,pad=0.5", fc="#fdf3e7", ec="#e0b080"))

    fig.suptitle("Energy conservation: delivered power, not just self-consistent power",
                 fontsize=13, y=1.02)
    fig.savefig(OUT / "slide5_conservation.png", dpi=150, bbox_inches="tight")
    print("  slide5: ratios all 1.000000; Galerkin residual 1e-9..1e-11")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", type=int, choices=[2, 3, 4, 5])
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [args.only] if args.only else [2, 3, 4, 5]
    if 2 in todo:
        slide2_validation()
    if 3 in todo:
        slide3_mesh()
    if 4 in todo:
        slide4_materials()
    if 5 in todo:
        slide5_conservation()
    print(f"\nwrote figures to {OUT}/")


if __name__ == "__main__":
    main()
