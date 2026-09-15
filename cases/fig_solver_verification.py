"""Figure: solver verification on a meshed 3D slab.

Three panels, meant to sit next to your own equations:
    (a) the 3D mesh actually being solved, with the BCs marked
    (b) steady  T(z)        vs the closed-form profile
    (c) transient T(t, z0)  vs the exact series solution

Geometry: a uniform slab, Robin-cooled on the bottom face, adiabatic on the
other five, with a uniform volumetric source. Solved on a genuine 3D
hexahedral mesh; the exact solution is 1-D in z, so any disagreement is the
3D assembly's fault and nothing else.

Exact transient (theta = T - T_amb, alpha = k/rho.cp):
    theta(z,t) = theta_ss(z) - sum_n A_n cos(lam_n (L-z)) exp(-alpha lam_n^2 t)
with lam_n L the roots of  x tan(x) = Bi = hL/k, and A_n the projection of
theta_ss onto each eigenfunction. Reduces to the lumped exponential only as
Bi -> 0; here Bi = 1, so the slab is NOT isothermal and the series is doing
real work.

    python cases/fig_solver_verification.py
"""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from mpi4py import MPI
from scipy.optimize import brentq

import dolfinx.mesh as dmesh
from dolfinx.fem import Function, functionspace
from dolfinx.fem.petsc import LinearProblem

from mesh.build import FACET_BOTTOM
from physics.forms import steady_form
from solve.steady import PETSC_OPTIONS
from solve.transient import TransientHeatSolver
from spec.chip import BoundaryConditions

pv.OFF_SCREEN = True
OUT = Path("out/presentation")

# Layered slab, so the conductivity map is meaningful and the solve is
# genuinely heterogeneous: Si / SiO2 / Si with a 100x conductivity contrast
# across each interface, source in the top layer only.
#   (name, k [W/m-K], rho*cp [J/m^3-K], thickness [m])
LAYERS = [("Si", 148.0, 2330.0 * 712.0, 120e-6),
          ("SiO2", 1.4, 2650.0 * 680.0, 10e-6),
          ("Si", 148.0, 2330.0 * 712.0, 70e-6)]
L = sum(t for *_x, t in LAYERS)
W = 80e-6             # lateral extent
T_AMB = 300.0
H_EFF = 7.4e5         # Robin sink; slab is NOT isothermal through its thickness
Q = 4.0e8             # uniform source in the TOP layer only
NZ, NXY = 60, 10


def _edges():
    z, out = 0.0, []
    for _n, _k, _rc, t in LAYERS:
        out.append((z, z + t)); z += t
    return out


EDGES = _edges()
P_FLUX = Q * LAYERS[-1][3]        # W/m^2 crossing every layer below the source


def theta_ss(z):
    """Exact steady rise: the full flux P crosses each passive layer (linear
    drop), and the source layer carries a parabola. Built by clamping into
    each layer so the domain's top node is handled correctly."""
    z = np.atleast_1d(z).astype(float)
    out = np.full_like(z, P_FLUX / H_EFF)
    for (z0, z1), (_n, kv, _rc, t) in zip(EDGES, LAYERS):
        zz = np.clip(z, z0, z1) - z0
        if (z0, z1) == EDGES[-1]:
            out += Q * (t * zz - zz ** 2 / 2) / kv
        else:
            out += P_FLUX * zz / kv
    return out


def theta_reference(z_probe, t_eval, n=4000):
    """Independent 1-D reference for the TRANSIENT on the layered slab.

    A closed-form series exists only for a uniform slab; across layers the
    eigenproblem couples through the interfaces and has no tidy form. So this
    is an independent NUMERICAL reference instead -- finite-volume in space
    with harmonic-mean interface conductances, integrated by scipy's BDF, i.e.
    a different discretisation AND a different time integrator from the code
    under test. Labelled as such rather than called 'analytic'.
    """
    from scipy.integrate import solve_ivp
    zc = (np.arange(n) + 0.5) * (L / n)
    dz = L / n
    kc = np.zeros(n); rcc = np.zeros(n); qc = np.zeros(n)
    for (z0, z1), (_nm, kv, rcv, _t) in zip(EDGES, LAYERS):
        m = (zc >= z0) & (zc < z1)
        kc[m] = kv; rcc[m] = rcv
    qc[zc >= EDGES[-1][0]] = Q
    k_face = 2 * kc[:-1] * kc[1:] / (kc[:-1] + kc[1:])      # harmonic mean

    def rhs(_t, T):
        flux = np.zeros(n + 1)                              # +z direction
        flux[1:-1] = -k_face * np.diff(T) / dz
        # Robin at z=0: heat leaves DOWNWARD, so the +z-directed flux at that
        # face is the negative of the outflow. Getting this sign wrong turns
        # the sink into a source and the reference runs away.
        flux[0] = -H_EFF * (T[0] - T_AMB)
        flux[-1] = 0.0                                      # adiabatic at z=L
        return (-(np.diff(flux) / dz) + qc) / rcc

    sol = solve_ivp(rhs, (0.0, float(t_eval[-1])), np.full(n, T_AMB),
                    t_eval=t_eval, method="BDF", rtol=1e-10, atol=1e-10)
    i = int(np.argmin(np.abs(zc - z_probe)))
    return sol.y[i] - T_AMB


def _build():
    mesh = dmesh.create_box(MPI.COMM_WORLD, [[0.0, 0.0, 0.0], [W, W, L]],
                             [NXY, NXY, NZ], cell_type=dmesh.CellType.hexahedron)
    fdim = mesh.topology.dim - 1
    facets = dmesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[2], 0.0))
    tags = dmesh.meshtags(mesh, fdim, facets,
                           np.full(facets.shape, FACET_BOTTOM, dtype=np.int32))
    dg0 = functionspace(mesh, ("DG", 0))
    k = Function(dg0); rc = Function(dg0); q = Function(dg0)
    zc = dg0.tabulate_dof_coordinates()[:, 2]
    q.x.array[:] = 0.0
    for (z0, z1), (_nm, kv, rcv, _t) in zip(EDGES, LAYERS):
        m = (zc >= z0) & (zc < z1)
        k.x.array[m] = kv
        rc.x.array[m] = rcv
    q.x.array[zc >= EDGES[-1][0]] = Q
    chip = types.SimpleNamespace(
        bcs=BoundaryConditions(ambient_t_k=T_AMB, backside_h_eff=H_EFF, top_face="adiabatic"))
    md = types.SimpleNamespace(mesh=mesh, cell_tags=None, facet_tags=tags)
    return md, k, rc, q, chip


def render_mesh(md, k):
    """The actual mesh, with the Robin face marked -- so the geometry being
    solved is visible rather than described."""
    mesh = md.mesh
    n = mesh.topology.index_map(mesh.topology.dim).size_local
    mesh.topology.create_connectivity(mesh.topology.dim, 0)
    conn = mesh.geometry.dofmap.reshape(n, -1)
    cells = np.hstack([np.full((n, 1), 8, dtype=np.int64), conn]).ravel()
    grid = pv.UnstructuredGrid(cells, np.full(n, 12, dtype=np.uint8),
                                mesh.geometry.x * 1e6)
    pl = pv.Plotter(off_screen=True, window_size=(900, 1250))
    pl.background_color = "white"
    grid.cell_data["k"] = np.asarray(k.x.array, dtype=float)
    pl.add_mesh(grid, scalars="k", cmap="viridis", show_edges=True,
                edge_color="#44515c", line_width=0.8, log_scale=True,
                scalar_bar_args=dict(title="k  (W/m-K)", n_labels=4,
                                      vertical=True, position_x=0.82,
                                      position_y=0.28, height=0.45, width=0.07,
                                      title_font_size=26, label_font_size=22,
                                      color="black"))
    # The Robin face sits UNDER the block, offset slightly, so it is actually
    # visible -- drawn coincident with the bottom face it is hidden by it.
    zb = grid.bounds
    dz = (zb[5] - zb[4]) * 0.06
    sink = pv.Box(bounds=(zb[0], zb[1], zb[2], zb[3], zb[4] - dz, zb[4] - dz * 0.35))
    pl.add_mesh(sink, color="#d62728", opacity=0.95)
    # view_isometric() frames from the DATA bounds; a hand-written
    # camera_position in unit coords points the camera at the origin instead
    # and renders the block nearly edge-on.
    pl.camera.parallel_projection = True
    pl.view_isometric()
    pl.reset_camera()
    pl.camera.zoom(1.15)
    path = OUT / "_mesh3d.png"
    pl.screenshot(str(path)); pl.close()
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    md, k, rc, q, chip = _build()
    mesh_png = render_mesh(md, k)

    # ---- steady ----
    V, a, Lf = steady_form(md.mesh, md.facet_tags, k, q, chip)
    T = LinearProblem(a, Lf, petsc_options_prefix="figver_",
                       petsc_options=PETSC_OPTIONS).solve()
    xyz = V.tabulate_dof_coordinates()
    # a single vertical line of dofs through the middle of the block
    cx, cy = W / 2, W / 2
    line = (np.abs(xyz[:, 0] - cx) < 1e-12) & (np.abs(xyz[:, 1] - cy) < 1e-12)
    if line.sum() < 5:      # fall back to nearest column if no exact hit
        d = (xyz[:, 0] - cx) ** 2 + (xyz[:, 1] - cy) ** 2
        line = d < (d.min() + 1e-18)
    z_line = xyz[line, 2]
    o = np.argsort(z_line)
    z_line, T_line = z_line[o], T.x.array[line][o]
    steady_err = float(np.max(np.abs(T_line - (T_AMB + theta_ss(z_line)))))

    # ---- transient, probed at a fixed depth ----
    z0 = L                                  # top face: largest signal
    i0 = int(np.argmin(np.abs(z_line - z0)))
    idx_all = np.where(line)[0][o][i0]
    alpha_eff = LAYERS[0][1] / LAYERS[0][2]
    tau = L ** 2 / alpha_eff
    # Run to ~4 diffusion times: the slowest mode decays as 1/(alpha lam_0^2)
    # ~ 1.35 tau at Bi=1, so stopping at 1.5 tau leaves the curve still
    # visibly climbing and the steady panel's value unreached.
    dt = tau / 300
    solver = TransientHeatSolver(md, k, rc, chip, dt=dt, T0=T_AMB)
    ts, probe = [0.0], [T_AMB]
    tt = 0.0
    for _ in range(1200):
        Tn = solver.step(q.x.array, dt=dt)
        tt += dt
        ts.append(tt); probe.append(float(Tn.x.array[idx_all]))
    ts, probe = np.array(ts), np.array(probe)
    exact_t = T_AMB + theta_reference(z0, ts)
    trans_err = np.max(np.abs(probe - exact_t))

    # ---- figure ----
    fig = plt.figure(figsize=(15.5, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.05, 1.05], wspace=0.28)

    ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
    ax.imshow(mpimg.imread(mesh_png))
    ax.set_title(f"geometry: {NXY}x{NXY}x{NZ} hex mesh, coloured by k\n"
                 f"red = Robin sink, other faces adiabatic", fontsize=11)

    ax = fig.add_subplot(gs[0, 1])
    zz = np.linspace(0, L, 400)
    ax.plot(theta_ss(zz), zz * 1e6, "-", color="#111", lw=2.4, label="analytic")
    for z0e, z1e in EDGES[:-1]:
        ax.axhline(z1e * 1e6, color="#9aa7b1", lw=1.0, ls="--")
    ax.plot(T_line[::3] - T_AMB, z_line[::3] * 1e6, "o", ms=6, mfc="none",
            mec="#d62728", mew=1.7, label="FEM")
    ax.set_xlabel("$T - T_{amb}$  (K)"); ax.set_ylabel("z  (um)")
    ax.set_title("steady state", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(ts * 1e3, exact_t - T_AMB, "-", color="#111", lw=2.4,
            label="1-D reference")
    ax.plot(ts[::50] * 1e3, probe[::50] - T_AMB, "o", ms=6, mfc="none",
            mec="#2a9d8f", mew=1.7, label="FEM")
    ss_val = float(theta_ss(z0)[0])
    ax.axhline(ss_val, color="#999", ls=":", lw=1.2)
    ax.text(ts[-1] * 1e3 * 0.98, ss_val * 0.97, "steady-state value",
            fontsize=8.5, color="#666", ha="right", va="top")
    ax.set_xlabel("time  (ms)"); ax.set_ylabel(f"$T - T_{{amb}}$ at z = {z0*1e6:.0f} um  (K)")
    ax.set_title("transient", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.savefig(OUT / "fig_solver_verification.png", dpi=170, bbox_inches="tight")
    rise = float(theta_ss(L)[0])
    print(f"  layers = {[n for n, *_ in LAYERS]},  rise = {rise:.4f} K")
    print(f"  steady    max error {steady_err:.3e} K  ({steady_err/rise:.2e} relative)")
    print(f"  transient max error {trans_err:.3e} K  ({trans_err/rise:.2e} relative)")
    print(f"  wrote {OUT}/fig_solver_verification.png")


if __name__ == "__main__":
    main()
