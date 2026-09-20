"""Steady linear conduction solve: CG + hypre BoomerAMG. PETSc options
centralized here per PLAN.md's module layout.
"""

from dolfinx.fem.petsc import LinearProblem

from physics.coeffs import build_coeffs
from physics.forms import steady_form
from physics.periodic import build_periodic_constraint

PETSC_OPTIONS = {
    "ksp_type": "cg",
    "pc_type": "hypre",
    "pc_hypre_type": "boomeramg",
    "ksp_rtol": 1e-10,
    "ksp_error_if_not_converged": True,
}


def _solve(V, a, L, chip, bcs=()):
    mpc = build_periodic_constraint(V, chip, bcs)
    if mpc is None:
        problem = LinearProblem(
            a, L, bcs=list(bcs), petsc_options_prefix="thermals_steady_", petsc_options=PETSC_OPTIONS
        )
    else:
        from dolfinx_mpc import LinearProblem as PeriodicLinearProblem

        problem = PeriodicLinearProblem(
            a, L, mpc, bcs=list(bcs), petsc_options_prefix="thermals_steady_", petsc_options=PETSC_OPTIONS
        )
    T = problem.solve()
    if problem.solver.getConvergedReason() <= 0:
        raise RuntimeError(f"Steady heat solve failed: KSP reason {problem.solver.getConvergedReason()}")
    T.name = "T"
    return T


def solve_steady(mesh_data, registry, chip, source_depth_m=None, bc_builder=None):
    """Returns (T, k, q) — temperature field plus the coefficient fields it
    was solved against (both post/ and the numeric gates need k and q).
    `source_depth_m`: see physics/coeffs.build_coeffs — leave None for a
    true 3D mesh, pass the device row pitch for a 2D cross-section.

    `bc_builder`: optional `V -> list[DirichletBC]`, for BCs beyond the
    Robin terms steady_form already builds (e.g. cases/run_gds_submodel.py
    pins the lateral cut faces FACET_X0/X1/Y0/Y1 to a coarse whole-die
    solve's temperature there). MUST be a callback, not a pre-built list of
    DirichletBC objects: a DirichletBC is tied to the specific
    FunctionSpace it was constructed against, and steady_form builds its
    own `V` internally — a BC built against a second, separately-created
    `functionspace(mesh, ("Lagrange", 1))` has identical dof coordinates
    but is NOT interchangeable with steady_form's `V` in `LinearProblem`,
    and silently produces a garbage solution (verified: values off by ~11
    orders of magnitude) instead of an error. The callback gets steady_form's
    actual `V`, so this can't happen.
    """
    mesh = mesh_data.mesh
    k, rho_cp, q = build_coeffs(mesh, mesh_data.cell_tags, registry, source_depth_m=source_depth_m)
    V, a, L = steady_form(mesh, mesh_data.facet_tags, k, q, chip)

    bcs = bc_builder(V) if bc_builder is not None else []
    T = _solve(V, a, L, chip, bcs)
    return T, k, q


def solve_steady_from_fields(mesh_data, k, q, chip):
    """Same solve as solve_steady, but for callers who already have k/q as
    DG0 Functions — e.g. mesh/gds_coarse.py's tile-grid-derived coefficient
    fields, which have no cell_tags/RegionRegistry to run build_coeffs
    against (every coarse cell has its own effectively-continuous value from
    gds.upscale, not a handful of discrete tagged regions)."""
    mesh = mesh_data.mesh
    V, a, L = steady_form(mesh, mesh_data.facet_tags, k, q, chip)
    return _solve(V, a, L, chip)
