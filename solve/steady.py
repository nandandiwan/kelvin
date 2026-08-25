"""Steady linear conduction solve: CG + hypre BoomerAMG. PETSc options
centralized here per PLAN.md's module layout.
"""

from dolfinx.fem.petsc import LinearProblem

from physics.coeffs import build_coeffs
from physics.forms import steady_form

PETSC_OPTIONS = {
    "ksp_type": "cg",
    "pc_type": "hypre",
    "pc_hypre_type": "boomeramg",
    "ksp_rtol": 1e-10,
}


def solve_steady(mesh_data, registry, chip, source_depth_m=None):
    """Returns (T, k, q) — temperature field plus the coefficient fields it
    was solved against (both post/ and the numeric gates need k and q).
    `source_depth_m`: see physics/coeffs.build_coeffs — leave None for a
    true 3D mesh, pass the device row pitch for a 2D cross-section.
    """
    mesh = mesh_data.mesh
    k, rho_cp, q = build_coeffs(mesh, mesh_data.cell_tags, registry, source_depth_m=source_depth_m)
    V, a, L = steady_form(mesh, mesh_data.facet_tags, k, q, chip)

    problem = LinearProblem(
        a, L, petsc_options_prefix="thermals_steady_", petsc_options=PETSC_OPTIONS
    )
    T = problem.solve()
    T.name = "T"
    return T, k, q
