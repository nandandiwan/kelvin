"""Analytic verification (PLAN.md's "numeric gates", first one): a uniform
single-material slab, Robin-cooled on one face, adiabatic elsewhere, with a
uniform volumetric source. This is exactly the BC/source structure the real
chip solve uses (Robin backside + adiabatic top/sides + volumetric q),
reduced to one material and one dimension of variation, so it's a direct
check of physics/forms.py + physics/bcs.py + solve/steady.py end to end —
not a bespoke reimplementation.

Derivation (y=0 Robin, y=t adiabatic, uniform k, uniform q):
  k*T'' = -q,  T'(t) = 0 (adiabatic top).
  Energy balance: all generated power q*t must exit through the one open
  face, so q*t = h*(T(0) - T_amb).
  => T(y) = T_amb + q*t/h + q*(t*y - y^2/2)/k
"""

import sys
import types
from pathlib import Path

import numpy as np
from mpi4py import MPI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dolfinx.mesh as dmesh
from dolfinx.fem import Function, functionspace
from dolfinx.fem.petsc import LinearProblem

from mesh.build import FACET_BOTTOM
from physics.forms import steady_form
from solve.steady import PETSC_OPTIONS
from spec.chip import BoundaryConditions

T_AMB = 300.0
H_EFF = 20000.0
K = 148.0
Q = 1e10
THICKNESS = 5e-6
WIDTH = 1e-6


def _build_slab_mesh():
    mesh = dmesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [WIDTH, THICKNESS]], [4, 64])
    fdim = mesh.topology.dim - 1
    bottom_facets = dmesh.locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[1], 0.0))
    values = np.full(bottom_facets.shape, FACET_BOTTOM, dtype=np.int32)
    facet_tags = dmesh.meshtags(mesh, fdim, bottom_facets, values)
    return mesh, facet_tags


def test_uniform_slab_matches_analytic_profile():
    mesh, facet_tags = _build_slab_mesh()

    dg0 = functionspace(mesh, ("DG", 0))
    k = Function(dg0, name="k")
    k.x.array[:] = K
    q = Function(dg0, name="q")
    q.x.array[:] = Q

    bcs = BoundaryConditions(ambient_t_k=T_AMB, backside_h_eff=H_EFF, top_face="adiabatic")
    chip = types.SimpleNamespace(bcs=bcs)

    V, a, L = steady_form(mesh, facet_tags, k, q, chip)
    problem = LinearProblem(
        a, L, petsc_options_prefix="thermals_test_slab_", petsc_options=PETSC_OPTIONS
    )
    T = problem.solve()

    y = V.tabulate_dof_coordinates()[:, 1]
    T_analytic = T_AMB + Q * THICKNESS / H_EFF + Q * (THICKNESS * y - y**2 / 2) / K

    rise = np.max(T_analytic) - T_AMB
    err = np.max(np.abs(T.x.array - T_analytic))
    assert err < 1e-3 * rise, f"max error {err} vs analytic rise {rise}"
