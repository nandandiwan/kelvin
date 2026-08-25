"""UFL: steady linear conduction form. Transient (theta-scheme) and
nonlinear k(T) are additive extensions of this, not a rewrite (PLAN.md).
"""

import ufl
from dolfinx.fem import functionspace

from .bcs import robin_terms


def steady_form(mesh, facet_tags, k, q, chip):
    V = functionspace(mesh, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
    dx = ufl.Measure("dx", domain=mesh)

    a = k * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    L = q * v * dx

    a_robin, l_robin = robin_terms(u, v, ds, chip)
    a = a + a_robin
    L = L + l_robin

    return V, a, L
