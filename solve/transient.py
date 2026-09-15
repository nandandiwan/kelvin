"""Transient heat conduction: backward Euler (implicit, unconditionally
stable). dt is a `dolfinx.fem.Constant`, not a baked-in Python float, so it
can change between steps (needed here: this project's own thermal time
constants span nanoseconds locally to ~28ms down to the backside heat sink
across 50um of stack — verified directly, not assumed: a fixed few-ns dt
here produced a temperature rise 1000x too small after 15ns, because 15ns is
nowhere close to the ~28ms constant that actually governs reaching steady
state through the full stack) without re-JIT-compiling the form — only a
cheap numeric re-assembly of the matrix is needed when dt changes.

    rho_cp (T^{n+1} - T^n)/dt - div(k grad T^{n+1}) = q^{n+1}

No strong Dirichlet BCs exist anywhere in this project's formulation
(everything is Robin/natural, see physics/bcs.py), so there's no
lifting/bc application to do here — unlike solve/steady.py's
LinearProblem-based one-shot solve, which hides that same fact.
"""

import ufl
from dolfinx.fem import Constant, Function, form, functionspace
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
from petsc4py import PETSc

from physics.bcs import robin_terms


class TransientHeatSolver:
    PETSC_OPTIONS = {
        "ksp_type": "cg",
        "pc_type": "hypre",
        "pc_hypre_type": "boomeramg",
        "ksp_rtol": 1e-10,
    }

    def __init__(self, mesh_data, k, rho_cp, chip, dt: float, T0: float = 300.0):
        self.mesh_data = mesh_data
        mesh = mesh_data.mesh
        self.V = functionspace(mesh, ("Lagrange", 1))
        self.T_prev = Function(self.V)
        self.T_prev.x.array[:] = T0
        self.q = Function(functionspace(mesh, ("DG", 0)))
        self._dt = Constant(mesh, PETSc.ScalarType(dt))

        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)
        dx = ufl.Measure("dx", domain=mesh)
        ds = ufl.Measure("ds", domain=mesh, subdomain_data=mesh_data.facet_tags)

        mass = (rho_cp / self._dt) * u * v * dx
        stiffness = k * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
        a_robin, l_robin = robin_terms(u, v, ds, chip, k=k)
        a = mass + stiffness + a_robin
        L = (rho_cp / self._dt) * self.T_prev * v * dx + self.q * v * dx + l_robin

        self._a_form = form(a)
        self._L_form = form(L)

        self._A = assemble_matrix(self._a_form)
        self._A.assemble()
        self._b = create_vector(self.V)

        self._ksp = PETSc.KSP().create(mesh.comm)
        self._ksp.setOperators(self._A)
        prefix = "thermals_transient_"
        self._ksp.setOptionsPrefix(prefix)
        opts = PETSc.Options()
        for key, val in self.PETSC_OPTIONS.items():
            opts[f"{prefix}{key}"] = val
        self._ksp.setFromOptions()
        self._dt_value = dt

    def step(self, q_array, dt: float = None) -> Function:
        """Advance one step using `q_array` (a full DG0 W/m^3 array) as this
        step's source term. Pass `dt` to change the timestep (e.g. a
        geometrically growing schedule spanning ns to ms) -- cheap: just a
        constant update + matrix re-assembly, no re-JIT."""
        if dt is not None and dt != self._dt_value:
            self._dt.value = dt
            self._dt_value = dt
            self._A = assemble_matrix(self._a_form)
            self._A.assemble()
            self._ksp.setOperators(self._A)

        self.q.x.array[:] = q_array
        with self._b.localForm() as loc:
            loc.set(0)
        assemble_vector(self._b, self._L_form)
        self._b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

        T_new = Function(self.V)
        T_new.name = "T"
        self._ksp.solve(self._b, T_new.x.petsc_vec)
        T_new.x.scatter_forward()
        self.T_prev.x.array[:] = T_new.x.array
        return T_new
