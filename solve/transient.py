"""Transient heat conduction: backward Euler (implicit, unconditionally
stable). dt is a `dolfinx.fem.Constant`, so it can change between steps
without re-JIT-compiling the form. Changing dt reassembles the matrix.

    rho_cp (T^{n+1} - T^n)/dt - div(k grad T^{n+1}) = q^{n+1}

This solver supports Robin/natural faces and optional x/y periodicity.
Periodic constraints apply to BOTH mass and stiffness and to each load
vector, with constrained temperatures reconstructed before the next step.
Strong Dirichlet data are not currently exposed by this transient API.
"""

import ufl
from dolfinx.fem import Constant, Function, form, functionspace
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
from petsc4py import PETSc

from physics.bcs import robin_terms
from physics.periodic import build_periodic_constraint


class TransientHeatSolver:
    PETSC_OPTIONS = {
        "ksp_type": "cg",
        "pc_type": "hypre",
        "pc_hypre_type": "boomeramg",
        "ksp_rtol": 1e-10,
        "ksp_error_if_not_converged": True,
    }

    def __init__(self, mesh_data, k, rho_cp, chip, dt: float, T0: float = 300.0):
        self.mesh_data = mesh_data
        mesh = mesh_data.mesh
        self.V = functionspace(mesh, ("Lagrange", 1))
        self._mpc = build_periodic_constraint(self.V, chip)
        if self._mpc is not None:
            self.V = self._mpc.function_space
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

        self._A = self._assemble_matrix()
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

    def _assemble_matrix(self):
        if self._mpc is None:
            return assemble_matrix(self._a_form)
        from dolfinx_mpc import assemble_matrix as assemble_periodic_matrix

        return assemble_periodic_matrix(self._a_form, self._mpc)

    def step(self, q_array, dt: float = None) -> Function:
        """Advance one step using `q_array` (a full DG0 W/m^3 array) as this
        step's source term. Pass `dt` to change the timestep (e.g. a
        geometrically growing schedule spanning ns to ms) -- cheap: just a
        constant update + matrix re-assembly, no re-JIT."""
        if dt is not None and dt != self._dt_value:
            self._dt.value = dt
            self._dt_value = dt
            previous_A = self._A
            self._A = self._assemble_matrix()
            self._A.assemble()
            self._ksp.setOperators(self._A)
            previous_A.destroy()

        self.q.x.array[:] = q_array
        self.q.x.scatter_forward()
        with self._b.localForm() as loc:
            loc.set(0)
        if self._mpc is None:
            assemble_vector(self._b, self._L_form)
        else:
            from dolfinx_mpc import assemble_vector as assemble_periodic_vector

            assemble_periodic_vector(self._L_form, self._mpc, b=self._b)
        self._b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

        T_new = Function(self.V)
        T_new.name = "T"
        self._ksp.solve(self._b, T_new.x.petsc_vec)
        if self._ksp.getConvergedReason() <= 0:
            raise RuntimeError(f"Transient heat solve failed: KSP reason {self._ksp.getConvergedReason()}")
        T_new.x.scatter_forward()
        if self._mpc is not None:
            self._mpc.backsubstitution(T_new)
            T_new.x.scatter_forward()
        self.T_prev.x.array[:] = T_new.x.array
        return T_new
