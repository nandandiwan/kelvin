"""Serial FEM operator export for thermal component-mode validation.

Applies the same periodic interpolation to mass, stiffness, and loads as the
full solver. Artificial slave diagonals are never included in diffusion modes.
MPI export is rejected: gathering a chip-scale operator to one rank is not a
scalable reduction strategy. Distributed basis construction is separate work.
"""

from dataclasses import dataclass

import numpy as np
from scipy import sparse
import ufl
from dolfinx import fem
from dolfinx.fem import petsc as fem_petsc

from physics.bcs import robin_terms
from physics.periodic import build_periodic_constraint


@dataclass
class ThermalOperators:
    mass: sparse.csc_matrix
    stiffness: sparse.csc_matrix
    prolongation: sparse.csc_matrix
    space: object
    independent_dofs: np.ndarray

    def load(self, source):
        """Assemble a DG0 volumetric source into independent nodal watts."""
        if source.function_space.mesh is not self.space.mesh:
            raise ValueError("Source must use the operator mesh")
        vector = fem_petsc.assemble_vector(fem.form(source * ufl.TestFunction(self.space) * ufl.dx))
        result = self.prolongation.T @ vector.array.copy()
        vector.destroy()
        return np.asarray(result)

    def reconstruct(self, independent_values):
        return np.asarray(self.prolongation @ independent_values)


def assemble_thermal_operators(mesh_data, conductivity, heat_capacity, chip):
    mesh = mesh_data.mesh
    if mesh.comm.size != 1:
        raise ValueError("Reference mode export supports one MPI rank only")
    space = fem.functionspace(mesh, ("Lagrange", 1))
    mpc = build_periodic_constraint(space, chip)
    if mpc is not None:
        space = mpc.function_space
    n = space.dofmap.index_map.size_local
    slaves = np.array([], dtype=int) if mpc is None else mpc.slaves[:mpc.num_local_slaves]
    independent = np.setdiff1d(np.arange(n), slaves)
    lookup = {int(dof): i for i, dof in enumerate(independent)}
    rows, columns, values = list(independent), list(range(len(independent))), [1.0] * len(independent)
    if mpc is not None:
        coefficients, offsets = mpc.coefficients()
        for slave in slaves:
            for master, weight in zip(mpc.masters.links(slave), coefficients[offsets[slave]:offsets[slave + 1]]):
                rows.append(int(slave))
                columns.append(lookup[int(master)])
                values.append(float(weight))
    prolongation = sparse.csc_matrix((values, (rows, columns)), shape=(n, len(independent)))
    u, v = ufl.TrialFunction(space), ufl.TestFunction(space)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=mesh_data.facet_tags)
    robin, _ = robin_terms(u, v, ds, chip, k=conductivity)
    forms = [heat_capacity * u * v * ufl.dx,
             conductivity * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + robin]
    operators = []
    for expression in forms:
        matrix = fem_petsc.assemble_matrix(fem.form(expression))
        matrix.assemble()
        indptr, indices, data = matrix.getValuesCSR()
        full = sparse.csr_matrix((data.copy(), indices.copy(), indptr.copy()), shape=(n, n))
        operators.append((prolongation.T @ full @ prolongation).tocsc())
        matrix.destroy()
    return ThermalOperators(*operators, prolongation, space, independent)
