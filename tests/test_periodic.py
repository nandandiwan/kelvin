"""Periodic thermal boundaries, including nonzero lateral heat transfer.

The manufactured solution is deliberately asymmetric about all side faces.
A uniform slab would also satisfy insulating conditions and cannot establish
that the solver actually imposed periodicity.  All comparisons use the public
steady/transient solvers, not a separate implementation of the weak form.
"""

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import ufl
from mpi4py import MPI
from petsc4py import PETSc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dolfinx.mesh as dmesh
from dolfinx.fem import (
    Function,
    assemble_scalar,
    dirichletbc,
    form,
    functionspace,
    locate_dofs_geometrical,
)

from mesh.build import (
    FACET_BOTTOM,
    FACET_TOP,
    FACET_X0,
    FACET_X1,
    FACET_Y0,
    FACET_Y1,
)
from physics.periodic import build_periodic_constraint
from solve.steady import solve_steady_from_fields
from solve.transient import TransientHeatSolver
from spec.chip import BoundaryConditions


T_AMB = 300.0
K = 2.0
H = 3.0
Q = 50.0
RHO_CP = 4.0
AMPLITUDE = 0.2
PERIOD = np.array([1.0, 1.5, 0.4])
ORIGIN = np.array([-0.3, -0.8, -0.4])
PHASE_X = 0.37
PHASE_Y = 0.61


def _exact_temperature(x, scale=1.0):
    local = x / scale - ORIGIN[:, None]
    wx, wy = 2 * np.pi / PERIOD[:2]
    z = local[2]
    f = 1 + H / K * z - H / (2 * K * PERIOD[2]) * z**2
    background = Q * PERIOD[2] / H + Q / K * (PERIOD[2] * z - z**2 / 2)
    return (
        T_AMB
        + background
        + AMPLITUDE * np.sin(wx * local[0] + PHASE_X)
        * np.cos(wy * local[1] + PHASE_Y) * f
    )


def _exact_source(x, scale=1.0):
    local = x / scale - ORIGIN[:, None]
    wx, wy = 2 * np.pi / PERIOD[:2]
    z = local[2]
    f = 1 + H / K * z - H / (2 * K * PERIOD[2]) * z**2
    return (
        Q
        + AMPLITUDE * (K * (wx**2 + wy**2) * f + H / PERIOD[2])
        * np.sin(wx * local[0] + PHASE_X)
        * np.cos(wy * local[1] + PHASE_Y)
    ) / scale**2


def _case(n=6, repeats=(1, 1), scale=1.0, nonmatching=False):
    """Constant material, negative origin, and tags for all six faces."""
    extent = PERIOD * np.array([repeats[0], repeats[1], 1])
    mesh = dmesh.create_box(
        MPI.COMM_WORLD,
        [ORIGIN * scale, (ORIGIN + extent) * scale],
        [n * repeats[0], n * repeats[1], max(2, n // 2)],
        cell_type=dmesh.CellType.tetrahedron,
    )
    if nonmatching:
        # Move y nodes on the high-x face without changing the rectangular
        # domain.  Their periodic partners now lie inside master triangles.
        x = mesh.geometry.x
        xi = (x[:, 0] / scale - ORIGIN[0]) / extent[0]
        eta = (x[:, 1] / scale - ORIGIN[1]) / extent[1]
        x[:, 1] += 0.19 * scale * PERIOD[1] / n * xi * np.sin(np.pi * eta)

    fdim = mesh.topology.dim - 1
    facet_indices, facet_values = [], []
    for axis, high, tag in (
        (0, False, FACET_X0), (0, True, FACET_X1),
        (1, False, FACET_Y0), (1, True, FACET_Y1),
        (2, False, FACET_BOTTOM), (2, True, FACET_TOP),
    ):
        value = scale * (ORIGIN[axis] + (extent[axis] if high else 0))
        facets = dmesh.locate_entities_boundary(
            mesh, fdim,
            lambda x, axis=axis, value=value: np.isclose(
                x[axis], value, atol=scale * 1e-10, rtol=0
            ),
        )
        facet_indices.extend(facets)
        facet_values.extend([tag] * len(facets))
    order = np.argsort(facet_indices)
    facet_tags = dmesh.meshtags(
        mesh, fdim, np.asarray(facet_indices, dtype=np.int32)[order],
        np.asarray(facet_values, dtype=np.int32)[order],
    )
    mesh_data = types.SimpleNamespace(mesh=mesh, facet_tags=facet_tags)
    dg0 = functionspace(mesh, ("DG", 0))
    k, rho_cp, q = (Function(dg0) for _ in range(3))
    k.x.array[:] = K
    rho_cp.x.array[:] = RHO_CP / scale**2
    q.interpolate(lambda x: _exact_source(x, scale))
    chip = types.SimpleNamespace(bcs=BoundaryConditions(
        ambient_t_k=T_AMB, backside_h_eff=H / scale,
        periodic_x=True, periodic_y=True,
    ))
    return mesh_data, k, rho_cp, q, chip


def _integral(mesh, expression):
    return mesh.comm.allreduce(assemble_scalar(form(expression)), op=MPI.SUM)


def _budget(mesh_data, q, temperature, chip):
    mesh = mesh_data.mesh
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=mesh_data.facet_tags)
    power = _integral(mesh, q * ufl.dx)
    cooling = _integral(
        mesh, chip.bcs.backside_h_eff * (temperature - T_AMB) * ds(FACET_BOTTOM)
    )
    return power, cooling


def _assert_periodic_nodes(temperature, repeats=(1, 1), scale=1.0, atol=1e-8):
    """Check matching faces and doubly constrained corner lines explicitly."""
    if temperature.function_space.mesh.comm.size != 1:
        pytest.skip("Coordinate dictionary comparisons are serial diagnostics")
    coordinates = temperature.function_space.tabulate_dof_coordinates() / scale
    values = temperature.x.array
    extent = PERIOD * np.array([repeats[0], repeats[1], 1])
    lookup = {tuple(np.round(x, 10)): value for x, value in zip(coordinates, values)}
    for axis in (0, 1):
        side = np.flatnonzero(np.isclose(
            coordinates[:, axis], ORIGIN[axis] + extent[axis], atol=1e-10, rtol=0
        ))
        assert len(side) > 0
        for dof in side:
            partner = coordinates[dof].copy()
            partner[axis] -= extent[axis]
            assert abs(values[dof] - lookup[tuple(np.round(partner, 10))]) < atol
    corners = np.flatnonzero(
        np.isclose(coordinates[:, 0], ORIGIN[0] + extent[0], atol=1e-10, rtol=0)
        & np.isclose(coordinates[:, 1], ORIGIN[1] + extent[1], atol=1e-10, rtol=0)
    )
    assert len(corners) > 1
    for dof in corners:
        partner = coordinates[dof] - np.array([extent[0], extent[1], 0])
        assert abs(values[dof] - lookup[tuple(np.round(partner, 10))]) < atol


def _weighted_lateral_fluxes(mesh_data, temperature, k):
    """Weights prevent a full-period sine/cosine integral hiding lateral flow."""
    mesh = mesh_data.mesh
    x = ufl.SpatialCoordinate(mesh)
    n = ufl.FacetNormal(mesh)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=mesh_data.facet_tags)
    wx, wy = 2 * np.pi / PERIOD[:2]
    weights = (
        ufl.cos(wy * (x[1] - ORIGIN[1]) + PHASE_Y),
        ufl.sin(wx * (x[0] - ORIGIN[0]) + PHASE_X),
    )
    normal_flux = -k * ufl.dot(ufl.grad(temperature), n)
    return np.array([
        _integral(mesh, weight * normal_flux * ds(tag))
        for weight, tag in (
            (weights[0], FACET_X0), (weights[0], FACET_X1),
            (weights[1], FACET_Y0), (weights[1], FACET_Y1),
        )
    ])


def _containing_cells_by_barycentric_coordinates(mesh, points, scale):
    """Independent P1 tetra containment oracle, including at micron scales.

    Do not use the same collision search as the MPC implementation to verify
    its interpolation.  Computing in normalized coordinates also avoids a
    squared-distance tolerance dominating a very small physical tetrahedron.
    """
    vertices = mesh.geometry.x[mesh.geometry.dofmap] / scale
    assert vertices.shape[1] == 4
    edge_matrices = np.transpose(vertices[:, 1:] - vertices[:, :1], (0, 2, 1))
    cells = []
    for point in points / scale:
        weights = np.linalg.solve(edge_matrices, (point - vertices[:, 0])[..., None])[..., 0]
        contained = np.flatnonzero(
            np.all(weights >= -1e-10, axis=1) & (np.sum(weights, axis=1) <= 1 + 1e-10)
        )
        assert len(contained), f"No tetrahedron contains translated point {point}"
        cells.append(contained[0])
    return np.asarray(cells, dtype=np.int32)


def test_asymmetric_manufactured_solution_and_flux_converge():
    errors, flux_errors = [], []
    wx, wy = 2 * np.pi / PERIOD[:2]
    integral_f = PERIOD[2] + H * PERIOD[2]**2 / (3 * K)
    outward_x = -K * AMPLITUDE * wx * np.cos(PHASE_X) * PERIOD[1] / 2 * integral_f
    outward_y = K * AMPLITUDE * wy * np.sin(PHASE_Y) * PERIOD[0] / 2 * integral_f
    exact_fluxes = np.array([-outward_x, outward_x, -outward_y, outward_y])
    for n in (6, 12):
        mesh_data, k, _, q, chip = _case(n)
        temperature = solve_steady_from_fields(mesh_data, k, q, chip)
        coordinates = temperature.function_space.tabulate_dof_coordinates()
        exact = _exact_temperature(coordinates.T)
        errors.append(np.max(np.abs(temperature.x.array - exact)))
        _assert_periodic_nodes(temperature)
        power, cooling = _budget(mesh_data, q, temperature, chip)
        assert abs(power - cooling) < 1e-7 * abs(power)
        fluxes = _weighted_lateral_fluxes(mesh_data, temperature, k)
        assert np.all(fluxes * exact_fluxes > 0), "Lateral flux must be nonzero with the correct sign"
        flux_errors.append(np.linalg.norm(fluxes - exact_fluxes))
    assert errors[1] < 0.45 * errors[0], errors
    assert errors[1] < 0.04
    assert flux_errors[1] < 0.75 * flux_errors[0], flux_errors


def test_two_by_two_repeated_domain_matches_one_tile():
    solutions = []
    for repeats in ((1, 1), (2, 2)):
        mesh_data, k, _, q, chip = _case(4, repeats=repeats)
        temperature = solve_steady_from_fields(mesh_data, k, q, chip)
        _assert_periodic_nodes(temperature, repeats)
        coordinates = temperature.function_space.tabulate_dof_coordinates()
        folded = coordinates.copy()
        for axis in (0, 1):
            folded[:, axis] = ORIGIN[axis] + np.mod(
                np.round(coordinates[:, axis] - ORIGIN[axis], 10), PERIOD[axis]
            )
            folded[np.isclose(folded[:, axis], ORIGIN[axis] + PERIOD[axis]), axis] = ORIGIN[axis]
        solutions.append((folded, temperature.x.array.copy()))
    one_tile = {tuple(np.round(x, 9)): value for x, value in zip(*solutions[0])}
    for x, value in zip(*solutions[1]):
        assert abs(value - one_tile[tuple(np.round(x, 9))]) < 2e-7


@pytest.mark.parametrize("scale", [1.0, 1e-6])
def test_variable_timestep_transient_preserves_periodicity_and_energy(scale):
    mesh_data, k, rho_cp, q, chip = _case(4, scale=scale)
    solver = TransientHeatSolver(mesh_data, k, rho_cp, chip, dt=0.005, T0=T_AMB)
    old_temperature = Function(solver.V)
    for dt in (0.005, 0.005, 0.02, 0.01, 0.04):
        old_temperature.x.array[:] = solver.T_prev.x.array
        temperature = solver.step(q.x.array, dt=dt)
        _assert_periodic_nodes(temperature, scale=scale)
        power, cooling = _budget(mesh_data, q, temperature, chip)
        storage = _integral(
            mesh_data.mesh, rho_cp * (temperature - old_temperature) / dt * ufl.dx
        )
        assert abs(power - cooling - storage) < 1e-6 * abs(power)
    # A changed dt must update both the matrix and the old-temperature RHS.
    steady = solve_steady_from_fields(mesh_data, k, q, chip)
    for _ in range(30):
        temperature = solver.step(q.x.array, dt=1.0)
    assert np.max(np.abs(temperature.x.array - steady.x.array)) < 2e-7


@pytest.mark.parametrize("scale", [1.0, 1e-6])
def test_nonmatching_faces_use_finite_element_interpolation(scale):
    errors = []
    for n in (6, 12):
        mesh_data, k, _, q, chip = _case(n, scale=scale, nonmatching=True)
        if mesh_data.mesh.comm.size != 1:
            pytest.skip("Pointwise boundary interpolation comparison is serial")
        V = functionspace(mesh_data.mesh, ("Lagrange", 1))
        original_coordinates = mesh_data.mesh.geometry.x.copy()
        mpc = build_periodic_constraint(V, chip)
        np.testing.assert_array_equal(mesh_data.mesh.geometry.x, original_coordinates)
        coefficients, offsets = mpc.coefficients()
        for dof in mpc.slaves:
            # Close to the high-y edge, x-face interpolation initially uses
            # y-slave nodes.  The final constraints must flatten that chain.
            assert not np.intersect1d(mpc.masters.links(dof), mpc.slaves).size
            assert abs(np.sum(coefficients[offsets[dof]:offsets[dof + 1]]) - 1) < 1e-12
        temperature = solve_steady_from_fields(mesh_data, k, q, chip)
        coordinates = temperature.function_space.tabulate_dof_coordinates()
        slave = np.flatnonzero(np.isclose(
            coordinates[:, 0], scale * (ORIGIN[0] + PERIOD[0]),
            atol=scale * 1e-10, rtol=0
        ))
        points = coordinates[slave].copy()
        points[:, 0] = scale * ORIGIN[0]
        cells = _containing_cells_by_barycentric_coordinates(mesh_data.mesh, points, scale)
        master_values = temperature.eval(points, cells).reshape(-1)
        assert np.max(np.abs(temperature.x.array[slave] - master_values)) < 1e-8
        power, cooling = _budget(mesh_data, q, temperature, chip)
        assert abs(power - cooling) < 1e-7 * abs(power)
        errors.append(np.max(np.abs(
            temperature.x.array - _exact_temperature(coordinates.T, scale)
        )))
    assert errors[1] < 0.5 * errors[0], errors


@pytest.mark.parametrize("periodic_axis", [0, 1])
def test_axes_can_be_enabled_independently(periodic_axis):
    mesh_data, k, _, q, chip = _case(6)
    chip.bcs = BoundaryConditions(
        ambient_t_k=T_AMB, backside_h_eff=H,
        periodic_x=periodic_axis == 0, periodic_y=periodic_axis == 1,
    )
    temperature = solve_steady_from_fields(mesh_data, k, q, chip)
    if mesh_data.mesh.comm.size != 1:
        pytest.skip("Coordinate dictionary comparisons are serial diagnostics")
    coordinates = temperature.function_space.tabulate_dof_coordinates()
    values = temperature.x.array
    lookup = {tuple(np.round(x, 10)): value for x, value in zip(coordinates, values)}
    for axis in (0, 1):
        side = np.flatnonzero(np.isclose(
            coordinates[:, axis], ORIGIN[axis] + PERIOD[axis], atol=1e-10, rtol=0
        ))
        differences = []
        for dof in side:
            partner = coordinates[dof].copy()
            partner[axis] -= PERIOD[axis]
            differences.append(abs(values[dof] - lookup[tuple(np.round(partner, 10))]))
        if axis == periodic_axis:
            assert max(differences) < 1e-8
        else:
            assert max(differences) > 1e-3, "The disabled axis must not be constrained"


def test_parallel_constraint_rows_and_energy():
    """Also run with mpiexec -n 2: masters can belong to another MPI rank."""
    mesh_data, k, rho_cp, q, chip = _case(4, scale=1e-6, nonmatching=True)
    mesh = mesh_data.mesh
    steady = solve_steady_from_fields(mesh_data, k, q, chip)
    power, cooling = _budget(mesh_data, q, steady, chip)
    assert abs(power - cooling) < 1e-7 * abs(power)

    solver = TransientHeatSolver(mesh_data, k, rho_cp, chip, dt=0.005, T0=T_AMB)
    mpc = solver._mpc
    index_map = mpc.function_space.dofmap.index_map
    global_ids = index_map.local_to_global(np.arange(
        index_map.size_local + index_map.num_ghosts, dtype=np.int32
    ))
    owned_slaves = global_ids[mpc.slaves[:mpc.num_local_slaves]].tolist()
    global_slaves = set().union(*mesh.comm.allgather(owned_slaves))
    coefficients, offsets = mpc.coefficients()
    for slave in mpc.slaves:
        assert not global_slaves.intersection(global_ids[mpc.masters.links(slave)])
        assert abs(np.sum(coefficients[offsets[slave]:offsets[slave + 1]]) - 1) < 1e-12

    old_temperature = Function(solver.V)
    for dt in (0.005, 0.02):
        old_temperature.x.array[:] = solver.T_prev.x.array
        temperature = solver.step(q.x.array, dt=dt)
        local_residual = 0.0
        for slave in mpc.slaves:
            weights = coefficients[offsets[slave]:offsets[slave + 1]]
            masters = mpc.masters.links(slave)
            residual = abs(temperature.x.array[slave] - weights @ temperature.x.array[masters])
            local_residual = max(local_residual, residual)
        assert mesh.comm.allreduce(local_residual, op=MPI.MAX) < 1e-8
        power, cooling = _budget(mesh_data, q, temperature, chip)
        storage = _integral(mesh, rho_cp * (temperature - old_temperature) / dt * ufl.dx)
        assert abs(power - cooling - storage) < 1e-6 * abs(power)


def test_geometry_is_restored_when_periodic_search_fails(monkeypatch):
    from dolfinx_mpc import MultiPointConstraint

    mesh_data, _, _, _, chip = _case(2, scale=1e-6, nonmatching=True)
    V = functionspace(mesh_data.mesh, ("Lagrange", 1))
    original_coordinates = mesh_data.mesh.geometry.x.copy()

    def fail_during_search(self, V, *args, **kwargs):
        normalized = V.mesh.geometry.x
        assert not np.array_equal(normalized, original_coordinates)
        minimum, maximum = np.empty(3), np.empty(3)
        V.mesh.comm.Allreduce(normalized.min(axis=0), minimum, op=MPI.MIN)
        V.mesh.comm.Allreduce(normalized.max(axis=0), maximum, op=MPI.MAX)
        np.testing.assert_allclose(minimum, 0, atol=1e-12)
        np.testing.assert_allclose(maximum, 1, atol=1e-12)
        raise RuntimeError("injected periodic search failure")

    monkeypatch.setattr(MultiPointConstraint, "create_periodic_constraint_geometrical", fail_during_search)
    with pytest.raises(RuntimeError, match="injected periodic search failure"):
        build_periodic_constraint(V, chip)
    np.testing.assert_array_equal(mesh_data.mesh.geometry.x, original_coordinates)


def test_periodic_lateral_robin_conflict_is_rejected():
    with pytest.raises(ValueError, match="(?i)periodic|lateral"):
        bcs = BoundaryConditions(periodic_x=True, lateral_radiation_r_m=1.0)
        mesh_data, _, _, _, _ = _case(2)
        V = functionspace(mesh_data.mesh, ("Lagrange", 1))
        build_periodic_constraint(V, types.SimpleNamespace(bcs=bcs))


def test_periodic_lateral_dirichlet_conflict_is_rejected():
    mesh_data, _, _, _, chip = _case(2)
    V = functionspace(mesh_data.mesh, ("Lagrange", 1))
    dofs = locate_dofs_geometrical(
        V, lambda x: np.isclose(x[0], ORIGIN[0], atol=1e-10, rtol=0)
    )
    bc = dirichletbc(PETSc.ScalarType(T_AMB), dofs, V)
    with pytest.raises(ValueError, match="(?i)periodic|Dirichlet"):
        build_periodic_constraint(V, chip, bcs=[bc])


def test_insulating_default_does_not_build_a_constraint():
    mesh_data, _, _, _, _ = _case(2)
    V = functionspace(mesh_data.mesh, ("Lagrange", 1))
    assert build_periodic_constraint(V, types.SimpleNamespace(bcs=BoundaryConditions())) is None
