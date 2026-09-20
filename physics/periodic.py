"""Translational x/y constraints for scalar P1 heat conduction in 3D.

The temperature AND test space are constrained, so paired boundary fluxes
balance weakly. No lateral Robin/Neumann term is added. Coordinates are in
metres, including when the mesh came from the notebook GDS importer.

Opposite meshes need not match: dolfinx_mpc evaluates the master-side FE
basis at the translated slave point. Near doubly periodic edges this can
reference another slave. We flatten those dependencies before assembly;
otherwise a standard one-level MPC would give an incorrect reduced system.
Only the boundary constraint graph is replicated across MPI ranks.
"""

import numpy as np
import ufl
from dolfinx import fem, mesh as dmesh
from mpi4py import MPI
from petsc4py import PETSc


def _check_face_areas(mesh, lower, upper, axes, tolerance):
    """Require complete rectangular opposing faces, not just a bounding box."""
    fdim = mesh.topology.dim - 1
    indices, values = [], []
    for axis in axes:
        for side, location in enumerate((lower[axis], upper[axis])):
            facets = dmesh.locate_entities_boundary(
                mesh, fdim,
                lambda x, a=axis, c=location: np.isclose(x[a], c, atol=tolerance, rtol=0),
            )
            indices.extend(facets)
            values.extend([2 * axis + side + 1] * len(facets))
    order = np.argsort(indices)
    tags = dmesh.meshtags(
        mesh, fdim, np.asarray(indices, dtype=np.int32)[order],
        np.asarray(values, dtype=np.int32)[order],
    )
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=tags)
    for axis in axes:
        expected = np.prod(np.delete(upper - lower, axis))
        for side in (0, 1):
            area = mesh.comm.allreduce(
                fem.assemble_scalar(fem.form(1.0 * ds(2 * axis + side + 1))), op=MPI.SUM
            )
            if not np.isclose(area, expected, rtol=1e-8, atol=0):
                raise ValueError(
                    f"Periodic {'xy'[axis]} faces must be complete rectangles: "
                    f"side {side} area {area}, expected {expected}"
                )


def _flatten_constraints(raw, original_V, constraint_type):
    """Rebuild a finalized MPC with independent masters only (public API)."""
    comm = original_V.mesh.comm
    im = raw.function_space.dofmap.index_map
    local_ids = np.arange(im.size_local + im.num_ghosts, dtype=np.int32)
    global_ids = im.local_to_global(local_ids)
    coefficients, offsets = raw.coefficients()
    local_rows = {}
    for slave in raw.slaves[:raw.num_local_slaves]:
        masters = raw.masters.links(slave)
        weights = coefficients[offsets[slave]:offsets[slave + 1]]
        local_rows[int(global_ids[slave])] = {
            int(global_ids[master]): weight.item() for master, weight in zip(masters, weights)
        }
    rows = {}
    for rank_rows in comm.allgather(local_rows):
        rows.update(rank_rows)

    resolved, visiting = {}, set()

    def expand(dof):
        if dof not in rows:
            return {dof: 1.0}
        if dof in resolved:
            return resolved[dof]
        if dof in visiting:
            raise ValueError("Cyclic periodic interpolation constraints; remesh the paired faces")
        visiting.add(dof)
        result = {}
        for master, weight in rows[dof].items():
            for independent, factor in expand(master).items():
                result[independent] = result.get(independent, 0.0) + weight * factor
        visiting.remove(dof)
        total = sum(result.values())
        if not result or not np.isclose(total, 1.0, rtol=0, atol=1e-10):
            raise ValueError("Periodic interpolation does not preserve constant temperature")
        # Correct only floating-point partition-of-unity roundoff.
        resolved[dof] = {master: weight / total for master, weight in result.items()}
        return resolved[dof]

    # Expand the whole (small, boundary-only) graph on every rank so errors
    # are collective and cannot leave another rank blocked in finalize().
    for dof in rows:
        expand(dof)
    if not any(master in rows for row in rows.values() for master in row):
        return raw

    original_im = original_V.dofmap.index_map
    rank_ends = np.asarray(comm.allgather(original_im.local_range[1]), dtype=np.int64)
    slaves, masters, weights, owners, new_offsets = [], [], [], [], [0]
    for slave in raw.slaves:
        slaves.append(slave)
        for master, weight in sorted(resolved[int(global_ids[slave])].items()):
            masters.append(master)
            weights.append(weight)
            owners.append(np.searchsorted(rank_ends, master, side="right"))
        new_offsets.append(len(masters))
    result = constraint_type(original_V)
    result.add_constraint(
        original_V, np.asarray(slaves, dtype=np.int32), np.asarray(masters, dtype=np.int64),
        np.asarray(weights, dtype=PETSc.ScalarType), np.asarray(owners, dtype=np.int32),
        np.asarray(new_offsets, dtype=np.int32),
    )
    result.finalize()
    return result


def build_periodic_constraint(V, chip, bcs=()):
    """Return a finalized x/y MPC, or None for the existing nonperiodic path.

    No implicit dependency for insulating cases. Periodicity currently
    supports Kelvin's scalar P1 3D spaces and full rectangular lateral
    faces. Dirichlet data on ANY paired boundary DOF are rejected explicitly;
    they must not silently override periodicity, including at corners.
    """
    axes = tuple(i for i, name in enumerate(("periodic_x", "periodic_y"))
                 if getattr(chip.bcs, name, False))
    if not axes:
        return None
    if getattr(chip.bcs, "lateral_radiation_r_m", None) is not None:
        raise ValueError("Periodic lateral faces cannot also use lateral Robin radiation")
    mesh = V.mesh
    if mesh.topology.dim != 3 or mesh.geometry.dim != 3:
        raise ValueError("x/y periodic heat boundaries currently require a 3D mesh")
    if V.dofmap.index_map_bs != 1 or V.element.basix_element.degree != 1:
        raise ValueError("Periodic heat constraints require a scalar P1 temperature space")
    xyz = mesh.geometry.x
    lower = np.array([mesh.comm.allreduce(np.min(xyz[:, a], initial=np.inf), op=MPI.MIN)
                      for a in range(3)])
    upper = np.array([mesh.comm.allreduce(np.max(xyz[:, a], initial=-np.inf), op=MPI.MAX)
                      for a in range(3)])
    lengths = upper - lower
    if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
        raise ValueError("Periodic mesh must have finite, positive x/y/z extents")
    # np.isclose's default 1e-8 metre tolerance would merge nanoscale features.
    eps = np.finfo(xyz.dtype).eps
    tolerance = max(128 * eps * max(np.max(np.abs(lower)), np.max(np.abs(upper)), np.max(lengths)),
                    1e-10 * min(lengths[a] for a in axes))

    def on_max(x, axis):
        return np.isclose(x[axis], upper[axis], atol=tolerance, rtol=0)

    def slaves(x):
        return np.logical_or.reduce([on_max(x, a) for a in axes])

    def all_paired(x):
        return np.logical_or.reduce([
            on_max(x, a) | np.isclose(x[a], lower[a], atol=tolerance, rtol=0) for a in axes
        ])

    paired_dofs = fem.locate_dofs_geometrical(V, all_paired)
    conflict = any(np.intersect1d(bc.dof_indices()[0], paired_dofs).size for bc in bcs)
    if mesh.comm.allreduce(bool(conflict), op=MPI.LOR):
        raise ValueError("Dirichlet BCs on periodic lateral faces/edges are not supported")
    _check_face_areas(mesh, lower, upper, axes, tolerance)
    try:
        from dolfinx_mpc import MultiPointConstraint
    except ImportError as exc:
        raise ImportError(
            "Periodic heat boundaries need dolfinx_mpc compatible with your DOLFINx version. "
            "See PERIODIC_BOUNDARIES.md; no insulating fallback is applied."
        ) from exc
    mpc = MultiPointConstraint(V)
    # MPC 0.10 uses the SAME absolute tolerance for tree padding, squared
    # distances in GJK point/cell searches, and dimensionless basis weights.
    # Running that search on metre-valued nanometre geometry can select the
    # wrong tetrahedron. Search in a unit bounding box instead. Affine
    # coordinate scaling preserves P1 interpolation weights and DOF numbers.
    # Restore the original geometry EXACTLY before any physics is assembled,
    # including on failure. No conductivity/source/time units are changed.
    original_coordinates = xyz.copy()
    normalized_tolerance = np.maximum(tolerance / lengths, 500 * eps)

    def normalized_high(x, axis):
        return np.isclose(x[axis], 1.0, atol=normalized_tolerance[axis], rtol=0)

    def normalized_slaves(x):
        return np.logical_or.reduce([normalized_high(x, a) for a in axes])

    def normalized_relation(x):
        mapped = x.copy()
        # Map the doubly periodic edge directly to (xmin,ymin), once.
        for axis in axes:
            mapped[axis, normalized_high(x, axis)] = 0.0
        return mapped

    try:
        xyz[:] = (original_coordinates - lower) / lengths
        mpc.create_periodic_constraint_geometrical(
            V, normalized_slaves, normalized_relation, [], tol=500 * eps
        )
    finally:
        xyz[:] = original_coordinates
    mpc.finalize()
    expected = fem.locate_dofs_geometrical(V, slaves)
    owned_expected = expected[expected < V.dofmap.index_map.size_local]
    actual = mpc.slaves[:mpc.num_local_slaves]
    missing = np.setdiff1d(owned_expected, actual).size
    if mesh.comm.allreduce(missing, op=MPI.SUM):
        raise ValueError("Some periodic boundary DOFs have no translated master; check the mesh")
    return _flatten_constraints(mpc, V, MultiPointConstraint)
