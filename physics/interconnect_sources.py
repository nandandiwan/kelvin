"""Conservative, layout-local Joule heat on a non-electrical thermal mesh.

The electrical sheet network is generally finer than the tetrahedral mesh.
Each resistor's declared polygon-prism heat support is intersected with the
correctly tagged tetrahedra, once.  The resulting DG0 projection preserves
power without depositing heat into the nearest, possibly unrelated, wire.
This is a cell-average projection: refining the thermal mesh is still needed
to resolve temperature gradients below its cell size.

Several electrical resistors can overlap one thermal cell. Their individual
contributions cannot be recovered uniquely from the summed field. Audits
therefore compare the *actual complete field* to the electrical energies'
spatial projection, plus independently integrated channel/layer/domain totals;
they do not pretend to measure each overlapping resistor separately.
"""

from __future__ import annotations

import math

import numpy as np
from dolfinx.fem import Function, functionspace
from mpi4py import MPI
from scipy.sparse import coo_matrix
from scipy.spatial import ConvexHull, QhullError

from physics.coeffs import _tag_by_cell
from physics.device_sources import DEVICE_NAMES, DeviceHeatSources, _device_values


_TET_EDGES = np.array(((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)))
_TET_FACES = np.array(((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)))
_GEOMETRIC_TOL_UM = 2e-11


def _cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _polygon_area(polygon):
    return 0.5 * float(np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1)
                              - polygon[:, 1] * np.roll(polygon[:, 0], -1)))


def _convex_parts(vertices):
    """Return nonoverlapping convex pieces; ear clipping handles concave tiles."""
    polygon = np.asarray(vertices, dtype=float)
    if (polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3
            or not np.isfinite(polygon).all()):
        raise ValueError("Heat polygon must have at least three finite xy vertices")
    if np.array_equal(polygon[0], polygon[-1]):
        polygon = polygon[:-1]
    # Drop duplicate and collinear points; they do not change a simple polygon.
    changed = True
    while changed and len(polygon) >= 3:
        changed = False
        for i in range(len(polygon)):
            before, here, after = polygon[i - 1], polygon[i], polygon[(i + 1) % len(polygon)]
            if (np.linalg.norm(here - before) < 1e-13
                    or abs(_cross2(here - before, after - here)) < 1e-15):
                polygon = np.delete(polygon, i, axis=0)
                changed = True
                break
    if len(polygon) < 3 or abs(_polygon_area(polygon)) <= 1e-16:
        raise ValueError("Heat polygon must have positive area")
    if _polygon_area(polygon) < 0:
        polygon = polygon[::-1].copy()
    turns = [_cross2(polygon[(i + 1) % len(polygon)] - polygon[i],
                     polygon[(i + 2) % len(polygon)] - polygon[(i + 1) % len(polygon)])
             for i in range(len(polygon))]
    if min(turns) >= -1e-15:
        return [polygon]
    pending = list(range(len(polygon)))
    result = []
    while len(pending) > 3:
        for i, current in enumerate(pending):
            previous, following = pending[i - 1], pending[(i + 1) % len(pending)]
            a, b, c = polygon[[previous, current, following]]
            if _cross2(b - a, c - b) <= 1e-15:
                continue
            other = [j for j in pending if j not in (previous, current, following)]
            if any(_cross2(b - a, polygon[j] - a) >= -1e-15
                   and _cross2(c - b, polygon[j] - b) >= -1e-15
                   and _cross2(a - c, polygon[j] - c) >= -1e-15 for j in other):
                continue
            result.append(polygon[[previous, current, following]])
            del pending[i]
            break
        else:
            raise ValueError("Heat polygon is not a valid simple polygon")
    result.append(polygon[pending])
    return result


def _prism(polygon, z_min, z_max):
    n = len(polygon)
    edges = np.roll(polygon, -1, axis=0) - polygon
    normals = np.column_stack((edges[:, 1], -edges[:, 0], np.zeros(n)))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    offsets = np.einsum("ij,ij->i", normals[:, :2], polygon)
    normals = np.vstack((normals, [0, 0, -1], [0, 0, 1]))
    offsets = np.r_[offsets, -z_min, z_max]
    points = np.vstack((np.column_stack((polygon, np.full(n, z_min))),
                        np.column_stack((polygon, np.full(n, z_max)))))
    indices = np.arange(n)
    edge_indices = np.vstack((np.column_stack((indices, (indices + 1) % n)),
                              np.column_stack((indices + n, (indices + 1) % n + n)),
                              np.column_stack((indices, indices + n))))
    return points, normals, offsets, edge_indices


def _edge_plane_points(points, edges, normals, offsets):
    starts = points[edges[:, 0]]
    directions = points[edges[:, 1]] - starts
    denominators = directions @ normals.T
    numerators = offsets[None, :] - starts @ normals.T
    with np.errstate(divide="ignore", invalid="ignore"):
        parameters = numerators / denominators
    usable = (np.abs(denominators) > 1e-14) & (parameters >= 0) & (parameters <= 1)
    edge, plane = np.nonzero(usable)
    return starts[edge] + directions[edge] * parameters[edge, plane, None]


def _tetra_prism_volume(tetrahedron, prism):
    """Exact convex-polyhedron intersection volume, in input-length cubed."""
    points, normals, offsets, prism_edges = prism
    inside_tet_vertices = np.all(tetrahedron @ normals.T <= offsets + _GEOMETRIC_TOL_UM, axis=1)
    if np.all(inside_tet_vertices):
        return abs(float(np.linalg.det(tetrahedron[1:] - tetrahedron[0]))) / 6
    if np.any(np.all(tetrahedron @ normals.T > offsets + _GEOMETRIC_TOL_UM, axis=0)):
        return 0.0
    faces = tetrahedron[_TET_FACES]
    tet_normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    tet_normals /= np.linalg.norm(tet_normals, axis=1)[:, None]
    tet_offsets = np.einsum("ij,ij->i", tet_normals, faces[:, 0])
    flip = np.einsum("ij,ij->i", tet_normals, tetrahedron) > tet_offsets
    tet_normals[flip] *= -1
    tet_offsets[flip] *= -1
    candidates = np.vstack((tetrahedron[inside_tet_vertices], points,
                            _edge_plane_points(tetrahedron, _TET_EDGES, normals, offsets),
                            _edge_plane_points(points, prism_edges, tet_normals, tet_offsets)))
    keep = (np.all(candidates @ normals.T <= offsets + _GEOMETRIC_TOL_UM, axis=1)
            & np.all(candidates @ tet_normals.T <= tet_offsets + _GEOMETRIC_TOL_UM, axis=1))
    candidates = candidates[keep]
    if len(candidates) < 4:
        return 0.0
    # Avoid asking QHull to construct a zero-volume face/edge intersection.
    if np.linalg.matrix_rank(candidates[1:] - candidates[0], tol=1e-12) < 3:
        return 0.0
    try:
        return float(ConvexHull(candidates).volume)
    except QhullError as exc:
        raise ValueError("Could not intersect a source prism with a thermal tetrahedron") from exc


def _values(values, names, quantity):
    if set(values) != set(names):
        raise ValueError(f"Interconnect {quantity} must contain exactly the network resistor names")
    result = {name: float(values[name]) for name in names}
    if any(not math.isfinite(value) or value < 0 for value in result.values()):
        raise ValueError(f"Interconnect {quantity} must be finite and nonnegative")
    return result


def _layer_region(manifest, heat_region):
    layer = heat_region["layer"]
    if layer == "licon1":
        lower = heat_region.get("lower_layer")
        if lower not in ("poly", "sd", "diff", "tap"):
            raise ValueError("LICON heat needs an explicit poly or source/drain lower landing")
        name = "licon_to_poly" if lower == "poly" else "licon_to_diff"
    else:
        name = layer
    matching = [region for region in manifest["regions"]
                if region.get("name") == name and region.get("role") == "conductor"]
    if len(matching) != 1:
        raise ValueError(f"Missing or ambiguous conductor layer {name!r} in the thermal manifest")
    return matching[0]


class InterconnectHeatSources:
    """Sparse DG0 projection of resistor watts into exact overlapping cells."""

    def __init__(self, mesh_data, registry, manifest, network, *, relative_tolerance=1e-6):
        self.mesh = mesh_data.mesh
        if self.mesh.topology.dim != 3:
            raise ValueError("Interconnect heat sources require a 3D thermal mesh")
        if not math.isfinite(relative_tolerance) or relative_tolerance <= 0:
            raise ValueError("Source audit relative tolerance must be finite and positive")
        self.relative_tolerance = float(relative_tolerance)
        self._owned_cells = self.mesh.topology.index_map(3).size_local
        geometry_dofs = self.mesh.geometry.dofmap[:self._owned_cells]
        if geometry_dofs.ndim != 2 or geometry_dofs.shape[1] != 4:
            raise ValueError("Exact interconnect projection requires first-order tetrahedra")
        self._tetrahedra = self.mesh.geometry.x[geometry_dofs, :3] * 1e6
        self._cell_volumes = np.abs(np.linalg.det(
            self._tetrahedra[:, 1:] - self._tetrahedra[:, :1])) / 6 * 1e-18
        if not np.isfinite(self._cell_volumes).all() or np.any(self._cell_volumes <= 0):
            raise ValueError("Thermal mesh contains degenerate tetrahedra")
        self._bbox_min = self._tetrahedra.min(axis=1)
        self._bbox_max = self._tetrahedra.max(axis=1)
        self._tags = _tag_by_cell(self.mesh, mesh_data.cell_tags)[:self._owned_cells]
        self._cells_by_tag = {int(tag): np.flatnonzero(self._tags == tag)
                              for tag in np.unique(self._tags)}
        space = functionspace(self.mesh, ("DG", 0))
        self.q = Function(space, name="interconnect_joule_heat")
        self._cell_dofs = np.array([space.dofmap.cell_dofs(i)[0]
                                   for i in range(self._owned_cells)], dtype=np.int32)
        registry_tags = {region.tag_id for region in registry.all()}
        resistors = network.get("resistors", [])
        self.resistor_names = tuple(resistor["name"] for resistor in resistors)
        if not resistors or len(set(self.resistor_names)) != len(self.resistor_names):
            raise ValueError("Interconnect network needs distinct, nonempty resistor names")
        if any(not isinstance(name, str) or not name for name in self.resistor_names):
            raise ValueError("Interconnect resistor names must be nonempty strings")
        self._layer_tags = {}
        self.volume_audit = {}
        projections = {}
        row_indices, column_indices, densities = [], [], []
        for column, resistor in enumerate(resistors):
            resistance = float(resistor["resistance_ohm"])
            if not math.isfinite(resistance) or resistance <= 0:
                raise ValueError("Interconnect resistance must be finite and positive")
            regions = resistor.get("heat_regions", [])
            weights = [float(region["weight"]) for region in regions]
            if (not weights or any(not math.isfinite(w) or w <= 0 for w in weights)
                    or not math.isclose(math.fsum(weights), 1, rel_tol=1e-12, abs_tol=1e-14)):
                raise ValueError("Resistor heat-region weights must be positive and sum to one")
            for region, weight in zip(regions, weights):
                layer = _layer_region(manifest, region)
                tag = int(layer["physical_tag"])
                if tag not in registry_tags:
                    raise ValueError(f"Conductor tag {tag} is absent from the source registry")
                self._layer_tags[layer["name"]] = tag
                polygon = np.asarray(region["polygon_um"], dtype=float)
                parts = _convex_parts(polygon)
                z_min, z_max = float(layer["z_min_um"]), float(layer["z_max_um"])
                if not math.isfinite(z_min + z_max) or z_max <= z_min:
                    raise ValueError("Conductor layer must have finite positive thickness")
                key = (tag, tuple(map(tuple, polygon)))
                if key not in projections:
                    cells, density, audit = self._project_region(tag, parts, z_min, z_max)
                    projections[key] = (cells, density)
                    self.volume_audit[f"region_{len(projections) - 1}"] = audit
                cells, density = projections[key]
                row_indices.extend(self._cell_dofs[cells])
                column_indices.extend([column] * len(cells))
                densities.extend(density * weight)
        self._projection = coo_matrix((densities, (row_indices, column_indices)),
                                      shape=(len(self.q.x.array), len(resistors))).tocsr()
        # Retain separately from the mutable live source field and sparse
        # update operator, for expected-energy forward projection at audit time.
        self._audit_projection = self._projection.copy()
        self.interconnect_power_w = dict.fromkeys(self.resistor_names, 0.0)
        self._column_integrals = self.mesh.comm.allreduce(np.asarray(
            self._projection[self._cell_dofs].T @ self._cell_volumes).ravel(), op=MPI.SUM)
        if not np.allclose(self._column_integrals, 1, rtol=relative_tolerance, atol=0):
            raise ValueError("Projected resistor watts are not conserved by the thermal mesh")

    def _project_region(self, tag, parts, z_min, z_max):
        vertices = np.vstack(parts)
        low = np.r_[vertices.min(axis=0), z_min]
        high = np.r_[vertices.max(axis=0), z_max]
        tagged_cells = self._cells_by_tag.get(tag, np.empty(0, dtype=np.int32))
        candidates = tagged_cells[np.all(self._bbox_max[tagged_cells] > low + 1e-13, axis=1)
                                  & np.all(self._bbox_min[tagged_cells] < high - 1e-13, axis=1)]
        prisms = [_prism(part, z_min, z_max) for part in parts]
        intersections = np.array([math.fsum(_tetra_prism_volume(self._tetrahedra[cell], p)
                                            for p in prisms) for cell in candidates])
        keep = intersections > 1e-20
        candidates, intersections = candidates[keep], intersections[keep]
        nominal_um3 = math.fsum(_polygon_area(part) for part in parts) * (z_max - z_min)
        meshed_um3 = float(self.mesh.comm.allreduce(float(intersections.sum()), op=MPI.SUM))
        if not math.isclose(meshed_um3, nominal_um3,
                            rel_tol=self.relative_tolerance, abs_tol=1e-17):
            raise ValueError(f"Interconnect heat polygon on tag {tag}: overlapping mesh volume "
                             f"{meshed_um3:.12g} um^3 does not match nominal "
                             f"{nominal_um3:.12g} um^3; source may leave its conductor")
        density = intersections / nominal_um3 / self._cell_volumes[candidates]
        return candidates, density, {"physical_tag": tag, "nominal_m3": nominal_um3 * 1e-18,
                                     "meshed_m3": meshed_um3 * 1e-18,
                                     "ratio": meshed_um3 / nominal_um3}

    def set_powers(self, interconnect_power_w):
        powers = _values(interconnect_power_w, self.resistor_names, "powers")
        density = self._projection @ np.array(list(powers.values()))
        if not np.isfinite(density).all():
            raise ValueError("Interconnect power density overflowed")
        self.q.x.array[:] = density
        self.q.x.scatter_forward()
        self.interconnect_power_w = powers
        return self.q

    def _integral(self, field, tag=None):
        selected = slice(None) if tag is None else self._tags == tag
        local = float(np.dot(field.x.array[self._cell_dofs[selected]], self._cell_volumes[selected]))
        return float(self.mesh.comm.allreduce(local, op=MPI.SUM))

    def integrated_powers(self):
        """Actual layer and total watts; overlapping resistors are not separable."""
        return {"per_layer": {name: self._integral(self.q, tag)
                               for name, tag in self._layer_tags.items()},
                "total_w": self._integral(self.q)}

    def _expected_field(self, values):
        return self._audit_projection @ np.array([values[name] for name in self.resistor_names])

    def _assert_actual_field(self, actual, expected, expected_total, cell_mask=None):
        if (actual.function_space.mesh is not self.mesh
                or actual.x.array.shape != self.q.x.array.shape
                or actual.function_space.ufl_element() != self.q.function_space.ufl_element()):
            raise ValueError("Accumulated source must use the same scalar DG0 thermal space")
        valid = bool(np.isfinite(actual.x.array).all() and np.all(actual.x.array >= 0))
        if not self.mesh.comm.allreduce(valid, op=MPI.LAND):
            raise ValueError("Accumulated source must be finite and nonnegative")
        selected = slice(None) if cell_mask is None else cell_mask
        dofs = self._cell_dofs[selected]
        difference = np.abs(actual.x.array[dofs] - expected[dofs])
        error = float(self.mesh.comm.allreduce(
            float(difference @ self._cell_volumes[selected]), op=MPI.SUM))
        if error > max(1e-30, self.relative_tolerance * expected_total):
            raise ValueError("Actual source field does not match the spatially projected electrical "
                             f"budget (volume-integrated absolute error {error:.12g})")
        return error

    def audit_energy(self, qdt, expected_interconnect_energy_j):
        expected = _values(expected_interconnect_energy_j, self.resistor_names, "energies")
        expected_total = math.fsum(expected.values())
        error = self._assert_actual_field(qdt, self._expected_field(expected), expected_total)
        return self._energy_report(qdt, expected, error)

    def _energy_report(self, actual, expected, spatial_error):
        expected_field = self._expected_field(expected)
        per_layer = {}
        for name, tag in self._layer_tags.items():
            selected = self._tags == tag
            expected_layer = float(self.mesh.comm.allreduce(float(np.dot(
                expected_field[self._cell_dofs[selected]], self._cell_volumes[selected])), op=MPI.SUM))
            deposited_layer = self._integral(actual, tag)
            if not math.isclose(deposited_layer, expected_layer,
                                rel_tol=self.relative_tolerance, abs_tol=1e-30):
                raise ValueError(f"{name}: deposited interconnect energy does not match electrical energy")
            per_layer[name] = {"expected_j": expected_layer, "deposited_j": deposited_layer,
                               "physical_tag": tag}
        return {
            "per_interconnect": {name: {"expected_j": expected[name],
                "projected_j": expected[name] * float(self._column_integrals[i])}
                for i, name in enumerate(self.resistor_names)},
            "per_layer": per_layer,
            "expected_total_j": math.fsum(expected.values()),
            "deposited_total_j": math.fsum(layer["deposited_j"] for layer in per_layer.values()),
            "spatial_absolute_error_j": spatial_error,
            "resistor_attribution": "Overlapping resistors are audited by the complete projected "
                                    "field, not individually recovered from summed DG0 heat.",
        }


class CombinedHeatSources:
    """One reusable DG0 field with channel and interconnect waveform heat."""

    def __init__(self, mesh_data, registry, manifest, network, *, relative_tolerance=1e-6):
        self.devices = DeviceHeatSources(mesh_data, registry, relative_tolerance=relative_tolerance)
        self.interconnects = InterconnectHeatSources(mesh_data, registry, manifest, network,
                                                    relative_tolerance=relative_tolerance)
        self.mesh = mesh_data.mesh
        self.q = Function(self.devices.q.function_space, name="device_and_interconnect_heat")
        self.device_names = self.devices.device_names
        self.resistor_names = self.interconnects.resistor_names
        self.volume_audit = {"channels": self.devices.volume_audit,
                             "interconnects": self.interconnects.volume_audit}
        self.projection_report = {
            "method": "exact polygon-prism/tetrahedron overlap projected to scalar DG0",
            "resistor_count": len(self.resistor_names),
            "unique_heat_region_count": len(self.interconnects.volume_audit),
            "volume_audit": self.volume_audit,
            "subcell_limitation": "The thermal mesh resolves cell-average sources, not subcell gradients.",
        }
        channel_tags = {r.tag_id for r in self.devices._channels.values()}
        if channel_tags & set(self.interconnects._layer_tags.values()):
            raise ValueError("Channel and interconnect physical tags must be disjoint")

    def set_powers(self, device_power_w, interconnect_power_w):
        # Validate both dictionaries before changing either component.
        devices = _device_values(device_power_w, "powers")
        interconnects = _values(interconnect_power_w, self.resistor_names, "powers")
        self.devices.set_powers(devices)
        self.interconnects.set_powers(interconnects)
        self.q.x.array[:] = self.devices.q.x.array + self.interconnects.q.x.array
        self.q.x.scatter_forward()
        return self.q

    def audit_energy(self, qdt, expected_device_energy_j, expected_interconnect_energy_j):
        devices = _device_values(expected_device_energy_j, "energies")
        wires = _values(expected_interconnect_energy_j, self.resistor_names, "energies")
        expected_wires = self.interconnects._expected_field(wires)
        expected = expected_wires.copy()
        for name in DEVICE_NAMES:
            expected[self.devices._dofs[name]] += devices[name] / self.devices._nominal_volume[name]
        total = math.fsum(devices.values()) + math.fsum(wires.values())
        error = self.interconnects._assert_actual_field(qdt, expected, total)
        wire_mask = np.isin(self.interconnects._tags, list(self.interconnects._layer_tags.values()))
        wire_error = self.interconnects._assert_actual_field(
            qdt, expected_wires, math.fsum(wires.values()), cell_mask=wire_mask)
        deposited_devices = self.devices.integrated_energy(qdt)
        channel_report = {}
        for name in DEVICE_NAMES:
            if not math.isclose(deposited_devices[name], devices[name],
                                rel_tol=self.devices.relative_tolerance, abs_tol=1e-30):
                raise ValueError(f"{name}: deposited channel energy does not match electrical energy")
            channel_report[name] = {"expected_j": devices[name],
                                    "deposited_j": deposited_devices[name],
                                    "physical_tag": self.devices._channels[name].tag_id}
        wire_report = self.interconnects._energy_report(qdt, wires, wire_error)
        actual_total = self.interconnects._integral(qdt)
        if not math.isclose(actual_total, total, rel_tol=self.devices.relative_tolerance, abs_tol=1e-30):
            raise ValueError("Total source energy does not match the electrical energy budget")
        return {"per_device": channel_report, "interconnects": wire_report,
                "expected_total_j": total, "deposited_total_j": actual_total,
                "spatial_absolute_error_j": error}

    def audit_powers(self, actual_q, expected_device_power_w, expected_interconnect_power_w):
        """The same spatial audit in watts; no time integration or unit disguise."""
        report = self.audit_energy(actual_q, expected_device_power_w, expected_interconnect_power_w)

        def watts(value):
            if isinstance(value, dict):
                return {(key[:-2] + "_w" if key.endswith("_j") else key): watts(item)
                        for key, item in value.items()}
            return value

        result = watts(report)
        result["electrical_total_w"] = result["expected_total_w"]
        result["meshed_total_w"] = result["deposited_total_w"]
        for device in result["per_device"].values():
            device["meshed_w"] = device["deposited_w"]
        return result


DeviceAndInterconnectHeatSources = CombinedHeatSources
