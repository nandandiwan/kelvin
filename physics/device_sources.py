"""Conservative, instance-resolved time-varying SRAM channel heat sources.

The electrical waveform supplies watts for each of X0--X7.  Geometry supplies
an independent nominal channel volume.  Their quotient is scattered into a
cached DG0 field, so updating a waveform does not rebuild material fields or
compile a new variational form.  A volume check fails closed before solving:
normalizing by a damaged mesh's volume would hide a geometry/mapping error.
"""

from __future__ import annotations

import math

import numpy as np
import ufl
from dolfinx.fem import Function, assemble_scalar, form, functionspace
from mpi4py import MPI

from physics.coeffs import _tag_by_cell


DEVICE_NAMES = tuple(f"X{i}" for i in range(8))


def _device_values(values, quantity):
    if set(values) != set(DEVICE_NAMES):
        raise ValueError(
            f"Device {quantity} must contain exactly X0 through X7: "
            f"missing={sorted(set(DEVICE_NAMES) - set(values))}, "
            f"extra={sorted(set(values) - set(DEVICE_NAMES))}")
    result = {name: float(values[name]) for name in DEVICE_NAMES}
    if any(not math.isfinite(value) or value < 0 for value in result.values()):
        raise ValueError(f"Device {quantity} must be finite and nonnegative")
    return result


class DeviceHeatSources:
    """Reusable source field for one eight-transistor, three-dimensional cell.

    ``q`` starts at zero. ``set_powers`` changes it in place and returns the
    same Function.  This adapter exclusively owns channel heat; a registry
    with any nonzero wire/contact/other source is rejected rather than silently
    dropping that heat.  Frozen registry powers remain untouched.
    """

    def __init__(self, mesh_data, registry, *, relative_tolerance=1e-6):
        self.mesh = mesh_data.mesh
        if self.mesh.topology.dim != 3:
            raise ValueError("Device heat sources require a 3D thermal mesh")
        if not math.isfinite(relative_tolerance) or relative_tolerance <= 0:
            raise ValueError("Source audit relative tolerance must be finite and positive")
        self.relative_tolerance = relative_tolerance
        regions = registry.all()
        tags = [region.tag_id for region in regions]
        if len(tags) != len(set(tags)):
            raise ValueError("Duplicate physical tags in the source registry")
        channels = []
        for region in regions:
            source = region.source
            if source is None:
                continue
            if source.kind == "channel":
                channels.append(region)
            elif source.power_uw != 0 or source.power_density_w_per_m3 != 0:
                raise ValueError(
                    f"Nonchannel source {region.label!r} has nonzero heat; "
                    "the device-waveform adapter cannot discard other sources")
        names = [region.source.device for region in channels]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate channel device IDs in the source registry")
        if set(names) != set(DEVICE_NAMES):
            raise ValueError("Channel registry must contain exactly X0 through X7")
        self._channels = {region.source.device: region for region in channels}
        self.device_names = DEVICE_NAMES
        self._dx = ufl.Measure("dx", domain=self.mesh, subdomain_data=mesh_data.cell_tags)
        self._whole_dx = ufl.Measure("dx", domain=self.mesh)
        self.volume_audit = {}
        self._nominal_volume = {}
        for name in DEVICE_NAMES:
            region = self._channels[name]
            source = region.source
            nominal = getattr(source, "volume_m3", None)
            if nominal is None:
                nominal = source.w_um * source.l_um * source.t_um * 1e-18
            nominal = float(nominal)
            if not math.isfinite(nominal) or nominal <= 0:
                raise ValueError(f"{name}: nominal channel volume must be finite and positive")
            meshed = self._integral(1.0 * self._dx(region.tag_id))
            if not math.isclose(meshed, nominal, rel_tol=relative_tolerance, abs_tol=0):
                raise ValueError(
                    f"{name}: meshed volume {meshed:.12g} m^3 does not match "
                    f"nominal channel volume {nominal:.12g} m^3")
            self._nominal_volume[name] = nominal
            self.volume_audit[name] = {
                "physical_tag": region.tag_id, "nominal_m3": nominal,
                "meshed_m3": meshed, "ratio": meshed / nominal,
            }

        space = functionspace(self.mesh, ("DG", 0))
        self.q = Function(space, name="device_channel_heat")
        tags_by_cell = _tag_by_cell(self.mesh, mesh_data.cell_tags)
        self._dofs = {}
        for name, region in self._channels.items():
            cells = np.flatnonzero(tags_by_cell == region.tag_id)
            # DG0 has one scalar DOF per cell, but its DOF numbering need not
            # equal the mesh's cell numbering (especially in distributed runs).
            self._dofs[name] = np.fromiter(
                (space.dofmap.cell_dofs(int(cell))[0] for cell in cells),
                dtype=np.int32, count=len(cells))
        self._power_forms = {
            name: form(self.q * self._dx(region.tag_id))
            for name, region in self._channels.items()
        }
        self.device_power_w = {name: 0.0 for name in DEVICE_NAMES}

    def _integral(self, expression):
        return float(self.mesh.comm.allreduce(assemble_scalar(form(expression)), op=MPI.SUM))

    def set_powers(self, device_power_w):
        """Scatter one interval's finite nonnegative device powers, in watts."""
        powers = _device_values(device_power_w, "powers")
        # Validate before modifying the previous valid field.
        densities = {name: powers[name] / self._nominal_volume[name] for name in DEVICE_NAMES}
        if any(not math.isfinite(value) for value in densities.values()):
            raise ValueError("Device power density overflowed; check electrical powers and units")
        self.q.x.array[:] = 0
        for name, density in densities.items():
            self.q.x.array[self._dofs[name]] = density
        self.q.x.scatter_forward()
        self.device_power_w = powers
        return self.q

    def integrated_powers(self):
        """Independently assemble each channel's actual current heat, in W."""
        return {
            name: float(self.mesh.comm.allreduce(assemble_scalar(compiled), op=MPI.SUM))
            for name, compiled in self._power_forms.items()
        }

    def integrated_energy(self, qdt):
        """Integrate a separately accumulated actual FEM load, in joules.

        Callers accumulate ``qdt += dt * solver.q`` for every thermal step,
        not the prescribed electrical energies, so this tests the load the
        thermal solver actually consumed, including timestep discretization.
        """
        if qdt.function_space.mesh is not self.mesh:
            raise ValueError("Accumulated source energy must use the same thermal mesh")
        valid = bool(np.isfinite(qdt.x.array).all() and np.all(qdt.x.array >= 0))
        if not self.mesh.comm.allreduce(valid, op=MPI.LAND):
            raise ValueError("Accumulated source energy must be finite and nonnegative")
        return {name: self._integral(qdt * self._dx(region.tag_id))
                for name, region in self._channels.items()}

    def audit_energy(self, qdt, expected_device_energy_j):
        """Fail unless every channel and the whole mesh match electrical energy."""
        expected = _device_values(expected_device_energy_j, "energies")
        deposited = self.integrated_energy(qdt)
        report = {}
        for name in DEVICE_NAMES:
            if not math.isclose(deposited[name], expected[name],
                                rel_tol=self.relative_tolerance, abs_tol=1e-30):
                raise ValueError(
                    f"{name}: deposited energy {deposited[name]:.12g} J does not "
                    f"match electrical energy {expected[name]:.12g} J")
            report[name] = {"expected_j": expected[name], "deposited_j": deposited[name],
                            "physical_tag": self._channels[name].tag_id}
        expected_total = math.fsum(expected.values())
        deposited_total = self._integral(qdt * self._whole_dx)
        if not math.isclose(deposited_total, expected_total,
                            rel_tol=self.relative_tolerance, abs_tol=1e-30):
            raise ValueError("Total source energy does not match the electrical energy budget")
        return {"per_device": report, "expected_total_j": expected_total,
                "deposited_total_j": deposited_total}
