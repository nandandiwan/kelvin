"""Short FEM smoke test on a completed, sealed notebook-derived SRAM mesh.

This does not rebuild a potentially expensive SRAM mesh during unit tests.
Set KELVIN_SRAM_TEST_MANIFEST to a generated manifest, or generate the default
out/sram_notebook_integration/sram_sp_cell_material_regions.json first. The
adapter checks geometry, mapping and mesh fingerprints before every import.
"""

import os
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip("dolfinx")
pytest.importorskip("dolfinx_mpc")

from dolfinx.fem import Function, assemble_scalar, form
from mpi4py import MPI
import ufl

from mesh.build import FACET_BOTTOM
from mesh.sram import load_sram_mesh
from physics.coeffs import build_coeffs
from post.budget import verify_device_source_powers
from solve.transient import TransientHeatSolver
from spec.chip import BoundaryConditions


@pytest.fixture(scope="module")
def notebook_sram_case():
    root = Path(__file__).resolve().parents[1]
    explicit_path = os.environ.get("KELVIN_SRAM_TEST_MANIFEST")
    path = Path(explicit_path) if explicit_path else root / (
        "out/sram_notebook_integration/sram_sp_cell_material_regions.json")
    if not path.is_file():
        if explicit_path:
            pytest.fail(f"KELVIN_SRAM_TEST_MANIFEST does not exist: {path}")
        pytest.skip("Generate the sealed notebook SRAM mesh or set KELVIN_SRAM_TEST_MANIFEST")
    # Deliberately asymmetric synthetic powers test bookkeeping and BCs;
    # these are not a SPICE operating-point or physical-temperature claim.
    powers = {f"X{i}": (i + 1) * 1e-7 for i in range(8)}
    case = load_sram_mesh(path, powers, bcs=BoundaryConditions(periodic_x=True, periodic_y=True))
    k, rho_cp, q = build_coeffs(case.mesh_data.mesh, case.mesh_data.cell_tags, case.registry)
    return case, powers, k, rho_cp, q


def test_notebook_sram_sources_preserve_all_eight_electrical_powers(notebook_sram_case):
    case, powers, _, _, q = notebook_sram_case
    audit = verify_device_source_powers(case.mesh_data, case.registry, q, powers)
    assert set(audit["per_device"]) == set(powers)
    assert audit["meshed_total_w"] == pytest.approx(sum(powers.values()), rel=1e-7, abs=0)
    assert case.report["unanchored_component_count"] == 0
    assert all(entry["ratio"] == pytest.approx(1.0, rel=1e-6)
               for entry in case.report["source_volumes"].values())


def test_notebook_sram_transient_preserves_periodicity_and_energy(notebook_sram_case):
    case, powers, k, rho_cp, q = notebook_sram_case
    mesh = case.mesh_data.mesh
    original_coordinates = mesh.geometry.x.copy()
    solver = TransientHeatSolver(case.mesh_data, k, rho_cp, case.chip, dt=1e-6, T0=300.0)
    np.testing.assert_array_equal(mesh.geometry.x, original_coordinates)
    assert solver._mpc is not None and solver._mpc.num_local_slaves > 0
    old_temperature = Function(solver.V)
    dx = ufl.Measure("dx", domain=mesh)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=case.mesh_data.facet_tags)

    def integral(expression):
        return float(mesh.comm.allreduce(assemble_scalar(form(expression)), op=MPI.SUM))

    generated = integral(q * dx)
    assert generated == pytest.approx(sum(powers.values()), rel=1e-7, abs=0)
    for dt in (1e-6, 4e-6):
        old_temperature.x.array[:] = solver.T_prev.x.array
        temperature = solver.step(q.x.array, dt=dt)
        storage = integral(rho_cp * (temperature - old_temperature) / dt * dx)
        cooling = integral(case.chip.bcs.backside_h_eff * (temperature - 300.0) * ds(FACET_BOTTOM))
        relative_energy_error = abs(generated - storage - cooling) / generated
        assert relative_energy_error < 1e-5
        coefficients, offsets = solver._mpc.coefficients()
        max_periodic_error = 0.0
        for slave in solver._mpc.slaves[:solver._mpc.num_local_slaves]:
            masters = solver._mpc.masters.links(slave)
            weights = coefficients[offsets[slave]:offsets[slave + 1]]
            error = abs(temperature.x.array[slave] - weights @ temperature.x.array[masters])
            max_periodic_error = max(max_periodic_error, float(error))
        assert max_periodic_error < 1e-8
        assert np.isfinite(temperature.x.array).all() and temperature.x.array.max() > 300.0
        print(f"dt={dt:g}s: energy relative error={relative_energy_error:.3e}, "
              f"periodic constraint error={max_periodic_error:.3e} K")
