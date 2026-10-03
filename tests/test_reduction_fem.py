from dataclasses import replace

import numpy as np
import pytest

from test_periodic import _case
from solve.reduction import component_basis, ReducedHeatSolver
from solve.reduction_fem import assemble_thermal_operators
from solve.transient import TransientHeatSolver


@pytest.mark.parametrize("periodic", [True, False])
def test_exported_operators_match_public_fem_transient(periodic):
    mesh, k, rho, q, chip = _case(2, nonmatching=periodic)
    chip.bcs = replace(chip.bcs, periodic_x=periodic, periodic_y=periodic)
    system = assemble_thermal_operators(mesh, k, rho, chip)
    size = system.mass.shape[0]
    retained = np.arange(size // 3, 2 * size // 3)
    basis = component_basis(system.mass, system.stiffness, retained, size - len(retained))
    reduced = ReducedHeatSolver(system.mass, system.stiffness, basis)
    fem = TransientHeatSolver(mesh, k, rho, chip, dt=.01)
    for dt in [.01, .02, .01]:
        result = reduced.step(dt, system.load(q))
        reference = fem.step(q.x.array, dt)
        actual = 300 + system.reconstruct(result.temperature_rise_k)
        np.testing.assert_allclose(actual, reference.x.array, atol=2e-7, rtol=0)
        assert abs(result.energy_residual_w) < 1e-9
