"""Compare operator reuse with forced reassembly for physical heat solves."""

import numpy as np
import pytest
from dataclasses import replace

from test_periodic import _case
from solve.transient import TransientHeatSolver


@pytest.mark.parametrize("periodic", [False, True])
def test_cached_steps_match_reassembly_and_preserve_prior_fields(periodic):
    mesh, k, rho, q, chip = _case(3)
    chip.bcs = replace(chip.bcs, periodic_x=periodic, periodic_y=periodic)
    cached = TransientHeatSolver(mesh, k, rho, chip, dt=.01, matrix_cache_size=2)
    reference = TransientHeatSolver(mesh, k, rho, chip, dt=.01, matrix_cache_size=1)
    saved = []
    for dt in [.01, .02, .01, .02, .01]:
        actual = cached.step(q.x.array, dt=dt)
        expected = reference.step(q.x.array, dt=dt)
        np.testing.assert_allclose(actual.x.array, expected.x.array, atol=1e-9, rtol=0)
        saved.append((actual, actual.x.array.copy()))
    for field, snapshot in saved:
        np.testing.assert_array_equal(field.x.array, snapshot)
    assert cached.statistics["matrix_assemblies"] == 2
    assert reference.statistics["matrix_assemblies"] == 5
    assert cached.statistics["steps"] == 5
    assert cached.statistics["krylov_iterations"] > 0
    cached.clear_operator_cache()
    assert len(cached._operators) == 1


def test_uniform_timestamp_roundoff_does_not_rebuild_matrix():
    mesh, k, rho, q, chip = _case(2)
    edges = np.linspace(0.0, 9.39519e-10, 189)
    solver = TransientHeatSolver(mesh, k, rho, chip, dt=edges[1])
    for dt in np.diff(edges):
        solver._select_operator(float(dt))
    assert solver.statistics["matrix_assemblies"] == 1
    solver._select_operator(float(edges[1] * 1.000001))
    assert solver.statistics["matrix_assemblies"] == 2


@pytest.mark.parametrize("dt", [0, -1, float("nan"), float("inf"), True])
def test_invalid_timestep_is_rejected(dt):
    with pytest.raises(ValueError, match="dt"):
        TransientHeatSolver._validate_dt(dt)
