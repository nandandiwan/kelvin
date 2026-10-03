"""Independent full linear solves, energy balance, and feedback failure checks."""

import numpy as np
import pytest
from scipy import sparse
from scipy.sparse.linalg import spsolve

from solve.reduction import component_basis, ReducedHeatSolver


def thermal_chain(n=30):
    # Heterogeneous finite-volume thermal bar: positive capacities, reciprocal
    # face conductances, Robin heat sinks at the two ends.
    c = np.linspace(1, 3, n) * 1e-9
    g = np.linspace(1, 4, n - 1) * 1e-5
    diagonal = np.r_[g, 0] + np.r_[0, g]
    diagonal[[0, -1]] += 2e-5
    return sparse.diags(c).tocsc(), sparse.diags([-g, diagonal, -g], [-1, 0, 1]).tocsc()


def test_complete_basis_recovers_full_transient_for_changing_loads_and_dt():
    m, k = thermal_chain(15)
    retained = np.arange(5, 10)
    v = component_basis(m, k, retained, 10)
    reduced = ReducedHeatSolver(m, k, v)
    expected = np.zeros(15)
    rng = np.random.default_rng(42)
    for dt in [1e-6, 3e-6, 1e-6, 2e-6]:
        load = rng.random(15) * 1e-5
        expected = spsolve(m / dt + k, m @ expected / dt + load)
        step = reduced.step(dt, load)
        np.testing.assert_allclose(step.temperature_rise_k, expected, atol=1e-10, rtol=1e-9)
        assert abs(step.energy_residual_w) < 1e-15


def test_patch_coordinates_preserved_and_constant_field_representable():
    m, k = thermal_chain()
    patch = np.arange(12, 18)
    v = component_basis(m, k, patch, 3)
    np.testing.assert_array_equal(v[patch, :len(patch)], np.eye(len(patch)))
    np.testing.assert_array_equal(v[patch, len(patch):], 0)
    coefficients = np.linalg.lstsq(v, np.ones(30), rcond=None)[0]
    np.testing.assert_allclose(v @ coefficients, 1, atol=1e-9)
    solver = ReducedHeatSolver(m, k, v)
    for _ in range(20):
        load = np.zeros(30)
        load[15] = 1e-4
        assert abs(solver.step(1e-5, load).energy_residual_w) < 1e-14


def test_mode_enrichment_converges_for_held_out_pulses():
    m, k = thermal_chain()
    patch = np.arange(12, 18)
    errors = []
    for modes in [0, 3, 12, 24]:
        solver = ReducedHeatSolver(m, k, component_basis(m, k, patch, modes))
        reference = np.zeros(30)
        error = 0
        for index in range(50):
            load = np.zeros(30)
            load[15 if index < 25 else 2] = 1e-4 if index % 7 < 3 else 0
            reference = spsolve(m / 1e-5 + k, m @ reference / 1e-5 + load)
            field = solver.step(1e-5, load).temperature_rise_k
            error = max(error, np.max(abs(field[patch] - reference[patch])))
        errors.append(error)
    assert errors[-1] < 1e-8
    assert errors[2] < errors[1] < errors[0]


def test_temperature_dependent_load_matches_independent_linear_reference():
    m, k = thermal_chain(12)
    solver = ReducedHeatSolver(m, k, np.eye(12))
    p0 = np.full(12, 1e-5)
    slope = sparse.eye(12) * 1e-6
    dt = 1e-5
    result = solver.step(dt, lambda theta: p0 + slope @ theta)
    expected = spsolve(m / dt + k - slope, p0)
    np.testing.assert_allclose(result.temperature_rise_k, expected, atol=1e-8, rtol=0)
    assert result.iterations > 1


def test_nonconverged_trial_does_not_advance_state():
    m, k = thermal_chain(8)
    solver = ReducedHeatSolver(m, k, np.eye(8))
    old = solver.state.copy()
    with pytest.raises(RuntimeError, match="did not converge"):
        solver.step(1e-4, lambda theta: np.full(8, 1e-3) + theta,
                    max_iterations=2)
    np.testing.assert_array_equal(solver.state, old)


def test_basis_allocation_guard_and_input_validation():
    m, k = thermal_chain()
    with pytest.raises(MemoryError):
        component_basis(m, k, [10], 2, max_basis_bytes=1)
    with pytest.raises(ValueError, match="distinct"):
        component_basis(m, k, [10, 10], 2)
    with pytest.raises(ValueError, match="integer"):
        component_basis(m, k, [1.5], 2)
    with pytest.raises(ValueError, match="mode_count"):
        component_basis(m, k, [10], -1)
    with pytest.raises(ValueError, match="symmetric"):
        ReducedHeatSolver(m, k + sparse.diags(np.ones(29), 1), np.eye(30))


def test_static_load_enrichment_recovers_steady_response():
    m, k = thermal_chain()
    load = np.zeros(30)
    load[2] = 1e-4
    v = component_basis(m, k, np.arange(12, 18), 0, load_vectors=load[:, None])
    solver = ReducedHeatSolver(m, k, v)
    reduced = v @ np.linalg.solve(solver.k, v.T @ load)
    np.testing.assert_allclose(reduced, spsolve(k, load), atol=1e-10, rtol=1e-10)
