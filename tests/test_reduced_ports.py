import numpy as np
import pytest
from scipy import sparse

from test_reduction import thermal_chain
from solve.reduction import component_basis, ReducedHeatSolver
from solve.reduced_ports import ProjectedThermalPorts


def models():
    m, k = thermal_chain()
    basis = component_basis(m, k, np.arange(12, 18), 6)
    reference = ReducedHeatSolver(m, k, basis)
    source = sparse.csc_matrix((np.ones(2), ([3, 15], [0, 1])), shape=(30, 2))
    observe = source.T
    return reference, ProjectedThermalPorts(reference, source, observe), source, observe


@pytest.mark.parametrize("feedback", [False, True])
def test_preprojected_ports_match_full_field_reference(feedback):
    reference, ports, source, observe = models()
    for index, dt in enumerate([1e-5, 2e-5, 1e-5] * 4):
        base = np.array([1e-5, 4e-5]) * (index % 3 > 0)
        power = lambda theta: base + 1e-6 * theta
        full = reference.step(dt, (lambda field: source @ power(observe @ field))
                              if feedback else source @ base)
        projected = ports.step(dt, power if feedback else base)
        np.testing.assert_allclose(projected.temperature_rise_k,
                                   observe @ full.temperature_rise_k, atol=1e-8, rtol=0)
        assert abs(projected.energy_residual_w) < 1e-12


def test_failed_port_feedback_rolls_back_and_rejects_invalid_power():
    _, ports, _, _ = models()
    original = ports.state.copy()
    with pytest.raises(RuntimeError, match="did not converge"):
        ports.step(1e-5, lambda theta: np.full(2, 1e-3) + theta, max_iterations=2)
    np.testing.assert_array_equal(ports.state, original)
    for values in [[-1, 0], [1], [0, np.nan]]:
        with pytest.raises(ValueError):
            ports.step(1e-5, values)
    np.testing.assert_array_equal(ports.state, original)


def test_normalized_maps_required():
    reference, _, source, observe = models()
    with pytest.raises(ValueError, match="normalized"):
        ProjectedThermalPorts(reference, 2 * source, observe)
    with pytest.raises(ValueError, match="normalized"):
        ProjectedThermalPorts(reference, source, -observe)
