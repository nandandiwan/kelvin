"""Online thermal solves using preprojected source and observation maps.

No full field is reconstructed during stepping. Sources must stay in the
specified spatial span, and nonlinear power must depend only on the specified
temperature observations. This is an acceleration of a supplied reduced model,
not a new power library or an error guarantee for that model.
"""

from collections import OrderedDict
from dataclasses import dataclass
import math

import numpy as np
from scipy import linalg, sparse


@dataclass(frozen=True)
class PortStep:
    temperature_rise_k: np.ndarray
    source_power_w: np.ndarray
    iterations: int
    energy_residual_w: float


class ProjectedThermalPorts:
    """Build from ReducedHeatSolver and full-order source/observation maps.

    Each column of sources distributes one watt to full-order nodal loads;
    each row of observations averages full-order temperature rise. Both maps
    must be nonnegative and normalized to sum to one. Their sparse forms are
    accepted. Only projected maps and operators are retained after setup.
    """

    def __init__(self, reference, sources, observations):
        sources = sparse.csc_matrix(sources, dtype=float)
        observations = sparse.csr_matrix(observations, dtype=float)
        n = reference.mass.shape[0]
        if (sources.shape[0] != n or observations.shape[1] != n
                or sources.shape[1] < 1 or observations.shape[0] < 1):
            raise ValueError("Source and observation maps must match the full system")
        for matrix, axis in [(sources, 0), (observations, 1)]:
            if (not np.isfinite(matrix.data).all() or np.any(matrix.data < 0)
                    or not np.allclose(np.asarray(matrix.sum(axis=axis)), 1, rtol=0, atol=1e-12)):
                raise ValueError("Maps must be finite, nonnegative and normalized")
        self.m = reference.m.copy()
        self.k = reference.k.copy()
        self.b = np.asarray(sources.T @ reference.basis).T
        self.c = np.asarray(observations @ reference.basis)
        self.energy_mass = np.asarray(reference.mass.sum(axis=0)) @ reference.basis
        self.energy_loss = np.asarray(reference.stiffness.sum(axis=0)) @ reference.basis
        self.state = reference.state.copy()
        self._factors = OrderedDict()
        self.cache_size = reference.cache_size

    def _power(self, power):
        values = np.asarray(power, dtype=float)
        if (values.shape != (self.b.shape[1],) or not np.isfinite(values).all()
                or np.any(values < 0)):
            raise ValueError("Expected finite nonnegative watts per source port")
        return values

    def step(self, dt, power, *, temperature_tolerance_k=1e-7,
             power_atol_w=1e-14, power_rtol=1e-7, max_iterations=30, relaxation=1.0):
        """Step with watts or pure power(observed temperature rise) callback.

        Callback trials always restart from the accepted previous state.
        Failure leaves state unchanged. Convergence is checked at ALL supplied
        temperature/source ports; unobserved full-field error is not checked.
        """
        controls = (dt, temperature_tolerance_k, power_atol_w, power_rtol, relaxation)
        if (any(isinstance(x, bool) or not math.isfinite(float(x)) for x in controls)
                or dt <= 0 or temperature_tolerance_k <= 0 or power_atol_w < 0
                or power_rtol < 0 or not 0 < relaxation <= 1
                or isinstance(max_iterations, bool) or not isinstance(max_iterations, int)
                or max_iterations < 1):
            raise ValueError("Invalid timestep or convergence controls")
        if dt not in self._factors:
            self._factors[dt] = linalg.cho_factor(self.m / dt + self.k)
            if len(self._factors) > self.cache_size:
                self._factors.popitem(last=False)
        self._factors.move_to_end(dt)
        previous = self.state.copy()
        history = self.m @ previous / dt
        trial = self.c @ previous
        nonlinear = callable(power)
        constant = None if nonlinear else self._power(power)
        for iteration in range(1, max_iterations + 1):
            supplied = self._power(power(trial.copy())) if nonlinear else constant
            state = linalg.cho_solve(self._factors[dt], history + self.b @ supplied)
            observed = self.c @ state
            if not np.isfinite(state).all() or not np.isfinite(observed).all():
                raise RuntimeError("Nonfinite reduced temperature")
            checked = self._power(power(observed.copy())) if nonlinear else supplied
            temperature_error = float(np.max(abs(observed - trial)))
            power_error = float(np.max(abs(checked - supplied)))
            power_limit = power_atol_w + power_rtol * max(np.max(abs(checked)), np.max(abs(supplied)))
            if not nonlinear or (temperature_error <= temperature_tolerance_k and power_error <= power_limit):
                residual = (self.energy_mass @ ((state - previous) / dt)
                            + self.energy_loss @ state).item() - float(checked.sum())
                self.state = state
                return PortStep(observed.copy(), checked.copy(), iteration, residual)
            trial += relaxation * (observed - trial)
        raise RuntimeError(f"Projected electrothermal solve did not converge after {max_iterations} iterations")
