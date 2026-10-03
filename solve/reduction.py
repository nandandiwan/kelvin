"""Galerkin thermal reduction with an unreduced patch and static interface lift.

Input: already assembled, constrained operators M [J/K], K [W/K]. Temperatures
are rises above a fixed ambient; loads [W] include any nonuniform boundary load
after subtracting K*T_ambient. No standard-cell power model is assumed here.

The exterior uses fixed-interface diffusion modes, augmented by the static
response to patch temperatures (component-mode synthesis). Patch coordinates
remain physical nodal temperatures. A constant-field enrichment preserves the
global energy test function. This is an offline, single-process reference;
dense bases are bounded by max_basis_bytes, not intended for full-chip DOFs.
"""

from collections import OrderedDict
from dataclasses import dataclass
import math

import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import eigsh, splu


def _symmetric(value, name):
    matrix = sparse.csc_matrix(value, dtype=float, copy=True)
    if matrix.shape[0] != matrix.shape[1] or not matrix.shape[0]:
        raise ValueError(f"{name} must be square and nonempty")
    if not np.isfinite(matrix.data).all():
        raise ValueError(f"{name} must be finite")
    error = matrix - matrix.T
    scale = max(np.max(abs(matrix.data), initial=0), np.finfo(float).tiny)
    if np.max(abs(error.data), initial=0) > 1e-12 * scale:
        raise ValueError(f"{name} must be symmetric")
    return matrix


def _mass_orthogonalize(columns, mass):
    """Normalize before Gram decomposition to avoid dropping SI-scaled modes."""
    if columns.shape[1] == 0:
        return columns
    norms = np.sqrt(np.maximum(np.sum(columns * (mass @ columns), axis=0), 0))
    keep = norms > np.finfo(float).tiny
    columns = columns[:, keep] / norms[keep]
    if not columns.shape[1]:
        return columns
    gram = columns.T @ (mass @ columns)
    values, vectors = linalg.eigh((gram + gram.T) / 2)
    keep = values > 1e-11 * max(values.max(), np.finfo(float).tiny)
    return columns @ (vectors[:, keep] / np.sqrt(values[keep]))


def component_basis(mass, stiffness, retained, mode_count, *, load_vectors=None,
                    max_basis_bytes=512 * 1024**2):
    """Retain patch/interface DOFs; reduce the remaining exterior.

    K_ee must be positive definite (the exterior is connected to the retained
    patch or a heat sink). Eigenvectors solve K_ee*phi=lambda*M_ee*phi.
    Optional load columns enrich the exterior with K_ee^-1 B_e, useful for
    distant sources. Counts are validated, and full-order exterior modes are
    supported for exact-reference comparisons on small problems.
    """
    mass, stiffness = _symmetric(mass, "mass"), _symmetric(stiffness, "stiffness")
    if mass.shape != stiffness.shape:
        raise ValueError("mass and stiffness shapes must match")
    n = mass.shape[0]
    raw = np.asarray(retained)
    if raw.ndim != 1 or (raw.size and raw.dtype.kind not in "iu"):
        raise ValueError("retained must contain integer DOF indices")
    retained = raw.astype(int)
    if (np.any(retained < 0) or np.any(retained >= n)
            or len(set(retained)) != len(retained)):
        raise ValueError("retained DOFs must be distinct and in range")
    exterior = np.setdiff1d(np.arange(n), retained)
    if (isinstance(mode_count, bool) or not isinstance(mode_count, (int, np.integer))
            or not 0 <= mode_count <= len(exterior)):
        raise ValueError("mode_count must lie between zero and the exterior size")
    loads = np.empty((n, 0)) if load_vectors is None else np.asarray(load_vectors, dtype=float)
    if loads.ndim != 2 or loads.shape[0] != n or not np.isfinite(loads).all():
        raise ValueError("load_vectors must be finite with one row per DOF")
    estimated_bytes = 8 * n * (len(retained) + mode_count + loads.shape[1] + 1)
    if estimated_bytes > max_basis_bytes:
        raise MemoryError("Requested dense basis exceeds max_basis_bytes; reduce the patch or use a distributed basis")
    if not len(exterior):
        return np.eye(n)[:, retained]
    mee = mass[exterior][:, exterior]
    kee = stiffness[exterior][:, exterior]
    ker = stiffness[exterior][:, retained]
    try:
        factor = splu(kee)
    except RuntimeError as error:
        raise ValueError("Exterior stiffness must be anchored and nonsingular") from error
    lift = np.zeros((len(exterior), len(retained)))
    # Interior patch columns have no exterior coupling: never solve for them.
    coupled = np.flatnonzero(np.asarray(abs(ker).sum(axis=0)).ravel())
    for column in coupled:
        lift[:, column] = factor.solve(-ker[:, column].toarray().ravel())
    if mode_count:
        if mode_count >= len(exterior) - 1:
            values, modes = linalg.eigh(kee.toarray(), mee.toarray(),
                                       subset_by_index=(0, mode_count - 1))
        else:
            values, modes = eigsh(kee, M=mee, k=mode_count, sigma=0, which="LM",
                                  v0=np.ones(len(exterior)))
        if np.any(values <= 0):
            raise ValueError("Exterior diffusion modes must have positive decay rates")
    else:
        modes = np.empty((len(exterior), 0))
    # The constant function must be in range(V): guarantees the full energy
    # balance is one of the reduced equations, even with Robin cooling.
    constant = np.ones(len(exterior)) - lift @ np.ones(len(retained))
    extras = [modes, constant[:, None]]
    if loads.shape[1]:
        extras.append(factor.solve(loads[exterior]))
    modes = _mass_orthogonalize(np.column_stack(extras), mee)
    basis = np.zeros((n, len(retained) + modes.shape[1]))
    basis[retained, np.arange(len(retained))] = 1
    basis[np.ix_(exterior, np.arange(len(retained)))] = lift
    basis[np.ix_(exterior, np.arange(len(retained), basis.shape[1]))] = modes
    return basis


@dataclass(frozen=True)
class ReducedStep:
    temperature_rise_k: np.ndarray
    iterations: int
    temperature_residual_k: float
    load_residual_w: float
    energy_residual_w: float


class ReducedHeatSolver:
    """Implicit transient in a supplied basis, with optional nonlinear power.

    power(theta) returns a full assembled nodal load [W] for the CURRENT trial
    temperature rise. It must be pure: rejected trials never advance state.
    On nonconvergence this solver retains its previous state. The full-space
    temperature is reconstructed for reference/patch validation each step.
    """

    def __init__(self, mass, stiffness, basis, initial_rise=None, cache_size=4):
        self.mass = _symmetric(mass, "mass")
        self.stiffness = _symmetric(stiffness, "stiffness")
        self.basis = np.array(basis, dtype=float, copy=True)
        n = self.mass.shape[0]
        if (self.stiffness.shape != self.mass.shape or self.basis.ndim != 2
                or self.basis.shape[0] != n or not self.basis.shape[1]
                or not np.isfinite(self.basis).all()):
            raise ValueError("Incompatible finite operators and basis required")
        if isinstance(cache_size, bool) or not isinstance(cache_size, int) or cache_size < 1:
            raise ValueError("cache_size must be a positive integer")
        self.m = self.basis.T @ (self.mass @ self.basis)
        self.k = self.basis.T @ (self.stiffness @ self.basis)
        self._mass_factor = linalg.cho_factor(self.m)
        self._factors = OrderedDict()
        self.cache_size = cache_size
        initial = np.zeros(n) if initial_rise is None else self._vector(initial_rise)
        self.state = linalg.cho_solve(self._mass_factor, self.basis.T @ (self.mass @ initial))
        self.initial_projection_error_k = float(np.max(abs(self.basis @ self.state - initial)))

    def _vector(self, value):
        array = np.asarray(value, dtype=float)
        if array.shape != (self.mass.shape[0],) or not np.isfinite(array).all():
            raise ValueError("Expected one finite entry per full-order DOF")
        return array

    def step(self, dt, power, *, temperature_tolerance_k=1e-7,
             load_atol_w=1e-14, load_rtol=1e-7, max_iterations=30, relaxation=1.0):
        controls = (dt, temperature_tolerance_k, load_atol_w, load_rtol, relaxation)
        if (any(isinstance(x, bool) or not math.isfinite(float(x)) for x in controls)
                or dt <= 0 or temperature_tolerance_k <= 0 or load_atol_w < 0
                or load_rtol < 0 or not 0 < relaxation <= 1
                or isinstance(max_iterations, bool) or not isinstance(max_iterations, int)
                or max_iterations < 1):
            raise ValueError("Invalid timestep or nonlinear convergence controls")
        if dt not in self._factors:
            self._factors[dt] = linalg.cho_factor(self.m / dt + self.k)
            if len(self._factors) > self.cache_size:
                self._factors.popitem(last=False)
        self._factors.move_to_end(dt)
        factor = self._factors[dt]
        previous = self.state.copy()
        old_field = self.basis @ previous
        trial = old_field.copy()
        constant_load = None if callable(power) else self._vector(power)
        for iteration in range(1, max_iterations + 1):
            load = self._vector(power(trial.copy())) if callable(power) else constant_load
            state = linalg.cho_solve(factor, self.m @ previous / dt + self.basis.T @ load)
            field = self.basis @ state
            if not np.isfinite(field).all():
                raise RuntimeError("Nonfinite reduced temperature")
            checked_load = self._vector(power(field.copy())) if callable(power) else load
            temperature_error = float(np.max(abs(field - trial)))
            load_error = float(np.max(abs(checked_load - load)))
            load_limit = load_atol_w + load_rtol * max(np.max(abs(load)), np.max(abs(checked_load)))
            if constant_load is not None or (temperature_error <= temperature_tolerance_k
                                             and load_error <= load_limit):
                residual = self.mass @ ((field - old_field) / dt) + self.stiffness @ field - checked_load
                self.state = state
                return ReducedStep(field.copy(), iteration, temperature_error, load_error,
                                   float(np.sum(residual)))
            trial += relaxation * (field - trial)
        raise RuntimeError(f"Reduced electrothermal solve did not converge after {max_iterations} iterations")
