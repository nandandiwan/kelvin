"""Compare a fully resolved 3D FEM slab against patch/exterior mode reduction.

This is a numerical benchmark, not an SRAM or CPU temperature prediction.
All runs share the same assembled FEM operators, source, and backward Euler
steps, isolating reduction error from mesh/time discretization error.
"""

import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from dolfinx import fem, mesh as dmesh
from mpi4py import MPI

from solve.reduction import component_basis, ReducedHeatSolver
from solve.reduction_fem import assemble_thermal_operators
from solve.reduced_ports import ProjectedThermalPorts
from spec.chip import BoundaryConditions


def benchmark(n=10, modes=(0, 8, 24), patch_fractions=(.2, .4)):
    from physics.bcs import FACET_BOTTOM
    if MPI.COMM_WORLD.size != 1:
        raise ValueError("Run this reference benchmark on one MPI rank")
    mesh = dmesh.create_box(MPI.COMM_SELF, [[0, 0, 0], [4e-6, 4e-6, 2e-6]], [n, n, max(2, n // 2)])
    facets = dmesh.locate_entities_boundary(mesh, 2, lambda x: np.isclose(x[2], 0, atol=1e-14))
    tags = dmesh.meshtags(mesh, 2, facets, np.full(len(facets), FACET_BOTTOM, dtype=np.int32))
    space = fem.functionspace(mesh, ("DG", 0))
    k, capacity, source = [fem.Function(space) for _ in range(3)]
    k.x.array[:] = 148
    capacity.x.array[:] = 1.63e6
    source.interpolate(lambda x: ((abs(x[0] - 2e-6) < .5e-6)
                                  & (abs(x[1] - 2e-6) < .5e-6)
                                  & (x[2] > 1.5e-6)).astype(float) * 1e14)
    case = SimpleNamespace(mesh=mesh, facet_tags=tags)
    chip = SimpleNamespace(bcs=BoundaryConditions(ambient_t_k=300, backside_h_eff=20000,
                                                periodic_x=False, periodic_y=False))
    system = assemble_thermal_operators(case, k, capacity, chip)
    load = system.load(source)
    coordinates = system.space.tabulate_dof_coordinates()[system.independent_dofs]
    # Source-local outputs stay identical when the retained patch is enlarged.
    outputs = np.flatnonzero((abs(coordinates[:, 0] - 2e-6) <= .4e-6)
                            & (abs(coordinates[:, 1] - 2e-6) <= .4e-6))
    dt = 2e-8
    amplitudes = [1 if i < 8 else (0 if i < 18 else (.4 if i < 35 else 0)) for i in range(60)]
    factor = splu(system.mass / dt + system.stiffness)
    state = np.zeros(system.mass.shape[0])
    full_history = []
    started = time.perf_counter()
    for amplitude in amplitudes:
        state = factor.solve(system.mass @ state / dt + amplitude * load)
        full_history.append(state.copy())
    full_seconds = time.perf_counter() - started
    reference = np.asarray(full_history)
    results = []
    for fraction in patch_fractions:
        patch = np.flatnonzero((abs(coordinates[:, 0] - 2e-6) <= fraction * 2e-6 + 1e-14)
                              & (abs(coordinates[:, 1] - 2e-6) <= fraction * 2e-6 + 1e-14))
        for count in modes:
            started = time.perf_counter()
            basis = component_basis(system.mass, system.stiffness, patch, count)
            solver = ReducedHeatSolver(system.mass, system.stiffness, basis)
            observations = sparse.csr_matrix((np.ones(len(outputs)),
                                               (np.arange(len(outputs)), outputs)),
                                              shape=(len(outputs), len(load)))
            ports = ProjectedThermalPorts(solver, (load / load.sum())[:, None], observations)
            setup_seconds = time.perf_counter() - started
            history, energy_residuals = [], []
            started = time.perf_counter()
            for amplitude in amplitudes:
                result = solver.step(dt, amplitude * load)
                history.append(result.temperature_rise_k)
                energy_residuals.append(abs(result.energy_residual_w))
            seconds = time.perf_counter() - started
            error = np.asarray(history) - reference
            port_history, port_energy = [], []
            started = time.perf_counter()
            for amplitude in amplitudes:
                result = ports.step(dt, [amplitude * load.sum()])
                port_history.append(result.temperature_rise_k)
                port_energy.append(abs(result.energy_residual_w))
            port_seconds = time.perf_counter() - started
            port_error = np.asarray(port_history) - np.asarray(history)[:, outputs]
            if np.max(abs(port_error)) > 1e-9:
                raise RuntimeError("Projected ports disagree with full-field reduced reference")
            results.append({
                "patch_fraction": fraction, "retained_dofs": len(patch), "exterior_modes": count,
                "reduced_dofs": basis.shape[1], "setup_seconds": setup_seconds,
                "advance_seconds": seconds, "max_patch_error_k": float(np.max(abs(error[:, outputs]))),
                "max_full_error_k": float(np.max(abs(error))),
                "max_energy_residual_w": max(energy_residuals),
                "projected_advance_seconds": port_seconds,
                "projected_vs_reconstructed_max_error_k": float(np.max(abs(port_error))),
                "projected_max_energy_residual_w": max(port_energy),
            })
    return {"scope": "uniform silicon 3D FEM numerical reduction benchmark, not chip calibration",
            "full_dofs": system.mass.shape[0], "full_advance_seconds": full_seconds,
            "reference_peak_rise_k": float(reference.max()), "source_power_w": float(load.sum()),
            "dt_s": dt, "steps": len(amplitudes), "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output exists; use a fresh evidence file")
    result = benchmark(args.n)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
