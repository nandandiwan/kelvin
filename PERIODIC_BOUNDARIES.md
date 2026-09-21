# Lateral periodic thermal boundaries

The compact SRAM steady and transient entry points now default to **periodic
x/y boundaries**. The backside remains Robin, with `h = 20000 W/(m² K)` and
`T_ambient = 300 K`; the top remains insulating. Geometry, material properties,
transistor power, and workload timing are unchanged by this boundary change.

Other cases retain their previous boundary conditions unless explicitly enabled
with `BoundaryConditions(periodic_x=True, periodic_y=True)`.

## What periodicity means

Opposite lateral faces are identified by translation:

```text
T(x_max, y, z) = T(x_min, y, z)
T(x, y_max, z) = T(x, y_min, z)
```

The temperature and test-function spaces obey these constraints. The variational
equations then balance outgoing heat on one face with incoming heat on its
partner. Unlike insulation, heat may cross either face. This is not a prescribed
temperature and does not impose a reflection symmetry.

The compact case represents an idealized infinite repetition of the **same
orientation and the same power history**. The real SRAM macro includes mirrored
cells and row variants; an authentic macro model needs its actual translational
repeat unit and matching workload. Adding `--pad-um` repeats the whole padded
window, including its background material, not just the active cell.

Periodicity adds no net lateral cooling. With an insulating top and uniform
backside Robin coefficient, steady conservation still gives

```text
mean(T_backside) - T_ambient = P_total / (h * A_backside).
```

Consequently the mean temperature can remain similar to an insulating run even
when local gradients and heat paths change. This boundary change does not by
itself calibrate the package cooling coefficient or fix the geometry and power
limitations described in `ACCURACY_REVIEW.md`.

## Implementation and dependencies

Kelvin applies periodic constraints to the finite-element system using
`dolfinx_mpc`, not by averaging opposite temperatures after a solve. Install a
`dolfinx_mpc` release compatible with the DOLFINx version in the **same Python
environment**, including its compiled dependencies. The environment used for
the previous accuracy review contains DOLFINx 0.10.0. Nonperiodic runs do not
require the optional periodic-constraint dependency.

For this checkout, `dolfinx_mpc` 0.10.0 and `gdstk` 1.0.1 were built/installed
in the ignored local prefix `.deps/dolfinx-mpc-0.10`, without changing the shared
DOLFINx environment. From `kelvin/`, use:

```bash
export KELVIN_PYTHON=/global/home/krishnabhattaram/Neural-Network-Materials/Electrostats/env/bin/python
export PYTHONPATH="$PWD/.deps/dolfinx-mpc-0.10/python${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME=/tmp/kelvin-periodic-cache
"$KELVIN_PYTHON" -c 'import dolfinx_mpc, gdstk'
```

Use `"$KELVIN_PYTHON"` instead of `python` in the commands below. In restricted
serial environments, `UCX_TLS=self UCX_LOG_LEVEL=error` avoids unavailable UCX
transports; **do not use `UCX_TLS=self` for multi-process execution**.

The local dependency is machine-specific and intentionally not committed.
To reproduce it elsewhere, follow the [upstream build instructions](https://github.com/jorgensd/dolfinx_mpc/tree/v0.10.0)
against your installed DOLFINx. Matching the version alone is insufficient:
this host's DOLFINx uses nanobind 2.9.2 / ABI 16 with stable Python ABI. The
MPC Python CMake build therefore needs `STABLE_ABI` on `nanobind_add_module`
and `Development.SABIModule` in Python discovery. Build provenance and local
settings are recorded in `.deps/dolfinx-mpc-0.10/BUILD_INFO.md`.

The compiled dependency may live in an isolated installation prefix instead of
modifying the DOLFINx environment. In that case its Python package and shared
library directories must be exposed to that environment at runtime. Do not use
an unrelated system Python or combine incompatible DOLFINx/PETSc/MPI builds.

Face selection uses the mesh's coordinate bounds, so the periodic option is
available to both the legacy GDS builder and the notebook mesh import. The
current helper supports **scalar P1 temperature spaces on 3D meshes only**;
it does not enable periodicity for Kelvin's 2D cross-section solver. Opposite
faces must be complete rectangles describing a translational cell; the helper
checks their areas against the coordinate bounds. Interpolation on the
partner finite-element surface supports nonmatching boundary nodes; it is not
nearest-node snapping. Periodic edge and corner constraints must map to unique
representatives without cyclic constraints. With nonmatching meshes, boundary
agreement is understood in the discrete interpolation sense and should be
checked under refinement.

The geometric search is performed temporarily in a unit bounding box to avoid
absolute point/cell-search tolerances selecting incorrect tetrahedra at
micrometre scales. This affine normalization preserves interpolation weights.
The original metre-valued coordinates are restored exactly, even if constraint
construction fails, **before any thermal matrix is assembled**. Conductivity,
volumetric power and time units are unchanged. Chained edge constraints are
expanded to independent master DOFs before either solver assembles its system.

In transient solves the constrained system includes both the heat-capacity
(mass) and conduction/Robin matrices, as well as the time-dependent right-hand
side. Slave temperatures are reconstructed before temperatures are returned or
used for the next step. Lateral Robin cooling cannot be combined with this
periodic option. The helper also rejects **any Dirichlet condition touching a
paired boundary degree of freedom**, including edges and corners, even if its
temperature values would be mathematically compatible. A top or bottom
Dirichlet condition that includes those edges therefore requires a different
constraint treatment; the compact cases use Robin/insulating top and bottom
conditions and are unaffected.

## Run the compact SRAM cases

Run from `kelvin/` using a Python environment with Kelvin's usual simulation
dependencies plus compatible `dolfinx_mpc`. Separate output directories prevent
one comparison from replacing another:

```bash
PYTHONDONTWRITEBYTECODE=1 python cases/run_bitcell_compact.py \
    --lateral-bc periodic --out-dir out/bitcell_compact_periodic
PYTHONDONTWRITEBYTECODE=1 python cases/run_bitcell_compact.py \
    --lateral-bc insulating --out-dir out/bitcell_compact_insulating

PYTHONDONTWRITEBYTECODE=1 python cases/run_bitcell_transient.py \
    --point crowbar --lateral-bc periodic --out-dir out/bitcell_transient_periodic
PYTHONDONTWRITEBYTECODE=1 python cases/run_bitcell_transient.py \
    --point crowbar --lateral-bc insulating --out-dir out/bitcell_transient_insulating
```

These entry points obtain transistor powers through ngspice. This checkout now
has ngspice 41 installed in `.deps/ngspice`, with its runtime libraries and no
changes to the shared Python environment. Executable discovery checks an
explicit `KELVIN_NGSPICE` override, then `PATH`, then this local installation,
and finally the old developer path if it exists. An invalid explicit override
is an error, not a silent fallback. To select the tested installation:

```bash
export KELVIN_NGSPICE="$PWD/.deps/ngspice/bin/ngspice"
"$KELVIN_NGSPICE" --version
```

To recreate it, use `conda create --prefix "$PWD/.deps/ngspice"
--override-channels -c conda-forge ngspice-exe=41` (as one command).
Provenance is in `.deps/ngspice/BUILD_INFO.md`. Model files, electrical decks,
and power formulas have not been changed. All four existing named DC bias
points execute successfully. This does **not** establish full transient
electrical characterization: the existing WRITE-transient deck still reports
`Timestep too small`; its circuit/convergence issue is outside this boundary
change. Thermal-transient periodicity is independently tested with prescribed
time-dependent heat input.

The steady Python API accepts the same selection while retaining its existing
positional arguments:

```python
from cases.run_bitcell_compact import run

run(out_dir="out/bitcell_compact_periodic", lateral_bc="periodic")
```

For a mesh from `read_gds.ipynb`, pass the boundary settings explicitly:

```python
from mesh.gds_import import load_tagged_msh
from solve.steady import solve_steady
from spec.chip import BoundaryConditions

# msh_path, manifest_path and source_density_by_tag come from the import workflow.
case = load_tagged_msh(
    msh_path,
    manifest_path,
    material_by_mask={(66, 20): "PolySi"},  # notebook poly_Si naming override
    source_density_by_tag=source_density_by_tag,
    bcs=BoundaryConditions(periodic_x=True, periodic_y=True),
)
case.assert_solver_ready()
T, k, q = solve_steady(case.mesh_data, case.registry, case.chip)
```

This enables solver periodicity; it does not repair an invalid imported
geometry or automatically map the SRAM SPICE device powers to source tags.

## Verification criteria

A meaningful periodic regression must allow **nonzero lateral heat flux**;
a uniform one-dimensional slab cannot distinguish periodicity from insulation.
Check an asymmetric manufactured solution, edge/corner temperature constraints,
opposite-face flux balance, steady power conservation, transient stored energy
plus cooling, and refinement of nonmatching-face interpolation. A repeated
multi-cell domain with the same power per cell should reproduce the single-tile
solution. Keep total power and backside cooling fixed when comparing the SRAM's
periodic and insulating results.

## Verified results (2026-09-15)

- All **34 tests passed**: 13 new periodic tests, 5 executable-discovery tests,
  and the 16 existing specification, analytical, energy, and GDS tests.
  The slow GDS subset was run separately (4 passed); the remaining 30 passed
  together. The nonmatching-face constraint/energy test also passed with
  **two MPI processes**.
- Manufactured periodic temperature error decreased from 0.040846 K to
  0.011067 K on refinement (about order 1.88). Nonzero lateral flux errors
  also decreased; this test would not pass with insulating faces.
- On the actual SRAM geometry, an independent boundary-only diagnostic used
  explicitly synthetic, unequal 1 microW channel powers on the SAME mesh.
  Periodic paired-face temperature mismatch was at most 3.82e-10 K, versus
  0.00172/0.00285 K on insulating x/y faces. Relative power imbalance was
  4.62e-11. These are numerical checks, not SPICE power predictions.
- The saved notebook inverter mesh also solved with x/y periodicity despite
  lacking Kelvin lateral facet tags; its synthetic 1 microW load balanced
  backside heat loss to relative error 5.13e-10.

Fresh ngspice compact-model **crowbar DC snapshot**, with the existing duty
scaling and all other physics unchanged, produced 1.184203 microW. Both compact
case entry-point runs completed on 346,785 tetrahedra. The two saved Gmsh files
have identical SHA-256 hashes, so the comparison does not mix different meshes:

| Lateral condition | Peak temperature | Relative source/sink imbalance |
| --- | ---: | ---: |
| Insulating | 331.456365932 K | 1.35e-10 |
| Periodic x/y | 331.455694151 K | 1.51e-9 |

The peak difference is only about **0.000672 K**. This is consistent with the
unchanged backside resistance dominating the overall rise; it does not mean
the lateral faces are still insulating. The independent trace checks above
establish the difference in boundary behavior. These runs validate the BC
implementation within the existing compact/legacy-geometry model, not all
physical assumptions in that model.

Mesh/tag outputs are in `out/periodic_verification_dos7Z2/{periodic,insulating}`.
The verification used `run(..., renders=False, lateral_bc=...)`; no plots were
generated or previous simulation outputs overwritten.
