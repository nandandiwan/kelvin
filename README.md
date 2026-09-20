# Kelvin: SRAM simulation review

Kelvin connects a SKY130 SRAM bitcell SPICE simulation to a three-dimensional
finite-element heat solver. The current review branch contains the integrated
single-bitcell pipeline, tests, and documentation. It is not a calibrated
absolute-temperature model of the entire SRAM macro.

## Start here

- [13-section code-review guide](SRAM_PIPELINE_CODE_REVIEW_AND_PRS.md): conceptual
  stages, files/functions, contracts, and acceptance criteria.
- [Draft PR scope and review checklist](SRAM_REVIEW_PR.md): one PR, 13 sections.
- [SRAM simulation overview](SRAM_SIMULATION_OVERVIEW.md): model and limitations.
- [Mesh notebook](read_gds.ipynb) and [notebook workflow](SRAM_NOTEBOOK_PIPELINE.md).
- [Outstanding accuracy work](SRAM_PIPELINE_CODE_REVIEW_AND_PRS.md#6-accuracy-follow-ups-separate-from-packaging-existing-code):
  complete workloads, target-macro loading, cooling/repeat unit, electrothermal
  feedback, process/interface fidelity, and numerical convergence.

The canonical drivers are `cases/run_bitcell_compact.py` (steady average) and
`cases/run_bitcell_transient.py` (transient). Other array/gallery/coarse studies
remain legacy routes; their presence does not imply they use the same model.

## Runtime requirements

Use a compatible scientific Python environment with DOLFINx and dolfinx_mpc
**0.10.0**, Gmsh, gdstk, NumPy, SciPy, mpi4py, petsc4py, IPython, and pytest.
Matplotlib, PyVista, and Pillow support plotting. The recorded review environment
uses Python 3.12.13, Gmsh 4.15.2, gdstk 1.0.1, and ngspice 41.
These are tested versions, not a portable environment lock or installation script.
MPC is required for the default periodic boundaries; the drivers fail rather
than silently substituting insulating sides.

An ngspice executable must be available on `PATH`, through `KELVIN_NGSPICE`, or
in the optional local `.deps/ngspice/bin/` installation. Bundled SKY130 model
inputs are under `data/`; see the included provenance/license files. Installed
dependencies in `.deps/` and generated results in `out/` are intentionally not
committed. A fresh clone therefore requires dependency setup and new simulations.

From the repository root, after activating the environment:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -c 'import dolfinx, dolfinx_mpc, gdstk, gmsh, scipy'
python -c 'from gds.spice_power import find_ngspice; print(find_ngspice())'

# Generate a sealed mesh bundle, without running SPICE.
python cases/build_sram_mesh.py --out-dir out/review_mesh

# Reuse the mesh with event-averaged READ transistor and local-wire heat.
python cases/run_bitcell_compact.py --point read_1 \
  --mesh-manifest out/review_mesh/sram_sp_cell_material_regions.json \
  --out-dir out/review_read --no-renders
```

Use a fresh output directory for each run. The notebook's final solve cell is
disabled by default; select `KELVIN_PYTHON` if its kernel lacks DOLFINx.

## Tests and interpretation

```bash
python -m pytest -q tests --ignore=tests/test_gds_pipeline.py
```

The excluded file exercises the slow legacy full-macro 2D path and can be run
separately. Some integration tests require ngspice or optional scientific
dependencies, and saved-full-mesh tests may skip when local artifacts are absent;
inspect the skip summary (`-ra`). Source/energy conservation, geometry checks,
and passing tests verify implementation contracts, not fabricated-chip accuracy.

Historical result links under `out/` in the review documents are local evidence,
not files delivered by this repository. In particular, steady/full-substrate
renderer prototypes in that directory have not yet been promoted to supported
entry points. No additional calibration or F01–F06 accuracy fix is claimed here.
