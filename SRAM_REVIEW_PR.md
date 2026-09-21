## Summary

Publish the existing SRAM SPICE-to-thermal implementation as **one draft PR**
with **13 conceptual review sections**. This is not a series of 13 implemented
accuracy fixes. The detailed guide describes code, functions, interfaces,
acceptance criteria, and future work for each section.

The model is the `sram_sp_cell` from `sram22_64x22m4w22.gds`, not an extracted
simulation of the entire macro. Local transistor and wire losses are coupled to
a polygon-preserving, conforming thermal mesh with lateral periodic boundaries.

## Base and compatibility

- Target: `nandandiwan/kelvin`, `main`.
- Review branch: `review/sram-pipeline`.
- Original local baseline: `ce0f1e2`; latest fetched upstream baseline:
  `21e6049` (including `a598ed1`). Upstream changes are preserved by merge.
- Legacy array/gallery examples receive per-instance-ID compatibility updates;
  they are not silently converted to the new transient-SPICE/notebook pipeline.
- The notebook is now included inside the repository with portable paths.
  No parent-workspace notebook or inverter fixture is required by its tests.

## Review sections

Each link opens the corresponding code/function review packet. Checkboxes are
for reviewer sign-off, not a claim that every proposed acceptance item is done.

- [ ] [01 — Input identity and transistor/layout correspondence](https://github.com/nandandiwan/kelvin/blob/review/sram-pipeline/SRAM_PIPELINE_CODE_REVIEW_AND_PRS.md#pr-01)
- [ ] [02 — Local interconnect/contact resistance extraction](https://github.com/nandandiwan/kelvin/blob/review/sram-pipeline/SRAM_PIPELINE_CODE_REVIEW_AND_PRS.md#pr-02)
- [ ] [03 — SPICE access decks and execution](https://github.com/nandandiwan/kelvin/blob/review/sram-pipeline/SRAM_PIPELINE_CODE_REVIEW_AND_PRS.md#pr-03)
- [ ] [04 — Waveform validation and heat extraction](https://github.com/nandandiwan/kelvin/blob/review/sram-pipeline/SRAM_PIPELINE_CODE_REVIEW_AND_PRS.md#pr-04)
- [ ] [05 — Workload and event-to-average power contract](https://github.com/nandandiwan/kelvin/blob/review/sram-pipeline/SRAM_PIPELINE_CODE_REVIEW_AND_PRS.md#pr-05)
- [ ] [06 — Physical-region construction from GDS polygons](https://github.com/nandandiwan/kelvin/blob/review/sram-pipeline/SRAM_PIPELINE_CODE_REVIEW_AND_PRS.md#pr-06)
- [ ] [07 — Conforming mesh generation and sealed bundles](https://github.com/nandandiwan/kelvin/blob/review/sram-pipeline/SRAM_PIPELINE_CODE_REVIEW_AND_PRS.md#pr-07)
- [ ] [08 — Mesh import, SI units, and material coefficients](https://github.com/nandandiwan/kelvin/blob/review/sram-pipeline/SRAM_PIPELINE_CODE_REVIEW_AND_PRS.md#pr-08)
- [ ] [09 — Conservative transistor/wire heat projection](https://github.com/nandandiwan/kelvin/blob/review/sram-pipeline/SRAM_PIPELINE_CODE_REVIEW_AND_PRS.md#pr-09)
- [ ] [10 — Robin and periodic boundary conditions](https://github.com/nandandiwan/kelvin/blob/review/sram-pipeline/SRAM_PIPELINE_CODE_REVIEW_AND_PRS.md#pr-10)
- [ ] [11 — Steady/transient solvers and energy budgets](https://github.com/nandandiwan/kelvin/blob/review/sram-pipeline/SRAM_PIPELINE_CODE_REVIEW_AND_PRS.md#pr-11)
- [ ] [12 — CLI/notebook integration and persistence](https://github.com/nandandiwan/kelvin/blob/review/sram-pipeline/SRAM_PIPELINE_CODE_REVIEW_AND_PRS.md#pr-12)
- [ ] [13 — Visualization, reproducibility, and review evidence](https://github.com/nandandiwan/kelvin/blob/review/sram-pipeline/SRAM_PIPELINE_CODE_REVIEW_AND_PRS.md#pr-13)

## Verification

Verified on 2026-09-20 using a clean local clone of code commit `ec96abf`:

```bash
python -m pytest -q -ra -p no:cacheprovider tests --ignore=tests/test_gds_pipeline.py
```

- **534 passed, 4 skipped in 159.02 seconds.** The four skips require a generated
  sealed full-SRAM mesh (device-source, wire-source, and two FEM integration tests).
- The four slow legacy full-macro 2D tests in `test_gds_pipeline.py` were excluded,
  not reported as passing. No full SRAM mesh rebuild or thermal production run
  was performed for publication.
- Real ngspice access/integration tests ran using an explicitly configured
  ngspice 41 executable. The clone contained no `.deps/`, saved `out/`, parent
  notebook, or parent PDK fixture. External installed Python/MPC dependencies
  were supplied separately; this is a source-portability check, not a clean
  dependency-installation/container test.
- Includes 18 notebook-bundle tests and 13 upstream-compatibility regression tests.
- Notebook configuration/extraction/region-preview checks retained all 19 cell
  IDs, constructed 30 regions with all eight mapped sources, and left the original
  parent-workspace notebook unchanged.
- `git diff --check origin/main...HEAD` passes; generated results and installed
  binaries remain excluded. A targeted common-token/private-key signature scan
  found no matches in the new source/docs/notebook; this is not a full security audit.

Earlier numerical evidence in the detailed guide is historical, not a claim
that every full thermal simulation was rerun for publication. Commits after
`ec96abf` that record these results change documentation only.

## Unresolved accuracy work

The six follow-ups in the guide remain open:

1. Complete repeating access/recharge/idle workloads and holding leakage.
2. Target-macro timing, extracted loading, and shared-wire representation.
3. Calibrated cooling, substrate domain, and actual array repeat geometry/activity.
4. Electrical/thermal temperature feedback and temperature-dependent properties.
5. Gate oxide, interfaces, calibrated process properties, and source-depth fidelity.
6. Thermal mesh/time/domain convergence and the known transient sub-ambient undershoot.

The historical READ steady result (~420.18 K) is a reproduction baseline, not a
validated silicon-temperature target; ~119.19 K of its ~120.18 K rise is set by
the assumed backside resistance. Source conservation does not validate that
boundary condition. PR sections 12–13 also describe future output/renderer
improvements that are explicitly not completed by publishing the existing code.

## Excluded artifacts

No generated waveforms, meshes, fields, GIFs, installed `.deps/` binaries, or
machine caches are included. Historical `out/` links may be absent in a clone.
Steady-renderer prototypes under `out/` remain unsupported local evidence.
