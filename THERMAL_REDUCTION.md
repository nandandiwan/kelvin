# Detailed patches and thermal mode reduction

The current implementation is a numerical reference for reducing a thermal
system, not a CPU or SRAM power library. It uses already assembled FEM mass,
conductance and source operators. No standard cells or empirical power models
are introduced.

For temperature rise above a fixed ambient, the full system is
`M dtheta/dt + K theta = f`. The basis V produces `Mr=V.T M V`, `Kr=V.T K V`
and `fr=V.T f`. Positive mass and dissipative conductance are preserved by this
Galerkin projection; nodal positivity and pointwise error bounds are not
guaranteed by modal truncation.

## Keeping a detailed patch

`solve/reduction.py:component_basis` retains every specified patch DOF. Only
exterior DOFs are reduced. The exterior part of each retained basis vector is
the static lift `-K_ee^-1 K_er`; additional modes solve
`K_ee phi = lambda M_ee phi`. This couples both regions through one system,
including interface mass contributions. It does not freeze a boundary
temperature from a preliminary coarse solve and does not double-count the
patch's capacity or sources.

The constant-field enrichment keeps the full energy test function in the
trial space. Optional static load-response vectors capture known distant
heat sources. Neither energy conservation nor nonlinear convergence alone
certifies local temperature accuracy. Compare additional exterior modes,
larger patches, refined meshes, shorter timesteps and unseen source histories.
Fast pulses can require fast modes omitted by a steady-state approximation.

`ReducedHeatSolver` uses backward Euler and bounded factorization reuse.
Temperature-dependent load callbacks are evaluated at each trial field;
failure to meet temperature and load tolerances leaves the old state intact.
Callbacks receive temperature rise, not absolute Kelvin. Initial conditions
are mass-projected and their maximum projection error is exposed explicitly.

## FEM connection and limits

`solve/reduction_fem.py` exports the existing FEM operators on one MPI rank.
Periodic constraints are represented by an explicit sparse prolongation P;
the operators and source are transformed consistently with `P.T A P` and
`P.T f`. Synthetic slave diagonals are excluded from the eigenproblem.
The exporter rejects multi-rank input instead of silently gathering large
operators. A distributed or hierarchical offline basis builder is still
needed for chip scale. Dense basis allocation has an explicit memory guard.
The current backend reconstructs the full field per step for verification;
production use should project source and observation maps in advance.

This formulation retains a detailed patch within one discretized system.
An independently meshed coarse chip plus fine patch needs conservative
interface interpolation, replacing (not adding to) the coarse patch.
Matching temperature and opposite normal heat flow gives a consistent
coupling; it does not establish a first-order error guarantee. Conductivity
contrasts, distant heat sources, package boundaries and patch extent can all
dominate the error.

## Reproducible checks

Use a compatible DOLFINx/dolfinx_mpc environment and a fresh output path:

```bash
python cases/limited_run.py --cores 4 --threads 1 -- \
  python -B -m pytest -q tests/test_reduction.py tests/test_reduction_fem.py
python cases/limited_run.py --cores 4 --threads 1 -- \
  python -B cases/verify_patch_reduction.py --out out/reduction.json
```

2026-10-03: nine reduction tests passed with DOLFINx/MPC 0.11. Full-mode
reconstruction agreed with the public FEM transient for both natural and
nonmatching periodic lateral boundaries. Tests also cover held-out pulses,
mode enrichment, static distant-source enrichment, energy conservation,
temperature-dependent loads and rollback on failed convergence.

The default 3D silicon-slab benchmark has 726 DOFs, 60 steps of 20 ns, and a
0.284998 K peak rise. The same source-local probes are used for every patch:

| Retained patch DOFs | Exterior modes | Total reduced DOFs | Maximum probe error (K) |
| --- | --- | --- | --- |
| 54 | 0 | 55 | 0.0057889 |
| 54 | 8 | 63 | 0.00015015 |
| 54 | 24 | 79 | 0.00005805 |
| 150 | 0 | 151 | 0.0034253 |
| 150 | 8 | 159 | 0.00015396 |
| 150 | 24 | 175 | 0.00003517 |

Maximum global energy residual was below 8e-19 W on a 4.48e-5 W source.
Errors are not strictly monotone with patch size at fixed mode count. The
full sparse solver took about 7 ms for the transient, while reduced solves
took 8–34 ms plus 16–48 ms setup on this small benchmark: no speedup is
claimed. The benchmark isolates reduction error; it is not a mesh-convergence
study, an extracted chip model, or evidence for calibrated SRAM temperature.

## Integration order

1. Verify full SRAM transient mesh/time/coupling convergence, with a 32-core
   total budget for our jobs.
2. Introduce continuous small-array electrical sequences using existing
   SKY130 compact models, then conservative source mapping into actual array
   geometry. Do not duplicate the single-cell lumped bitline load per cell.
3. Validate mode/patch approximations against those exact small-array cases.
4. Use existing published cell/memory models or direct simulation for larger
   workloads; defer a new standard-cell power/thermal library.
5. Connect Chipyard activity and placement. Ingest supplied extracted netlists
   and their model/RC collateral when the licensed extraction flow is ready.

External netlists require their PDK models, pin mapping and extraction corner;
a Virtuoso license alone does not establish an extraction capability or model
compatibility. Exact SPICE remains appropriate for selected small regions.
