# GDS/Gmsh → Kelvin bridge

Kelvin has no single `Kelvin` class. Its steady solver consumes:

```python
solve_steady(mesh_data, registry, chip)
```

- `mesh_data`: DOLFINx `MeshData` containing the mesh, cell tags, and facet tags.
- `registry.all()`: `Region(tag_id, label, material, source)` records.
- `chip.bcs`: Kelvin `BoundaryConditions`.

`mesh/gds_import.py` constructs that object graph from the MSH and JSON manifest written by
`read_gds.ipynb`.

For the bundled SRAM, use the integrated [notebook pipeline](SRAM_NOTEBOOK_PIPELINE.md)
instead of manually distributing power over tags. Its eight transistor instances
are matched to exact source polygons, and compact/general-transient drivers can
reuse a sealed notebook manifest with `--mesh-manifest`.

## Physical inputs required

GDS supplies lateral masks, not thermal process semantics.  The notebook's
recognized SKY130 profile supplies a documented working geometry and explicit
engineering material defaults; for an unrecognized process, provide:

1. A material for every `(layer, datatype)`, using a key from `spec/materials.py`.
2. Real z extents in the notebook's `PROCESS_STACK_RECORDS`.
3. Dedicated source volume tags and either `source_power_by_tag` in watts or
   `source_density_by_tag` in W/m³. Optional `source_device_by_tag` preserves
   instance IDs; power-based inputs are normalized by polygon volume, not bounds.
4. A connected substrate/dielectric/background domain.
5. The physical package backside that receives Kelvin facet tag `101`.

Do not map every exposed lower metal face to tag 101. That would cool every floating island
as though it were bonded directly to the heat sink.

## Import example

Run with the DOLFINx environment and place `kelvin/` on `PYTHONPATH`:

```python
import json
from pathlib import Path

from mesh.gds_import import load_tagged_msh, write_xdmf
from solve.steady import solve_steady

root = Path(
    "out/gds_geometry/sky130_fd_sc_hd__inv_1/sky130_fd_sc_hd__inv_1"
)
manifest = root / (
    "sky130_fd_sc_hd__inv_1_sky130_fd_sc_hd__inv_1_material_regions.json"
)
mesh = root / (
    "sky130_fd_sc_hd__inv_1_sky130_fd_sc_hd__inv_1_material_regions.msh"
)

# Demonstration only: distribute 1 µW uniformly over dedicated channel tags.
# Replace this with extracted per-footprint power for a real workload.
metadata = json.loads(manifest.read_text())
source_tags = set(metadata["thermal_source_candidate_tags"])
source_volume_m3 = sum(
    sum(volume["area_um2"] for volume in region["volumes"])
    * (region["z_max_um"] - region["z_min_um"]) * 1e-18
    for region in metadata["regions"]
    if region["physical_tag"] in source_tags
)
source_density_by_tag = {
    tag: 1.0e-6 / source_volume_m3 for tag in source_tags
}

case = load_tagged_msh(
    mesh,
    manifest,
    source_density_by_tag=source_density_by_tag,
)

print(case.report)
case.assert_solver_ready()
T, k, q = solve_steady(case.mesh_data, case.registry, case.chip)
write_xdmf(case, root / "sky130_fd_sc_hd__inv_1_kelvin_tags.xdmf")
```

The adapter refuses unknown materials, mismatched volume tags, incomplete cell tagging,
missing backside facets, and unanchored connected components. It maps only regions at the
global minimum z to Kelvin's `FACET_BOTTOM=101` by default.  It also rejects heat on tags
outside `thermal_source_candidate_tags`; set `allow_non_candidate_sources=True` only when
deliberately importing another loss mechanism such as interconnect Joule heating.

## Current SKY130 inverter status

The notebook validates the pinned inverter end to end with gdstk/Gmsh:

- 76,482 tetrahedra, each owned by exactly one of 19 physical volume tags.
- One connected CAD thermal domain with conformal material interfaces.
- Magic-derived well, diffusion, poly, contact, LI, MCON, and Metal-1 z extents.
- Explicit pwell/nwell, source-drain, channel-source, substrate, dielectric, and
  interconnect regions.
- Local refinement near semiconductor sources and conductors, with all volumes
  and minimum `minSICN` quality checked.

The final inverter MSH has also been loaded with DOLFINx 0.10 and solved with
Kelvin/PETSc using the notebook's 1 µW demonstration source: 76,482 cells, one
connected/backside-anchored component, no unanchored component, and
308.8778–308.8849 K.  `venvThermal` itself does not contain DOLFINx, so repeat
that last stage in Kelvin's DOLFINx environment.  Material values remain
engineering assumptions until replaced by calibrated data.

## Visualization artifacts

The notebook writes:

- `*_mesh_diagnostics.svg`: a clean physical-region envelope, one tag-diverse
  tetrahedral cut, quality distribution, and realized sizes. The cut uses physical
  lateral scale and prints its explicit z exaggeration.
- `*_material_regions_quality.msh`: the full Gmsh model plus a tetrahedral `minSICN`
  ElementData view for interactive clipping and coloring.

Open the enriched mesh with:

```bash
source venvThermal/bin/activate
gmsh out/gds_geometry/sky130_fd_sc_hd__inv_1/sky130_fd_sc_hd__inv_1/\
  sky130_fd_sc_hd__inv_1_sky130_fd_sc_hd__inv_1_material_regions_quality.msh
```

The notebook SVG is dependency-free.  The quality-enriched MSH can also be
clipped and colored interactively in Gmsh by physical volume or `minSICN`.
