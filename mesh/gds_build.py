"""GDS-derived ChipSpec -> gmsh OCC -> dolfinx Mesh + cell_tags + facet_tags
(2D cross-section path). Mirrors mesh/build.py's build_2d_mesh, reusing its
dimension-generic helpers (facet/material tagging, raw-mesh extraction) —
only geometry emission and sizing are GDS-specific (mesh/gds_section.py,
mesh/gds_sizing.py).

Same GATE 0 policy as the synthetic chip: every build writes its visual-
verification deliverables to out/ unconditionally.
"""

import os

import gmsh
from mpi4py import MPI

from dolfinx.io import XDMFFile
from dolfinx.io.gmsh import model_to_mesh

from gds import techmap
from gds.sources import build_heat_sources

from .build import (
    _extract_raw_mesh,
    _extract_raw_mesh_3d,
    _invert_surfaces,
    _surface_to_region,
    _tag_facets,
    _tag_facets_3d,
    _tag_material_regions,
    _write_stack_table,
    _write_stack_table_3d,
)
from . import gds_section as section_mod
from . import gds_sizing as sizing_mod
from . import gds_volume as volume_mod


def build_gds_2d_mesh(by_layer, cut_y_um: float, total_power_w: float, out_dir: str = "out/gds",
                       refine: float = 1.0, renders: bool = True, stack=None):
    """`stack` selects a gds.techmap.StackProfile (default: the frontside
    stack) — the BSPDN study passes techmap.BSPDN_STACK.
    """
    os.makedirs(out_dir, exist_ok=True)
    stack_obj = stack or techmap.FRONTSIDE_STACK

    channel_sources, contact_sources = build_heat_sources(by_layer, total_power_w)

    gmsh.initialize()
    gmsh.model.add("gds_section")

    registry, label_to_surfaces, layer_to_surfaces, x_bounds, z_bounds = \
        section_mod.emit_gds_2d_geometry(by_layer, channel_sources, contact_sources, cut_y_um, stack=stack)
    sizing_mod.apply_gds_2d_sizing(*x_bounds, refine=refine, stack=stack)

    _tag_material_regions(registry, label_to_surfaces, dim=2)
    _tag_facets(x_bounds, z_bounds)

    gmsh.model.mesh.generate(2)
    gmsh.write(os.path.join(out_dir, "gds_section.msh"))

    surface_to_region = _surface_to_region(registry, label_to_surfaces)
    surface_to_layer = _invert_surfaces(layer_to_surfaces)
    xz, tri_nodes, tri_region, tri_layer = _extract_raw_mesh(surface_to_region, surface_to_layer)
    _write_stack_table(stack_obj, xz, tri_nodes, tri_region, tri_layer,
                        os.path.join(out_dir, "stack_table.txt"))

    # Rendering ~1M triangles costs ~40s a panel — skippable for convergence
    # sweeps, where only Tmax matters and GATE 0 has already been eyeballed.
    if renders:
        from . import gds_viz as viz_mod
        viz_mod.render_regions(xz, tri_nodes, tri_region, x_bounds, z_bounds, cut_y_um,
                                os.path.join(out_dir, "regions_2d.png"), stack=stack_obj)
        viz_mod.render_sources(xz, tri_nodes, tri_region, x_bounds,
                                os.path.join(out_dir, "sources.png"))

    mesh_data = model_to_mesh(gmsh.model, MPI.COMM_SELF, 0, gdim=2)
    gmsh.finalize()

    # Geometry was built in micrometres; physics is SI (see mesh/build.py's
    # identical note — same trap, same fix).
    mesh_data.mesh.geometry.x[:] *= 1e-6

    with XDMFFile(MPI.COMM_SELF, os.path.join(out_dir, "mesh_tags.xdmf"), "w") as xf:
        xf.write_mesh(mesh_data.mesh)
        if mesh_data.cell_tags is not None:
            xf.write_meshtags(mesh_data.cell_tags, mesh_data.mesh.geometry)
        if mesh_data.facet_tags is not None:
            xf.write_meshtags(mesh_data.facet_tags, mesh_data.mesh.geometry)

    return mesh_data, registry


def build_gds_3d_mesh(by_layer, window, total_power_w: float, out_dir: str = "out/gds3d",
                       refine: float = 1.0, renders: bool = True, stack=None,
                       channel_sources=None, contact_sources=None):
    """3D analogue of build_gds_2d_mesh: extrudes real device geometry within
    a bounded lateral `window=(x0,x1,y0,y1)` (not the whole die — see
    cases/run_gds_3d.py for sizing it) into full 3D. `stack` selects a
    gds.techmap.StackProfile (default: frontside).

    `channel_sources`/`contact_sources`: optional pre-built (SourceBox, poly)
    lists (gds.sources.build_heat_sources's return shape) to use verbatim
    instead of the default flat/proportional-to-width split — e.g.
    cases/run_bitcell_compact.py maps real per-device SPICE compact-model
    power onto the real channel positions instead of assuming a flat split.
    `total_power_w` is ignored when these are given.
    """
    os.makedirs(out_dir, exist_ok=True)
    stack_obj = stack or techmap.FRONTSIDE_STACK

    if channel_sources is None or contact_sources is None:
        channel_sources, contact_sources = build_heat_sources(by_layer, total_power_w)

    gmsh.initialize()
    gmsh.model.add("gds_volume")

    registry, label_to_volumes, layer_to_volumes, x_bounds, y_bounds, z_bounds, feature_points = \
        volume_mod.emit_gds_3d_geometry(by_layer, channel_sources, contact_sources, window, stack_obj)
    sizing_mod.apply_gds_3d_sizing(*x_bounds, *y_bounds, feature_points, refine=refine, stack=stack_obj)

    _tag_material_regions(registry, label_to_volumes, dim=3)
    _tag_facets_3d(x_bounds, y_bounds, z_bounds)

    import time
    t0 = time.time()
    gmsh.model.mesh.generate(3)
    print(f"[gds_volume] mesh.generate(3): {time.time()-t0:.1f}s", flush=True)
    gmsh.write(os.path.join(out_dir, "gds_volume.msh"))

    volume_to_region = _surface_to_region(registry, label_to_volumes)
    volume_to_layer = _invert_surfaces(layer_to_volumes)
    xyz, tet_nodes, region_idx, region_list, layer_idx, layer_list = \
        _extract_raw_mesh_3d(volume_to_region, volume_to_layer)
    _write_stack_table_3d(stack_obj, xyz, tet_nodes, region_idx, region_list, layer_idx, layer_list,
                           os.path.join(out_dir, "stack_table.txt"))

    if renders:
        from . import gds_viz3d as viz3d_mod
        viz3d_mod.render_gds_3d_gate0(xyz, tet_nodes, region_idx, region_list, stack_obj, window, out_dir)

    mesh_data = model_to_mesh(gmsh.model, MPI.COMM_SELF, 0, gdim=3)
    gmsh.finalize()

    mesh_data.mesh.geometry.x[:] *= 1e-6

    with XDMFFile(MPI.COMM_SELF, os.path.join(out_dir, "mesh_tags.xdmf"), "w") as xf:
        xf.write_mesh(mesh_data.mesh)
        if mesh_data.cell_tags is not None:
            xf.write_meshtags(mesh_data.cell_tags, mesh_data.mesh.geometry)
        if mesh_data.facet_tags is not None:
            xf.write_meshtags(mesh_data.facet_tags, mesh_data.mesh.geometry)

    return mesh_data, registry
