"""ChipSpec -> gmsh OCC -> dolfinx Mesh + cell_tags + facet_tags (2D path).

Every build writes the GATE 0 visual-verification deliverables to out/mesh/
unconditionally (PLAN.md: geometry is checked by looking at it, not by a
passing solve) — chip_section.msh, mesh_tags.xdmf/.h5, regions_2d.png,
sources.png, stack_table.txt.
"""

import os
import time

import gmsh
import numpy as np
from mpi4py import MPI

from dolfinx.io import XDMFFile
from dolfinx.io.gmsh import model_to_mesh

from spec.chip import ChipSpec

from . import section as section_mod
from . import sizing as sizing_mod
from . import volume as volume_mod
from .boxes import RegionRegistry

FACET_BOTTOM = 101  # backside, z=0, Robin h_eff
FACET_TOP = 102      # top face, per chip.bcs.top_face
FACET_LEFT = 103     # 2D: tile-edge cut, x=0, symmetry/adiabatic
FACET_RIGHT = 104    # 2D: tile-edge cut, x=tile_um, symmetry/adiabatic
FACET_X0 = FACET_LEFT    # 3D alias: x=0 face
FACET_X1 = FACET_RIGHT   # 3D alias: x=tile_um face
FACET_Y0 = 105       # 3D: y=0 face, symmetry/adiabatic
FACET_Y1 = 106       # 3D: y=tile_um face, symmetry/adiabatic

_EPS = 1e-6


def _tag_material_regions(registry: RegionRegistry, label_to_entities, dim: int):
    for region in registry.all():
        entities = label_to_entities.get(region.label, [])
        if not entities:
            continue
        gmsh.model.addPhysicalGroup(dim, entities, tag=region.tag_id)
        gmsh.model.setPhysicalName(dim, region.tag_id, region.label)


def _tag_facets(x_bounds, z_bounds):
    x0, x1 = x_bounds
    z0, z1 = z_bounds
    sides = {FACET_BOTTOM: [], FACET_TOP: [], FACET_LEFT: [], FACET_RIGHT: []}
    for dim, tag in gmsh.model.getEntities(1):
        xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(1, tag)
        if abs(xmin - x0) < _EPS and abs(xmax - x0) < _EPS:
            sides[FACET_LEFT].append(tag)
        elif abs(xmin - x1) < _EPS and abs(xmax - x1) < _EPS:
            sides[FACET_RIGHT].append(tag)
        elif abs(ymin - z0) < _EPS and abs(ymax - z0) < _EPS:
            sides[FACET_BOTTOM].append(tag)
        elif abs(ymin - z1) < _EPS and abs(ymax - z1) < _EPS:
            sides[FACET_TOP].append(tag)
    names = {FACET_BOTTOM: "bottom", FACET_TOP: "top", FACET_LEFT: "left", FACET_RIGHT: "right"}
    for tag_id, curves in sides.items():
        if not curves:
            raise RuntimeError(f"no boundary curves found for facet {names[tag_id]!r}")
        gmsh.model.addPhysicalGroup(1, curves, tag=tag_id)
        gmsh.model.setPhysicalName(1, tag_id, names[tag_id])


def _tag_facets_3d(x_bounds, y_bounds, z_bounds):
    x0, x1 = x_bounds
    y0, y1 = y_bounds
    z0, z1 = z_bounds
    sides = {FACET_BOTTOM: [], FACET_TOP: [], FACET_X0: [], FACET_X1: [], FACET_Y0: [], FACET_Y1: []}
    for dim, tag in gmsh.model.getEntities(2):
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, tag)
        if abs(xmin - x0) < _EPS and abs(xmax - x0) < _EPS:
            sides[FACET_X0].append(tag)
        elif abs(xmin - x1) < _EPS and abs(xmax - x1) < _EPS:
            sides[FACET_X1].append(tag)
        elif abs(ymin - y0) < _EPS and abs(ymax - y0) < _EPS:
            sides[FACET_Y0].append(tag)
        elif abs(ymin - y1) < _EPS and abs(ymax - y1) < _EPS:
            sides[FACET_Y1].append(tag)
        elif abs(zmin - z0) < _EPS and abs(zmax - z0) < _EPS:
            sides[FACET_BOTTOM].append(tag)
        elif abs(zmin - z1) < _EPS and abs(zmax - z1) < _EPS:
            sides[FACET_TOP].append(tag)
    names = {FACET_BOTTOM: "bottom", FACET_TOP: "top", FACET_X0: "x0", FACET_X1: "x1",
             FACET_Y0: "y0", FACET_Y1: "y1"}
    for tag_id, faces in sides.items():
        if not faces:
            raise RuntimeError(f"no boundary faces found for facet {names[tag_id]!r}")
        gmsh.model.addPhysicalGroup(2, faces, tag=tag_id)
        gmsh.model.setPhysicalName(2, tag_id, names[tag_id])


def _invert_surfaces(surfaces_by_key):
    out = {}
    for key, surfaces in surfaces_by_key.items():
        for s in surfaces:
            out[s] = key
    return out


def _surface_to_region(registry: RegionRegistry, label_to_surfaces):
    by_label = {r.label: r for r in registry.all()}
    label_of_surface = _invert_surfaces(label_to_surfaces)
    return {s: by_label[label] for s, label in label_of_surface.items()}


def _extract_raw_mesh(surface_to_region, surface_to_layer):
    """Read back node coords + per-triangle region/layer straight from the
    gmsh mesh, before dolfinx/model_to_mesh touch it. Node coords are (x, z);
    gmsh's own y axis is our physical z (see mesh/section.py). Layer
    membership comes from mesh/section.py's own bookkeeping (which layer
    each surface was built for), not from re-deriving it via coordinates.
    """
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    coords = np.asarray(coords).reshape(-1, 3)
    tag_to_index = {int(t): i for i, t in enumerate(node_tags)}
    xz = coords[:, [0, 1]]

    tri_nodes = []
    tri_region = []
    tri_layer = []
    for surf_tag, region in surface_to_region.items():
        layer_name = surface_to_layer[surf_tag]
        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2, tag=surf_tag)
        for etype, enodes in zip(elem_types, elem_node_tags):
            enodes = np.asarray(enodes).reshape(-1, 3)  # triangles only (linear P1)
            for tri in enodes:
                tri_nodes.append([tag_to_index[int(n)] for n in tri])
                tri_region.append(region)
                tri_layer.append(layer_name)

    return xz, np.asarray(tri_nodes, dtype=np.int64), tri_region, tri_layer


def _write_stack_table(stack, xz, tri_nodes, tri_region, tri_layer, out_path):
    """Read the realized z-extent and material mix back out of the mesh, per
    layer, and compare against the intended z-bounds. This is the check
    PLAN.md calls out specifically: a thin layer silently absorbed by a gmsh
    `fragment` failure shows up here as a missing cell count, even though
    the solve would otherwise run cleanly. Layer membership is exact (from
    section.py's own bookkeeping), so any z-range mismatch here is a real
    geometry bug, not a binning artifact.
    """
    tri_layer = np.asarray(tri_layer)

    header = f"{'layer':<18}{'z_expected':<16}{'z_realized':<16}{'cells':>8}  materials"
    lines = [header]
    for z0, z1, layer in stack.z_bounds():
        idx = np.nonzero(tri_layer == layer.name)[0]
        if len(idx) == 0:
            lines.append(f"{layer.name:<18}{f'{z0:.4f}-{z1:.4f}':<16}{'MISSING':<16}{0:>8}  -")
            continue
        node_idx = tri_nodes[idx].ravel()
        z_real = xz[node_idx, 1]
        materials = sorted({tri_region[i].material for i in idx})
        lines.append(
            f"{layer.name:<18}{f'{z0:.4f}-{z1:.4f}':<16}"
            f"{f'{z_real.min():.4f}-{z_real.max():.4f}':<16}{len(idx):>8}  {', '.join(materials)}"
        )

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _extract_raw_mesh_3d(volume_to_region, volume_to_layer):
    """Vectorized read-back of node coords + per-tet region/layer: 3D
    element counts can run into the millions, so unlike the 2D extractor
    this avoids a per-element Python loop (only loops over volumes, of
    which there are hundreds, not tets, of which there can be millions).
    """
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    coords = np.asarray(coords).reshape(-1, 3)
    node_tags = np.asarray(node_tags, dtype=np.int64)
    tag_to_index = np.full(int(node_tags.max()) + 1, -1, dtype=np.int64)
    tag_to_index[node_tags] = np.arange(len(node_tags))

    tet_node_chunks = []
    region_idx_chunks = []
    layer_idx_chunks = []
    region_list = []
    layer_list = []

    for vol_tag, region in volume_to_region.items():
        layer_name = volume_to_layer[vol_tag]
        region_list.append(region)
        layer_list.append(layer_name)
        r_i, l_i = len(region_list) - 1, len(layer_list) - 1

        _, _, elem_node_tags = gmsh.model.mesh.getElements(dim=3, tag=vol_tag)
        for enodes in elem_node_tags:
            enodes = np.asarray(enodes, dtype=np.int64).reshape(-1, 4)  # tets only (linear P1)
            idx = tag_to_index[enodes]
            tet_node_chunks.append(idx)
            region_idx_chunks.append(np.full(idx.shape[0], r_i, dtype=np.int32))
            layer_idx_chunks.append(np.full(idx.shape[0], l_i, dtype=np.int32))

    tet_nodes = np.concatenate(tet_node_chunks, axis=0) if tet_node_chunks else np.zeros((0, 4), dtype=np.int64)
    region_idx = np.concatenate(region_idx_chunks) if region_idx_chunks else np.zeros((0,), dtype=np.int32)
    layer_idx = np.concatenate(layer_idx_chunks) if layer_idx_chunks else np.zeros((0,), dtype=np.int32)
    return coords, tet_nodes, region_idx, region_list, layer_idx, layer_list


def _write_stack_table_3d(stack, xyz, tet_nodes, region_idx, region_list, layer_idx, layer_list, out_path):
    materials_arr = np.array([r.material for r in region_list])

    header = f"{'layer':<18}{'z_expected':<16}{'z_realized':<16}{'cells':>10}  materials"
    lines = [header]
    for z0, z1, layer in stack.z_bounds():
        li = [i for i, name in enumerate(layer_list) if name == layer.name]
        if not li:
            lines.append(f"{layer.name:<18}{f'{z0:.4f}-{z1:.4f}':<16}{'MISSING':<16}{0:>10}  -")
            continue
        mask = np.isin(layer_idx, li)
        n = int(mask.sum())
        if n == 0:
            lines.append(f"{layer.name:<18}{f'{z0:.4f}-{z1:.4f}':<16}{'MISSING':<16}{0:>10}  -")
            continue
        node_idx = tet_nodes[mask].ravel()
        z_real = xyz[node_idx, 2]
        mats = sorted(set(materials_arr[region_idx[mask]]))
        lines.append(
            f"{layer.name:<18}{f'{z0:.4f}-{z1:.4f}':<16}"
            f"{f'{z_real.min():.4f}-{z_real.max():.4f}':<16}{n:>10}  {', '.join(mats)}"
        )

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def build_3d_mesh(chip: ChipSpec, out_dir: str = "out/mesh3d"):
    os.makedirs(out_dir, exist_ok=True)

    def _t(msg, t0):
        t1 = time.time()
        print(f"[build_3d] {msg}: {t1 - t0:.1f}s", flush=True)
        return t1

    t = time.time()
    gmsh.initialize()
    gmsh.model.add("chip_volume")

    registry, label_to_volumes, layer_to_volumes, x_bounds, y_bounds, z_bounds = \
        volume_mod.emit_3d_geometry(chip)
    t = _t("geometry (boxes + removeAllDuplicates)", t)

    sizing_mod.apply_3d_sizing(chip)
    t = _t("sizing fields", t)

    _tag_material_regions(registry, label_to_volumes, dim=3)
    _tag_facets_3d(x_bounds, y_bounds, z_bounds)
    t = _t("physical group tagging", t)

    gmsh.model.mesh.generate(3)
    t = _t("mesh.generate(3)", t)
    gmsh.write(os.path.join(out_dir, "chip_volume.msh"))
    t = _t("write .msh", t)

    volume_to_region = _surface_to_region(registry, label_to_volumes)
    volume_to_layer = _invert_surfaces(layer_to_volumes)
    xyz, tet_nodes, region_idx, region_list, layer_idx, layer_list = \
        _extract_raw_mesh_3d(volume_to_region, volume_to_layer)
    t = _t("extract raw mesh", t)
    _write_stack_table_3d(chip.stack, xyz, tet_nodes, region_idx, region_list, layer_idx, layer_list,
                           os.path.join(out_dir, "stack_table.txt"))
    t = _t("stack table", t)

    from . import viz3d as viz3d_mod
    viz3d_mod.render_3d_gate0(xyz, tet_nodes, region_idx, region_list, chip, out_dir)
    t = _t("viz3d renders", t)

    mesh_data = model_to_mesh(gmsh.model, MPI.COMM_SELF, 0, gdim=3)
    t = _t("model_to_mesh", t)
    gmsh.finalize()

    mesh_data.mesh.geometry.x[:] *= 1e-6

    with XDMFFile(MPI.COMM_SELF, os.path.join(out_dir, "mesh_tags.xdmf"), "w") as xf:
        xf.write_mesh(mesh_data.mesh)
        if mesh_data.cell_tags is not None:
            xf.write_meshtags(mesh_data.cell_tags, mesh_data.mesh.geometry)
        if mesh_data.facet_tags is not None:
            xf.write_meshtags(mesh_data.facet_tags, mesh_data.mesh.geometry)

    return mesh_data, registry


def build_2d_mesh(chip: ChipSpec, row: int = 0, out_dir: str = "out/mesh"):
    os.makedirs(out_dir, exist_ok=True)

    gmsh.initialize()
    gmsh.model.add("chip_section")

    registry, label_to_surfaces, layer_to_surfaces, x_bounds, z_bounds = \
        section_mod.emit_2d_geometry(chip, row=row)
    sizing_mod.apply_2d_sizing(chip.stack, *x_bounds)

    _tag_material_regions(registry, label_to_surfaces, dim=2)
    _tag_facets(x_bounds, z_bounds)

    gmsh.model.mesh.generate(2)
    gmsh.write(os.path.join(out_dir, "chip_section.msh"))

    surface_to_region = _surface_to_region(registry, label_to_surfaces)
    surface_to_layer = _invert_surfaces(layer_to_surfaces)
    xz, tri_nodes, tri_region, tri_layer = _extract_raw_mesh(surface_to_region, surface_to_layer)
    _write_stack_table(chip.stack, xz, tri_nodes, tri_region, tri_layer,
                        os.path.join(out_dir, "stack_table.txt"))

    from . import viz as viz_mod
    viz_mod.render_regions(xz, tri_nodes, tri_region, chip, row,
                            os.path.join(out_dir, "regions_2d.png"))
    viz_mod.render_sources(xz, tri_nodes, tri_region, chip, row,
                            os.path.join(out_dir, "sources.png"))

    mesh_data = model_to_mesh(gmsh.model, MPI.COMM_SELF, 0, gdim=2)
    gmsh.finalize()

    # Geometry was built in micrometres (keeps gmsh size fields well-conditioned,
    # PLAN.md); everything downstream (k, h_eff, q, ...) is proper SI, so the
    # dolfinx mesh itself must be in metres before any physics touches it.
    mesh_data.mesh.geometry.x[:] *= 1e-6

    with XDMFFile(MPI.COMM_SELF, os.path.join(out_dir, "mesh_tags.xdmf"), "w") as xf:
        xf.write_mesh(mesh_data.mesh)
        if mesh_data.cell_tags is not None:
            xf.write_meshtags(mesh_data.cell_tags, mesh_data.mesh.geometry)
        if mesh_data.facet_tags is not None:
            xf.write_meshtags(mesh_data.facet_tags, mesh_data.mesh.geometry)

    return mesh_data, registry
