"""Build exact-polygon SRAM mask meshes for an old/new geometry comparison.

Run with a Python environment containing gdstk, gmsh, numpy, and IPython::

    python cases/build_sram_polygon_preview.py

This script meshes individual, disconnected mask layers for visualization
ONLY. These are not solver-ready thermal meshes:
there is no substrate, dielectric fill, assembled contact model, or heat source.

Polygon extraction and prism construction use the notebook's shared Python
backend. Prism heights retain the active-top z=0 coordinate origin. Raw licon
footprints remain 2D here because poly/diff plugs have different z spans.
Use cases/build_sram_mesh.py for the assembled, solver-ready thermal mesh.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import gmsh
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "read_gds.ipynb"
LAYER_NAMES = ("nwell", "diff", "poly", "licon1", "li1", "mcon", "met1", "via", "met2")


def load_notebook_functions():
    """Compatibility name for older visualization scripts; no notebook AST."""
    sys.path.insert(0, str(ROOT / "kelvin"))
    from mesh.gds_notebook import create_context
    return create_context(ROOT=ROOT)


def model_arrays(dimension):
    node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
    points = np.asarray(coordinates).reshape(-1, 3)
    index = {int(tag): i for i, tag in enumerate(node_tags)}
    types, element_tags, connectivity = gmsh.model.mesh.getElements(dimension)
    expected_type, arity = (2, 3) if dimension == 2 else (4, 4)
    cells, tags = [], []
    for etype, ids, flat in zip(types, element_tags, connectivity):
        if int(etype) != expected_type:
            raise RuntimeError(f"Unexpected element type {etype}")
        cells.extend([[index[int(tag)] for tag in cell] for cell in np.asarray(flat).reshape(-1, arity)])
        tags.extend(map(int, ids))
    return points, np.asarray(cells, dtype=np.int64), tags


def build_layer(namespace, layout, name, out, mesh_size):
    components = layout["polygons_by_spec"][namespace["SKY130_SPECS"][name]]
    exact_area = sum(component["area_um2"] for component in components)
    gmsh.clear()
    gmsh.model.add(name + "_footprint")
    surfaces = []
    for component in components:
        points = [gmsh.model.occ.addPoint(x, y, 0, mesh_size) for x, y in component["footprint_xy_um"]]
        lines = [gmsh.model.occ.addLine(a, b) for a, b in zip(points, points[1:] + points[:1])]
        loop = gmsh.model.occ.addCurveLoop(lines)
        surfaces.append(gmsh.model.occ.addPlaneSurface([loop]))
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(2, surfaces, 1)
    gmsh.model.setPhysicalName(2, 1, name)
    gmsh.model.mesh.generate(2)
    points, triangles, _ = model_arrays(2)
    tri_points = points[triangles, :2]
    ab, ac = tri_points[:, 1] - tri_points[:, 0], tri_points[:, 2] - tri_points[:, 0]
    triangle_area = float(np.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]).sum() / 2)
    if not np.isclose(triangle_area, exact_area, rtol=1e-10, atol=1e-12):
        raise AssertionError((name, triangle_area, exact_area))
    footprint_path = out / f"new_exact_{name}_footprint.msh"
    gmsh.write(str(footprint_path))
    item = {
        "name": name,
        "gds_spec": namespace["SKY130_SPECS"][name],
        "color": namespace["SKY130_LAYER_COLORS"][name],
        "rings": [component["footprint_xy_um"] for component in components],
        "vertices_xy_um": points[:, :2].tolist(),
        "triangles": triangles.tolist(),
        "area_um2": exact_area,
        "triangle_area_sum_um2": triangle_area,
        "node_count": len(points),
        "triangle_count": len(triangles),
        "msh_path": str(footprint_path),
    }
    if name == "licon1":
        item["prism_omitted_reason"] = "Separate-mask preview does not assemble differing poly/diff contact spans."
        return item

    z0, z1 = namespace["SKY130_SOLID_Z_UM"]["well" if name == "nwell" else name]
    gmsh.clear()
    gmsh.model.add(name + "_prisms")
    volumes = [(3, namespace["add_polygon_prism"](gmsh, component["footprint_xy_um"], z0, z1, mesh_size))
               for component in components]
    if len(volumes) > 1:
        volumes, _ = gmsh.model.occ.fragment(volumes[:1], volumes[1:])
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(3, [tag for dim, tag in volumes if dim == 3], 1)
    gmsh.model.setPhysicalName(3, 1, name)
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize("Netgen")
    points, tetrahedra, tags = model_arrays(3)
    tet_points = points[tetrahedra]
    determinants = np.linalg.det(tet_points[:, 1:] - tet_points[:, :1])
    tet_volume = float(np.abs(determinants).sum() / 6)
    exact_volume = exact_area * (z1 - z0)
    if not np.isclose(tet_volume, exact_volume, rtol=1e-9, atol=1e-12):
        raise AssertionError((name, tet_volume, exact_volume))
    qualities = gmsh.model.mesh.getElementQualities(tags, "minSICN")
    if np.min(qualities) <= 0:
        raise AssertionError(f"Nonpositive prism mesh quality for {name}")
    surface_points, surface_triangles, _ = model_arrays(2)
    if not np.array_equal(surface_points, points):
        raise AssertionError("Node ordering changed between mesh reads")
    prism_path = out / f"new_exact_{name}_prisms.msh"
    array_path = out / f"new_exact_{name}_prisms.npz"
    gmsh.write(str(prism_path))
    np.savez_compressed(array_path, points_um=points, tetrahedra=tetrahedra, surface_triangles=surface_triangles)
    item.update({
        "z_range_um": [z0, z1],
        "prism_msh_path": str(prism_path),
        "prism_arrays_path": str(array_path),
        "prism_node_count": len(points),
        "tetrahedron_count": len(tetrahedra),
        "exact_volume_um3": exact_volume,
        "tetrahedron_volume_sum_um3": tet_volume,
        "minimum_minSICN_quality": float(np.min(qualities)),
    })
    return item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gds", type=Path, default=ROOT / "kelvin/data/sram22_64x22m4w22.gds")
    parser.add_argument("--cell", default="sram_sp_cell")
    parser.add_argument("--out", type=Path, default=ROOT / "kelvin/out/sram_mesh_comparison")
    parser.add_argument("--mesh-size-um", type=float, default=0.06)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    namespace = load_notebook_functions()
    layout = namespace["extract_layout"](args.gds.resolve(), args.cell)
    namespace.update(layout=layout, layer_profile="sky130")
    try:
        namespace["build_sky130_regions"]()
    except ValueError as exc:
        failure = str(exc)
    else:
        failure = None
    specs, components = namespace["SKY130_SPECS"], layout["polygons_by_spec"]
    boolean, precision = namespace["boolean_components"], layout["boolean_grid_um"]
    claimed = boolean(components[specs["poly"]], components[specs["diff"]], "or", precision)
    residual = boolean(components[specs["licon1"]], claimed, "not", precision, allow_empty=True)
    result = {
        "status": "visualization_only_disconnected_exact_mask_meshes",
        "not_a_solver_ready_thermal_mesh": True,
        "warning": "Separate well/active/conductor prisms only: no substrate, dielectric, contact assembly, or sources.",
        "source_gds": str(args.gds.resolve()), "source_sha256": layout["source_sha256"],
        "notebook": str(NOTEBOOK), "cell": args.cell, "bbox_um": layout["bbox_um"],
        "z_origin": "active_top (notebook default)",
        "new_thermal_flow_failure": failure,
        "new_thermal_flow_function": "mesh.gds_notebook.build_sky130_regions (shared with read_gds.ipynb)",
        "residual_licon_area_um2": sum(item["area_um2"] for item in residual),
        "residual_licon_components": residual,
        "mesh_size_um": args.mesh_size_um, "layers": [],
    }
    gmsh.initialize()
    try:
        for key, value in {
            "General.Terminal": 0, "General.NumThreads": 1,
            "Mesh.MeshSizeMin": args.mesh_size_um / 2,
            "Mesh.MeshSizeMax": args.mesh_size_um,
            "Mesh.Algorithm": 6, "Mesh.Algorithm3D": 1,
            "Mesh.ElementOrder": 1, "Mesh.RandomSeed": 1,
        }.items():
            gmsh.option.setNumber(key, value)
        for name in LAYER_NAMES:
            item = build_layer(namespace, layout, name, args.out.resolve(), args.mesh_size_um)
            result["layers"].append(item)
            print(name, "area_um2=", item["area_um2"], "triangles=", item["triangle_count"],
                  "tetrahedra=", item.get("tetrahedron_count", "omitted"), flush=True)
    finally:
        gmsh.finalize()
    destination = args.out / "new_exact_footprint_meshes.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print("Complete thermal mesh status:", failure or "region construction succeeded (not assembled here)")
    print("Wrote", destination)


if __name__ == "__main__":
    main()
