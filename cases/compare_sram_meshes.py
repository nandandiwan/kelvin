"""Render an existing legacy SRAM mesh against exact-polygon layer previews.

The new previews come from build_sram_polygon_preview.py, using the notebook's
actual polygon normalization/extrusion functions. They are NOT a connected,
solver-ready replacement for the newer thermal flow, which currently rejects
this bitcell's contact landing geometry. No solver/geometry model is modified.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import gdstk
import gmsh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pyvista as pv

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from gds.techmap import FRONTSIDE_STACK, z_bounds
from mesh.gds_volume import _window_rects
from mesh.gds_svg_viz import _LAYER_COLOR

LAYERS = ("diff", "li1", "met1")
COLORS = {name: _LAYER_COLOR[name] for name in LAYERS}
NAMES = {"diff": "DIFF / active silicon", "li1": "LI1 / local interconnect", "met1": "M1 / metal 1"}


def read_tetrahedra(path):
    """Read coordinates/physical-volume tags without importing the heat solver."""
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(path))
        node_ids, xyz, _ = gmsh.model.mesh.getNodes()
        order = np.argsort(node_ids)
        node_ids = node_ids[order]
        xyz = np.asarray(xyz).reshape(-1, 3)[order]
        blocks, tags, names = [], [], {}
        for dim, physical in gmsh.model.getPhysicalGroups(3):
            names[physical] = gmsh.model.getPhysicalName(dim, physical)
            for entity in gmsh.model.getEntitiesForPhysicalGroup(3, physical):
                types, _, connectivity = gmsh.model.mesh.getElements(3, int(entity))
                for kind, nodes in zip(types, connectivity):
                    if kind != 4:
                        raise ValueError(f"Expected linear tetrahedra, found Gmsh type {kind}")
                    block = np.searchsorted(node_ids, np.asarray(nodes).reshape(-1, 4))
                    blocks.append(block)
                    tags.extend([physical] * len(block))
        tets = np.vstack(blocks)
        cells = np.column_stack((np.full(len(tets), 4), tets)).ravel()
        grid = pv.UnstructuredGrid(cells, np.full(len(tets), 10, dtype=np.uint8), xyz)
        grid.cell_data["physical_tag"] = np.asarray(tags, dtype=np.int32)
        return grid, names
    finally:
        gmsh.finalize()


def polygons(surface):
    faces = surface.faces
    result = []
    start = 0
    while start < len(faces):
        count = faces[start]
        result.append(surface.points[faces[start + 1:start + count + 1]])
        start += count + 1
    return result


def footprint_area(shapes):
    return sum(abs(np.dot(p[:, 0], np.roll(p[:, 1], 1)) -
                   np.dot(p[:, 1], np.roll(p[:, 0], 1))) / 2 for p in shapes)


def old_layer(grid, names, name, bands):
    z0, z1 = bands[name]
    centers = grid.cell_centers().points
    if name == "diff":
        # Include the old 10 nm channel band in the 3D active-layer view.
        z1 = bands["channel"][1]
        allowed = {tag for tag, label in names.items()
                   if label == "Si_SD_doped" or label.startswith("src_")}
    else:
        material = {"li1": "TiN", "met1": "Al"}[name]
        allowed = {tag for tag, label in names.items() if label == material}
    keep = ((centers[:, 2] > z0) & (centers[:, 2] < z1)
            & np.isin(grid.cell_data["physical_tag"], list(allowed)))
    return grid.extract_cells(keep)


def outline(ax, rings, color="#243345", width=0.9, linestyle="-"):
    for ring in rings:
        points = np.asarray(ring)
        points = np.vstack((points, points[0]))
        ax.plot(points[:, 0], points[:, 1], color=color, linewidth=width,
                linestyle=linestyle, zorder=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-msh", type=Path, required=True)
    parser.add_argument("--preview-json", type=Path,
                        default=REPO / "out/sram_mesh_comparison/new_exact_footprint_meshes.json")
    parser.add_argument("--out-dir", type=Path, default=REPO / "out/sram_mesh_comparison")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(args.preview_json.read_text())
    layers = {entry["name"]: entry for entry in metadata["layers"]}
    xmin, ymin, xmax, ymax = metadata["bbox_um"]
    grid, names = read_tetrahedra(args.old_msh)
    if not np.allclose(np.array(grid.bounds)[:4], [xmin, xmax, ymin, ymax], rtol=0, atol=1e-7):
        raise ValueError("The old mesh and exact-polygon preview are not the same bitcell window")
    bands = {band.name: (z0, z1) for z0, z1, band in z_bounds(FRONTSIDE_STACK)}
    active_top = bands["channel"][1]
    old_grids, new_grids, reports = {}, {}, {}
    for name in LAYERS:
        entry = layers[name]
        old_grids[name] = old_layer(grid, names, name, bands)
        new_grids[name], _ = read_tetrahedra(Path(entry["prism_msh_path"]))
        old_z = sum(bands[name]) / 2
        new_z = sum(entry["z_range_um"]) / 2
        old_shapes = polygons(old_grids[name].slice(normal="z", origin=(0, 0, old_z)))
        new_shapes = polygons(new_grids[name].slice(normal="z", origin=(0, 0, new_z)))
        old_area, new_area = footprint_area(old_shapes), footprint_area(new_shapes)
        rings = entry["rings"]
        exact = [gdstk.Polygon(ring) for ring in rings]
        window = gdstk.rectangle((xmin, ymin), (xmax, ymax))
        rectangles = _window_rects(exact, window, xmin, ymin, xmax, ymax)
        box_polys = [gdstk.rectangle((a, b), (c, d)) for a, b, c, d in rectangles]
        extra = gdstk.boolean(box_polys, exact, "not", precision=1e-6)
        expected_old = sum(p.area() for p in gdstk.boolean(box_polys, [], "or", precision=1e-6))
        if not np.isclose(old_area, expected_old, rtol=1e-8, atol=1e-10):
            raise AssertionError(f"Old {name} mesh slice doesn't match the bounding-box geometry")
        if not np.isclose(new_area, entry["area_um2"], rtol=1e-8, atol=1e-10):
            raise AssertionError(f"New {name} mesh slice doesn't preserve the GDS footprint")
        reports[name] = dict(old_slice_area_um2=old_area, exact_slice_area_um2=new_area,
                             excess_area_percent=100 * (old_area / new_area - 1),
                             old_tetrahedra=old_grids[name].n_cells,
                             new_preview_tetrahedra=new_grids[name].n_cells,
                             old_slice_z_um=old_z, new_slice_z_um=new_z,
                             old_shapes=old_shapes, new_shapes=new_shapes,
                             extra=[p.points for p in extra])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 15.5))
    fig.patch.set_facecolor("white")
    fig.suptitle("SRAM bitcell: old vs polygon-preserving meshes", y=0.986, fontsize=18, weight="bold")
    fig.text(0.5, 0.957, "sram_sp_cell  ·  1.20 × 1.58 µm  ·  identical x/y scales", ha="center", color="#44556B")
    fig.text(0.295, 0.925, "OLD: bounding-box thermal mesh", ha="center", fontsize=12, weight="bold")
    fig.text(0.75, 0.925, "NEW: exact-polygon prism preview", ha="center", fontsize=12, weight="bold")
    for row, name in enumerate(LAYERS):
        entry, report = layers[name], reports[name]
        for col, key in enumerate(("old_shapes", "new_shapes")):
            ax = axes[row, col]
            ax.set_facecolor("#F5F7FA")
            shapes = [p[:, :2] for p in report[key]]
            ax.add_collection(PolyCollection(shapes, facecolors=COLORS[name], edgecolors="#314154",
                                             linewidths=0.16, alpha=0.93))
            if col == 0:
                ax.add_collection(PolyCollection(report["extra"], facecolors="#D95B5B",
                                                 edgecolors="none", alpha=0.62, zorder=3))
            outline(ax, entry["rings"], width=0.9)
            ax.set_xlim(xmin - .035, xmax + .035)
            ax.set_ylim(ymin - .035, ymax + .035)
            ax.set_aspect("equal")
            ax.set_xlabel("x (µm)")
            ax.set_ylabel("y (µm)")
            area = report["old_slice_area_um2" if col == 0 else "exact_slice_area_um2"]
            suffix = f"  (+{report['excess_area_percent']:.1f}% area)" if col == 0 else "  (GDS footprint preserved)"
            ax.set_title(f"{NAMES[name]}\n{area:.4f} µm²{suffix}", fontsize=10.5, pad=8)
    fig.legend(handles=[Patch(facecolor="#D95B5B", label="Material added by bounding-box approximation"),
                        Patch(facecolor="none", edgecolor="#243345", label="Exact GDS outline")],
               loc="lower center", bbox_to_anchor=(0.5, 0.051), ncol=2, frameon=False)
    fig.text(0.5, .022, "NEW column: layer-only mesh preview, NOT a solver-ready thermal domain.\n"
             "The full new flow rejects 0.02975 µm² of unresolved contact landing geometry.",
             ha="center", color="#8A3030", fontsize=10)
    fig.subplots_adjust(left=.1, right=.97, top=.897, bottom=.105, hspace=.32, wspace=.28)
    fig.savefig(args.out_dir / "old_vs_new_mesh_slices.png", dpi=190)
    fig.savefig(args.out_dir / "old_vs_new_mesh_slices.svg")
    plt.close(fig)

    fig = plt.figure(figsize=(14, 8.8))
    for col, (title, collection) in enumerate((("OLD: selected layers of thermal mesh", old_grids),
                                              ("NEW: exact-polygon prism previews", new_grids))):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        for name in LAYERS:
            surface = collection[name].extract_surface(algorithm="dataset_surface").triangulate()
            points = surface.points.copy()
            if col == 0:
                points[:, 2] -= active_top
            triangles = surface.faces.reshape(-1, 4)[:, 1:]
            ax.add_collection3d(Poly3DCollection(points[triangles], facecolors=COLORS[name],
                                                edgecolors=(.08, .14, .2, .3), linewidths=.13,
                                                alpha=0.98, rasterized=True))
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(-.13, 1.45)
        ax.set_box_aspect((xmax-xmin, ymax-ymin, 1.58))
        ax.view_init(elev=24, azim=-56)
        ax.set_xlabel("x (µm)", labelpad=8)
        ax.set_ylabel("y (µm)", labelpad=8)
        ax.set_zlabel("z from active top (µm)", labelpad=9)
        ax.set_title(title, pad=14, fontsize=12, weight="bold")
    fig.suptitle("Same SRAM bitcell · actual tetrahedral surface meshes", y=.965, fontsize=17, weight="bold")
    fig.legend(handles=[Patch(facecolor=COLORS[name], label=NAMES[name]) for name in LAYERS],
               loc="lower center", bbox_to_anchor=(.5, .085), ncol=3, frameon=False)
    fig.text(.5, .059, "Selected layers only: substrate, dielectric, gates and contacts hidden. Physical aspect ratio; no z exaggeration.",
             ha="center", fontsize=10, color="#44556B")
    fig.text(.5, .028, "New previews are separately meshed polygon prisms, not the full validated thermal stack.",
             ha="center", fontsize=10, color="#8A3030")
    fig.subplots_adjust(left=.025, right=.96, top=.89, bottom=.155, wspace=.08)
    fig.savefig(args.out_dir / "old_vs_new_mesh_3d.png", dpi=190)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 8))
    for name in ("diff", "poly", "licon1"):
        color = {"diff": "#95CDAA", "poly": "#E6A088", "licon1": "#607488"}[name]
        ax.add_collection(PolyCollection(layers[name]["rings"], facecolors=color,
                                         edgecolors="#425264", linewidths=.45, alpha=.68))
    residual = [c["footprint_xy_um"] for c in metadata["residual_licon_components"]]
    ax.add_collection(PolyCollection(residual, facecolors="#E32C3F", edgecolors="#99192C", linewidths=.5))
    ax.set_xlim(xmin-.03, xmax+.03)
    ax.set_ylim(ymin-.03, ymax+.03)
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title("Why the full new SRAM mesh stops\n0.02975 µm² of licon is outside poly/diff", pad=12)
    ax.legend(handles=[Patch(facecolor=c, label=l) for c, l in (
        ("#95CDAA", "Diffusion"), ("#E6A088", "Poly"), ("#607488", "Licon"),
        ("#E32C3F", "Outside modeled poly/diff"))], loc="upper center",
        bbox_to_anchor=(.5, -.11), ncol=2, frameon=False)
    fig.subplots_adjust(bottom=.2, top=.88)
    fig.savefig(args.out_dir / "new_flow_contact_blocker.png", dpi=170)
    plt.close(fig)

    for report in reports.values():
        for key in ("old_shapes", "new_shapes", "extra"):
            del report[key]
    summary = {"cell": metadata["cell"], "gds_sha256": metadata["source_sha256"],
               "old_mesh": str(args.old_msh.resolve()),
               "old_mesh_sha256": hashlib.sha256(args.old_msh.read_bytes()).hexdigest(),
               "old_total_tetrahedra": grid.n_cells, "layers": reports,
               "new_mesh_status": "visualization-only layer prisms; not solver ready",
               "new_full_flow_failure": metadata["new_thermal_flow_failure"]}
    (args.out_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
