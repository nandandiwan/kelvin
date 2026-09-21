"""Verify and visualize LI1-only connectivity in GDS, legacy, and new meshes.

The legacy footprint is recovered from OCC boxes whose saved tetrahedron
volumes are validated by old_entity_boxes. New exact rings are cross-checked
against source GDS, saved tetrahedron volumes, and face-connected components.
No thermal simulation or source-geometry modification is performed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from html import escape
import hashlib
import itertools
import json
from pathlib import Path

import gdstk
import numpy as np

from compare_sram_meshes import read_tetrahedra
from render_sram_notebook_comparison import old_entity_boxes
from gds.techmap import FRONTSIDE_STACK, z_bounds


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "out/sram_mesh_comparison"
RED = "#d1342a"


def area(polygons):
    return float(sum(polygon.area() for polygon in polygons))


def face_components(points, tetrahedra):
    """Find tetrahedron components through shared triangular faces."""
    parent = np.arange(len(tetrahedra))
    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return int(index)
    face_owner = {}
    for index, tet in enumerate(tetrahedra):
        for face in itertools.combinations(tet, 3):
            key = tuple(sorted(map(int, face)))
            if key in face_owner:
                parent[find(index)] = find(face_owner[key])
            else:
                face_owner[key] = index
    groups = defaultdict(list)
    for index in range(len(tetrahedra)):
        groups[find(index)].append(index)
    records = []
    for indices in groups.values():
        cells = tetrahedra[indices]
        vertices = points[cells]
        volume = float(np.abs(np.linalg.det(vertices[:, 1:] - vertices[:, :1])).sum() / 6)
        used = points[np.unique(cells)]
        records.append({
            "tetrahedron_count": len(indices), "volume_um3": volume,
            "bbox_min_um": used.min(axis=0).tolist(), "bbox_max_um": used.max(axis=0).tolist(),
        })
    return sorted(records, key=lambda record: (record["bbox_min_um"][1], record["bbox_min_um"][0]))


def render_svg(report, exact, old, new):
    width, height = 1400, 830
    xmin, ymin, xmax, ymax = report["bbox_um"]
    panels = ((20, "Exact GDS", exact), (480, "Actual legacy mesh footprint", old),
              (940, "Actual new mesh footprint", new))
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<style>text{font-family:Inter,Arial,sans-serif;fill:#263238}.title{font-size:22px;font-weight:700}.head{font-size:16px;font-weight:700}.small{font-size:13px}.note{font-size:12px;fill:#546e7a}</style>',
           '<defs><marker id="arrow" viewBox="0 0 6 6" refX="3" refY="3" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0L6 3L0 6Z" fill="#1565c0"/></marker></defs>',
           '<text x="20" y="35" class="title">SRAM LI1: real gaps versus bounding-box merging</text>',
           '<text x="20" y="61" class="small">sram_sp_cell · same 1.20 × 1.58 µm footprint and x/y scale · LI1 is red in the reference palette</text>',
           '<text x="20" y="85" class="note">Counts are connected components of LI1 alone—not whole-chip thermal connectivity or necessarily different electrical nets.</text>']
    for index, (px, title, shapes) in enumerate(panels):
        svg.append(f'<rect x="{px}" y="105" width="440" height="630" rx="8" fill="#fafafa" stroke="#cfd8dc"/>')
        svg.append(f'<text x="{px+18}" y="133" class="head">{escape(title)}</text>')
        if index == 0:
            subtitle = "8 raw polygons → 6 genuine connected pieces"
        else:
            count = report["old_mesh" if index == 1 else "new_mesh"]["component_count"]
            subtitle = f"{count} components verified from saved tetrahedra"
        svg.append(f'<text x="{px+18}" y="155" class="small">{subtitle}</text>')
        scale = min(350 / (xmax-xmin), 485 / (ymax-ymin))
        xoff, yoff = px + (440-(xmax-xmin)*scale)/2, 190
        def project(point):
            return xoff+(point[0]-xmin)*scale, yoff+(ymax-point[1])*scale
        bx0, by0 = project((xmin, ymax))
        svg.append(f'<rect x="{bx0}" y="{by0}" width="{(xmax-xmin)*scale}" height="{(ymax-ymin)*scale}" fill="white" stroke="#90a4ae" stroke-dasharray="4 3"/>')
        for shape in shapes:
            rendered = " ".join(f"{x:.3f},{y:.3f}" for x,y in map(project, shape.points))
            svg.append(f'<polygon points="{rendered}" fill="{RED}" fill-opacity=".76" stroke="#8d251f" stroke-width=".8"/>')
        if index in (0, 2):
            a, b = (project(point) for point in report["example_gap"]["endpoints_xy_um"])
            svg.append(f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" stroke="#1565c0" stroke-width="1.4" marker-start="url(#arrow)" marker-end="url(#arrow)"/>')
            # Separate halo and fill work in SVG renderers without paint-order.
            gap_label = (f'x="{(a[0]+b[0])/2}" y="{a[1]-11}" '
                         'text-anchor="middle" font-size="12"')
            svg.append(f'<text {gap_label} stroke="white" stroke-width="3">140 nm</text>')
            svg.append(f'<text {gap_label} style="fill:#1565c0">140 nm</text>')
        for value in (xmin, xmax):
            x,y = project((value,ymin))
            svg.append(f'<text x="{x}" y="{y+20}" class="note" text-anchor="middle">{value:g}</text>')
        svg.append(f'<text x="{px+220}" y="{yoff+(ymax-ymin)*scale+36}" class="note" text-anchor="middle">x (µm); y spans −1.58 to 0 µm</text>')
        result_area = report["exact_gds"]["union_area_um2"] if index == 0 else report["old_mesh" if index == 1 else "new_mesh"]["footprint_area_um2"]
        svg.append(f'<text x="{px+18}" y="707" class="small">LI1 area: {result_area:.4f} µm²' + (" (+59.15%)" if index == 1 else " (gaps preserved)") + '</text>')
    svg += [
        '<text x="20" y="763" class="small">Legacy rectangles join two bent traces and both side pads into one central piece. The new mesh retains all six exact LI1 pieces.</text>',
        '<text x="20" y="787" class="note">The two raw polygon overlaps occur at the top/bottom rails; their union is legitimate. New prism meshes remain visualization-only.</text>',
        '<text x="20" y="809" class="note">Footprints are shown without tetrahedral edges; connectivity was measured from saved tetrahedra, not inferred from this picture.</text>',
        '</svg>',
    ]
    return "\n".join(svg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gds", type=Path, default=REPO / "data/sram22_64x22m4w22.gds")
    parser.add_argument("--old-msh", type=Path, default=REPO / "out/periodic_verification_dos7Z2/periodic/gds_volume.msh")
    parser.add_argument("--preview-json", type=Path, default=OUT / "new_exact_footprint_meshes.json")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    metadata = json.loads(args.preview_json.read_text())
    source_hash = hashlib.sha256(args.gds.read_bytes()).hexdigest()
    assert source_hash == metadata["source_sha256"]
    library = gdstk.read_gds(args.gds, unit=1e-6)
    cell = next(cell for cell in library.cells if cell.name == metadata["cell"])
    raw = [polygon for polygon in cell.get_polygons() if (polygon.layer, polygon.datatype) == (67, 20)]
    precision = gdstk.gds_units(args.gds)[1] * 1e6
    exact = gdstk.boolean(raw, [], "or", precision=precision)
    overlaps = []
    for (i, left), (j, right) in itertools.combinations(enumerate(raw), 2):
        overlap = area(gdstk.boolean([left], [right], "and", precision=precision))
        if overlap > 0:
            overlaps.append({"raw_indices": [i, j], "overlap_area_um2": overlap})
    old_grid, names = read_tetrahedra(args.old_msh)
    xmin, ymin, xmax, ymax = metadata["bbox_um"]
    assert np.allclose(np.asarray(old_grid.bounds)[:4], [xmin, xmax, ymin, ymax], rtol=0, atol=1e-7)
    old_boxes = old_entity_boxes(args.old_msh)
    z0, z1 = next((z0,z1) for z0,z1,band in z_bounds(FRONTSIDE_STACK) if band.name == "li1")
    tags = {tag for tag,name in names.items() if name == "TiN"}
    centers = old_grid.cell_centers().points
    keep = np.isin(old_grid.cell_data["physical_tag"], list(tags)) & (centers[:,2]>z0) & (centers[:,2]<z1)
    old_tets = old_grid.cells.reshape(-1,5)[keep,1:]
    old_components = face_components(old_grid.points, old_tets)
    old = gdstk.boolean([gdstk.rectangle(lo[:2],hi[:2]) for tag,lo,hi in old_boxes
                        if tag in tags and z0 < (lo[2]+hi[2])/2 < z1], [], "or", precision=precision)
    entry = next(entry for entry in metadata["layers"] if entry["name"] == "li1")
    new_grid, _ = read_tetrahedra(Path(entry["prism_msh_path"]))
    new_tets = new_grid.cells.reshape(-1,5)[:,1:]
    new_components = face_components(new_grid.points,new_tets)
    new = [gdstk.Polygon(ring) for ring in entry["rings"]]
    assert not gdstk.boolean(exact,new,"xor",precision=precision)
    assert len(raw) == 8 and len(exact) == 6 and len(overlaps) == 2
    assert all(np.isclose(item["overlap_area_um2"], .01275) for item in overlaps)
    assert np.isclose(area(raw)-sum(item["overlap_area_um2"] for item in overlaps), area(exact))
    assert len(old_components) == len(old) == 3 and len(new_components) == len(new) == 6
    assert np.isclose(sum(c["volume_um3"] for c in old_components)/(z1-z0),area(old),rtol=1e-9)
    assert np.isclose(sum(c["volume_um3"] for c in new_components)/(entry["z_range_um"][1]-entry["z_range_um"][0]),area(new),rtol=1e-9)
    assert np.isclose(area(exact), .8949) and np.isclose(area(old), 1.4242)
    # Interior gap sample avoids boundary-rounding ambiguity.
    gap = [[-.835,-.875],[-.695,-.875]]
    interior = [(-.835+.14*t,-.875) for t in np.linspace(.05,.95,19)]
    assert not any(gdstk.inside(interior,exact)) and not any(gdstk.inside(interior,new))
    assert all(gdstk.inside(interior,old))
    report = {
        "cell":metadata["cell"],"source_gds":str(args.gds.resolve()),"source_sha256":source_hash,
        "bbox_um":metadata["bbox_um"],"layer_spec":[67,20],"color":RED,
        "component_definition":"Tetrahedra connected by shared triangular faces within LI1 only; not net connectivity or whole-domain connectivity.",
        "exact_gds":{"raw_polygon_count":len(raw),"raw_area_sum_um2":area(raw),"component_count":len(exact),
                     "union_area_um2":area(exact),"legitimate_raw_overlaps":overlaps},
        "old_mesh":{"path":str(args.old_msh.resolve()),"component_count":len(old_components),
                    "tetrahedron_count":len(old_tets),"footprint_area_um2":area(old),"components":old_components,
                    "footprint_method":"Union of legacy entity boxes, each validated against saved tetrahedron volume."},
        "new_mesh":{"path":entry["prism_msh_path"],"component_count":len(new_components),
                    "tetrahedron_count":len(new_tets),"footprint_area_um2":area(new),"components":new_components,
                    "footprint_method":"Exact prism rings validated against source GDS, saved tetrahedron volumes, and connected components.",
                    "solver_ready":False},
        "legacy_excess_area_percent":100*(area(old)/area(exact)-1),
        "example_gap":{"endpoints_xy_um":gap,"width_nm":140.,"inside_exact_material":False,"inside_new_material":False,"filled_by_legacy_bbox":True,
                       "interior_sample_count":len(interior)},
        "checks_passed":["source hash match","new footprint XOR source GDS empty","legacy boxes match tetrahedron volumes",
                         "old/new mesh volumes match footprints","raw overlap accounting","6 new versus 3 old face-connected LI1 components",
                         "140 nm example gap empty in GDS/new and filled in old"],
    }
    args.out_dir.mkdir(parents=True,exist_ok=True)
    (args.out_dir/"li1_connectivity.json").write_text(json.dumps(report,indent=2)+"\n")
    (args.out_dir/"li1_connectivity.svg").write_text(render_svg(report,exact,old,new))
    print(json.dumps({"old_components":len(old_components),"new_components":len(new_components),
                      "exact_area_um2":area(exact),"old_area_um2":area(old),"gap_nm":140.,"out":str(args.out_dir)},indent=2))


if __name__ == "__main__":
    main()
