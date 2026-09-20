"""Native notebook-style SVG views of ALL SRAM GDS masks and old/new meshes.

The geometry overview has clean, colored prism envelopes, like the notebook;
the companion section view alone draws actual tetrahedral edges.  New prisms
are disconnected visualization previews: the full thermal builder still
rejects this bitcell's unresolved licon landing geometry.
"""
from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path

import gdstk
import gmsh
import numpy as np

from build_sram_polygon_preview import load_notebook_functions
from compare_sram_meshes import read_tetrahedra, polygons
from gds.techmap import FRONTSIDE_STACK, z_bounds
from mesh.gds_svg_viz import _LAYER_COLOR, render_regions_geometry_svg


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "out/sram_mesh_comparison"
LAYER_ORDER = ("nwell", "diff", "poly", "licon1", "li1", "mcon", "met1", "via", "met2")
WIDTH = 1530
PANELS = [(25, 128, 480, 600), (525, 128, 480, 600), (1025, 128, 480, 600)]
ZLOW = -0.3262
STYLE = """text{font-family:Inter,Arial,sans-serif;fill:#263238}
.title{font-size:21px;font-weight:700}.sub{font-size:13px;fill:#546e7a}
.head{font-size:15px;font-weight:700}.small{font-size:12px}
.legend{font-size:11px}.warn{font-size:13px;font-weight:700;fill:#a85110}"""


def points_attribute(points):
    return " ".join(f"{x:.3f},{y:.3f}" for x, y in points)


def polygon(points, color, opacity=.5, stroke=None, stroke_width=.5, extra=""):
    return (f'<polygon points="{points_attribute(points)}" fill="{color}" '
            f'fill-opacity="{opacity}" stroke="{stroke or color}" '
            f'stroke-width="{stroke_width}" {extra}/>')


def label(x, y, text, css="small", extra=""):
    return f'<text class="{css}" x="{x:.2f}" y="{y:.2f}" {extra}>{escape(str(text))}</text>'


def start_svg(height, title, subtitle, warning):
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
           f'viewBox="0 0 {WIDTH} {height}" style="max-width:100%;height:auto">',
           '<rect width="100%" height="100%" fill="#ffffff"/>', f'<style>{STYLE}</style>',
           label(25, 34, title, "title"), label(25, 59, subtitle, "sub"),
           label(25, 84, warning, "warn")]
    for x, y, w, h in PANELS:
        out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" '
                   'fill="#fafafa" stroke="#cfd8dc"/>')
    return out


def plan_projector(bbox, panel=PANELS[0]):
    xmin, ymin, xmax, ymax = bbox
    x, y, w, h = panel
    scale = min((w - 85) / (xmax - xmin), (h - 155) / (ymax - ymin))
    px = x + (w - scale * (xmax - xmin)) / 2
    py = y + 95 + (h - 155 - scale * (ymax - ymin)) / 2
    return lambda point: (px + (point[0] - xmin) * scale, py + (ymax - point[1]) * scale)


def draw_gds(out, audit, cut=None):
    x, y, w, h = PANELS[0]
    out += [label(x + 17, y + 27, "GDS: all mask layers", "head"),
            label(x + 17, y + 48,
                  f"{audit['polygon_count']} polygons · {audit['polygon_spec_count']} layer/datatype pairs"),
            label(x + 17, y + 67, f"{audit['label_count']} text labels · original 2D geometry")]
    project = plan_projector(audit["bbox_um"])
    xmin, ymin, xmax, ymax = audit["bbox_um"]
    box = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    out.append(polygon([project(p) for p in box], "#eaf2f6", .24, "#90a4ae", 1))
    # Large masks first; keep every original shape, even coincident purposes.
    layers = sorted(audit["layers"], key=lambda item: item["union_area_um2"], reverse=True)
    for item in layers:
        physical = item["role"] in {"modeled_physical_mask", "well_region"}
        color = item.get("color", "#78909c")
        for ring in item.get("raw_rings", item["rings"]):
            out.append(polygon([project(p) for p in ring], color, .27 if physical else .025,
                               color, .95 if physical else .65,
                               extra="" if physical else 'stroke-dasharray="3 2"'))
    for group in audit["labels"]:
        for item in group["entries"]:
            px, py = project(item["origin_xy_um"])
            rotation = -float(item.get("rotation", 0)) * 180 / np.pi
            attributes = (f'x="{px:.3f}" y="{py:.3f}" font-size="9" '
                          f'transform="rotate({rotation:.3f},{px:.3f},{py:.3f})" '
                          'text-anchor="middle"')
            # Two passes also work in older librsvg engines without paint-order.
            out.append(f'<text {attributes} style="fill:white" stroke="white" '
                       f'stroke-width="2">{escape(item["text"])}</text>')
            out.append(f'<text {attributes}>{escape(item["text"])}</text>')
    if cut is not None:
        axis, value = cut
        a, b = (((value, ymin), (value, ymax)) if axis == 0
                else ((xmin, value), (xmax, value)))
        aa, bb = project(a), project(b)
        out.append(f'<line x1="{aa[0]}" y1="{aa[1]}" x2="{bb[0]}" y2="{bb[1]}" '
                   'stroke="#1565c0" stroke-width="2" stroke-dasharray="6 4"/>')
    out += [label(x + 17, y + h - 37, f"Footprint: {xmax-xmin:.2f} × {ymax-ymin:.2f} µm"),
            label(x + 17, y + h - 18, "Dashed: non-solid purposes; these are not extra films.")]


def prism_faces(rings, z0, z1, name):
    result = []
    for ring in rings:
        bottom = [(x, y, z0) for x, y in ring]
        top = [(x, y, z1) for x, y in ring]
        result.extend([(name, bottom), (name, top)])
        for i in range(len(ring)):
            j = (i + 1) % len(ring)
            result.append((name, [bottom[i], bottom[j], top[j], top[i]]))
    return result


def make_projector(bbox, zhigh, panel):
    """Same clean orthographic projection as notebook mesh diagnostics; fitted bounds."""
    xmin, ymin, xmax, ymax = bbox
    center = np.array(((xmin+xmax)/2, (ymin+ymax)/2, (ZLOW+zhigh)/2))
    lateral = max(xmax-xmin, ymax-ymin)
    def raw(point):
        x, y, z = (np.asarray(point) - center) / lateral
        return .90*x-.48*y, .30*x+.52*y-z, .42*x+.80*y+.24*z
    box = [(x, y, z) for x in (xmin, xmax) for y in (ymin, ymax) for z in (ZLOW, zhigh)]
    bounds = np.array([raw(p)[:2] for p in box])
    low, high = bounds.min(axis=0), bounds.max(axis=0)
    px, py, w, h = panel
    dw, dh = w-55, h-155
    scale = min(dw/(high[0]-low[0]), dh/(high[1]-low[1]))
    offset = np.array((px+(w-(high[0]-low[0])*scale)/2,
                       py+88+(dh-(high[1]-low[1])*scale)/2))
    def project(point):
        return tuple(offset+(np.array(raw(point)[:2])-low)*scale)
    return raw, project


def draw_envelope(out, faces, colors, bbox, zhigh, panel, title, details, new=False):
    x, y, w, h = panel
    out += [label(x+17, y+27, title, "head"), label(x+17, y+48, details),
            label(x+17, y+67, "Clean region envelopes · z ×1 · no tetrahedral edges")]
    raw, project = make_projector(bbox, zhigh, panel)
    for name, face in sorted(faces, key=lambda tagged:
                             np.mean([raw(point)[2] for point in tagged[1]])):
        color = colors[name]
        opacity = .23 if name == "nwell" else .91
        out.append(polygon([project(p) for p in face], color, opacity, "#455a64", .42))
    xmin, ymin, xmax, ymax = bbox
    box = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    lower = [project((x, y, ZLOW)) for x, y in box]
    upper = [project((x, y, zhigh)) for x, y in box]
    for ring in (lower, upper):
        out.append(polygon(ring, "none", 0, "#78909c", .8, 'stroke-dasharray="4 3"'))
    for a, b in zip(lower, upper):
        out.append(f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" '
                   'stroke="#78909c" stroke-width=".8" stroke-dasharray="4 3"/>')
    out += [label(x+17, y+h-37, "licon1: 2D only; landing remains unresolved" if new
                  else "nwell is folded into bulk silicon in this old model"),
            label(x+17, y+h-18, "Dashed box is a view guide, not meshed fill" if new
                  else "Substrate/dielectric hidden; device-stack crop only")]


def draw_legend(out, audit, start=784):
    out.append(label(25, start-23, "All GDS polygon purposes (same colors in every panel)", "head"))
    for index, item in enumerate(audit["layers"]):
        col, row = index % 3, index // 3
        x, y = 25+col*500, start+row*23
        color = item.get("color", "#78909c")
        out.append(f'<rect x="{x}" y="{y-10}" width="12" height="12" fill="{color}"/>')
        purpose = str(item.get("purpose") or "unmapped")
        text = f"{item['spec'][0]}/{item['spec'][1]}  {item['name']}.{purpose}  ({item['raw_polygon_count']})"
        out.append(label(x+19, y, text, "legend"))
    return start+((len(audit["layers"])+2)//3)*23


def old_entity_boxes(path):
    """Recover the legacy OCC boxes, validating each against its saved tets."""
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(path))
        node_tags, xyz, _ = gmsh.model.mesh.getNodes()
        order = np.argsort(node_tags)
        node_tags = node_tags[order]
        xyz = np.asarray(xyz).reshape(-1, 3)[order]
        result = []
        for dim, tag in gmsh.model.getPhysicalGroups(3):
            for entity in gmsh.model.getEntitiesForPhysicalGroup(3, tag):
                types, _, groups = gmsh.model.mesh.getElements(3, int(entity))
                volume, all_points = 0., []
                for kind, connectivity in zip(types, groups):
                    if kind != 4:
                        raise AssertionError("Legacy box recovery needs linear tetrahedra")
                    points = xyz[np.searchsorted(node_tags, np.asarray(connectivity).reshape(-1, 4))]
                    all_points.append(points.reshape(-1, 3))
                    volume += float(np.abs(np.linalg.det(points[:, 1:]-points[:, :1])).sum()/6)
                points = np.vstack(all_points)
                lo, hi = points.min(axis=0), points.max(axis=0)
                box_volume = float(np.prod(hi-lo))
                if not np.isclose(volume, box_volume, rtol=1e-7, atol=1e-11):
                    raise AssertionError(f"Legacy entity {entity} is not its enclosing box: {volume} vs {box_volume}")
                result.append((int(tag), lo, hi))
        return result
    finally:
        gmsh.finalize()


def region_record(name, rings, z0, z1, color, tag=0):
    """Region schema consumed by the original shared two-panel SVG renderer."""
    return {"physical_tag": tag, "name": name, "role": "conductor",
            "color": color, "z_min_um": z0, "z_max_um": z1,
            "volumes": [{"footprint_xy_um": ring, "area_um2": float(gdstk.Polygon(ring).area())}
                        for ring in rings],
            "source_polygon_count": len(rings), "placeholder": False, "material": name}


def merge_identical_adjacent_regions(regions):
    """Join only identical-footprint adjacent bands; preserve actual old geometry."""
    result = []
    for region in sorted(regions, key=lambda item: (LAYER_ORDER.index(item["name"]), item["z_min_um"])):
        if result and result[-1]["name"] == region["name"] and np.isclose(
                result[-1]["z_max_um"], region["z_min_um"], rtol=0, atol=1e-10):
            previous = result[-1]
            left = [gdstk.Polygon(v["footprint_xy_um"]) for v in previous["volumes"]]
            right = [gdstk.Polygon(v["footprint_xy_um"]) for v in region["volumes"]]
            difference = gdstk.boolean(left, right, "xor", precision=1e-6)
            if not difference:
                previous["z_max_um"] = region["z_max_um"]
                continue
        result.append(region)
    for index, region in enumerate(result):
        region["physical_tag"] = index
    return result


def old_views(grid, names, boxes, colors):
    """Recover exact legacy box envelopes; union boxes, never tiny slice fragments."""
    bands = {band.name: (z0, z1) for z0, z1, band in z_bounds(FRONTSIDE_STACK)}
    offset = bands["channel"][1]
    centers = grid.cell_centers().points
    physical = grid.cell_data["physical_tag"]
    material = {"diff": "Si_SD_doped", "channel": "Si_SD_doped", "poly": "PolySi",
                "licon1": "W", "li1": "TiN", "mcon": "W", "met1": "Al", "via": "W", "met2": "Al"}
    faces, layer_grids, regions = [], {}, []
    for band, target in material.items():
        z0, z1 = bands[band]
        allowed = {tag for tag, text in names.items() if text == target}
        if band == "channel":
            allowed |= {tag for tag, text in names.items() if text.startswith("src_") and not text.startswith("src_co")}
        if band == "licon1":
            allowed |= {tag for tag, text in names.items() if text.startswith("src_co")}
        keep = (centers[:, 2]>z0) & (centers[:, 2]<z1) & np.isin(physical, list(allowed))
        part = grid.extract_cells(keep)
        if not part.n_cells:
            raise AssertionError(f"Old {band} material selection is empty")
        footprints = [gdstk.rectangle(lo[:2], hi[:2]) for tag, lo, hi in boxes
                      if tag in allowed and z0 < (lo[2]+hi[2])/2 < z1]
        union = gdstk.boolean(footprints, [], "or", precision=1e-6)
        expected = {"diff": .9917, "li1": 1.4242, "met1": 1.1692}.get(band)
        if expected is not None and not np.isclose(sum(p.area() for p in union), expected, rtol=1e-8):
            raise AssertionError(f"Legacy {band} footprint changed unexpectedly")
        name = "diff" if band == "channel" else band
        faces += prism_faces([p.points.tolist() for p in union], z0-offset, z1-offset, name)
        regions.append(region_record(name, [p.points.tolist() for p in union],
                                     z0-offset, z1-offset, colors[name]))
        print(f"Old {band}: {part.n_cells:,} tetrahedra, {len(union)} envelope components", flush=True)
        part.points[:, 2] -= offset
        layer_grids[band] = part
        # Licon diffusion contacts also occupy the old poly band.  Recover
        # them separately; neither the poly nor licon material is guessed.
        if band == "poly":
            allowed_w = {tag for tag, text in names.items() if text == "W"}
            keep_w = (centers[:, 2]>z0) & (centers[:, 2]<z1) & np.isin(physical, list(allowed_w))
            lower_contact = grid.extract_cells(keep_w)
            if lower_contact.n_cells:
                footprints = [gdstk.rectangle(lo[:2], hi[:2]) for tag, lo, hi in boxes
                              if tag in allowed_w and z0 < (lo[2]+hi[2])/2 < z1]
                union = gdstk.boolean(footprints, [], "or", precision=1e-6)
                faces += prism_faces([p.points.tolist() for p in union], z0-offset, z1-offset, "licon1")
                regions.append(region_record("licon1", [p.points.tolist() for p in union],
                                             z0-offset, z1-offset, colors["licon1"]))
                lower_contact.points[:, 2] -= offset
                layer_grids["licon_lower"] = lower_contact
    return faces, layer_grids, offset, merge_identical_adjacent_regions(regions)


def render_reference_style_views(audit, metadata, old_regions, new_regions, out_dir):
    """Use the original two-panel renderer and palette, not a similar reimplementation."""
    plan_regions = []
    for index, item in enumerate(sorted(audit["layers"], key=lambda layer: layer["union_area_um2"], reverse=True)):
        rings = item.get("raw_rings", item["rings"])
        region = region_record(item["name"], rings, 0., 0., item["color"], index)
        region.update(role=item["role"], opacity=1.0 if item["role"] in
                      {"modeled_physical_mask", "well_region"} else .08,
                      gds_spec=item["spec"], purpose=item["purpose"])
        plan_regions.append(region)
    paths = {}
    for method, regions in (("old", old_regions), ("new", new_regions)):
        preview = method == "new"
        banner = ("NEW: POLYGON-PRESERVING LAYER PREVIEWS — NOT A SOLVER-READY THERMAL DOMAIN"
                  if preview else "OLD: VERIFIED REGION ENVELOPES OF THE SAVED BOUNDING-BOX THERMAL MESH")
        notes = [
            "Same complete GDS at left: all 26 polygon purposes; non-solid pins, implants and process masks are faint context.",
            "Right: device-stack region envelopes only; substrate/dielectric/passivation hidden. Tetrahedral edges are in the separate section figure.",
            "Legend tags index displayed layer bands, not solver physical tags. Colors and rendering are the original shared SVG renderer.",
            "LI1 is red: the exact layout has six connected pieces; the old bounding-box model joins them into three and closes a 140 nm gap.",
        ]
        if preview:
            notes += [
                "licon1 appears only in the GDS: its 3D extrusion is intentionally omitted because contact landing is unresolved.",
                "The complete new thermal flow rejects 0.02975 µm² of licon landing geometry; these separate prisms are not a heat-solver domain.",
                "nwell is a silicon-region preview below active, not an additional film stacked on the substrate.",
            ]
        else:
            notes += [
                "Old footprints come from 190 saved mesh entities; each entity's tetrahedral volume equals its recovered box volume.",
                "The screenshot's original mask-prism illustration is not the old FEM geometry: this view shows the actual old bounding-box approximation.",
            ]
        text = render_regions_geometry_svg(
            regions,
            title=f"{escape(Path(metadata['source_gds']).name)} / {escape(metadata['cell'])} — {method.upper()} method",
            subtitle=f"{audit['polygon_count']} raw GDS polygons · {audit['polygon_spec_count']} GDS purposes · {len(regions)} displayed 3D layer bands",
            banner=banner,
            plan_label="2D masks (complete real GDS; all 26 layer/datatype pairs)",
            width=1180, z_exaggeration=1.0, notes=notes,
            plan_regions=plan_regions,
            iso_label="3D isometric — exact-polygon previews" if preview
                      else "3D isometric — actual old mesh region envelopes")
        destination = out_dir/f"sram_reference_style_{method}.svg"
        destination.write_text(text, encoding="utf-8")
        paths[method] = str(destination)
    return paths


def choose_cut(layer_grids, bbox):
    """Pick a shared audit plane that intersects as many shown layers as possible."""
    xmin, ymin, xmax, ymax = bbox
    extents = {}
    for name, grid in layer_grids.items():
        vertices = grid.points[grid.cells.reshape(-1, 5)[:, 1:]]
        extents[name] = (vertices.min(axis=1), vertices.max(axis=1))
    best = None
    for axis, low, high in ((0, xmin, xmax), (1, ymin, ymax)):
        for value in np.linspace(low+.1*(high-low), high-.1*(high-low), 19):
            hits = {name: int(np.sum((a[:, axis]<value) & (b[:, axis]>value)))
                    for name, (a, b) in extents.items()}
            score = (sum(count>0 for count in hits.values()), sum(hits.values()))
            if best is None or score > best[0]:
                best = score, axis, float(value)
    return best[1:]


def section_shapes(grid, axis, value, zhigh):
    normal = (1, 0, 0) if axis == 0 else (0, 1, 0)
    origin = [0., 0., 0.]
    origin[axis] = value
    surface = grid.slice(normal=normal, origin=origin)
    surface = surface.clip(normal=(0, 0, 1), origin=(0, 0, ZLOW), invert=False)
    surface = surface.clip(normal=(0, 0, 1), origin=(0, 0, zhigh), invert=True)
    return surface, polygons(surface)


def draw_sections(out, panel, sections, bbox, zhigh, cut, title, detail, footer):
    x, y, w, h = panel
    axis, value = cut
    out += [label(x+17, y+27, title, "head"), label(x+17, y+48, detail),
            label(x+17, y+67, f"{'xy'[axis]} = {value:.4f} µm · actual tetrahedron intersections")]
    xmin, ymin, xmax, ymax = bbox
    low, high = (ymin, ymax) if axis == 0 else (xmin, xmax)
    scale = min((w-95)/(high-low), (h-210)/(zhigh-ZLOW))
    ox = x+(w-(high-low)*scale)/2
    oy = y+105+(h-210-(zhigh-ZLOW)*scale)/2
    project = lambda p: (ox+(p[1-axis]-low)*scale, oy+(zhigh-p[2])*scale)
    for color, shape in sections:
        out.append(polygon([project(p) for p in shape], color, .60, "#263238", .30))
    base = oy+(zhigh-ZLOW)*scale
    out += [label(ox, base+19, f"{low:g}"), label(ox+(high-low)*scale-24, base+19, f"{high:g}"),
            label(ox+(high-low)*scale/2, base+35, f"{'yx'[axis]} (µm)", extra='text-anchor="middle"'),
            label(x+17, y+h-37, "Device-stack crop; identical physical scales · z ×1"),
            label(x+17, y+h-18, footer)]
    for z in (ZLOW, 0, 1, 2):
        py = oy+(zhigh-z)*scale
        out.append(label(ox-9, py+4, f"{z:.2f}", "legend", 'text-anchor="end"'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-msh", type=Path,
                        default=REPO/"out/periodic_verification_dos7Z2/periodic/gds_volume.msh")
    parser.add_argument("--preview-json", type=Path, default=OUT/"new_exact_footprint_meshes.json")
    parser.add_argument("--audit-json", type=Path, default=OUT/"sram_all_layer_audit.json")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(args.preview_json.read_text())
    audit = json.loads(args.audit_json.read_text())
    namespace = load_notebook_functions()
    colors = {**namespace["SKY130_LAYER_COLORS"], **_LAYER_COLOR, "nwell": "#8ECAE6"}
    bbox = metadata["bbox_um"]
    if not np.allclose(bbox, audit["bbox_um"]):
        raise AssertionError("GDS audit and preview windows differ")
    if metadata["source_sha256"] != audit["source_sha256"]:
        raise AssertionError("GDS audit and preview source hashes differ")
    # Native notebook helpers are loaded without executing notebook side effects;
    # this view extends their fitted projection and SVG styling to three panels.
    if "make_mesh_diagnostics_svg" not in namespace:
        raise RuntimeError("Notebook visualization helpers were not loaded")
    for item in audit["layers"]:
        if item["name"] in colors:
            item["color"] = colors[item["name"]]
    old_grid, names = read_tetrahedra(args.old_msh)
    print(f"Read old mesh: {old_grid.n_cells:,} tetrahedra", flush=True)
    if not np.allclose(np.asarray(old_grid.bounds)[:4], [bbox[0], bbox[2], bbox[1], bbox[3]], atol=1e-7, rtol=0):
        raise AssertionError("Old mesh is not the requested GDS bitcell window")
    boxes = old_entity_boxes(args.old_msh)
    print(f"Verified {len(boxes)} legacy entity box volumes against actual tetrahedra", flush=True)
    old_faces, old_parts, offset, old_regions = old_views(old_grid, names, boxes, colors)
    new_faces, new_grids, new_regions = [], {}, []
    for item in metadata["layers"]:
        if "prism_msh_path" not in item:
            continue
        name = item["name"]
        new_faces += prism_faces(item["rings"], *item["z_range_um"], name)
        record = region_record(name, item["rings"], *item["z_range_um"], colors[name], len(new_regions))
        if name == "nwell":
            record.update(role="silicon well", opacity=.3)
        new_regions.append(record)
        new_grids[name], _ = read_tetrahedra(Path(item["prism_msh_path"]))
        print(f"Read new {name}: {new_grids[name].n_cells:,} tetrahedra", flush=True)
    reference_paths = render_reference_style_views(audit, metadata, old_regions, new_regions, args.out_dir)
    zhigh = max(max(point[2] for point in face) for _, face in old_faces+new_faces)
    height = 850+((len(audit["layers"])+2)//3)*23
    warning = "NEW = separate polygon-preserving prism previews; full thermal flow still rejects 0.02975 µm² of licon landing geometry."
    overview = start_svg(height, "SRAM bitcell — complete GDS and old/new region construction",
                         f"sram_sp_cell · one shared footprint · active-top z = 0 · old mesh has {old_grid.n_cells:,} tetrahedra", warning)
    overview.append(label(25, 108, "A GDS purpose is not necessarily a separate physical film: pins, implants and process masks are included at left, not extruded blindly.", "sub"))
    draw_gds(overview, audit)
    draw_envelope(overview, old_faces, colors, bbox, zhigh, PANELS[1],
                  "OLD: bounding-box thermal geometry", "All eight device / interconnect mask families")
    draw_envelope(overview, new_faces, colors, bbox, zhigh, PANELS[2],
                  "NEW: exact-polygon layer previews", "nwell + active + poly + LI1 + M1/M2 + contacts", new=True)
    bottom = draw_legend(overview, audit)
    overview += [label(25, bottom+8, "Envelope view only. Mesh topology is shown in the companion actual-tetrahedron section, not inferred from these clean faces.", "sub"),
                 label(25, bottom+31, "nwell is silicon below active, not an additional film on the substrate. Deep substrate and passivation are outside this device-stack crop.", "sub"), "</svg>"]
    geometry_path = args.out_dir/"sram_notebook_geometry_comparison.svg"
    geometry_path.write_text("\n".join(overview), encoding="utf-8")

    cut = choose_cut({**{"old_"+k: v for k, v in old_parts.items()},
                      **{"new_"+k: v for k, v in new_grids.items()}}, bbox)
    old_grid.points[:, 2] -= offset
    old_surface, old_shapes = section_shapes(old_grid, *cut, zhigh)
    if len(old_shapes) != old_surface.n_cells:
        raise AssertionError("Old section polygons and physical-tag cells differ")
    old_colors = {"Si_bulk": "#D7E8EF", "SiO2": "#E8ECEF", "SiN": "#D6D6E0",
                  "Si_SD_doped": colors["diff"], "PolySi": colors["poly"],
                  "TiN": colors["li1"], "W": colors["licon1"], "Al": colors["met1"]}
    # Use z to distinguish same-material metal/contact families in this real cut.
    bands = [(z0-offset, z1-offset, band.name) for z0, z1, band in z_bounds(FRONTSIDE_STACK)]
    sections_old = []
    for physical, shape in zip(old_surface.cell_data["physical_tag"], old_shapes):
        name = names[int(physical)]
        color = old_colors.get(name, colors["licon1"] if name.startswith("src_co") else colors["diff"])
        center_z = np.mean(shape[:, 2])
        for z0, z1, band in bands:
            if z0-1e-9 <= center_z <= z1+1e-9 and band in colors and name not in ("SiO2", "Si_bulk", "SiN"):
                color = colors["licon1"] if name == "W" and band == "poly" else colors[band]
                break
        sections_old.append((color, shape))
    sections_new, counts_new = [], {}
    for name, grid in new_grids.items():
        _, shapes = section_shapes(grid, *cut, zhigh)
        counts_new[name] = len(shapes)
        sections_new += [(colors[name], shape) for shape in shapes]
    section = start_svg(height, "SRAM bitcell — actual old/new tetrahedral sections",
                        "GDS at left locates the blue cut; both mesh sections use the same plane and physical coordinate scale.", warning)
    section.append(label(25, 108, "Cut chosen to intersect many layers; one plane cannot prove full connectivity or accuracy. The old and preview mesh densities are not matched.", "sub"))
    draw_gds(section, audit, cut)
    draw_sections(section, PANELS[1], sections_old, bbox, zhigh, cut,
                  "OLD: connected thermal mesh", f"{len(sections_old):,} cut-cell polygons including thermal fill",
                  "Pale gray = dielectric; pale blue = bulk silicon")
    draw_sections(section, PANELS[2], sections_new, bbox, zhigh, cut,
                  "NEW: disconnected layer meshes", f"{len(sections_new):,} cut-cell polygons; no assembled thermal fill",
                  "licon1 is intentionally absent from the new 3D preview")
    bottom = draw_legend(section, audit)
    missed = [name for name, count in counts_new.items() if count == 0]
    missed_note = ("This chosen plane does not intersect new " + ", ".join(missed)
                   + "; those layers are present in the geometry overview.") if missed else "This chosen plane intersects every available new prism family."
    section += [label(25, bottom+8, "Black lines are intersections of saved tetrahedra with this cut. Extra lines at the lower/top limits are viewport clipping boundaries.", "sub"),
                label(25, bottom+31, "New blank areas are unassembled space, not insulating material or a valid thermal domain. No heat solve was performed for these previews.", "sub"),
                label(25, bottom+54, missed_note, "sub"), "</svg>"]
    section_path = args.out_dir/"sram_notebook_mesh_sections.svg"
    section_path.write_text("\n".join(section), encoding="utf-8")
    report = {"geometry_svg": str(geometry_path), "actual_sections_svg": str(section_path),
              "reference_style_old_svg": reference_paths["old"],
              "reference_style_new_svg": reference_paths["new"],
              "reference_renderer": "mesh.gds_svg_viz.render_regions_geometry_svg",
              "reference_colors": {name: colors[name] for name in LAYER_ORDER},
              "svg_size_px": [WIDTH, height], "old_mesh": str(args.old_msh),
              "old_envelope_source": "Saved axis-aligned mesh entities; each tetrahedron-volume sum validated against its enclosing box",
              "verified_old_entity_boxes": len(boxes),
              "new_preview_only": True, "gds_polygon_specs_shown": audit["polygon_spec_count"],
              "old_mesh_tetrahedra": old_grid.n_cells, "new_prism_layers": list(new_grids),
              "old_z_offset_um": offset, "z_crop_um": [ZLOW, zhigh],
              "cut_axis": "xy"[cut[0]], "cut_value_um": cut[1],
              "old_cut_polygons": len(sections_old), "new_cut_polygons_by_layer": counts_new,
              "new_thermal_flow_failure": metadata["new_thermal_flow_failure"]}
    (args.out_dir/"notebook_render_summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
