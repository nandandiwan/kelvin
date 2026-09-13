"""SVG "physical-region construction" visualizer: a 2D mask-plan panel plus a
proper isometric 3D cutaway (painter's-algorithm face sorting, real polygon
outlines rather than bounding boxes, per-layer legend) — vector output, so it
stays crisp at any size and needs no VTK/offscreen-rendering context.

Adapted from a reference notebook visualizer the user supplied; rebuilt here
against this project's own data (gds.techmap's real z-bounds + real GDS
polygons) instead of that notebook's `regions`/`layout` schema.
"""

import math
from html import escape
from typing import Dict, List, Tuple

import gdstk
import numpy as np

LayerKey = Tuple[int, int]

# (color) per real layer name — reused from mesh/gds_isometric.py's palette
# for visual consistency across both renderers.
_LAYER_COLOR = {
    "poly": "#c9a0dc",
    "licon1": "#2b2b2b",
    "li1": "#d1342a",
    "mcon": "#2b2b2b",
    "met1": "#e0a840",
    "via": "#2b2b2b",
    "met2": "#3d6fd6",
}
_LAYER_ORDER = ("poly", "licon1", "li1", "mcon", "met1", "via", "met2")


def _build_regions(by_layer: Dict[LayerKey, List], stack, window=None) -> List[dict]:
    """One region per real layer, each holding its real (clipped) polygons —
    actual point lists, not bounding-box rectangles, so non-rectangular
    shapes render faithfully."""
    from gds.techmap import DIFF, LI1, LICON1, MCON, MET1, MET2, POLY, VIA, z_bounds

    layer_keys = {"poly": POLY, "licon1": LICON1, "li1": LI1, "mcon": MCON,
                  "met1": MET1, "via": VIA, "met2": MET2}
    z_by_name = {band.name: (z0, z1) for z0, z1, band in z_bounds(stack)}

    win_rect = None
    if window is not None:
        wx0, wx1, wy0, wy1 = window
        win_rect = gdstk.rectangle((wx0, wy0), (wx1, wy1))

    regions = []
    for tag, name in enumerate(_LAYER_ORDER):
        key = layer_keys[name]
        polys = by_layer.get(key, [])
        if not polys:
            continue
        n_raw = len(polys)
        if win_rect is not None:
            polys = gdstk.boolean(polys, [win_rect], "and")
        if not polys:
            continue
        z0, z1 = z_by_name[name]
        volumes = [{"footprint_xy_um": [tuple(p) for p in poly.points],
                    "area_um2": abs(gdstk.Polygon(poly.points).area())} for poly in polys]
        regions.append({
            "physical_tag": tag, "name": name, "role": "conductor",
            "color": _LAYER_COLOR[name], "z_min_um": z0, "z_max_um": z1,
            "volumes": volumes, "source_polygon_count": n_raw, "placeholder": False,
            "material": name,
        })
    return regions


def make_geometry_svg(by_layer, stack, source_name: str, top_cell_name: str,
                       window=None, width: int = 1180) -> str:
    regions = _build_regions(by_layer, stack, window)
    if not regions:
        raise ValueError("no layer geometry found in the given window")

    all_xy = [point for region in regions for volume in region["volumes"] for point in volume["footprint_xy_um"]]
    xs, ys = zip(*all_xy)
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    xspan, yspan = max(xmax - xmin, 1e-12), max(ymax - ymin, 1e-12)
    lateral = max(xspan, yspan)
    zmin = min(r["z_min_um"] for r in regions)
    zmax = max(r["z_max_um"] for r in regions)
    zspan = max(zmax - zmin, 1e-12)
    z_exaggeration = min(3.0, max(1.0, 0.42 * lateral / zspan))
    meshed_component_count = sum(len(r["volumes"]) for r in regions)
    flattened_polygon_count = sum(r["source_polygon_count"] for r in regions)

    panel_y, panel_h, panel_w, margin = 98, 470, 520, 55
    legend_columns = 3
    legend_rows = math.ceil(len(regions) / legend_columns)
    height = 625 + legend_rows * 25
    plan_scale = min((panel_w - 2 * margin) / xspan, (panel_h - 2 * margin) / yspan)

    def plan(point):
        x, y = point
        return (35 + margin + (x - xmin) * plan_scale, panel_y + margin + (ymax - y) * plan_scale)

    def iso(point, z):
        x = (point[0] - xmin) / lateral
        y = (point[1] - ymin) / lateral
        zz = (z - zmin) * z_exaggeration / lateral
        return (875 + 215 * (x - y), 415 + 105 * (x + y) - 205 * zz)

    def pts(points):
        return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Inter,Arial,sans-serif;fill:#263238}'
        '.title{font-size:19px;font-weight:700}.sub{font-size:12px;fill:#546e7a}'
        '.lab{font-size:12px;font-weight:600}.legend{font-size:11px}'
        '.warn{font-size:12px;font-weight:700;fill:#a85110}</style>',
        f'<text class="title" x="35" y="32">{escape(source_name)} / {escape(top_cell_name)} '
        f'&#8212; physical-region construction</text>',
        f'<text class="sub" x="35" y="54">{flattened_polygon_count} source polygons &#183; '
        f'{len(regions)} tagged layers &#183; {meshed_component_count} shapes &#183; z &#215;{z_exaggeration:.1f}</text>',
        f'<text class="warn" x="35" y="76">REAL SKY130 STACK HEIGHTS + REAL GDS GEOMETRY '
        f'(see spec/materials.py for material assumptions)</text>',
        f'<rect x="35" y="{panel_y}" width="{panel_w}" height="{panel_h}" rx="8" fill="#fafafa" stroke="#cfd8dc"/>',
        f'<rect x="585" y="{panel_y}" width="560" height="{panel_h}" rx="8" fill="#fafafa" stroke="#cfd8dc"/>',
        f'<text class="lab" x="53" y="{panel_y + 27}">2D masks (real GDS polygons)</text>',
        f'<text class="lab" x="603" y="{panel_y + 27}">3D isometric &#8212; painter-sorted faces</text>',
    ]

    domain_plan = [plan((xmin, ymin)), plan((xmax, ymin)), plan((xmax, ymax)), plan((xmin, ymax))]
    out.append(f'<polygon points="{pts(domain_plan)}" fill="#eaf2f6" fill-opacity="0.35" stroke="#90a4ae" stroke-width="1"/>')
    for region in regions:
        for volume in region["volumes"]:
            polygon = [plan(p) for p in volume["footprint_xy_um"]]
            out.append(f'<polygon points="{pts(polygon)}" fill="{region["color"]}" fill-opacity="0.4" '
                       f'stroke="{region["color"]}" stroke-width="1.15"/>')

    faces, top_faces = [], []
    for region in regions:
        z0, z1 = region["z_min_um"], region["z_max_um"]
        for volume in region["volumes"]:
            points = volume["footprint_xy_um"]
            for p0, p1 in zip(points, points[1:] + points[:1]):
                face = [iso(p0, z0), iso(p1, z0), iso(p1, z1), iso(p0, z1)]
                faces.append((p0[0] + p0[1] + p1[0] + p1[1] + z1, region["color"], face))
            top_faces.append((z1, region["color"], [iso(p, z1) for p in points]))
    for _, color, face in sorted(faces, reverse=True):
        out.append(f'<polygon points="{pts(face)}" fill="{color}" fill-opacity="0.62" stroke="#455a64" stroke-width="0.45"/>')
    for _, color, face in sorted(top_faces):
        out.append(f'<polygon points="{pts(face)}" fill="{color}" fill-opacity="0.82" stroke="#37474f" stroke-width="0.60"/>')

    box = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    bottom = [iso(p, zmin) for p in box]
    top = [iso(p, zmax) for p in box]
    out.append(f'<polygon points="{pts(bottom)}" fill="#dceaf2" fill-opacity="0.07" stroke="#78909c" '
               f'stroke-width="0.8" stroke-dasharray="4 3"/>')
    out.append(f'<polygon points="{pts(top)}" fill="none" stroke="#78909c" stroke-width="0.8" stroke-dasharray="4 3"/>')
    for a, b in zip(bottom, top):
        out.append(f'<line x1="{a[0]:.2f}" y1="{a[1]:.2f}" x2="{b[0]:.2f}" y2="{b[1]:.2f}" '
                   f'stroke="#78909c" stroke-width="0.8" stroke-dasharray="4 3"/>')

    legend_y = panel_y + panel_h + 28
    column_width = 380
    for index, region in enumerate(regions):
        column, row = index % legend_columns, index // legend_columns
        x, y = 45 + column * column_width, legend_y + row * 25
        out.append(f'<rect x="{x}" y="{y - 12}" width="14" height="14" fill="{region["color"]}"/>')
        label = (f'tag {region["physical_tag"]} &#183; {region["name"]} &#183; {region["role"]} &#183; '
                 f'z {region["z_min_um"]:g}..{region["z_max_um"]:g}um')
        out.append(f'<text class="legend" x="{x + 20}" y="{y}">{label}</text>')
    out.append("</svg>")
    return "\n".join(out)


def _inferno_hex(t: float) -> str:
    """t in [0, 1] -> '#rrggbb' via matplotlib's inferno, no per-call import
    cost worth avoiding (matplotlib is already a hard dependency here)."""
    import matplotlib

    r, g, b, _ = matplotlib.colormaps["inferno"](max(0.0, min(1.0, t)))
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"




def build_layer_regions(by_layer, stack, window=None) -> List[dict]:
    """Public wrapper over _build_regions: one region per real layer, each
    with its real clipped polygons. Built ONCE and reused across animation
    frames -- the geometry never moves, only the per-volume values do."""
    return _build_regions(by_layer, stack, window)


def render_layer_regions_svg(regions: List[dict], clim, title: str, scale_label: str,
                              width: int = 1180) -> str:
    """Two-panel figure in the same style as make_geometry_svg -- 2D mask
    plan on the left, painter-sorted isometric stack on the right -- but
    each individual polygon is filled from its own `volume["value"]` via
    the inferno colormap over `clim`, instead of by material color.

    The caller supplies the values, deliberately: what makes per-transistor
    structure visible is WHICH quantity is colored, and that is a physics
    decision, not a drawing one. Coloring absolute dT hides the devices
    entirely here -- measured directly on this project's own bitcell
    transient, the device-layer spatial spread is a ~0.0167K ripple that
    stays essentially CONSTANT while the bulk baseline climbs from 0.0003K
    to 31.4K, so on an absolute 0..31K scale every transistor maps to the
    same color and the figure degenerates into one flat tint changing over
    time. Coloring each frame's LOCAL EXCESS above that frame's own
    coolest element instead keeps the devices legible at every instant,
    and still fades honestly to dark when the power stops (the excess
    itself collapses to ~0.0009K, ~19x smaller, once nothing is driving).
    """
    if not regions:
        raise ValueError("no layer geometry to render")

    clim_lo, clim_hi = clim
    clim_span = max(clim_hi - clim_lo, 1e-12)

    def color_of(volume):
        return _inferno_hex((volume["value"] - clim_lo) / clim_span)

    all_xy = [point for region in regions for volume in region["volumes"] for point in volume["footprint_xy_um"]]
    xs, ys = zip(*all_xy)
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    xspan, yspan = max(xmax - xmin, 1e-12), max(ymax - ymin, 1e-12)
    lateral = max(xspan, yspan)
    zmin = min(r["z_min_um"] for r in regions)
    zmax = max(r["z_max_um"] for r in regions)
    z_exaggeration = min(3.0, max(1.0, 0.42 * lateral / max(zmax - zmin, 1e-12)))

    panel_y, panel_h, panel_w, margin = 92, 500, 520, 55
    height = 660
    plan_scale = min((panel_w - 2 * margin) / xspan, (panel_h - 2 * margin) / yspan)

    def plan(point):
        x, y = point
        return (35 + margin + (x - xmin) * plan_scale, panel_y + margin + (ymax - y) * plan_scale)

    def iso(point, z):
        x = (point[0] - xmin) / lateral
        y = (point[1] - ymin) / lateral
        zz = (z - zmin) * z_exaggeration / lateral
        return (865 + 215 * (x - y), 395 + 105 * (x + y) - 195 * zz)

    def pts(points):
        return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Inter,Arial,sans-serif;fill:#263238}'
        '.title{font-size:17px;font-weight:700}.lab{font-size:12px;font-weight:600}'
        '.sc{font-size:11px}</style>',
        f'<text class="title" x="35" y="34">{escape(title)}</text>',
        f'<rect x="35" y="{panel_y}" width="{panel_w}" height="{panel_h}" rx="8" fill="#0d0d12" stroke="#cfd8dc"/>',
        f'<rect x="585" y="{panel_y}" width="560" height="{panel_h}" rx="8" fill="#0d0d12" stroke="#cfd8dc"/>',
        f'<text class="lab" x="53" y="{panel_y - 8}">2D mask plan &#8212; per-device detail</text>',
        f'<text class="lab" x="603" y="{panel_y - 8}">3D isometric &#8212; painter-sorted faces</text>',
    ]

    for region in regions:
        for volume in region["volumes"]:
            polygon = [plan(p) for p in volume["footprint_xy_um"]]
            out.append(f'<polygon points="{pts(polygon)}" fill="{color_of(volume)}" fill-opacity="0.92" '
                       f'stroke="#546e7a" stroke-width="0.5"/>')

    faces, top_faces = [], []
    for region in regions:
        z0, z1 = region["z_min_um"], region["z_max_um"]
        for volume in region["volumes"]:
            color = color_of(volume)
            points = volume["footprint_xy_um"]
            for p0, p1 in zip(points, points[1:] + points[:1]):
                face = [iso(p0, z0), iso(p1, z0), iso(p1, z1), iso(p0, z1)]
                faces.append((p0[0] + p0[1] + p1[0] + p1[1] + z1, color, face))
            top_faces.append((z1, color, [iso(p, z1) for p in points]))
    for _, color, face in sorted(faces, reverse=True):
        out.append(f'<polygon points="{pts(face)}" fill="{color}" fill-opacity="0.9" stroke="#455a64" stroke-width="0.3"/>')
    for _, color, face in sorted(top_faces):
        out.append(f'<polygon points="{pts(face)}" fill="{color}" fill-opacity="0.98" stroke="#546e7a" stroke-width="0.4"/>')

    box = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    bottom, top = [iso(p, zmin) for p in box], [iso(p, zmax) for p in box]
    for ring in (bottom, top):
        out.append(f'<polygon points="{pts(ring)}" fill="none" stroke="#607d8b" stroke-width="0.7" stroke-dasharray="4 3"/>')
    for a, b in zip(bottom, top):
        out.append(f'<line x1="{a[0]:.2f}" y1="{a[1]:.2f}" x2="{b[0]:.2f}" y2="{b[1]:.2f}" '
                   f'stroke="#607d8b" stroke-width="0.7" stroke-dasharray="4 3"/>')

    bar_x, bar_y, bar_w, bar_h = 35, height - 34, 300, 15
    n_stops = 24
    for i in range(n_stops):
        out.append(f'<rect x="{bar_x + i/n_stops*bar_w:.1f}" y="{bar_y}" width="{bar_w/n_stops + 0.6:.1f}" '
                   f'height="{bar_h}" fill="{_inferno_hex(i/n_stops)}"/>')
    out.append(f'<rect x="{bar_x}" y="{bar_y}" width="{bar_w}" height="{bar_h}" fill="none" stroke="#263238" stroke-width="0.6"/>')
    out.append(f'<text class="sc" x="{bar_x}" y="{bar_y - 5}">{escape(scale_label)}</text>')
    out.append(f'<text class="sc" x="{bar_x}" y="{bar_y + bar_h + 13}">{clim_lo:.4f}</text>')
    out.append(f'<text class="sc" x="{bar_x + bar_w}" y="{bar_y + bar_h + 13}" text-anchor="end">{clim_hi:.4f}</text>')

    out.append("</svg>")
    return "\n".join(out)
