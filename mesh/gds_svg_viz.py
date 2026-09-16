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
    "diff": "#7fc08a",
    "poly": "#c9a0dc",
    "licon1": "#2b2b2b",
    "li1": "#d1342a",
    "mcon": "#2b2b2b",
    "met1": "#e0a840",
    "via": "#2b2b2b",
    "met2": "#3d6fd6",
}
# Bottom-up, so the painter's-algorithm sort starts from the real substrate
# side. `diff` (the S/D active area) is the FEOL layer everything else sits
# on; without it the isometric stack appears to float above nothing.
#
# The `channel` band is deliberately NOT listed: it is drawn from the same
# (65,20) DIFF polygons as `diff`, just at the thin z-slice where DIFF meets
# POLY, so including both would draw the identical footprint twice at two
# heights and read as a duplicated layer.
_LAYER_ORDER = ("diff", "poly", "licon1", "li1", "mcon", "met1", "via", "met2")


def _build_regions(by_layer: Dict[LayerKey, List], stack, window=None) -> List[dict]:
    """One region per real layer, each holding its real (clipped) polygons —
    actual point lists, not bounding-box rectangles, so non-rectangular
    shapes render faithfully."""
    from gds.techmap import DIFF, LI1, LICON1, MCON, MET1, MET2, POLY, VIA, z_bounds

    layer_keys = {"diff": DIFF, "poly": POLY, "licon1": LICON1, "li1": LI1,
                  "mcon": MCON, "met1": MET1, "via": VIA, "met2": MET2}
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
    meshed = sum(len(r["volumes"]) for r in regions)
    flattened = sum(r["source_polygon_count"] for r in regions)
    return render_regions_geometry_svg(
        regions,
        title=f"{source_name} / {top_cell_name} &#8212; physical-region construction",
        subtitle=f"{flattened} source polygons &#183; {len(regions)} tagged layers &#183; "
                 f"{meshed} shapes",
        banner="REAL SKY130 STACK HEIGHTS + REAL GDS GEOMETRY "
               "(see spec/materials.py for material assumptions)",
        plan_label="2D masks (real GDS polygons)",
        width=width)


def render_regions_geometry_svg(regions: List[dict], title: str, subtitle: str,
                                 banner: str, plan_label: str = "2D plan",
                                 width: int = 1180, z_exaggeration: float = None,
                                 notes: List[str] = None) -> str:
    """Draw an already-built `regions` list: 2D plan on the left, painter-sorted
    isometric on the right.

    Split out of make_geometry_svg so geometry that exists in no GDS layer can
    reuse the same renderer -- the BS-PDN stack's uTSVs, buried power rails and
    backside metal tracks are SYNTHETIC grids (gds.techmap.SYNTHETIC), generated
    from a pitch/width rule rather than read from a mask, so they can never go
    through _build_regions.
    """
    if not regions:
        raise ValueError("no layer geometry to render")

    all_xy = [point for region in regions for volume in region["volumes"] for point in volume["footprint_xy_um"]]
    xs, ys = zip(*all_xy)
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    xspan, yspan = max(xmax - xmin, 1e-12), max(ymax - ymin, 1e-12)
    lateral = max(xspan, yspan)
    zmin = min(r["z_min_um"] for r in regions)
    zmax = max(r["z_max_um"] for r in regions)
    zspan = max(zmax - zmin, 1e-12)
    # The auto rule is tuned for a stack whose height is comparable to its
    # lateral extent (the SKY130 bitcell: 3.2um of stack over 1.6um of cell).
    # A BS-PDN unit cell is the opposite shape -- 0.86um of stack over a
    # several-um window -- and the 3.0 cap then leaves the isometric so flat
    # that it runs off the bottom of its panel. Callers with a flat stack pass
    # their own value.
    if z_exaggeration is None:
        z_exaggeration = min(3.0, max(1.0, 0.42 * lateral / zspan))

    panel_y, panel_h, panel_w, margin = 98, 470, 520, 55
    legend_columns = 3
    legend_rows = math.ceil(len(regions) / legend_columns)
    height = 625 + legend_rows * 25 + (22 * len(notes) + 14 if notes else 0)
    plan_scale = min((panel_w - 2 * margin) / xspan, (panel_h - 2 * margin) / yspan)

    def plan(point):
        x, y = point
        return (35 + margin + (x - xmin) * plan_scale, panel_y + margin + (ymax - y) * plan_scale)

    # Auto-fit the isometric to its panel. The projection's vertical extent is
    # 210px of plan depth plus 205*zz_max of stack height, and with a tall
    # z_exaggeration that runs off the bottom of the panel box (the fixed
    # y-origin was tuned for the bitcell's own proportions). Compute the
    # extent, then shift and, if needed, shrink to fit.
    zz_max = (zmax - zmin) * z_exaggeration / lateral
    iso_h = 210.0 + 205.0 * zz_max
    iso_fit = min(1.0, (panel_h - 70.0) / iso_h)
    iso_y_mid = (415.0 - 205.0 * zz_max + 625.0) / 2.0
    iso_dy = (panel_y + panel_h / 2.0) - iso_y_mid * iso_fit

    def iso(point, z):
        x = (point[0] - xmin) / lateral
        y = (point[1] - ymin) / lateral
        zz = (z - zmin) * z_exaggeration / lateral
        return (875 + 215 * iso_fit * (x - y),
                (415 + 105 * (x + y) - 205 * zz) * iso_fit + iso_dy)

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
        f'<text class="title" x="35" y="32">{title}</text>',
        f'<text class="sub" x="35" y="54">{subtitle} &#183; z &#215;{z_exaggeration:.1f}</text>',
        f'<text class="warn" x="35" y="76">{banner}</text>',
        f'<rect x="35" y="{panel_y}" width="{panel_w}" height="{panel_h}" rx="8" fill="#fafafa" stroke="#cfd8dc"/>',
        f'<rect x="585" y="{panel_y}" width="560" height="{panel_h}" rx="8" fill="#fafafa" stroke="#cfd8dc"/>',
        f'<text class="lab" x="53" y="{panel_y + 27}">{plan_label}</text>',
        f'<text class="lab" x="603" y="{panel_y + 27}">3D isometric &#8212; painter-sorted faces</text>',
    ]

    domain_plan = [plan((xmin, ymin)), plan((xmax, ymin)), plan((xmax, ymax)), plan((xmin, ymax))]
    out.append(f'<polygon points="{pts(domain_plan)}" fill="#eaf2f6" fill-opacity="0.35" stroke="#90a4ae" stroke-width="1"/>')
    for region in regions:
        # A region may declare `opacity` as a multiplier -- used for "context"
        # bodies (the silicon a via passes THROUGH) that must be visible enough
        # to show the features are embedded, but not so opaque they hide them.
        op = region.get("opacity", 1.0)
        for volume in region["volumes"]:
            polygon = [plan(p) for p in volume["footprint_xy_um"]]
            out.append(f'<polygon points="{pts(polygon)}" fill="{region["color"]}" '
                       f'fill-opacity="{0.4*op:.3f}" '
                       f'stroke="{region["color"]}" stroke-width="{1.15*op:.2f}"/>')

    faces, top_faces = [], []
    for region in regions:
        z0, z1 = region["z_min_um"], region["z_max_um"]
        op = region.get("opacity", 1.0)
        for volume in region["volumes"]:
            points = volume["footprint_xy_um"]
            for p0, p1 in zip(points, points[1:] + points[:1]):
                face = [iso(p0, z0), iso(p1, z0), iso(p1, z1), iso(p0, z1)]
                faces.append((p0[0] + p0[1] + p1[0] + p1[1] + z1, region["color"], face, op))
            top_faces.append((z1, region["color"], [iso(p, z1) for p in points], op))
    for _, color, face, op in sorted(faces, reverse=True, key=lambda f: f[0]):
        out.append(f'<polygon points="{pts(face)}" fill="{color}" fill-opacity="{0.62*op:.3f}" '
                   f'stroke="#455a64" stroke-width="{0.45*op:.2f}"/>')
    for _, color, face, op in sorted(top_faces, key=lambda f: f[0]):
        out.append(f'<polygon points="{pts(face)}" fill="{color}" fill-opacity="{0.82*op:.3f}" '
                   f'stroke="#37474f" stroke-width="{0.60*op:.2f}"/>')

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

    if notes:
        note_y = legend_y + legend_rows * 25 + 12
        for i, note in enumerate(notes):
            out.append(f'<text class="sub" x="45" y="{note_y + i * 22}">{note}</text>')
    out.append("</svg>")
    return "\n".join(out)


def _inferno_hex(t: float) -> str:
    """t in [0, 1] -> '#rrggbb' via matplotlib's inferno, no per-call import
    cost worth avoiding (matplotlib is already a hard dependency here)."""
    import matplotlib

    r, g, b, _ = matplotlib.colormaps["inferno"](max(0.0, min(1.0, t)))
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"




SUBSTRATE_DEPTHS_UM = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def substrate_regions(stack, window, tag_start: int = 0,
                      depths_um=SUBSTRATE_DEPTHS_UM, color: str = "#6b7a85") -> List[dict]:
    """Slabs of the SILICON SUBSTRATE beneath the device stack, as extra
    regions in the same schema the layer regions use.

    Why this exists: the drawn GDS layers span only ~3.2um (z 50.21 -> 53.37
    for the frontside stack) while the substrate under them is **50.2um** --
    94% of the model, and the entire path the heat actually takes. Rendering
    only the tagged layers makes every heat animation look like the heat is
    "stuck" in the bottom drawn layer, when really it is leaving downward
    through silicon that was never drawn.

    Sliced into progressively thicker slabs (fine near the devices, coarse
    with depth) rather than one block, so an animation can show the thermal
    front moving DOWN through them: at nanosecond timescales the diffusion
    length is sub-micron, so only the top slab warms; approaching steady
    state the whole depth develops the near-linear conduction profile.

    `depths_um` are depths BELOW the top of the substrate, cumulative.
    """
    from gds.techmap import z_bounds

    x0, x1, y0, y1 = window
    z_sub_top = z_sub_bot = None
    for z0, z1, band in z_bounds(stack):
        if band.name == "Si_substrate":
            z_sub_bot, z_sub_top = z0, z1
            break
    if z_sub_top is None:
        return []

    regions, tag, prev = [], tag_start, z_sub_top
    for d in depths_um:
        z_lo = max(z_sub_bot, z_sub_top - d)
        if z_lo >= prev - 1e-9:
            continue
        regions.append({
            "physical_tag": tag,
            "name": f"substrate {z_sub_top - prev:g}-{z_sub_top - z_lo:g}um deep",
            "role": "substrate", "color": color,
            "z_min_um": round(z_lo, 4), "z_max_um": round(prev, 4),
            "volumes": [{"footprint_xy_um": [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                         "area_um2": (x1 - x0) * (y1 - y0)}],
            "source_polygon_count": 1, "placeholder": False, "material": "Si_bulk",
        })
        tag += 1
        prev = z_lo
        if z_lo <= z_sub_bot + 1e-9:
            break
    return regions


def build_layer_regions(by_layer, stack, window=None, include_substrate: bool = False,
                        substrate_depths_um=SUBSTRATE_DEPTHS_UM) -> List[dict]:
    """Public wrapper over _build_regions: one region per real layer, each
    with its real clipped polygons. Built ONCE and reused across animation
    frames -- the geometry never moves, only the per-volume values do.

    `include_substrate` appends `substrate_regions` slabs below the stack --
    needed for any figure meant to show where the heat actually GOES, since
    the drawn layers are only the top 6% of the model (see that function).
    Requires an explicit `window`: the substrate has no polygons of its own
    to infer a footprint from.
    """
    regions = _build_regions(by_layer, stack, window)
    if include_substrate:
        if window is None:
            raise ValueError("include_substrate needs an explicit window "
                             "(the substrate has no polygons to infer one from)")
        regions = regions + substrate_regions(stack, window, tag_start=len(regions),
                                              depths_um=substrate_depths_um)
    return regions


def dof_indices_per_volume(regions: List[dict], dof_coords_um, z_pad_um: float = 0.05):
    """For each drawn shape (every gate stripe, contact, metal segment -- one
    per `volume` in `regions`), the dofs sitting inside its xy bounding box
    and within its layer's own z band. Computed ONCE (geometry doesn't move)
    and reused every animation frame.

    Sampling a shape's HOTTEST dof rather than its centroid matters: a gate
    stripe's centroid can fall between two devices and miss the hot channel
    entirely. First proved out per-cell in cases/render_transient_layers.py;
    lives here now that a second caller (a tiled multi-cell array) needs the
    identical rule and a third copy would just be drift waiting to happen.

    Returns a flat list of int index arrays, one per volume, in the same
    (region, volume) traversal order `regions` itself iterates in -- callers
    that rebuild `regions` from the same by_layer/stack/window get back
    exactly matching order, so a saved per-volume history array lines up
    with a freshly rebuilt regions list without needing its own manifest.
    """
    x, y, z = dof_coords_um[:, 0], dof_coords_um[:, 1], dof_coords_um[:, 2]
    per_volume = []
    for region in regions:
        in_z = (z >= region["z_min_um"] - z_pad_um) & (z <= region["z_max_um"] + z_pad_um)
        for volume in region["volumes"]:
            pts = volume["footprint_xy_um"]
            x0 = min(p[0] for p in pts); x1 = max(p[0] for p in pts)
            y0 = min(p[1] for p in pts); y1 = max(p[1] for p in pts)
            mask = in_z & (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
            idx = np.flatnonzero(mask)
            if idx.size == 0:  # sliver thinner than the mesh -- fall back to nearest dof
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                cz = (region["z_min_um"] + region["z_max_um"]) / 2
                idx = np.array([int(np.argmin((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2))])
            per_volume.append(idx)
    return per_volume


def render_layer_regions_svg(regions: List[dict], clim, title: str, scale_label: str,
                              width: int = 1180, z_exaggeration: float = None) -> str:
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
    # Same override as render_regions_geometry_svg: the automatic rule assumes
    # a stack roughly as tall as it is wide, and returns 1.0 once the substrate
    # is included (53um of depth over a 1.6-6um footprint), which renders as an
    # unreadable needle. Callers drawing substrate pass their own value.
    if z_exaggeration is None:
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
