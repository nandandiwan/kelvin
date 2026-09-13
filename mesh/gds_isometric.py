"""Clean isometric "layer stack" box render — real per-layer geometry
extruded to its real z-thickness and drawn as semi-transparent boxes, no
tet-meshing needed (this is a picture of the geometry, not a solve). Matches
the reference look in data/image.png: colored boxes per interconnect layer,
dark contacts/vias, thin wireframe bounding box, isometric view.
"""

import os

import gdstk
import pyvista as pv

pv.OFF_SCREEN = True

# name -> (color, opacity). Contacts/vias share one dark tone (matches the
# reference image's convention of "contacts are always dark, metals are
# colored"); met2 blue / met1 tan / li1 red loosely echoes the reference's
# top-to-bottom blue/tan/red layering.
_LAYER_STYLE = {
    "poly": ("#c9a0dc", 0.45),
    "licon1": ("#2b2b2b", 0.85),
    "li1": ("#d1342a", 0.55),
    "mcon": ("#2b2b2b", 0.85),
    "met1": ("#e0a840", 0.55),
    "via": ("#2b2b2b", 0.85),
    "met2": ("#3d6fd6", 0.55),
}
# rendered back-to-front (bottom layer first) so transparency looks right
_LAYER_ORDER = ("poly", "licon1", "li1", "mcon", "met1", "via", "met2")


def _poly_bbox_um(poly):
    (x0, y0), (x1, y1) = poly.bounding_box()
    return x0, x1, y0, y1


def render_layer_stack_iso(by_layer, stack, out_path, title="", window=None,
                            window_size=(1400, 1100), show_die_outline=True):
    """`window=(x0,x1,y0,y1)` clips which polygons to draw (e.g. one bitcell,
    or a small tile of several) — cheap regardless of how many real polygons
    exist elsewhere, since this only touches the ones inside it.
    """
    from gds.techmap import DIFF, LI1, LICON1, MCON, MET1, MET2, POLY, VIA, z_bounds

    layer_keys = {"poly": POLY, "licon1": LICON1, "li1": LI1, "mcon": MCON,
                  "met1": MET1, "via": VIA, "met2": MET2}
    z_by_name = {band.name: (z0, z1) for z0, z1, band in z_bounds(stack)}

    if window is not None:
        wx0, wx1, wy0, wy1 = window
        win_rect = gdstk.rectangle((wx0, wy0), (wx1, wy1))

    pl = pv.Plotter(off_screen=True, window_size=window_size)
    pl.background_color = "white"

    all_bounds = None
    for name in _LAYER_ORDER:
        key = layer_keys[name]
        polys = by_layer.get(key, [])
        if not polys:
            continue
        if window is not None:
            polys = gdstk.boolean(polys, [win_rect], "and")
        z0, z1 = z_by_name[name]
        color, opacity = _LAYER_STYLE[name]
        for poly in polys:
            x0, x1, y0, y1 = _poly_bbox_um(poly)
            box = pv.Box(bounds=(x0, x1, y0, y1, z0, z1))
            pl.add_mesh(box, color=color, opacity=opacity, show_edges=True,
                        edge_color="black", line_width=1)
            b = box.bounds
            if all_bounds is None:
                all_bounds = list(b)
            else:
                all_bounds[0] = min(all_bounds[0], b[0]); all_bounds[1] = max(all_bounds[1], b[1])
                all_bounds[2] = min(all_bounds[2], b[2]); all_bounds[3] = max(all_bounds[3], b[3])
                all_bounds[4] = min(all_bounds[4], b[4]); all_bounds[5] = max(all_bounds[5], b[5])

    if all_bounds is None:
        raise ValueError("no polygons found in the given window/layers")

    if show_die_outline:
        outline = pv.Box(bounds=all_bounds)
        pl.add_mesh(outline, style="wireframe", color="gray", line_width=1, opacity=0.6)

    if title:
        pl.add_text(title, font_size=10, color="black")

    pl.view_isometric()
    pl.camera.parallel_projection = True
    pl.reset_camera()
    pl.camera.zoom(1.15)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pl.screenshot(out_path)
    pl.close()
    return out_path
