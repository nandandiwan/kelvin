"""2D cross-section emitter for a GDS-derived chip: cuts the real polygon
layout at y=cut_y_um using the same background/override interval-insertion
pattern mesh/section.py uses for the synthetic chip (mesh/boxes.insert_interval),
but sourcing override x-intervals from actual GDS polygons via a thin-strip
boolean intersection instead of synthetic device positions.

Assumes gmsh.initialize() and gmsh.model.add(...) have already been called;
this module only adds OCC entities to the current model.
"""

import dataclasses
from typing import Dict, List, Tuple

import gdstk
import gmsh

from gds.techmap import SYNTHETIC, z_bounds

from .boxes import RegionRegistry, insert_interval

_EPS = 1e-9
# GDS precision here is 1nm (0.001um) -- a strip thinner than that quantizes
# to zero height and every boolean intersection silently comes back empty
# (verified: 0.1nm strip -> 0 hits, 1nm -> 18 hits, no further change above
# that). 10nm gives comfortable margin above the grid while staying tiny
# next to the smallest real feature (90nm gate length).
_STRIP_HALF_HEIGHT_UM = 1e-2


def die_bounds(by_layer) -> Tuple[float, float, float, float]:
    boxes = [p.bounding_box() for polys in by_layer.values() for p in polys]
    x0 = min(b[0][0] for b in boxes)
    x1 = max(b[1][0] for b in boxes)
    y0 = min(b[0][1] for b in boxes)
    y1 = max(b[1][1] for b in boxes)
    return x0, x1, y0, y1


def _crosses_cut(poly: "gdstk.Polygon", cut_y_um: float) -> bool:
    """Cheap bbox pre-check before paying for a gdstk.boolean call. A real
    macro can carry tens of thousands of per-source polygons (this one has
    54,618) and only a handful ever cross any single cutline — checking
    each one's bounding box first turned a ~2.5 minute per-source boolean
    sweep into a sub-second one."""
    (_, y0), (_, y1) = poly.bounding_box()
    return y0 - _STRIP_HALF_HEIGHT_UM <= cut_y_um <= y1 + _STRIP_HALF_HEIGHT_UM


def _cutline_intervals(polys: List["gdstk.Polygon"], cut_y_um: float, x0: float, x1: float):
    """[(x0, x1), ...] where `polys` cross the horizontal line y=cut_y_um."""
    if not polys:
        return []
    strip = gdstk.rectangle((x0 - 1.0, cut_y_um - _STRIP_HALF_HEIGHT_UM),
                             (x1 + 1.0, cut_y_um + _STRIP_HALF_HEIGHT_UM))
    hits = gdstk.boolean(polys, [strip], "and")
    out = []
    for p in hits:
        (px0, _), (px1, _) = p.bounding_box()
        if px1 - px0 > _EPS:
            out.append((px0, px1))
    return out


def _synthetic_grid_intervals(x0: float, x1: float, pitch_um: float, width_um: float):
    """[(a, b), ...] — a feature of `width_um` every `pitch_um` across
    [x0, x1], for bands with no real GDS layer to read from (backside power
    delivery: SKY130 has no BSPDN process, so nano-TSVs/backside rails don't
    exist in any layer — see gds/techmap.py's SYNTHETIC sentinel)."""
    out = []
    x = x0
    while x < x1:
        a, b = x, min(x + width_um, x1)
        if b - a > _EPS:
            out.append((a, b))
        x += pitch_um
    return out


def emit_gds_2d_geometry(by_layer, channel_sources, contact_sources, cut_y_um: float, stack=None):
    """Returns (registry, label_to_surfaces, layer_to_surfaces, x_bounds, z_bounds_tuple).
    `stack` selects a gds.techmap.StackProfile (default: frontside) — the
    BSPDN study passes techmap.BSPDN_STACK.
    """
    tile_x0, tile_x1, _, _ = die_bounds(by_layer)

    registry = RegionRegistry()
    input_dimtags: List[Tuple[int, int]] = []
    input_labels: List[str] = []
    input_layers: List[str] = []

    for z0, z1, band in z_bounds(stack):
        segments = [(tile_x0, tile_x1, (band.background, None, band.background))]

        # Drawn-layer overrides, in the band's own order (later wins on
        # overlap) — a band can carry several layers at different materials,
        # e.g. poly draws both PolySi and the licon plugs running past it.
        for key, material in band.drawn:
            if key is SYNTHETIC:
                pitch_um, width_um = band.synthetic_grid
                intervals = _synthetic_grid_intervals(tile_x0, tile_x1, pitch_um, width_um)
            else:
                intervals = _cutline_intervals(by_layer.get(key, []), cut_y_um, tile_x0, tile_x1)
            for x0, x1 in intervals:
                segments = insert_interval(segments, x0, x1, (material, None, material))

        # Punch the S/D-contact subset of licon back in as tagged heat
        # sources — same material as the plug around them, higher priority.
        if band.name == "licon1":
            for box, poly in contact_sources:
                if not _crosses_cut(poly, cut_y_um):
                    continue
                hit = _cutline_intervals([poly], cut_y_um, tile_x0, tile_x1)
                if not hit:
                    continue
                x0, x1 = hit[0]
                label = f"src_{box.device}"
                dz0_box = dataclasses.replace(box, z0_um=z0)
                segments = insert_interval(segments, x0, x1, ("W", dz0_box, label))

        # channel: the inversion layer at the top of the silicon, where
        # hot-carrier dissipation physically happens — silicon, not the gate
        # oxide above it (which SKY130 makes ~4.15nm of SiO2, too thin to mesh
        # and thermally near-transparent; see gds/techmap.py).
        if band.name == "channel":
            for box, poly in channel_sources:
                if not _crosses_cut(poly, cut_y_um):
                    continue
                hit = _cutline_intervals([poly], cut_y_um, tile_x0, tile_x1)
                if not hit:
                    continue
                x0, x1 = hit[0]
                label = f"src_{box.device}"
                dz0_box = dataclasses.replace(box, z0_um=z0)
                segments = insert_interval(segments, x0, x1, ("Si_channel", dz0_box, label))

        for x0, x1, (material, source, label) in segments:
            if x1 - x0 < _EPS:
                continue
            registry.get(label, material=material, source=source)
            tag = gmsh.model.occ.addRectangle(x0, z0, 0.0, x1 - x0, z1 - z0)
            input_dimtags.append((2, tag))
            input_labels.append(label)
            input_layers.append(band.name)

    gmsh.model.occ.synchronize()
    out_dimtags, out_map = gmsh.model.occ.fragment(input_dimtags, [])
    gmsh.model.occ.synchronize()

    label_to_surfaces: Dict[str, List[int]] = {}
    layer_to_surfaces: Dict[str, List[int]] = {}
    for label, layer_name, mapped in zip(input_labels, input_layers, out_map):
        surfaces = [t for (d, t) in mapped if d == 2]
        label_to_surfaces.setdefault(label, []).extend(surfaces)
        layer_to_surfaces.setdefault(layer_name, []).extend(surfaces)

    z_max = z_bounds(stack)[-1][1]
    return registry, label_to_surfaces, layer_to_surfaces, (tile_x0, tile_x1), (0.0, z_max)
