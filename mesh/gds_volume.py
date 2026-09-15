"""3D geometry emitter for a GDS-derived chip: extrudes real device polygons
within a bounded lateral (x0,x1,y0,y1) window — not the whole die, which is
intractable at real polygon density (see cases/run_gds_3d.py's docstring) —
into full 3D boxes, one per z-band per polygon. Mirrors mesh/volume.py's
pattern for the synthetic chip (mesh/boxes.insert_rect, removeAllDuplicates
not fragment — same hard-won lesson, see that module's comment) but sources
geometry from real GDS polygons via a 2D window intersection, the lateral
analogue of mesh/gds_section.py's 1D cutline strip.

Each real GDS polygon is approximated by its own bounding box (documented
simplification — most shapes in this layout are already simple rectangles,
verified in SRAM_THERMAL_REPORT.md's geometry-accuracy check; a non-rectangular
shape's exact footprint is not preserved, only its lateral extent).

Assumes gmsh.initialize() and gmsh.model.add(...) have already been called;
this module only adds OCC entities to the current model.
"""

import math
import time
from typing import Dict, List, Tuple

import gdstk
import gmsh

from gds.techmap import SYNTHETIC, z_bounds

from .boxes import RegionRegistry, insert_rect

_EPS = 1e-9


def _window_rects(polys: List["gdstk.Polygon"], window: "gdstk.Polygon",
                   x0: float, y0: float, x1: float, y1: float):
    """[(rx0, ry0, rx1, ry1), ...] bounding boxes of `polys` clipped to the
    window — the 2D-lateral analogue of mesh/gds_section._cutline_intervals.
    """
    if not polys:
        return []
    hits = gdstk.boolean(polys, [window], "and")
    out = []
    for p in hits:
        (px0, py0), (px1, py1) = p.bounding_box()
        rx0, ry0 = max(px0, x0), max(py0, y0)
        rx1, ry1 = min(px1, x1), min(py1, y1)
        if rx1 - rx0 > _EPS and ry1 - ry0 > _EPS:
            out.append((rx0, ry0, rx1, ry1))
    return out


def _grid_starts(lo: float, hi: float, pitch: float, width: float, anchor: float = None):
    """Feature start coordinates tiling [lo, hi] at `pitch`. With `anchor`,
    one feature is CENTRED on it and the rest step outward by pitch; without,
    tiling simply begins at `lo`.

    The anchor matters physically, not cosmetically: a via grid phased from
    the window edge puts a via under the heat source only by luck, and the
    whole backside-metal mechanism is about getting heat from the source into
    a nearby via. [Oprins] Fig. 6 defines their small heat source AS "the
    area contacted by a single uTSV to BPR", i.e. exactly one via under the
    heater -- so the alignment is part of their test case, not a free
    parameter.
    """
    if anchor is None:
        starts, x = [], lo
        while x < hi:
            starts.append(x)
            x += pitch
        return starts
    n_lo = int(math.floor((lo - (anchor - width / 2)) / pitch)) - 1
    n_hi = int(math.ceil((hi - (anchor - width / 2)) / pitch)) + 1
    return [anchor - width / 2 + n * pitch for n in range(n_lo, n_hi + 1)]


def _synthetic_grid_rects(x0: float, y0: float, x1: float, y1: float,
                           pitch_um: float, width_um: float, shape: str = "lines",
                           width_y_um: float = None, anchor_um=None,
                           pitch_y_um: float = None, stagger_y_um: float = 0.0):
    """2D analogue of mesh/gds_section._synthetic_grid_intervals, for backside
    power delivery geometry that exists in no real GDS layer to read.

    `shape` ("lines" | "posts") is load-bearing physics, not styling — see
    gds.techmap.GdsLayerBand.synthetic_shape:
      - "lines": continuous strips spanning y, repeating every pitch_um in x.
        A backside metal rail conducts along its length; drawing it as
        disconnected islands removes that lateral path entirely.
      - "posts": width x width islands on a pitch grid in BOTH x and y. A
        via/nTSV is a vertical connector and is genuinely not continuous.
    The 2D cross-section path has no equivalent choice: its x-intervals
    extrude along the unresolved axis, so they are always effectively lines.

    `stagger_y_um` (posts only): shift every OTHER column by this much in y.
    [Oprins] Fig. 5 shows a PWR backside track dropping uTSVs only onto PWR
    buried rails and a GND track only onto GND rails, so vias on neighbouring
    tracks sit on DIFFERENT rails -- a checkerboard, not a square grid. Since
    the rails alternate every buried-rail pitch, the natural offset is one
    rail pitch. This changes how much via area sits under a given heat
    source, so it is geometry, not decoration.

    `width_y_um` (posts only): a rectangular post, width_um x width_y_um,
    for geometry that is genuinely not square -- [Oprins]' uTSV is 180 x 250
    nm, whose 18% area fraction differs materially from the 25% a 250nm
    square would give.
    """
    out = []
    wy = width_um if width_y_um is None else width_y_um
    py = pitch_um if pitch_y_um is None else pitch_y_um
    ax = ay = None
    if anchor_um is not None:
        ax, ay = anchor_um
    if shape == "lines_perp":
        # rails running along x, repeating in y -- the transpose of "lines"
        for y in _grid_starts(y0, y1, pitch_um, width_um, ay):
            ry0, ry1 = max(y, y0), min(y + width_um, y1)
            if ry1 - ry0 > _EPS and x1 - x0 > _EPS:
                out.append((x0, ry0, x1, ry1))
        return out
    x_starts = _grid_starts(x0, x1, pitch_um, width_um, ax)
    for x in x_starts:
        rx0, rx1 = max(x, x0), min(x + width_um, x1)
        if rx1 - rx0 <= _EPS:
            continue
        if shape == "lines":
            if y1 - y0 > _EPS:
                out.append((rx0, y0, rx1, y1))
            continue
        # Column parity is taken from the ANCHORED index, not this window's
        # first column, so the checkerboard phase is a property of the
        # structure rather than of where the window happens to be cut.
        if ax is None:
            col = int(round((x - x_starts[0]) / pitch_um))
        else:
            col = int(round((x - (ax - width_um / 2)) / pitch_um))
        ay_col = None if ay is None else ay + (col % 2) * stagger_y_um
        if True:
            for y in _grid_starts(y0, y1, py, wy, ay_col):
                ry0, ry1 = max(y, y0), min(y + wy, y1)
                if ry1 - ry0 > _EPS:
                    out.append((rx0, ry0, rx1, ry1))
    return out


def _window_contains(poly: "gdstk.Polygon", x0: float, y0: float, x1: float, y1: float) -> bool:
    (px0, py0), (px1, py1) = poly.bounding_box()
    return not (px1 < x0 or px0 > x1 or py1 < y0 or py0 > y1)


def emit_gds_3d_geometry(by_layer, channel_sources, contact_sources,
                          window: Tuple[float, float, float, float], stack,
                          grid_anchor_um=None):
    """`window` = (x0, x1, y0, y1) in um, the bounded lateral region to
    extrude (see cases/run_gds_3d.py for how it's sized). `stack` is a
    required gds.techmap.StackProfile (frontside or BSPDN).

    Returns (registry, label_to_volumes, layer_to_volumes, x_bounds,
    y_bounds, z_bounds, feature_points) — feature_points is
    {band_name: [(x_um, y_um), ...]} for mesh/gds_sizing.py's 3D refinement,
    the centroids of every non-background box actually emitted, so local
    refinement doesn't need a second, redundant polygon extraction pass.
    """
    x0, x1, y0, y1 = window
    window_poly = gdstk.rectangle((x0, y0), (x1, y1))

    registry = RegionRegistry()
    label_to_volumes: Dict[str, List[int]] = {}
    layer_to_volumes: Dict[str, List[int]] = {}
    feature_points: Dict[str, List[Tuple[float, float]]] = {}

    t0 = time.time()
    for z0, z1, band in z_bounds(stack):
        rects = [(x0, y0, x1, y1, (band.background, None, band.background))]

        for key, material in band.drawn:
            if key is SYNTHETIC:
                pitch_um, width_um = band.synthetic_grid
                grid_rects = _synthetic_grid_rects(x0, y0, x1, y1, pitch_um, width_um,
                                                    band.synthetic_shape,
                                                    band.synthetic_width_y_um,
                                                    grid_anchor_um,
                                                    band.synthetic_pitch_y_um,
                                                    band.synthetic_stagger_y_um)
            else:
                grid_rects = _window_rects(by_layer.get(key, []), window_poly, x0, y0, x1, y1)
            for rx0, ry0, rx1, ry1 in grid_rects:
                rects = insert_rect(rects, rx0, ry0, rx1, ry1, (material, None, material))

        # Contact sources punched into licon1, channel sources into channel —
        # same nesting pattern as mesh/gds_section.py's 2D cutline path.
        if band.name == "licon1":
            for box, poly in contact_sources:
                if not _window_contains(poly, x0, y0, x1, y1):
                    continue
                hits = _window_rects([poly], window_poly, x0, y0, x1, y1)
                if not hits:
                    continue
                rx0, ry0, rx1, ry1 = hits[0]
                label = f"src_{box.device}"
                import dataclasses
                dz0_box = dataclasses.replace(box, z0_um=z0)
                rects = insert_rect(rects, rx0, ry0, rx1, ry1, ("W", dz0_box, label))
        if band.name == "channel":
            for box, poly in channel_sources:
                if not _window_contains(poly, x0, y0, x1, y1):
                    continue
                hits = _window_rects([poly], window_poly, x0, y0, x1, y1)
                if not hits:
                    continue
                rx0, ry0, rx1, ry1 = hits[0]
                label = f"src_{box.device}"
                import dataclasses
                dz0_box = dataclasses.replace(box, z0_um=z0)
                rects = insert_rect(rects, rx0, ry0, rx1, ry1, ("Si_channel", dz0_box, label))

        band_points = []
        for rx0, ry0, rx1, ry1, (material, source, label) in rects:
            if rx1 - rx0 < _EPS or ry1 - ry0 < _EPS:
                continue
            registry.get(label, material=material, source=source)
            tag = gmsh.model.occ.addBox(rx0, ry0, z0, rx1 - rx0, ry1 - ry0, z1 - z0)
            label_to_volumes.setdefault(label, []).append(tag)
            layer_to_volumes.setdefault(band.name, []).append(tag)
            # Skip the full-window background rect (label == background
            # material, spans the whole band) as a refinement point — only
            # real, smaller features need local mesh refinement.
            if not (rx0 <= x0 + _EPS and ry0 <= y0 + _EPS and rx1 >= x1 - _EPS and ry1 >= y1 - _EPS):
                band_points.append(((rx0 + rx1) / 2, (ry0 + ry1) / 2))
        if band_points:
            feature_points[band.name] = band_points

    total_boxes = sum(len(v) for v in layer_to_volumes.values())
    print(f"[gds_volume] {total_boxes} boxes built: {time.time() - t0:.1f}s", flush=True)

    t1 = time.time()
    gmsh.model.occ.synchronize()
    print(f"[gds_volume] synchronize: {time.time() - t1:.1f}s", flush=True)

    # Our boxes never overlap by construction (insert_rect always pre-splits
    # into non-overlapping remainder pieces) — removeAllDuplicates (a
    # coincidence merge), not fragment (general boolean intersection), same
    # lesson as mesh/volume.py: fragment on ~1000+ boxes there never
    # finished (killed after 15+ min, RAM past 7GB).
    t2 = time.time()
    gmsh.model.occ.removeAllDuplicates()
    print(f"[gds_volume] removeAllDuplicates: {time.time() - t2:.1f}s", flush=True)
    gmsh.model.occ.synchronize()

    z_max = z_bounds(stack)[-1][1]
    return (registry, label_to_volumes, layer_to_volumes,
            (x0, x1), (y0, y1), (0.0, z_max), feature_points)
