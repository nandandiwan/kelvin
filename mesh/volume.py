"""3D volume emitter: ChipSpec -> gmsh OCC geometry for the full tile (all
30 devices), extruding the same per-layer background/device_fill/via_fill/
source structure mesh/section.py uses for the 2D cut — now genuinely
resolved in both lateral dimensions via mesh/boxes.insert_rect, rather than
one row.

Assumes gmsh.initialize() and gmsh.model.add(...) have already been called;
this module only adds OCC entities to the current model.
"""

from typing import Dict, List

import gmsh

from spec.chip import ChipSpec, SourceBox
from spec.layout import DEVICE_L_UM, DEVICE_W_UM, VIA_WIDTH_UM
from spec.stack import Layer

from .boxes import RegionRegistry, insert_rect

_EPS = 1e-9


def _sources_by_device(chip: ChipSpec) -> Dict[str, Dict[str, SourceBox]]:
    out: Dict[str, Dict[str, SourceBox]] = {}
    for s in chip.sources:
        out.setdefault(s.device, {})[s.kind] = s
    return out


def _layer_rects(layer: Layer, chip: ChipSpec, sources_by_device,
                  tile_x0: float, tile_x1: float, tile_y0: float, tile_y1: float):
    """[(x0, y0, x1, y1, (material, source_or_None, label)), ...] tiling the
    full tile for this layer.
    """
    rects = [(tile_x0, tile_y0, tile_x1, tile_y1, (layer.material, None, layer.material))]

    if layer.device_fill is not None:
        payload = (layer.device_fill, None, layer.device_fill)
        for d in chip.layout.devices:
            x0, x1 = d.x_um - DEVICE_W_UM / 2, d.x_um + DEVICE_W_UM / 2
            y0, y1 = d.y_um - DEVICE_L_UM / 2, d.y_um + DEVICE_L_UM / 2
            rects = insert_rect(rects, x0, y0, x1, y1, payload)

    if layer.via_fill is not None:
        payload = (layer.via_fill, None, layer.via_fill)
        for d in chip.layout.devices:
            if not d.has_via_stack:
                continue
            x0, x1 = d.x_um - VIA_WIDTH_UM / 2, d.x_um + VIA_WIDTH_UM / 2
            y0, y1 = d.y_um - VIA_WIDTH_UM / 2, d.y_um + VIA_WIDTH_UM / 2
            rects = insert_rect(rects, x0, y0, x1, y1, payload)
        # thermal-via / dummy-metal farm: one more solid block of the same material
        fx0, fy0 = chip.layout.via_farm_origin_um
        fs = chip.layout.via_farm_size_um
        rects = insert_rect(rects, fx0, fy0, fx0 + fs, fy0 + fs, payload)

    if layer.name == "STI_channel":
        for d in chip.layout.devices:
            src = sources_by_device[d.name]["channel"]
            x0, x1 = src.x_um - src.w_um / 2, src.x_um + src.w_um / 2
            y0, y1 = src.y_um - src.l_um / 2, src.y_um + src.l_um / 2
            label = f"src_channel_{d.name}"
            rects = insert_rect(rects, x0, y0, x1, y1, (layer.device_fill, src, label))

    if layer.name == "silicide":
        for d in chip.layout.devices:
            src = sources_by_device[d.name]["contact"]
            x0, x1 = src.x_um - src.w_um / 2, src.x_um + src.w_um / 2
            y0, y1 = src.y_um - src.l_um / 2, src.y_um + src.l_um / 2
            label = f"src_contact_{d.name}"
            rects = insert_rect(rects, x0, y0, x1, y1, (layer.device_fill, src, label))

    return rects


def emit_3d_geometry(chip: ChipSpec):
    """Build the gmsh OCC geometry for the full 3D tile.

    Returns (registry, label_to_volumes, layer_to_volumes, x_bounds, y_bounds, z_bounds).
    """
    stack = chip.stack
    tile_x0, tile_x1 = 0.0, chip.layout.tile_um
    tile_y0, tile_y1 = 0.0, chip.layout.tile_um
    sources_by_device = _sources_by_device(chip)

    import time
    t0 = time.time()

    registry = RegionRegistry()
    label_to_volumes: Dict[str, List[int]] = {}
    layer_to_volumes: Dict[str, List[int]] = {}

    for z0, z1, layer in stack.z_bounds():
        rects = _layer_rects(layer, chip, sources_by_device, tile_x0, tile_x1, tile_y0, tile_y1)
        for x0, y0, x1, y1, (material, source, label) in rects:
            if x1 - x0 < _EPS or y1 - y0 < _EPS:
                continue
            registry.get(label, material=material, source=source)
            tag = gmsh.model.occ.addBox(x0, y0, z0, x1 - x0, y1 - y0, z1 - z0)
            label_to_volumes.setdefault(label, []).append(tag)
            layer_to_volumes.setdefault(layer.name, []).append(tag)

    total_boxes = sum(len(v) for v in layer_to_volumes.values())
    print(f"[volume] {total_boxes} boxes built: {time.time() - t0:.1f}s", flush=True)

    t1 = time.time()
    gmsh.model.occ.synchronize()
    print(f"[volume] synchronize: {time.time() - t1:.1f}s", flush=True)

    # Our boxes never overlap in volume by construction (insert_rect always
    # pre-splits into non-overlapping remainder pieces) — they only ever
    # touch. That makes removeAllDuplicates (a coincidence merge) the
    # intended tool, not fragment (general boolean intersection): fragment
    # on ~1000+ boxes here was catastrophic (still running after 15+
    # minutes, RAM climbing past 7GB), and was verified separately to
    # produce genuinely shared nodes at every interface on a small case (0
    # coincident-but-separate node pairs, exact total volume) — but see the
    # timing print right after this call for whether it actually scales
    # here; if not, the next lever is fragmenting per-layer instead.
    t2 = time.time()
    gmsh.model.occ.removeAllDuplicates()
    print(f"[volume] removeAllDuplicates: {time.time() - t2:.1f}s", flush=True)
    gmsh.model.occ.synchronize()

    z_max = stack.total_thickness_um
    return (registry, label_to_volumes, layer_to_volumes,
            (tile_x0, tile_x1), (tile_y0, tile_y1), (0.0, z_max))
