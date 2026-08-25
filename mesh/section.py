"""2D cross-section emitter: ChipSpec -> gmsh OCC geometry in the (x, z)
plane, cut through one row of devices. Physical z is mapped onto gmsh's y
axis (gmsh 2D meshing needs geometry in a plane; z is left unused/near-zero).

Assumes gmsh.initialize() and gmsh.model.add(...) have already been called;
this module only adds OCC entities to the current model and returns the
bookkeeping mesh/build.py needs to turn them into physical groups.
"""

from typing import Dict, List, Tuple

import gmsh

from spec.chip import ChipSpec, SourceBox
from spec.layout import DEVICE_W_UM, VIA_WIDTH_UM
from spec.stack import Layer, LayerStack

from .boxes import RegionRegistry, insert_interval

_EPS = 1e-9


def _devices_in_row(chip: ChipSpec, row: int):
    return sorted((d for d in chip.layout.devices if d.row == row), key=lambda d: d.x_um)


def _sources_by_device(chip: ChipSpec) -> Dict[str, Dict[str, SourceBox]]:
    out: Dict[str, Dict[str, SourceBox]] = {}
    for s in chip.sources:
        out.setdefault(s.device, {})[s.kind] = s
    return out


def _layer_segments(layer: Layer, devices, sources_by_device, tile_x0: float, tile_x1: float):
    """[(x0, x1, (material, source_or_None, label)), ...] tiling [tile_x0, tile_x1]."""
    segments = [(tile_x0, tile_x1, (layer.material, None, layer.material))]

    if layer.device_fill is not None:
        for d in devices:
            x0, x1 = d.x_um - DEVICE_W_UM / 2, d.x_um + DEVICE_W_UM / 2
            payload = (layer.device_fill, None, layer.device_fill)
            segments = insert_interval(segments, x0, x1, payload)

    if layer.via_fill is not None:
        for d in devices:
            if not d.has_via_stack:
                continue
            x0, x1 = d.x_um - VIA_WIDTH_UM / 2, d.x_um + VIA_WIDTH_UM / 2
            payload = (layer.via_fill, None, layer.via_fill)
            segments = insert_interval(segments, x0, x1, payload)

    # Heat-source sub-boxes: same material as the device_fill they sit in,
    # but their own label/tag so post-processing can see per-device power.
    if layer.name == "STI_channel":
        for d in devices:
            src = sources_by_device[d.name]["channel"]
            x0, x1 = src.x_um - src.w_um / 2, src.x_um + src.w_um / 2
            label = f"src_channel_{d.name}"
            segments = insert_interval(segments, x0, x1, (layer.device_fill, src, label))

    if layer.name == "silicide":
        for d in devices:
            src = sources_by_device[d.name]["contact"]
            x0, x1 = src.x_um - src.w_um / 2, src.x_um + src.w_um / 2
            label = f"src_contact_{d.name}"
            segments = insert_interval(segments, x0, x1, (layer.device_fill, src, label))

    return segments


def emit_2d_geometry(chip: ChipSpec, row: int = 0):
    """Build the gmsh OCC geometry for the row-`row` cross-section.

    Returns (registry, label_to_surfaces, x_bounds, z_bounds).
    """
    stack = chip.stack
    tile_x0, tile_x1 = 0.0, chip.layout.tile_um
    devices = _devices_in_row(chip, row)
    if not devices:
        raise ValueError(f"no devices in row {row}")
    sources_by_device = _sources_by_device(chip)

    registry = RegionRegistry()
    input_dimtags: List[Tuple[int, int]] = []
    input_labels: List[str] = []
    input_layers: List[str] = []

    for z0, z1, layer in stack.z_bounds():
        segments = _layer_segments(layer, devices, sources_by_device, tile_x0, tile_x1)
        for x0, x1, (material, source, label) in segments:
            if x1 - x0 < _EPS:
                continue
            registry.get(label, material=material, source=source)
            tag = gmsh.model.occ.addRectangle(x0, z0, 0.0, x1 - x0, z1 - z0)
            input_dimtags.append((2, tag))
            input_labels.append(label)
            input_layers.append(layer.name)

    gmsh.model.occ.synchronize()
    out_dimtags, out_map = gmsh.model.occ.fragment(input_dimtags, [])
    gmsh.model.occ.synchronize()

    # Every input rectangle was built for a known (label, layer) — track both
    # mappings off the same fragment map so downstream code never has to
    # re-derive "which layer is this cell in" from geometry/tolerances.
    label_to_surfaces: Dict[str, List[int]] = {}
    layer_to_surfaces: Dict[str, List[int]] = {}
    for label, layer_name, mapped in zip(input_labels, input_layers, out_map):
        surfaces = [t for (d, t) in mapped if d == 2]
        label_to_surfaces.setdefault(label, []).extend(surfaces)
        layer_to_surfaces.setdefault(layer_name, []).extend(surfaces)

    z_max = stack.total_thickness_um
    return registry, label_to_surfaces, layer_to_surfaces, (tile_x0, tile_x1), (0.0, z_max)
