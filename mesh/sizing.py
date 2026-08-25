"""Mesh-size grading: one gmsh Box field per z-layer (VIn = the layer's
mesh_size_um, sharp in z, full width in x), combined with Min. This directly
follows PLAN.md's "Box + Distance/Threshold size fields for grading" — a
per-layer Box is enough here since every feature of interest is already
resolved as its own thin z-band; devices don't need extra per-column fields
because the whole band they sit in is already sized for them.

Assumes the 2D geometry lives in gmsh's (x, y) plane with physical z mapped
onto gmsh's y (see mesh/section.py).
"""

import gmsh

from spec.chip import ChipSpec
from spec.stack import LayerStack

COARSE_UM = 5.0  # VOut: size far from any layer (never actually reached inside the domain)

# 3D-only: how far from a device/via centre the fine size persists, and how
# far out the ramp to background size finishes. Kept well under half the x
# pitch (2um) and y pitch (3um) so neighbouring devices' refinement zones
# don't merge into one contiguous fine region across the whole array.
FEATURE_DIST_MIN_UM = 0.5
FEATURE_DIST_MAX_UM = 0.9
BACKGROUND_UM = 1.0  # lateral size once far from any feature, for 3D


def apply_2d_sizing(stack: LayerStack, x0: float, x1: float) -> None:
    field_ids = []
    for z0, z1, layer in stack.z_bounds():
        f = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(f, "VIn", layer.mesh_size_um)
        gmsh.model.mesh.field.setNumber(f, "VOut", COARSE_UM)
        gmsh.model.mesh.field.setNumber(f, "XMin", x0)
        gmsh.model.mesh.field.setNumber(f, "XMax", x1)
        gmsh.model.mesh.field.setNumber(f, "YMin", z0)
        gmsh.model.mesh.field.setNumber(f, "YMax", z1)
        gmsh.model.mesh.field.setNumber(f, "ZMin", -1.0)
        gmsh.model.mesh.field.setNumber(f, "ZMax", 1.0)
        # smooth transition into the next layer's target size, capped at
        # a fraction of this layer's own thickness so thin layers don't
        # bleed their fine size far into neighbours
        gmsh.model.mesh.field.setNumber(f, "Thickness", max(layer.thickness_um * 0.5, 1e-3))
        field_ids.append(f)

    combined = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(combined, "FieldsList", field_ids)
    gmsh.model.mesh.field.setAsBackgroundMesh(combined)

    # size is entirely field-driven, not extended from point/curve/boundary defaults
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)


def _feature_points(layer, chip: ChipSpec, z_mid: float):
    """gmsh points at every device/via/via-farm centre this layer actually
    has a small feature at, for a Distance field to refine around. Floating
    (unembedded) OCC points are enough — Distance only needs their location,
    not model topology.
    """
    points = []
    if layer.device_fill is not None:
        for d in chip.layout.devices:
            points.append(gmsh.model.occ.addPoint(d.x_um, d.y_um, z_mid))
    if layer.via_fill is not None:
        for d in chip.layout.devices:
            if d.has_via_stack:
                points.append(gmsh.model.occ.addPoint(d.x_um, d.y_um, z_mid))
        fx0, fy0 = chip.layout.via_farm_origin_um
        fs = chip.layout.via_farm_size_um
        points.append(gmsh.model.occ.addPoint(fx0 + fs / 2, fy0 + fs / 2, z_mid))
    return points


def apply_3d_sizing(chip: ChipSpec) -> None:
    """Per z-layer: a Box field caps sizing to that z-band (as in 2D), and —
    only where the layer actually has small lateral features (devices/vias/
    the via farm) — a Distance+Threshold pair refines to the layer's target
    size near each feature centre, ramping out to BACKGROUND_UM elsewhere.
    Reusing the full per-layer fine size everywhere laterally (as the 2D
    path does across its single row) is intractable in 3D across a 24x24um
    tile with 30 devices; this keeps element count near the actual features
    instead of the whole layer.
    """
    stack = chip.stack
    x0, x1 = 0.0, chip.layout.tile_um
    y0, y1 = 0.0, chip.layout.tile_um

    field_ids = []
    for z0, z1, layer in stack.z_bounds():
        has_features = layer.device_fill is not None or layer.via_fill is not None
        band_size = min(layer.mesh_size_um * 4, BACKGROUND_UM) if has_features else layer.mesh_size_um

        box = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(box, "VIn", band_size)
        gmsh.model.mesh.field.setNumber(box, "VOut", COARSE_UM)
        gmsh.model.mesh.field.setNumber(box, "XMin", x0)
        gmsh.model.mesh.field.setNumber(box, "XMax", x1)
        gmsh.model.mesh.field.setNumber(box, "YMin", y0)
        gmsh.model.mesh.field.setNumber(box, "YMax", y1)
        gmsh.model.mesh.field.setNumber(box, "ZMin", z0)
        gmsh.model.mesh.field.setNumber(box, "ZMax", z1)
        gmsh.model.mesh.field.setNumber(box, "Thickness", max(layer.thickness_um * 0.5, 1e-3))
        field_ids.append(box)

        points = _feature_points(layer, chip, (z0 + z1) / 2)
        if points:
            # OCC and the general model keep separate tag namespaces until
            # synchronized — Distance's PointsList looks entities up by tag
            # in the general model, so without this it silently can't find
            # the points just created ("Unknown point N" warnings, and the
            # refinement never actually applies).
            gmsh.model.occ.synchronize()
            dist = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(dist, "PointsList", points)

            thresh = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(thresh, "InField", dist)
            gmsh.model.mesh.field.setNumber(thresh, "SizeMin", layer.mesh_size_um)
            gmsh.model.mesh.field.setNumber(thresh, "SizeMax", band_size)
            gmsh.model.mesh.field.setNumber(thresh, "DistMin", FEATURE_DIST_MIN_UM)
            gmsh.model.mesh.field.setNumber(thresh, "DistMax", FEATURE_DIST_MAX_UM)

            # Distance is full 3D Euclidean, so on its own the Threshold's
            # fine size would bleed vertically into neighbouring z-layers
            # (radius up to FEATURE_DIST_MAX_UM=0.9um, far more than many
            # layers' own thickness of 10s-100s of nm) — and because
            # everything below is combined with Min, that leaked fine size
            # would win over a neighbouring layer's own, coarser Box field.
            # Gate it: a z-only Box mask that's 0 inside this layer's own
            # z-band and huge outside, Max'd with the threshold, so this
            # layer's refinement only ever applies within its own z-band.
            gate = gmsh.model.mesh.field.add("Box")
            gmsh.model.mesh.field.setNumber(gate, "VIn", 0.0)
            gmsh.model.mesh.field.setNumber(gate, "VOut", 1e6)
            gmsh.model.mesh.field.setNumber(gate, "XMin", x0)
            gmsh.model.mesh.field.setNumber(gate, "XMax", x1)
            gmsh.model.mesh.field.setNumber(gate, "YMin", y0)
            gmsh.model.mesh.field.setNumber(gate, "YMax", y1)
            gmsh.model.mesh.field.setNumber(gate, "ZMin", z0)
            gmsh.model.mesh.field.setNumber(gate, "ZMax", z1)

            gated = gmsh.model.mesh.field.add("Max")
            gmsh.model.mesh.field.setNumbers(gated, "FieldsList", [thresh, gate])
            field_ids.append(gated)

    combined = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(combined, "FieldsList", field_ids)
    gmsh.model.mesh.field.setAsBackgroundMesh(combined)

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
