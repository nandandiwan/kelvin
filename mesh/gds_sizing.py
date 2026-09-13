"""Mesh-size grading for the GDS 2D cross-section: one gmsh Box field per
z-band (VIn = the band's mesh_size_um, sharp in z, full width in x),
combined with Min. Same pattern as mesh/sizing.py's apply_2d_sizing, reading
from gds.techmap.z_bounds() instead of a spec.stack.LayerStack.
"""

import gmsh

from gds.techmap import z_bounds

COARSE_UM = 5.0


def apply_gds_2d_sizing(x0: float, x1: float, refine: float = 1.0, stack=None) -> None:
    """`refine` scales every band's target element size (and the coarse
    background) uniformly: 0.5 halves every element edge, so ~4x the cells in
    2D. Used for h-refinement convergence checks — see
    cases/run_gds_convergence.py. `stack` selects a gds.techmap.StackProfile
    (default: the frontside stack) — the BSPDN study passes techmap.BSPDN_STACK.
    """
    field_ids = []
    for z0, z1, band in z_bounds(stack):
        f = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(f, "VIn", band.mesh_size_um * refine)
        gmsh.model.mesh.field.setNumber(f, "VOut", COARSE_UM * refine)
        gmsh.model.mesh.field.setNumber(f, "XMin", x0)
        gmsh.model.mesh.field.setNumber(f, "XMax", x1)
        gmsh.model.mesh.field.setNumber(f, "YMin", z0)
        gmsh.model.mesh.field.setNumber(f, "YMax", z1)
        gmsh.model.mesh.field.setNumber(f, "ZMin", -1.0)
        gmsh.model.mesh.field.setNumber(f, "ZMax", 1.0)
        gmsh.model.mesh.field.setNumber(f, "Thickness", max(band.thickness_um * 0.5, 1e-3))
        field_ids.append(f)

    combined = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(combined, "FieldsList", field_ids)
    gmsh.model.mesh.field.setAsBackgroundMesh(combined)

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)


# 3D only: real GDS features (contacts, vias, gate fingers) are sparse
# within a window, not filling it — unlike the 2D cutline path, uniform-fine
# across the whole lateral window is NOT tractable even for a small window
# (the mesh_size_um values below are nm-scale, tuned for through-thickness
# resolution, not lateral extent; applied uniformly over even a 6x6um window
# across ~15 z-bands this would run to millions of cells). Same pattern as
# mesh/sizing.py's apply_3d_sizing for the synthetic chip: coarse background
# per band, local Distance+Threshold refinement only at real feature
# centroids (mesh/gds_volume.py already collects these while emitting
# geometry — no second extraction pass).
COARSE_3D_UM = 1.0
FEATURE_DIST_MIN_UM = 0.10
FEATURE_DIST_MAX_UM = 0.20


def apply_gds_3d_sizing(x0: float, x1: float, y0: float, y1: float,
                         feature_points, refine: float = 1.0, stack=None) -> None:
    """`feature_points`: {band_name: [(x_um, y_um), ...]} from
    mesh.gds_volume.emit_gds_3d_geometry's return value.
    """
    field_ids = []
    for z0, z1, band in z_bounds(stack):
        feature_size = band.mesh_size_um * refine
        points_xy = feature_points.get(band.name, [])
        band_size = min(feature_size * 4, COARSE_3D_UM) if points_xy else feature_size

        box = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(box, "VIn", band_size)
        gmsh.model.mesh.field.setNumber(box, "VOut", COARSE_UM * refine)
        gmsh.model.mesh.field.setNumber(box, "XMin", x0)
        gmsh.model.mesh.field.setNumber(box, "XMax", x1)
        gmsh.model.mesh.field.setNumber(box, "YMin", y0)
        gmsh.model.mesh.field.setNumber(box, "YMax", y1)
        gmsh.model.mesh.field.setNumber(box, "ZMin", z0)
        gmsh.model.mesh.field.setNumber(box, "ZMax", z1)
        gmsh.model.mesh.field.setNumber(box, "Thickness", max(band.thickness_um * 0.5, 1e-3))
        field_ids.append(box)

        if points_xy:
            z_mid = (z0 + z1) / 2
            pts = [gmsh.model.occ.addPoint(px, py, z_mid) for px, py in points_xy]
            # OCC/general-model tag namespaces are separate until synchronized
            # — Distance's PointsList looks entities up in the general model
            # (see mesh/sizing.py's identical note; same bug, same fix).
            gmsh.model.occ.synchronize()
            dist = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(dist, "PointsList", pts)

            thresh = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(thresh, "InField", dist)
            gmsh.model.mesh.field.setNumber(thresh, "SizeMin", feature_size)
            gmsh.model.mesh.field.setNumber(thresh, "SizeMax", band_size)
            gmsh.model.mesh.field.setNumber(thresh, "DistMin", FEATURE_DIST_MIN_UM)
            gmsh.model.mesh.field.setNumber(thresh, "DistMax", FEATURE_DIST_MAX_UM)

            # Distance is full 3D Euclidean — gate it to this band's own
            # z-range so the fine size can't bleed into a neighbouring,
            # coarser-target band (same z-bleed bug/fix as mesh/sizing.py).
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
