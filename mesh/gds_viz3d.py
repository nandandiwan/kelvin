"""GATE 0 visual verification for the GDS 3D path — thin wrapper around
mesh/viz3d.py's already dimension/spec-generic slice renderers (they take
raw xyz/tet_nodes/region_idx/region_list + a slice location, nothing
ChipSpec-specific), picking slice locations from the real stack instead of
the synthetic chip's hardcoded z-values.
"""

from gds import techmap
from .viz3d import render_xy_slice, render_xz_slice


def render_gds_3d_gate0(xyz_um, tet_nodes, region_idx, region_list, stack, window, out_dir):
    x0, x1, y0, y1 = window
    y_mid = (y0 + y1) / 2

    render_xz_slice(xyz_um, tet_nodes, region_idx, region_list, y_um=y_mid,
                     title=f"GDS 3D mesh, XZ slice at y={y_mid:.2f}um — cross-check vs 2D GATE 0",
                     out_path=f"{out_dir}/slice_xz.png")

    for name in ("channel", "licon1", "met1"):
        try:
            band = techmap.find(name, stack)
        except KeyError:
            continue
        z_mid = None
        for zb0, zb1, b in techmap.z_bounds(stack):
            if b is band:
                z_mid = (zb0 + zb1) / 2
                break
        if z_mid is None:
            continue
        render_xy_slice(xyz_um, tet_nodes, region_idx, region_list, z_um=z_mid,
                         title=f"GDS 3D mesh, XY slice through {name} (z={z_mid:.3f}um)",
                         out_path=f"{out_dir}/slice_xy_{name}.png")
