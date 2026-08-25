"""GATE 0 visual verification for the 3D path, via pyvista (off-screen).
Builds an UnstructuredGrid directly from the raw tet data mesh/build.py
extracts (before dolfinx touches it), colors by material, and renders slices
— a full 3D render of ~millions of tets isn't legible anyway; slices are
what actually let you check the geometry by eye, same as the 2D GATE 0.
"""

import numpy as np
import pyvista as pv

from mesh.viz import MATERIAL_COLORS

pv.OFF_SCREEN = True

_TET_VTK_TYPE = 10  # vtk.VTK_TETRA


def _grid(xyz_m, tet_nodes, region_idx, region_list):
    n = tet_nodes.shape[0]
    cells = np.hstack([np.full((n, 1), 4, dtype=np.int64), tet_nodes]).ravel()
    cell_types = np.full(n, _TET_VTK_TYPE, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, cell_types, xyz_m)

    materials = [r.material for r in region_list]
    palette = sorted(set(materials))
    color_idx = {m: i for i, m in enumerate(palette)}
    grid.cell_data["material_idx"] = np.array([color_idx[materials[i]] for i in region_idx], dtype=np.int32)
    return grid, palette


def _material_cmap(palette):
    return [MATERIAL_COLORS.get(m, "#999999") for m in palette]


def render_xy_slice(xyz_um, tet_nodes, region_idx, region_list, z_um, title, out_path):
    grid, palette = _grid(xyz_um, tet_nodes, region_idx, region_list)
    sliced = grid.slice(normal="z", origin=(0, 0, z_um))

    pl = pv.Plotter(off_screen=True, window_size=(1400, 1400))
    pl.add_mesh(sliced, scalars="material_idx", cmap=_material_cmap(palette),
                show_scalar_bar=False, show_edges=True, line_width=0.3)
    pl.add_text(title, font_size=10)
    pl.view_xy()
    pl.camera.parallel_projection = True
    pl.reset_camera()
    pl.screenshot(out_path)
    pl.close()


def render_xz_slice(xyz_um, tet_nodes, region_idx, region_list, y_um, title, out_path):
    grid, palette = _grid(xyz_um, tet_nodes, region_idx, region_list)
    sliced = grid.slice(normal="y", origin=(0, y_um, 0))

    pl = pv.Plotter(off_screen=True, window_size=(1800, 900))
    pl.add_mesh(sliced, scalars="material_idx", cmap=_material_cmap(palette),
                show_scalar_bar=False, show_edges=True, line_width=0.3)
    pl.add_text(title, font_size=10)
    pl.view_xz()
    pl.camera.parallel_projection = True
    pl.reset_camera()
    pl.screenshot(out_path)
    pl.close()


def render_3d_gate0(xyz_um, tet_nodes, region_idx, region_list, chip, out_dir):
    render_xz_slice(xyz_um, tet_nodes, region_idx, region_list, y_um=4.5,
                     title="3D mesh, XZ slice at y=4.5um (row 0) — cross-check vs 2D GATE 0",
                     out_path=f"{out_dir}/slice_xz_row0.png")

    render_xy_slice(xyz_um, tet_nodes, region_idx, region_list, z_um=50.12,
                     title="3D mesh, XY slice through silicide (z=50.12um) — full device array",
                     out_path=f"{out_dir}/slice_xy_devices.png")

    render_xy_slice(xyz_um, tet_nodes, region_idx, region_list, z_um=50.7,
                     title="3D mesh, XY slice through M2 (z=50.7um) — via pattern + via farm",
                     out_path=f"{out_dir}/slice_xy_vias.png")
