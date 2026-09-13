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
                show_scalar_bar=False, show_edges=False)
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


def _temperature_grid(T):
    """UnstructuredGrid built from the P1 FunctionSpace's own dof
    coordinates + dofmap, not mesh.geometry — self-consistent with
    T.x.array by construction (same reasoning as post/viz.py's 2D
    triangulation; dolfinx's dof order isn't guaranteed to match the raw
    geometry node order)."""
    V = T.function_space
    coords_um = V.tabulate_dof_coordinates() * 1e6
    cells_conn = V.dofmap.list
    n = cells_conn.shape[0]
    cells = np.hstack([np.full((n, 1), 4, dtype=np.int64), cells_conn]).ravel()
    cell_types = np.full(n, _TET_VTK_TYPE, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, cell_types, coords_um)
    grid.point_data["dT"] = T.x.array - 300.0
    return grid


def _scalar_bar_args(grid, scalars="dT"):
    """Explicit tick-label precision: pyvista's default format duplicates
    ticks (e.g. "31.4  31.4  31.4  31.5  31.5") when the data range is
    narrow relative to its absolute value, as it is for a sub-1K-wide dT
    range sitting at ~300+K -- verified directly on this project's own
    bitcell renders."""
    rng = grid.get_data_range(scalars)
    tick_spacing = max((rng[1] - rng[0]) / 4, 1e-12)  # n_labels=5 -> 4 intervals
    # enough decimals that adjacent tick labels are visibly distinct even
    # when the range is a tiny fraction of the absolute value (a sub-1K dT
    # sitting at ~300+K -- verified directly: 2 decimals still rendered
    # visually-identical repeated labels on this project's own bitcell render)
    decimals = max(2, min(8, int(np.ceil(-np.log10(tick_spacing))) + 1))
    return {"title": "dT (K)", "fmt": f"%.{decimals}f", "n_labels": 5}


def render_temperature_xy_slice(T, z_um, title, out_path, clim=None):
    grid = _temperature_grid(T)
    sliced = grid.slice(normal="z", origin=(0, 0, z_um))
    pl = pv.Plotter(off_screen=True, window_size=(1400, 1200))
    pl.add_mesh(sliced, scalars="dT", cmap="inferno", clim=clim,
                show_scalar_bar=True, scalar_bar_args=_scalar_bar_args(grid))
    pl.add_text(title, font_size=10)
    pl.view_xy()
    pl.camera.parallel_projection = True
    pl.reset_camera()
    pl.screenshot(out_path)
    pl.close()


def render_temperature_xz_slice(T, y_um, title, out_path, clim=None, zoom_z_um=None):
    """`zoom_z_um=(z0, z1)` frames the camera on that z-band instead of the
    full stack -- `reset_camera()` fits the WHOLE z-extent by default, which
    for a small lateral window (a few um) against the full ~50+um stack
    squashes the actual device-layer region into an invisible sliver
    (verified directly on this project's own bitcell render)."""
    grid = _temperature_grid(T)
    sliced = grid.slice(normal="y", origin=(0, y_um, 0))
    pl = pv.Plotter(off_screen=True, window_size=(1800, 900))
    pl.add_mesh(sliced, scalars="dT", cmap="inferno", clim=clim,
                show_scalar_bar=True, scalar_bar_args=_scalar_bar_args(grid))
    pl.add_text(title, font_size=10)
    pl.view_xz()
    pl.camera.parallel_projection = True
    if zoom_z_um is not None:
        xb = grid.bounds
        pl.reset_camera(bounds=(xb[0], xb[1], xb[2], xb[3], zoom_z_um[0], zoom_z_um[1]))
    else:
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
