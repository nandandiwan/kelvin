"""Whole-die (or any sub-region) coarse 3D solve, built directly as a
structured dolfinx mesh from gds.upscale.TileGrid — no gmsh, no OCC, no
`removeAllDuplicates`. Cell count is a knob (tile size x per-slab z-cell
count), not a consequence of how many real polygons exist, which is what
makes the whole die tractable at all (cases/run_gds_3d.py's probe
extrapolates full-resolution 3D to ~27 billion tets / ~0.87TB of mesh
connectivity alone — see SRAM_THERMAL_REPORT.md's upscaling section).

Non-uniform z (band thickness spans 10nm to 50um) is handled by building a
uniform box mesh with `sum(nz_per_slab)` z-subdivisions, then warping the
z-coordinate through a monotonic piecewise-linear map from the uniform grid
onto each slab's real z-range — cheap (dolfinx builds nodes at exact
uniform-fraction positions, so remapping by value is correct regardless of
the mesh's internal node ordering).
"""

from typing import Dict, List

import numpy as np
from mpi4py import MPI

from dolfinx.fem import Function, functionspace
from dolfinx.mesh import CellType, compute_midpoints, create_box, locate_entities_boundary, meshtags

from gds.upscale import TileGrid
from .build import FACET_BOTTOM, FACET_TOP

_EPS = 1e-9


def _z_warp(slab_order: List[str], slabs: Dict[str, tuple], nz_per_slab: List[int]):
    """(uniform_breakpoints, real_breakpoints_m) for np.interp — see module
    docstring. `slabs[name] = (z0_um, z1_um, ...)`.
    """
    total_nz = sum(nz_per_slab)
    z0_first = slabs[slab_order[0]][0] * 1e-6
    uniform_bp = [0.0]
    real_bp = [z0_first]
    cursor = 0
    for name, n in zip(slab_order, nz_per_slab):
        z0_um, z1_um = slabs[name][0], slabs[name][1]
        z0_m, z1_m = z0_um * 1e-6, z1_um * 1e-6
        for k in range(1, n + 1):
            cursor += 1
            uniform_bp.append(cursor / total_nz)
            real_bp.append(z0_m + (z1_m - z0_m) * k / n)
    return np.array(uniform_bp), np.array(real_bp)


_MAX_NZ_PER_SLAB = 12  # cap even for heterogeneous slabs — see below


def _default_nz_per_slab(tile_grid: TileGrid, min_aspect_um: float) -> List[int]:
    """One z-cell-thickness near `min_aspect_um` (the lateral tile size —
    near-cubic cells), but ONLY for slabs with real heterogeneity (nonzero
    k variation or any source); a uniform slab (e.g. the 50um bulk-Si
    substrate) has no sharp jump to oscillate around regardless of aspect
    ratio, so it keeps a small fixed count instead of scaling to hundreds
    of z-cells at a fine lateral tile.

    Why this matters at all: a *fixed* nz produces increasingly
    needle-shaped cells as the lateral tile shrinks (z-thickness unchanged,
    footprint shrinking) — and high-aspect-ratio hex cells combined with
    sharp material/source jumps between adjacent cells is a textbook recipe
    for violating the discrete maximum principle: verified directly
    (cases/run_gds_coarse_solve.py --validate at fixed nz=3, shrinking tile
    from 0.5um to 0.05um) — Tmax went from a sane 5.4e-7 K above ambient to
    *below* ambient by -1.9e-5 K, worse as aspect ratio grew, impossible for
    a q>=0 problem with a Robin sink referenced to ambient.
    """
    nz = []
    for name in tile_grid.slab_order:
        z0_um, z1_um, k_grid, _rho_cp, q_grid = tile_grid.slabs[name]
        heterogeneous = (k_grid.max() - k_grid.min() > 1e-6 * k_grid.max()) or (q_grid.max() > 0)
        if not heterogeneous:
            nz.append(2)
            continue
        n = max(1, round((z1_um - z0_um) / min_aspect_um))
        nz.append(min(n, _MAX_NZ_PER_SLAB))
    return nz


def build_gds_coarse_mesh(tile_grid: TileGrid, nz_per_slab=None):
    """Returns (mesh_data-like namespace with .mesh/.cell_tags(None)/.facet_tags,
    registry=None, k, rho_cp, q) — DG0 coefficient Functions built DIRECTLY
    from the tile grid (no RegionRegistry/cell_tags indirection: every coarse
    cell has its own effectively-continuous value, not a small number of
    discrete tagged regions, so there's nothing a region-tag layer would add).
    """
    import types

    slab_order = tile_grid.slab_order
    if nz_per_slab is None:
        tile_um = tile_grid.x_edges[1] - tile_grid.x_edges[0]
        nz_per_slab = _default_nz_per_slab(tile_grid, tile_um)

    x0_m, x1_m = tile_grid.x_edges[0] * 1e-6, tile_grid.x_edges[-1] * 1e-6
    y0_m, y1_m = tile_grid.y_edges[0] * 1e-6, tile_grid.y_edges[-1] * 1e-6
    nz_total = sum(nz_per_slab)

    mesh = create_box(
        MPI.COMM_SELF, [[x0_m, y0_m, 0.0], [x1_m, y1_m, 1.0]],
        [tile_grid.nx, tile_grid.ny, nz_total], cell_type=CellType.hexahedron,
    )

    uniform_bp, real_bp = _z_warp(slab_order, tile_grid.slabs, nz_per_slab)
    mesh.geometry.x[:, 2] = np.interp(mesh.geometry.x[:, 2], uniform_bp, real_bp)

    # Facet tags: bottom (z=0, Robin sink) and top (z=z_max, optional
    # dual-sided Robin) by coordinate — no physical-group bookkeeping needed
    # for a structured mesh with no interior material boundaries to track.
    z_max = real_bp[-1]
    fdim = mesh.topology.dim - 1
    bottom_facets = locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[2], 0.0, atol=1e-12))
    top_facets = locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[2], z_max, atol=z_max * 1e-9))
    facets = np.concatenate([bottom_facets, top_facets])
    values = np.concatenate([np.full(len(bottom_facets), FACET_BOTTOM, dtype=np.int32),
                              np.full(len(top_facets), FACET_TOP, dtype=np.int32)])
    order = np.argsort(facets)
    facet_tags = meshtags(mesh, fdim, facets[order], values[order])

    # DG0 coefficient fields, assigned directly from the tile grid by cell
    # centroid — no cell_tags/RegionRegistry indirection (see docstring).
    V = functionspace(mesh, ("DG", 0))
    cell_dim = mesh.topology.dim
    num_cells = mesh.topology.index_map(cell_dim).size_local
    midpoints = compute_midpoints(mesh, cell_dim, np.arange(num_cells, dtype=np.int32))

    dx_m = (tile_grid.x_edges[1] - tile_grid.x_edges[0]) * 1e-6
    dy_m = (tile_grid.y_edges[1] - tile_grid.y_edges[0]) * 1e-6
    tile_i = np.clip(((midpoints[:, 0] - x0_m) / dx_m).astype(np.int64), 0, tile_grid.nx - 1)
    tile_j = np.clip(((midpoints[:, 1] - y0_m) / dy_m).astype(np.int64), 0, tile_grid.ny - 1)

    slab_z0_m = np.array([tile_grid.slabs[n][0] * 1e-6 for n in slab_order])
    slab_idx = np.searchsorted(slab_z0_m, midpoints[:, 2], side="right") - 1
    slab_idx = np.clip(slab_idx, 0, len(slab_order) - 1)

    k = Function(V, name="k")
    rho_cp = Function(V, name="rho_cp")
    q = Function(V, name="q")
    for si, name in enumerate(slab_order):
        mask = slab_idx == si
        if not mask.any():
            continue
        _z0, _z1, k_grid, rho_cp_grid, q_grid = tile_grid.slabs[name]
        k.x.array[np.where(mask)[0]] = k_grid[tile_i[mask], tile_j[mask]]
        rho_cp.x.array[np.where(mask)[0]] = rho_cp_grid[tile_i[mask], tile_j[mask]]
        q.x.array[np.where(mask)[0]] = q_grid[tile_i[mask], tile_j[mask]]

    mesh_data = types.SimpleNamespace(mesh=mesh, cell_tags=None, facet_tags=facet_tags)
    return mesh_data, k, rho_cp, q
