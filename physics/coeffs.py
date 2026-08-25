"""cell_tags -> DG0 k / rho*cp / q coefficient fields.

One `dx` for the whole domain: adding a material or a region never touches
the variational form, only this lookup table (PLAN.md's core architectural
idea). Everything here is a single scatter through the region registry that
mesh/build.py already built, no dolfinx-specific tagging logic beyond it.
"""

import numpy as np
from dolfinx.fem import Function, functionspace

from mesh.boxes import RegionRegistry
from spec.materials import get as get_material


def _tag_by_cell(mesh, cell_tags):
    dim = mesh.topology.dim
    im = mesh.topology.index_map(dim)
    num_cells = im.size_local + im.num_ghosts
    tag_by_cell = np.zeros(num_cells, dtype=np.int32)
    tag_by_cell[cell_tags.indices] = cell_tags.values
    return tag_by_cell


def build_coeffs(mesh, cell_tags, registry: RegionRegistry, source_depth_m=None):
    """Returns (k, rho_cp, q) DG0 Functions, at T=300K reference (linear
    steady solve — k(T) nonlinearity is a later, additive step).

    `source_depth_m`: a 2D cross-section is translationally invariant in the
    unmodeled third dimension, i.e. every source is implicitly extruded to
    infinite depth there. Using the source's true (tiny, ~20-100nm) 3D depth
    for q in that setting hugely overstates injected power — it says "this
    much power, every metre, forever" instead of "this much power, once, in
    a sliver a few tens of nm thick". Pass the real 3D `power_density_w_per_m3`
    only when the mesh actually resolves all three dimensions (the future 3D
    path: leave this None). For a 2D cross-section through a periodic device
    row, pass the row pitch instead: it re-derives q as
    `power_uw / (w_m * t_m * source_depth_m)`, i.e. "smear this device's
    power uniformly over the periodic cell it actually repeats in" — the
    correct homogenization for a translationally-invariant slice of a
    periodic array, and the only choice under which the 2D solve's absolute
    temperatures are physically meaningful rather than a qualitative-only
    sanity check.
    """
    V = functionspace(mesh, ("DG", 0))
    tag_by_cell = _tag_by_cell(mesh, cell_tags)
    max_tag = int(tag_by_cell.max())

    k_by_tag = np.zeros(max_tag + 1)
    rho_cp_by_tag = np.zeros(max_tag + 1)
    q_by_tag = np.zeros(max_tag + 1)
    for region in registry.all():
        mat = get_material(region.material)
        k_by_tag[region.tag_id] = mat.k
        rho_cp_by_tag[region.tag_id] = mat.rho_cp
        if region.source is not None:
            if source_depth_m is None:
                q_by_tag[region.tag_id] = region.source.power_density_w_per_m3
            else:
                w_m = region.source.w_um * 1e-6
                t_m = region.source.t_um * 1e-6
                power_w = region.source.power_uw * 1e-6
                q_by_tag[region.tag_id] = power_w / (w_m * t_m * source_depth_m)

    k = Function(V, name="k")
    rho_cp = Function(V, name="rho_cp")
    q = Function(V, name="q")
    k.x.array[:] = k_by_tag[tag_by_cell]
    rho_cp.x.array[:] = rho_cp_by_tag[tag_by_cell]
    q.x.array[:] = q_by_tag[tag_by_cell]
    return k, rho_cp, q
