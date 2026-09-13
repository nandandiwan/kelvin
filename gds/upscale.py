"""Per-tile effective material properties from real GDS geometry, computed
by exact area-fraction arithmetic — no meshing, no PDE solve. This is what
makes a whole-die 3D solve memory-tractable: cases/run_gds_3d.py's probe
extrapolates full-resolution 3D to ~27 billion tets and ~0.87TB of mesh
connectivity alone, on a 502GB machine. A die-scale coarse mesh built from
this module has a cell count WE choose (one per tile), not one dictated by
how many real polygons exist.

Physical basis: within one real z-band, several materials sit side by side
laterally (e.g. PolySi gates next to the SiO2 field). For heat flowing
straight down through that band — the dominant direction throughout this
project's Bi<<1 regime, see SRAM_THERMAL_REPORT.md section 3 — each material
is a parallel conduction path, and the area-fraction-weighted arithmetic
mean is the *exact* effective in-band conductivity for that direction
(parallel resistors). Stacking several real bands in the z-direction (e.g.
lumping the whole FEOL+BEOL stack into one coarse slab) is a *series*
combination instead — thickness-weighted harmonic mean, also exact for
layers stacked in the flow direction.

The SAME per-tile value is used for the lateral (in-plane) conductivity too,
which is only an upper-bound approximation there (real lateral spreading
crosses phase boundaries in series, not parallel, and the true effective
value depends on percolation/geometry a simple mixing rule can't capture).
This is stated as an approximation to be validated against a direct
fine-mesh solve on the same footprint (cases/run_gds_3d.py), not asserted
as exact — see SRAM_THERMAL_REPORT.md's upscaling section for that check.
"""

from typing import Dict, List, Tuple

import numpy as np

from gds.techmap import SYNTHETIC, z_bounds
from spec.materials import get as get_material

# Real bands lumped into one coarse "feol_beol" slab for the global solve —
# band thickness in this stack spans 10nm (channel) to 50um (substrate), a
# 5000x range no single uniform-z structured mesh can resolve directly. The
# individual real bands are still resolved exactly by the fine windowed
# builds (cases/run_gds_3d.py); this lump is only for the coarse tier.
_FEOL_BEOL_NAMES = {"diff", "channel", "poly", "licon1", "li1", "mcon", "met1", "via", "met2"}


def default_lump_map(stack) -> Dict[str, List[str]]:
    """{coarse_slab_name: [real_band_name, ...]} — every FEOL/BEOL band
    lumped together; every other real band (substrate, BSPDN backside
    bands, passivation) keeps its own coarse slab unchanged. Generalizes
    across any StackProfile without hardcoding per-stack maps.
    """
    lump: Dict[str, List[str]] = {}
    for _z0, _z1, band in z_bounds(stack):
        if band.name in _FEOL_BEOL_NAMES:
            lump.setdefault("feol_beol", []).append(band.name)
        else:
            lump[band.name] = [band.name]
    return lump


def _rect_bboxes(polys) -> List[Tuple[float, float, float, float]]:
    """Each real polygon approximated by its own bounding box — the same
    simplification cases/run_gds_3d.py's windowed path uses (most shapes in
    this layout are already simple rectangles, verified in
    SRAM_THERMAL_REPORT.md's geometry-accuracy check)."""
    out = []
    for p in polys:
        (x0, y0), (x1, y1) = p.bounding_box()
        out.append((x0, y0, x1, y1))
    return out


def rasterize_area_fraction(rects: List[Tuple[float, float, float, float]],
                             x_edges: np.ndarray, y_edges: np.ndarray) -> np.ndarray:
    """Fraction of each tile's area covered by `rects` — vectorized
    rectangle/grid overlap, no gdstk boolean calls (the reason this scales
    to hundreds of thousands of real shapes across a whole die in seconds,
    not the ~1min-per-2400-boxes wall cases/run_gds_3d.py hit). Assumes a
    UNIFORM tile grid (x_edges, y_edges evenly spaced). Clipped to [0, 1] —
    can exceed 1 slightly from the bounding-box approximation on adjacent
    non-rectangular shapes; real GDS layers don't self-overlap.
    """
    nx, ny = len(x_edges) - 1, len(y_edges) - 1
    area = np.zeros((nx, ny))
    x0g, y0g = x_edges[0], y_edges[0]
    dx, dy = x_edges[1] - x_edges[0], y_edges[1] - y_edges[0]
    tile_area = dx * dy

    for x0, y0, x1, y1 in rects:
        i0 = max(0, int(np.floor((x0 - x0g) / dx)))
        i1 = min(nx, int(np.ceil((x1 - x0g) / dx)))
        j0 = max(0, int(np.floor((y0 - y0g) / dy)))
        j1 = min(ny, int(np.ceil((y1 - y0g) / dy)))
        if i1 <= i0 or j1 <= j0:
            continue
        xs = x_edges[i0:i1 + 1]
        ys = y_edges[j0:j1 + 1]
        ox = np.clip(np.minimum(xs[1:], x1) - np.maximum(xs[:-1], x0), 0, None)
        oy = np.clip(np.minimum(ys[1:], y1) - np.maximum(ys[:-1], y0), 0, None)
        area[i0:i1, j0:j1] += np.outer(ox, oy)

    return np.clip(area / tile_area, 0.0, 1.0)


def _band_field(by_layer, band, x_edges, y_edges, prop: str) -> np.ndarray:
    """Per-tile effective value of `prop` ("k" or "rho_cp") for one real
    band — background everywhere, each drawn (layer, material) overriding
    proportionally to its own area fraction, in the band's own order
    (later wins on overlap, same semantics as mesh/gds_section.py's real
    geometry emission — e.g. a licon1 plug overwriting PolySi where a
    contact lands on a gate).
    """
    nx, ny = len(x_edges) - 1, len(y_edges) - 1
    field = np.full((nx, ny), getattr(get_material(band.background), prop))
    for key, material in band.drawn:
        mat_val = getattr(get_material(material), prop)
        if key is SYNTHETIC:
            pitch, width = band.synthetic_grid
            frac = min(1.0, (width / pitch) ** 2)
            frac_field = frac  # scalar broadcasts
        else:
            frac_field = rasterize_area_fraction(_rect_bboxes(by_layer.get(key, [])), x_edges, y_edges)
        field = field * (1 - frac_field) + mat_val * frac_field
    return field


def _accumulate_source_power(channel_sources, contact_sources, real_names,
                              x_edges, y_edges) -> np.ndarray:
    """Total real source power (uW) landing in each tile, for whichever of
    "channel"/"licon1" (contact) bands this lump actually contains. Exact
    energy conservation by construction — same "assume some heating per
    element, sum matches" principle as the rest of this project — no
    source_depth_m homogenization needed: a true 3D coarse mesh resolves x,
    y, AND z structurally, just at tile resolution instead of per-device.
    """
    nx, ny = len(x_edges) - 1, len(y_edges) - 1
    acc = np.zeros((nx, ny))
    x0g, y0g = x_edges[0], y_edges[0]
    dx, dy = x_edges[1] - x_edges[0], y_edges[1] - y_edges[0]

    def add(sources, wanted_band):
        if wanted_band not in real_names:
            return
        for box, _poly in sources:
            i = int((box.x_um - x0g) // dx)
            j = int((box.y_um - y0g) // dy)
            if 0 <= i < nx and 0 <= j < ny:
                acc[i, j] += box.power_uw

    add(channel_sources, "channel")
    add(contact_sources, "licon1")
    return acc


class TileGrid:
    """Per-(coarse slab) effective properties on a uniform (x, y) tile grid.
    `slabs[name] = (z0_um, z1_um, k_eff[NX,NY], rho_cp_eff[NX,NY], q_eff_w_m3[NX,NY])`,
    ordered bottom (z=0, Robin sink) to top.
    """

    def __init__(self, x_edges, y_edges, slab_order: List[str], slabs: Dict[str, tuple]):
        self.x_edges = x_edges
        self.y_edges = y_edges
        self.slab_order = slab_order
        self.slabs = slabs

    @property
    def nx(self):
        return len(self.x_edges) - 1

    @property
    def ny(self):
        return len(self.y_edges) - 1


def build_tile_grid(by_layer, channel_sources, contact_sources, stack,
                     window: Tuple[float, float, float, float], tile_um: float,
                     lump_map: Dict[str, List[str]] = None) -> TileGrid:
    """`window` = (x0, x1, y0, y1) in um — the full die, or any sub-region.
    `tile_um`: lateral tile size, the single knob controlling coarse cell
    count (and therefore memory) independent of real polygon density.
    """
    lump_map = lump_map or default_lump_map(stack)
    x0, x1, y0, y1 = window
    nx = max(1, round((x1 - x0) / tile_um))
    ny = max(1, round((y1 - y0) / tile_um))
    x_edges = np.linspace(x0, x1, nx + 1)
    y_edges = np.linspace(y0, y1, ny + 1)

    bounds = {b.name: (zb0, zb1, b) for zb0, zb1, b in z_bounds(stack)}
    slab_order = sorted(lump_map, key=lambda name: min(bounds[n][0] for n in lump_map[name]))

    slabs = {}
    for name in slab_order:
        real_names = lump_map[name]
        z0s = min(bounds[n][0] for n in real_names)
        z1s = max(bounds[n][1] for n in real_names)

        inv_k_over_t = np.zeros((nx, ny))
        rho_cp_num = np.zeros((nx, ny))
        total_t = 0.0
        for n in real_names:
            zb0, zb1, band = bounds[n]
            t = zb1 - zb0
            total_t += t
            k_tile = _band_field(by_layer, band, x_edges, y_edges, "k")
            rho_cp_tile = _band_field(by_layer, band, x_edges, y_edges, "rho_cp")
            inv_k_over_t += t / k_tile
            rho_cp_num += rho_cp_tile * t
        k_eff = total_t / inv_k_over_t
        rho_cp_eff = rho_cp_num / total_t

        power_uw = _accumulate_source_power(channel_sources, contact_sources, real_names, x_edges, y_edges)
        tile_area_um2 = (x_edges[1] - x_edges[0]) * (y_edges[1] - y_edges[0])
        vol_m3 = tile_area_um2 * total_t * 1e-18
        q_eff = (power_uw * 1e-6) / vol_m3  # W/m^3, per tile

        slabs[name] = (z0s, z1s, k_eff, rho_cp_eff, q_eff)

    return TileGrid(x_edges, y_edges, slab_order, slabs)
