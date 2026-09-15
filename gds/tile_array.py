"""Build an N-row x M-column real bitcell array by tiling the single
validated `sram_sp_cell` (data/sram22_64x22m4w22.gds), mirrored at cell
boundaries the way a real SRAM array abuts columns/rows -- rather than
trusting the GDS's own `sp_cell_array_center`/`_left`/`_right` cells.

Those pre-built array cells were checked and rejected: `sp_cell_array_center`
contains TWO full sets of transistor references (`sram_sp_cell` and
`sram_sp_cell_opt1a`) at the IDENTICAL (x, y) origin for every site --
physically impossible for real silicon. Cross-checked against the actual
macro: the top-level `sram22_64x22m4w22` cell has 1622 references, each with
a UNIQUE name, and references NEITHER `sp_cell_array_center` NOR
`sram_sp_cell` anywhere. These array-assembly cells are leftover generator
staging artifacts (an OpenRAM/sram22-style flow builds the array from a
template, then flattens+renames per-instance for the final macro), not real
tileable geometry. Building on them would silently double every device's
power.

Tiling ourselves from the single already-validated cell (the same one
cases/run_bitcell_compact.py uses) avoids that risk entirely: it is known-good
geometry, and the transform is explicit and checkable.
"""

from typing import Dict, List, Tuple

import gdstk

from gds import techmap
from gds.read import flatten_by_layer
from gds.sources import _dims_um, extract_channels, extract_contacts
from spec.chip import SourceBox

GDS_PATH = "data/sram22_64x22m4w22.gds"
CELL_NAME = "sram_sp_cell"
CELL_W_UM, CELL_H_UM = 1.2, 1.58     # the cell's own bbox extent (see build_gds_3d_mesh calls)
_NWELL_SPLIT_X_UM = -0.72             # PMOS (nwell) at x < this, in the cell's own local coords

LayerKey = Tuple[int, int]


def _classify(x_ext, y_ext, is_left):
    if abs(y_ext - 0.025) < 0.01:
        return "parasitic"
    if abs(x_ext - 0.21) < 0.01:
        return "latch"
    return "pullup" if is_left else "access"


def _transform_points(pts, mirror_x, mirror_y):
    """Local (-1.2..0, -1.58..0) points -> the (0..1.2, 0..1.58) unit square,
    optionally mirrored about the cell's own center, as a plain point list
    (translation to the (row, col) tile happens separately in `tile_array` so
    this stays reusable for both layer polygons and channel/contact polygons).

    Real SRAM arrays mirror adjacent columns/rows at their shared edge so the
    two cells' wells and taps line up (a checkerboard, mirroring both x and y
    on alternating cells) -- reproduced here as the standard, not a guess:
    verified against the real (unused) sp_cell_array_center hierarchy, whose
    reference list alternates `rotation=180` (an x+y mirror) on exactly this
    pattern before it was rejected for the overlap bug above.

    A single-axis mirror reverses polygon winding; a double mirror (both axes)
    is a rotation and preserves it. Reversed here for consistency, though
    nothing downstream currently depends on winding order.
    """
    out = [(x + CELL_W_UM, y + CELL_H_UM) for x, y in pts]
    if mirror_x:
        out = [(CELL_W_UM - x, y) for x, y in out]
    if mirror_y:
        out = [(x, CELL_H_UM - y) for x, y in out]
    if mirror_x != mirror_y:
        out = out[::-1]
    return out


def tile_array(n_rows: int, n_cols: int):
    """Returns (by_layer, window, channel_sources, contact_sources) for an
    n_rows x n_cols array of real bitcells, checkerboard-mirrored at every
    boundary, in one absolute coordinate frame ready for build_gds_3d_mesh.

    Each channel SourceBox's `device` field is "r{row}c{col}_{cls}{i}" so a
    caller can bin per-cell power/temperature by (row, col) afterward.
    """
    lib = gdstk.read_gds(GDS_PATH)
    cell = next(c for c in lib.cells if c.name == CELL_NAME)
    by_layer_local = flatten_by_layer(cell)
    (bx0, by0), (bx1, by1) = cell.bounding_box()
    assert abs((bx1 - bx0) - CELL_W_UM) < 1e-6 and abs((by1 - by0) - CELL_H_UM) < 1e-6, (
        f"cell bbox {(bx1-bx0, by1-by0)} != assumed {(CELL_W_UM, CELL_H_UM)}")

    channels_local = extract_channels(by_layer_local)
    contacts_local = extract_contacts(by_layer_local)
    ch_classified = []
    for i, poly in enumerate(channels_local):
        cx, cy, xe, ye = _dims_um(poly)
        cls = _classify(xe, ye, is_left=cx < _NWELL_SPLIT_X_UM)
        ch_classified.append((cls, poly))

    by_layer_big: Dict[LayerKey, List] = {}
    channel_sources, contact_sources = [], []

    for row in range(n_rows):
        mirror_y = bool(row % 2)
        for col in range(n_cols):
            mirror_x = bool(col % 2)
            ox, oy = col * CELL_W_UM, row * CELL_H_UM

            for key, polys in by_layer_local.items():
                dst = by_layer_big.setdefault(key, [])
                for poly in polys:
                    pts = _transform_points(list(poly.points), mirror_x, mirror_y)
                    pts = [(x + ox, y + oy) for x, y in pts]
                    dst.append(gdstk.Polygon(pts, layer=key[0], datatype=key[1]))

            for i, (cls, poly) in enumerate(ch_classified):
                pts = _transform_points(list(poly.points), mirror_x, mirror_y)
                pts = [(x + ox, y + oy) for x, y in pts]
                tpoly = gdstk.Polygon(pts, layer=poly.layer, datatype=poly.datatype)
                cx, cy, xe, ye = _dims_um(tpoly)
                box = SourceBox(device=f"r{row}c{col}_{cls}{i}", kind="channel",
                                x_um=cx, y_um=cy, z0_um=0.0, w_um=xe, l_um=ye,
                                t_um=techmap.CHANNEL_THICKNESS_UM, power_uw=0.0)
                channel_sources.append((box, tpoly))

            for i, poly in enumerate(contacts_local):
                pts = _transform_points(list(poly.points), mirror_x, mirror_y)
                pts = [(x + ox, y + oy) for x, y in pts]
                tpoly = gdstk.Polygon(pts, layer=poly.layer, datatype=poly.datatype)
                cx, cy, xe, ye = _dims_um(tpoly)
                box = SourceBox(device=f"r{row}c{col}_co{i}", kind="contact",
                                x_um=cx, y_um=cy, z0_um=0.0, w_um=xe, l_um=ye,
                                t_um=techmap.LICON1_THICKNESS_UM, power_uw=0.0)
                contact_sources.append((box, tpoly))

    window = (0.0, n_cols * CELL_W_UM, 0.0, n_rows * CELL_H_UM)
    return by_layer_big, window, channel_sources, contact_sources
