"""Build an N-row x M-column bitcell array by tiling the single validated
`sram_sp_cell` (data/sram22_64x22m4w22.gds), mirrored at cell boundaries the
way a real SRAM array abuts columns/rows.

PITCH IS VERIFIED CORRECT: real cells abut at exactly 1.2 x 1.58um. Checked
against the macro's own `sp_cell_array_center`, whose 16 bitcells occupy
x 1.30-6.10 (= 4 x 1.2) and y 0-6.32 (= 4 x 1.58), with zero overlapping
pairs once each reference's rotation/x_reflection is applied.

KNOWN LIMITATION -- STRAP CELLS ARE OMITTED. The real array interleaves
wordline-strap and well-tap cells that this tiling does not reproduce: inside
each 6.1 x 7.9um tile the 16 bitcells occupy only 4.8 x 6.32um, leaving a
1.30um strap column every 4 columns and a 1.58um strap row every 4 rows.

    real:  48.19 um^2 per 16 cells = 3.012 um^2 per cell
    here:  1.2 x 1.58              = 1.896 um^2 per cell
    -> this tiling is 1.589x DENSER, so array power density (and hence the
       temperature rise) is overestimated by that factor unless the per-cell
       power is scaled by 1/1.589 to match real areal density.

(An earlier version of this docstring claimed the GDS's own array cells were
unused staging artifacts that stacked two transistor sets per site. That was
a transform-reading error -- origins were read without applying rotation and
x_reflection. The array cells are real and used: `sp_cell_array` is
instantiated by `sram22_inner` and holds 1620 bitcells, 1408 functional plus
212 dummy edge cells.)
"""

from typing import Dict, List, Tuple

import gdstk

from gds import techmap
from gds.bitcell_mapping import map_bitcell_channels
from gds.read import flatten_by_layer
from gds.sources import _dims_um, extract_channels, extract_contacts
from spec.chip import SourceBox

GDS_PATH = "data/sram22_64x22m4w22.gds"
CELL_NAME = "sram_sp_cell"
CELL_W_UM, CELL_H_UM = 1.2, 1.58     # the cell's own bbox extent (see build_gds_3d_mesh calls)

LayerKey = Tuple[int, int]


def _transform_points(pts, mirror_x, mirror_y):
    """Local (-1.2..0, -1.58..0) points -> the (0..1.2, 0..1.58) unit square,
    optionally mirrored about the cell's own center, as a plain point list
    (translation to the (row, col) tile happens separately in `tile_array` so
    this stays reusable for both layer polygons and channel/contact polygons).

    Real SRAM arrays mirror adjacent columns/rows at their shared edge so the
    two cells' wells and taps line up. The macro's own sp_cell_array_center
    does exactly this: its refs alternate `x_reflection=True, rotation=0`
    against `rotation=pi` along a row, which is what places consecutive cells
    side by side at 1.2um pitch rather than on top of each other.

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

    Each channel SourceBox's `device` field is "r{row}c{col}_{instance}",
    where instance is the original SPICE name X0..X7. Mapping happens in the
    original cell frame before any mirrors, so transforms preserve device
    identity and asymmetric per-instance powers.
    """
    lib = gdstk.read_gds(GDS_PATH)
    cell = next(c for c in lib.cells if c.name == CELL_NAME)
    by_layer_local = flatten_by_layer(cell)
    (bx0, by0), (bx1, by1) = cell.bounding_box()
    assert abs((bx1 - bx0) - CELL_W_UM) < 1e-6 and abs((by1 - by0) - CELL_H_UM) < 1e-6, (
        f"cell bbox {(bx1-bx0, by1-by0)} != assumed {(CELL_W_UM, CELL_H_UM)}")

    channels_local = map_bitcell_channels(by_layer_local)
    contacts_local = extract_contacts(by_layer_local)

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

            for channel in channels_local:
                poly = channel.polygon
                pts = _transform_points(list(poly.points), mirror_x, mirror_y)
                pts = [(x + ox, y + oy) for x, y in pts]
                tpoly = gdstk.Polygon(pts, layer=poly.layer, datatype=poly.datatype)
                cx, cy, xe, ye = _dims_um(tpoly)
                box = SourceBox(device=f"r{row}c{col}_{channel.instance}", kind="channel",
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


# --- Tiling the macro's OWN array tile (straps included) --------------------
ARRAY_TILE_NAME = "sp_cell_array_center"
TILE_W_UM, TILE_H_UM = 6.10, 7.90        # verified from sp_cell_array's center-tile origins
TILE_CELL_X_UM = (1.30, 2.50, 3.70, 4.90)  # bitcell left edges within the tile
TILE_CELL_Y_UM = (0.00, 1.58, 3.16, 4.74)


def _classify_dims(x_ext, y_ext, in_nwell):
    if abs(y_ext - 0.025) < 0.01:
        return "parasitic"
    if abs(x_ext - 0.21) < 0.01:
        return "latch"
    return "pullup" if in_nwell else "access"


def tile_array_real(n_tiles_x: int, n_tiles_y: int):
    """Tile the macro's OWN `sp_cell_array_center`, which carries the wordline
    strap column and well-tap strap row that `tile_array` omits.

    Each tile is 6.10 x 7.90um and holds a 4x4 block of bitcells occupying
    only 4.80 x 6.32um of it -- verified: flattening the tile and binning
    DIFF n POLY into the 16 known cell boxes gives exactly 8 channels each
    (128 total), plus 9 strap-region tap structures which are NOT driven
    devices and are given zero power.

    Real areal density (3.012 um^2/cell) instead of tile_array's abutted
    1.896 um^2/cell, so array power density is right rather than 1.59x high.

    Returns (by_layer, window, channel_sources, contact_sources); channel
    device names are "r{row}c{col}_{cls}{i}" with GLOBAL cell indices, and
    "strap{i}" for the zero-power tap structures.
    """
    lib = gdstk.read_gds(GDS_PATH)
    tile = next(c for c in lib.cells if c.name == ARRAY_TILE_NAME)
    tile_by_layer = flatten_by_layer(tile)
    nwell = tile_by_layer.get(techmap.NWELL, [])

    tile_channels = extract_channels(tile_by_layer)
    tile_contacts = extract_contacts(tile_by_layer)

    # Classify each tile channel once: which (local row, col) bitcell it is in
    # (or None = strap tap), and its device class.
    classified = []
    for poly in tile_channels:
        cx, cy, xe, ye = _dims_um(poly)
        ci = next((i for i, x in enumerate(TILE_CELL_X_UM)
                   if x - 1e-6 <= cx <= x + CELL_W_UM + 1e-6), None)
        ri = next((j for j, y in enumerate(TILE_CELL_Y_UM)
                   if y - 1e-6 <= cy <= y + CELL_H_UM + 1e-6), None)
        if ci is None or ri is None:
            classified.append((None, None, "strap", poly))
            continue
        in_nw = any(gdstk.inside([(cx, cy)], [p])[0] for p in nwell)
        classified.append((ri, ci, _classify_dims(xe, ye, in_nw), poly))

    by_layer_big: Dict[LayerKey, List] = {}
    channel_sources, contact_sources = [], []
    n_strap = 0
    for tj in range(n_tiles_y):
        for ti in range(n_tiles_x):
            ox, oy = ti * TILE_W_UM, tj * TILE_H_UM

            for key, polys in tile_by_layer.items():
                dst = by_layer_big.setdefault(key, [])
                for poly in polys:
                    dst.append(gdstk.Polygon(
                        [(x + ox, y + oy) for x, y in poly.points],
                        layer=key[0], datatype=key[1]))

            for ri, ci, cls, poly in classified:
                tp = gdstk.Polygon([(x + ox, y + oy) for x, y in poly.points],
                                   layer=poly.layer, datatype=poly.datatype)
                cx, cy, xe, ye = _dims_um(tp)
                if cls == "strap":
                    name = f"strap{n_strap}"; n_strap += 1
                else:
                    name = f"r{tj*4 + ri}c{ti*4 + ci}_{cls}{ri}{ci}"
                channel_sources.append((SourceBox(
                    device=name, kind="channel", x_um=cx, y_um=cy, z0_um=0.0,
                    w_um=xe, l_um=ye, t_um=techmap.CHANNEL_THICKNESS_UM,
                    power_uw=0.0), tp))

            for i, poly in enumerate(tile_contacts):
                tp = gdstk.Polygon([(x + ox, y + oy) for x, y in poly.points],
                                   layer=poly.layer, datatype=poly.datatype)
                cx, cy, xe, ye = _dims_um(tp)
                contact_sources.append((SourceBox(
                    device=f"t{tj}_{ti}_co{i}", kind="contact", x_um=cx, y_um=cy,
                    z0_um=0.0, w_um=xe, l_um=ye,
                    t_um=techmap.LICON1_THICKNESS_UM, power_uw=0.0), tp))

    window = (0.0, n_tiles_x * TILE_W_UM, 0.0, n_tiles_y * TILE_H_UM)
    return by_layer_big, window, channel_sources, contact_sources
