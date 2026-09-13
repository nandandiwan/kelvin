"""Load a GDSII file and reduce it to per-(layer, datatype) polygon lists.

`union_merge` (boolean-OR each layer's polygons together) turned out to be
the wrong lever for the 2D cutline pipeline, not the right one it first
looked like: it doesn't scale (superlinear in entity count — merging
sram22_2048x8m8w1's ~426k unmerged licon1 polygons was still running after
200s, whereas a single boolean-AND of that same list against a thin cutline
strip takes well under a second, see mesh/gds_section.py). The 2D cutline
path never needed pre-merged geometry for correctness — a boolean AND
against a strip is geometrically local and cheap regardless of how many
separate-but-touching polygons the layer has; merging first only ever saved
*gmsh* a few extra (functionally identical) touching surfaces per band, not
enough to be worth its own now-dominant cost at real macro scale. Kept here
as a still-usable utility (e.g. for cheaper rendering of a small file), but
the 2D pipeline (mesh/gds_build.py, gds/sources.py) calls `flatten_by_layer`
directly and skips it.
"""

from typing import Dict, Iterable, List, Optional, Tuple

import gdstk

LayerKey = Tuple[int, int]


def load_top_cell(path: str):
    lib = gdstk.read_gds(path)
    tops = lib.top_level()
    if len(tops) != 1:
        raise ValueError(f"expected exactly one top-level cell, got {[c.name for c in tops]}")
    return lib, tops[0]


def flatten_by_layer(top_cell) -> Dict[LayerKey, List["gdstk.Polygon"]]:
    """{(layer, datatype): [gdstk.Polygon, ...]} for the flattened top cell."""
    flat = top_cell.copy(f"{top_cell.name}$flat").flatten()
    by_layer: Dict[LayerKey, List] = {}
    for p in flat.polygons:
        by_layer.setdefault((p.layer, p.datatype), []).append(p)
    return by_layer


def union_merge(
    by_layer: Dict[LayerKey, List], layers: Optional[Iterable[LayerKey]] = None
) -> Dict[LayerKey, List]:
    """Boolean-OR each layer's polygons together, merging abutting/overlapping
    shapes into fewer, larger ones. Same total covered area, far fewer
    entities for gmsh to stitch. With `layers` given, only those keys are
    processed (and returned) — everything else in `by_layer` is dropped.
    """
    keys = by_layer.keys() if layers is None else (k for k in layers if k in by_layer)
    return {key: gdstk.boolean(by_layer[key], [], "or") for key in keys}


def named_cell_bbox_um(lib, cell_name: str) -> Tuple[float, float, float, float]:
    """(x0, y0, x1, y1) of one specific named cell in the library — for
    measuring a real repeat pitch (e.g. a bitcell's height) directly from the
    GDS hierarchy, instead of falling back to the whole die's extent when a
    real periodic pitch is actually available to measure (see the 2D-cut
    homogenization notes in GDS_PLAN.md/PLAN.md: source_depth_m should be the
    real row pitch for a periodic array, not the full die height — the
    latter is only the defensible choice when there's no repeating structure
    to measure a pitch from at all).
    """
    for cell in lib.cells:
        if cell.name == cell_name:
            (x0, y0), (x1, y1) = cell.bounding_box()
            return x0, y0, x1, y1
    raise KeyError(f"no cell named {cell_name!r} in this library")


def load_and_prepare(path: str, layers: Optional[Iterable[LayerKey]] = None):
    """load_and_prepare(path) -> (lib, top_cell, by_layer_merged)."""
    lib, top = load_top_cell(path)
    by_layer = flatten_by_layer(top)
    merged = union_merge(by_layer, layers)
    return lib, top, merged
