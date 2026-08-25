"""Interval/rectangle-partition primitives (1D for the 2D cross-section's
x-axis, 2D for the 3D path's lateral x-y plane), and the Region bookkeeping
that tracks which gmsh entity ends up representing which material/tag. Pure
data + geometry bookkeeping, no gmsh calls here.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from spec.chip import SourceBox


@dataclass(frozen=True)
class Region:
    """One distinguishable cell region: a physical-group id plus what it means.

    Most regions share `label == material` (e.g. every plain SiO2 cell in the
    model is one physical group — there's nothing to distinguish between two
    SiO2 cells). Heat-source cells are the exception: each device's channel
    and contact source get their own label (and therefore their own tag_id)
    even though the material is the same as the surrounding fill, because
    they carry a distinct per-cell power density.
    """
    tag_id: int
    label: str
    material: str
    source: Optional[SourceBox] = None


class RegionRegistry:
    """Assigns stable integer ids to region labels in first-seen order."""

    def __init__(self):
        self._regions: Dict[str, Region] = {}
        self._next_id = 1

    def get(self, label: str, material: str, source: Optional[SourceBox] = None) -> Region:
        r = self._regions.get(label)
        if r is None:
            r = Region(tag_id=self._next_id, label=label, material=material, source=source)
            self._regions[label] = r
            self._next_id += 1
        return r

    def all(self) -> List[Region]:
        return list(self._regions.values())


# A segment is (x0, x1, payload) where payload = (material, source_or_None, label).
Segment = Tuple[float, float, Tuple[str, Optional[SourceBox], str]]


def insert_interval(segments: List[Segment], x0: float, x1: float,
                     payload: Tuple[str, Optional[SourceBox], str]) -> List[Segment]:
    """Clip any existing segment overlapping [x0, x1) and insert `payload`
    there. Later calls win over earlier ones on overlap — this is how a
    device_fill column punches through a layer's background, and a source
    sub-box then punches through the device_fill column on top of that.
    """
    out = []
    for s0, s1, p in segments:
        if s1 <= x0 or s0 >= x1:
            out.append((s0, s1, p))
            continue
        if s0 < x0:
            out.append((s0, x0, p))
        if s1 > x1:
            out.append((x1, s1, p))
    out.append((x0, x1, payload))
    out.sort(key=lambda t: t[0])
    return out


# A rect is (x0, y0, x1, y1, payload) where payload = (material, source_or_None, label).
Rect = Tuple[float, float, float, float, Tuple[str, Optional[SourceBox], str]]


def insert_rect(rects: List[Rect], x0: float, y0: float, x1: float, y1: float,
                 payload: Tuple[str, Optional[SourceBox], str]) -> List[Rect]:
    """2D analogue of insert_interval, for the 3D path's lateral (x, y)
    plane: clip any existing rect overlapping [x0,x1]x[y0,y1] into up to 4
    axis-aligned remainder pieces (top/bottom/left/right slabs around the
    hole), then add the new rect. Same later-wins-on-overlap priority.
    """
    out = []
    for rx0, ry0, rx1, ry1, p in rects:
        if rx1 <= x0 or rx0 >= x1 or ry1 <= y0 or ry0 >= y1:
            out.append((rx0, ry0, rx1, ry1, p))
            continue
        ox0, oy0 = max(rx0, x0), max(ry0, y0)
        ox1, oy1 = min(rx1, x1), min(ry1, y1)
        if ry0 < oy0:  # bottom slab, full width
            out.append((rx0, ry0, rx1, oy0, p))
        if oy1 < ry1:  # top slab, full width
            out.append((rx0, oy1, rx1, ry1, p))
        if rx0 < ox0:  # left slab, clipped to the hole's y-range
            out.append((rx0, oy0, ox0, oy1, p))
        if ox1 < rx1:  # right slab, clipped to the hole's y-range
            out.append((ox1, oy0, rx1, oy1, p))
    out.append((x0, y0, x1, y1, payload))
    return out
