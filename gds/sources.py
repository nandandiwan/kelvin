"""Heat-source extraction from real layout geometry.

Two mechanisms, same split the synthetic chip uses (spec/chip.py):
  channel = DIFF ∩ POLY     -- hot-carrier dissipation in the driven channel,
                                70% of a device's power (CHANNEL_POWER_FRAC).
  contact = LICON1 ∩ DIFF   -- S/D contact-resistance heating, 30%
                                (CONTACT_POWER_FRAC).

Per-channel power is proportional to channel width (drive current ~ W/L at
fixed L, V) -- see GDS_PLAN.md section 3 for the derivation and worked
numbers on this exact layout. Contact power is split uniformly per contact
instance -- attributing a specific contact to a specific channel needs
LVS-level netlist extraction, out of scope for "assume some heating per
element".

Sources are represented as spec.chip.SourceBox. physics/coeffs.build_coeffs's
`source_depth_m` homogenization expects `w_um`/`l_um` in a specific role: w
is the dimension actually resolved in the 2D mesh (here, X — the cutline
sweeps x), and l is the dimension that ISN'T resolved (Y, replaced by
`source_depth_m` at build time) — the same roles PITCH_X/PITCH_Y-style
values play for the synthetic chip. Get this backwards and q is computed
against the wrong axis, silently overstating power the same way an
unhomogenized 2D cut always does (see PLAN.md's Tmax=4135K incident).
"""

from typing import Dict, List, Tuple

import gdstk

from spec.chip import SourceBox

from .techmap import CHANNEL_THICKNESS_UM, DIFF, LICON1, LICON1_THICKNESS_UM, POLY

CHANNEL_POWER_FRAC = 0.7
CONTACT_POWER_FRAC = 0.3

LayerKey = Tuple[int, int]


def extract_channels(by_layer: Dict[LayerKey, List]) -> List["gdstk.Polygon"]:
    diff = by_layer.get(DIFF, [])
    poly = by_layer.get(POLY, [])
    if not diff or not poly:
        return []
    return gdstk.boolean(diff, poly, "and")


def extract_contacts(by_layer: Dict[LayerKey, List]) -> List["gdstk.Polygon"]:
    diff = by_layer.get(DIFF, [])
    licon1 = by_layer.get(LICON1, [])
    if not diff or not licon1:
        return []
    return gdstk.boolean(licon1, diff, "and")


def _dims_um(poly) -> Tuple[float, float, float, float]:
    """(cx, cy, x_extent_um, y_extent_um) — plain bounding-box measurement,
    axis-explicit rather than "whichever is longer". Two different things
    both come from this: the real device width (the Y-extent, for the power
    model below) and the SourceBox w_um/l_um mesh-axis roles (assigned by
    the caller, since which axis is "resolved" depends on the cut).
    """
    (x0, y0), (x1, y1) = poly.bounding_box()
    return (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0


def build_heat_sources(by_layer: Dict[LayerKey, List], total_power_w: float):
    """Returns (channel_sources, contact_sources): lists of (SourceBox, gdstk.Polygon)
    pairs — the polygon is kept alongside for mesh emission (cutline / extrusion),
    since SourceBox itself only carries the box's numeric extent, not its shape.

    Note real device width (Y-extent, used for power proportionality) and
    SourceBox.w_um (X-extent — the mesh-resolved axis, per the module
    docstring) are different things that happen to both come off the same
    bounding box; don't conflate them.
    """
    channels = extract_channels(by_layer)
    contacts = extract_contacts(by_layer)

    channel_dims = [_dims_um(p) for p in channels]
    total_device_width = sum(d[3] for d in channel_dims) or 1.0  # sum of Y-extents
    channel_budget = total_power_w * CHANNEL_POWER_FRAC

    channel_sources = []
    for i, (poly, (cx, cy, x_ext, y_ext)) in enumerate(zip(channels, channel_dims)):
        power_uw = channel_budget * 1e6 * y_ext / total_device_width
        box = SourceBox(
            device=f"ch{i}", kind="channel",
            x_um=cx, y_um=cy,
            z0_um=0.0,  # placed by the mesh emitter, which knows the actual band z0
            w_um=x_ext, l_um=y_ext, t_um=CHANNEL_THICKNESS_UM,
            power_uw=power_uw,
        )
        channel_sources.append((box, poly))

    n_contacts = len(contacts) or 1
    contact_power_uw = total_power_w * CONTACT_POWER_FRAC * 1e6 / n_contacts
    contact_sources = []
    for i, poly in enumerate(contacts):
        cx, cy, x_ext, y_ext = _dims_um(poly)
        box = SourceBox(
            device=f"co{i}", kind="contact",
            x_um=cx, y_um=cy,
            z0_um=0.0,
            w_um=x_ext, l_um=y_ext, t_um=LICON1_THICKNESS_UM,
            power_uw=contact_power_uw,
        )
        contact_sources.append((box, poly))

    return channel_sources, contact_sources
