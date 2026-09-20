"""Instance-preserving heat sources for the bundled SRAM bitcell.

This is a deliberately cell-specific, netlist-assisted connectivity map, not
a general LVS extractor. The manifest is independently reproducible with
cases/verify_bitcell_mapping.py. Geometry and circuit fingerprints fail closed
when its assumptions no longer apply. Mapping precedes any array transforms.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import gdstk

from gds.sources import _dims_um, extract_channels
from gds.techmap import CHANNEL_THICKNESS_UM
from spec.chip import SourceBox


REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "data/sram_sp_cell_channel_map.json"
INSTANCE_NAMES = frozenset(f"X{i}" for i in range(8))
# Include both conducting masks and pin purposes: a changed wire/pin must not
# silently retain an old instance assignment. Non-electrical annotations are
# irrelevant to this cell-specific correspondence.
MAPPING_SPECS = ((64, 20), (65, 20), (65, 44), (66, 20), (66, 44),
                 (67, 20), (67, 16), (68, 20), (68, 16), (67, 44),
                 (68, 44), (69, 20), (69, 16))


def mapping_revision():
    """Cache identity: old class-assigned temperature files are incompatible."""
    return hashlib.sha256(MANIFEST.read_bytes()).hexdigest()


def _canonical_ring(points):
    """Order/winding-independent coordinates, rounded below the GDS grid."""
    ring = [tuple(round(float(v), 9) for v in point) for point in points]
    if len(ring) > 1 and ring[-1] == ring[0]:
        ring.pop()
    if not ring:
        raise ValueError("Empty polygon in bitcell mapping")
    candidates = []
    for order in (ring, list(reversed(ring))):
        for i, point in enumerate(order):
            if point == min(order):
                candidates.append(tuple(order[i:] + order[:i]))
    return min(candidates)


def geometry_fingerprint(by_layer):
    """Fingerprint raw physical/pin polygons independent of list order."""
    records = [(spec, sorted(_canonical_ring(p.points)
                            for p in by_layer.get(spec, [])))
               for spec in sorted(MAPPING_SPECS)]
    return hashlib.sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class MappedChannel:
    instance: str
    device_class: str
    polygon: gdstk.Polygon
    gate: str
    drain: str
    source: str
    body: str


def map_bitcell_channels(by_layer):
    """Return eight individually identified channels, or reject stale mapping.

    Accepts the original local ``sram_sp_cell`` masks only. Array placement
    transforms the returned polygons and carries their names along; a polygon
    enumeration, class representative, or guessed top/bottom ordering is never
    used to select electrical power.
    """
    manifest = json.loads(MANIFEST.read_text())
    # Polygon dictionaries omit label text. Bind the known BL/BR/WL port
    # semantics to the complete original file as well as the actual masks.
    source_gds = REPO / manifest["source_gds"]
    if hashlib.sha256(source_gds.read_bytes()).hexdigest() != manifest["source_gds_sha256"]:
        raise ValueError("Source GDS/port labels changed; reverify the SRAM instance map")
    if geometry_fingerprint(by_layer) != manifest["geometry_sha256"]:
        raise ValueError("SRAM geometry does not match the verified instance map; "
                         "re-extract and verify connectivity before assigning power")
    for relative_path, expected_hash in manifest["netlist_sha256"].items():
        if hashlib.sha256((REPO / relative_path).read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"SRAM netlist changed: {relative_path}; reverify the instance map")
    channels = extract_channels(by_layer)
    records = manifest["channels"]
    if len(channels) != 8 or len(records) != 8 or {r["instance"] for r in records} != INSTANCE_NAMES:
        raise ValueError("SRAM instance map must contain exactly eight unique channels")
    unused = set(range(len(channels)))
    result = []
    for record in sorted(records, key=lambda item: item["instance"]):
        expected = gdstk.rectangle(record["bbox_um"][:2], record["bbox_um"][2:])
        matches = [i for i in unused
                   if not gdstk.boolean([channels[i]], [expected], "xor", precision=1e-6)]
        if len(matches) != 1:
            raise ValueError(f"Channel footprint for {record['instance']} is missing or ambiguous")
        index = matches[0]
        unused.remove(index)
        result.append(MappedChannel(
            instance=record["instance"], device_class=record["device_class"],
            polygon=channels[index], gate=record["gate"], drain=record["drain"],
            source=record["source"], body=record["body"],
        ))
    if unused:
        raise ValueError("Unmapped SRAM channels remain")
    return result


def validate_device_powers(device_power_w, expected_names=INSTANCE_NAMES):
    """Reject missing/extra IDs and nonphysical values; preserve every value."""
    expected_names = set(expected_names)
    if set(device_power_w) != expected_names:
        raise ValueError(f"Device-power names do not match mapped instances: "
                         f"missing={sorted(expected_names - set(device_power_w))}, "
                         f"extra={sorted(set(device_power_w) - expected_names)}")
    powers = {name: float(value) for name, value in device_power_w.items()}
    if any(not math.isfinite(value) or value < 0 for value in powers.values()):
        raise ValueError("Device powers must be finite and nonnegative")
    return powers


def channel_sources_for(mapped_channels, device_power_w):
    """Build individually named SourceBoxes from the full SPICE power dict."""
    names = [channel.instance for channel in mapped_channels]
    if len(names) != 8 or set(names) != INSTANCE_NAMES:
        raise ValueError("Expected each of X0 through X7 exactly once")
    powers = validate_device_powers(device_power_w)
    result = []
    for channel in mapped_channels:
        cx, cy, width, length = _dims_um(channel.polygon)
        if not math.isclose(channel.polygon.area(), width * length, rel_tol=1e-9):
            raise ValueError(f"Nonrectangular source for {channel.instance}; "
                             "SourceBox power normalization is not applicable")
        box = SourceBox(
            device=channel.instance, kind="channel", x_um=cx, y_um=cy,
            z0_um=0.0, w_um=width, l_um=length, t_um=CHANNEL_THICKNESS_UM,
            power_uw=powers[channel.instance] * 1e6,
        )
        result.append((box, channel.polygon))
    return result
