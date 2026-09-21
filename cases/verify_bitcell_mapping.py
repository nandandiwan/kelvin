"""Netlist-assisted, exact-mask connectivity audit for the bundled SRAM cell.

This is NOT a general LVS extractor. The six functional devices are identified
from original conductor masks, named ports, gate connectivity and the bundled
netlist. Two cell-extraction contracts remain explicit: physically separate
access-gate stripes denote the same logical WL, and each short PMOS edge overlap
has one adjacent diffusion node which the netlist assigns to both D and S.
Neither contract is presented as independently established local connectivity.

No thermal mesh or electrical/thermal solver is used. The optional JSON report
is suitable evidence for a separately fingerprinted, cell-specific mapping.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import gdstk

ROOT = Path(__file__).resolve().parents[1]
PRECISION_UM = 1e-6
ADJACENCY_UM = 1e-5
AREA_EPS_UM2 = 1e-10


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _union(polygons):
    return gdstk.boolean(polygons, [], "or", precision=PRECISION_UM) if polygons else []


def _area_overlap(a, b):
    return sum(p.area() for p in gdstk.boolean(a, b, "and", precision=PRECISION_UM))


def _bbox(polygon):
    (x0, y0), (x1, y1) = polygon.bounding_box()
    return [round(v, 9) for v in (x0, y0, x1, y1)]


def _read_devices(path):
    devices = {}
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if not fields or not re.fullmatch(r"X\d+", fields[0]):
            continue
        instance, drain, gate, source, body, model = fields[:6]
        params = dict(token.split("=", 1) for token in fields[6:] if "=" in token)
        devices[instance] = {
            "instance": instance, "drain": drain, "gate": gate,
            "source": source, "body": body, "model": model,
            "w_um": float(params["w"]), "l_um": float(params["l"]),
        }
    _require(set(devices) == {f"X{i}" for i in range(8)}, "Expected exactly X0 through X7")
    return devices


def verify(gds_path, netlist_path, cell_name="sram_sp_cell"):
    devices = _read_devices(netlist_path)
    lib = gdstk.read_gds(str(gds_path))
    cell = next(c for c in lib.cells if c.name == cell_name)
    by = {}
    for polygon in cell.get_polygons():
        by.setdefault((polygon.layer, polygon.datatype), []).append(polygon)

    poly = _union(by[(66, 20)])
    diff = _union(by[(65, 20)])
    channels = gdstk.boolean(diff, poly, "and", precision=PRECISION_UM)
    sd = gdstk.boolean(diff, poly, "not", precision=PRECISION_UM)
    _require(len(channels) == 8, "Expected eight channel overlaps")
    nodes, layers = [], {}
    for layer_name, polygons in [
        ("poly", poly), ("sd", sd), ("li1", _union(by[(67, 20)])),
        ("met1", _union(by[(68, 20)])), ("met2", _union(by[(69, 20)])),
    ]:
        layers[layer_name] = []
        for polygon in polygons:
            layers[layer_name].append(len(nodes))
            nodes.append((layer_name, polygon))
    parents = list(range(len(nodes)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def join(i, j):
        parents[root(i)] = root(j)

    bridges = []
    for bridge_name, key, lower, upper in [
        ("licon1", (66, 44), ["poly", "sd"], ["li1"]),
        ("mcon", (67, 44), ["li1"], ["met1"]),
        ("via", (68, 44), ["met1"], ["met2"]),
    ]:
        for polygon in by[key]:
            lo = [i for name in lower for i in layers[name]
                  if _area_overlap(polygon, nodes[i][1]) > AREA_EPS_UM2]
            hi = [i for name in upper for i in layers[name]
                  if _area_overlap(polygon, nodes[i][1]) > AREA_EPS_UM2]
            _require(lo and hi, f"Unconnected {bridge_name} at {_bbox(polygon)}")
            for i in lo + hi:
                join(i, lo[0])
            bridges.append({"layer": bridge_name, "bbox_um": _bbox(polygon),
                            "lower_nodes": lo, "upper_nodes": hi})

    names, port_evidence = {}, []

    def name_net(net, name):
        _require(net not in names or names[net] == name,
                 f"Conflicting logical names for physical component {net}")
        names[net] = name

    aliases = {"bl": "BL", "br": "BR", "vpwr": "VDD", "vgnd": "VSS", "wl": "WL"}
    for label in cell.get_labels():
        key = (label.layer, label.texttype)
        if label.text.lower() not in aliases or key not in ((68, 5), (66, 5)):
            continue
        layer_name = "met1" if label.layer == 68 else "poly"
        hits = [i for i in layers[layer_name]
                if gdstk.inside([label.origin], [nodes[i][1]])[0]]
        _require(len(hits) == 1, f"Port {label.text} does not identify exactly one conductor")
        net = root(hits[0])
        name_net(net, aliases[label.text.lower()])
        port_evidence.append({"port": aliases[label.text.lower()],
                              "position_um": list(label.origin), "component": net})
    _require({"BL", "BR", "VDD", "VSS", "WL"} <= set(names.values()), "Missing named ports")

    extracted = []
    for polygon in sorted(channels, key=_bbox):
        bbox = _bbox(polygon)
        x0, y0, x1, y1 = bbox
        _require(abs(polygon.area() - (x1 - x0) * (y1 - y0)) < AREA_EPS_UM2,
                 "Cell-specific audit requires the known rectangular channel overlaps")
        gates = [i for i in layers["poly"] if _area_overlap(polygon, nodes[i][1]) > AREA_EPS_UM2]
        expanded = gdstk.offset(polygon, ADJACENCY_UM, precision=PRECISION_UM)
        terminals = [i for i in layers["sd"] if _area_overlap(expanded, nodes[i][1]) > AREA_EPS_UM2]
        _require(len(gates) == 1 and len(terminals) in (1, 2), "Unexpected channel adjacency")
        nwell_area = _area_overlap(polygon, by[(64, 20)])
        if abs(nwell_area - polygon.area()) < AREA_EPS_UM2:
            body, mos_type = "VPB", "pmos"
        else:
            # The small (64,44) shape is a VNB port marker, not the full
            # p-body footprint. This known cell's NMOS body is implicit
            # outside nwell; do not turn the port marker into a well mask.
            _require(nwell_area < AREA_EPS_UM2, "Channel partially crosses the nwell boundary")
            body, mos_type = "VNB", "nmos"
        extracted.append({
            "bbox_um": bbox, "centroid_um": [round((x0+x1)/2, 9), round((y0+y1)/2, 9)],
            "area_um2": round(polygon.area(), 12), "w_um": round(x1-x0, 9),
            "l_um": round(y1-y0, 9), "body": body, "mos_type": mos_type,
            "gate_component": root(gates[0]),
            "adjacent_sd_nodes": terminals,
            "adjacent_sd_components": [root(i) for i in terminals],
        })

    # Named bitlines establish Q/QB from the bundled access-device netlist.
    access_gate_components = set()
    for channel in extracted:
        known = [names.get(net) for net in channel["adjacent_sd_components"]]
        bitlines = set(known) & {"BL", "BR"}
        if not bitlines:
            continue
        _require(channel["mos_type"] == "nmos" and len(bitlines) == 1 and len(known) == 2,
                 "Unexpected bitline access geometry")
        bitline = next(iter(bitlines))
        matches = [d for d in devices.values() if bitline in (d["drain"], d["source"]) and d["gate"] == "WL"]
        _require(len(matches) == 1, "Netlist does not identify one access device per bitline")
        storage = next(n for n in (matches[0]["drain"], matches[0]["source"]) if n != bitline)
        unknown = [net for net in channel["adjacent_sd_components"] if names.get(net) != bitline]
        _require(len(unknown) == 1, "Cannot establish the storage node")
        name_net(unknown[0], storage)
        # A NETLIST CONTRACT, not an assertion that these poly stripes join locally.
        name_net(channel["gate_component"], "WL")
        access_gate_components.add(channel["gate_component"])
    _require({"Q", "QB"} <= set(names.values()) and len(access_gate_components) == 2,
             "Expected two distinct access-gate components and two storage nodes")

    result = []
    for channel in extracted:
        gate = names.get(channel["gate_component"])
        terminal_names = [names.get(net) for net in channel["adjacent_sd_components"]]
        _require(gate is not None and all(terminal_names), "Unidentified terminal net")
        short_edge = len(terminal_names) == 1
        if short_edge:
            _require(channel["mos_type"] == "pmos" and abs(channel["l_um"] - 0.025) < 1e-8
                     and gate == "WL", "Unexpected one-sided diffusion overlap")
            logical_ds = terminal_names * 2
        else:
            logical_ds = terminal_names
        matches = [d for d in devices.values()
                   if d["gate"] == gate and sorted([d["drain"], d["source"]]) == sorted(logical_ds)
                   and d["body"] == channel["body"]
                   and abs(d["w_um"] - channel["w_um"]) < 1e-8
                   and abs(d["l_um"] - channel["l_um"]) < 1e-8]
        _require(len(matches) == 1, f"Ambiguous or missing instance for {channel}")
        device = matches[0]
        _require(("nfet" in device["model"]) == (channel["mos_type"] == "nmos"), "MOS type mismatch")
        role = ("parasitic" if short_edge else "pullup" if channel["mos_type"] == "pmos"
                else "access" if gate == "WL" else "latch")
        result.append({**channel, **device, "device_class": role,
                       "adjacent_sd_nets": terminal_names,
                       "short_ds_from_netlist_contract": short_edge})
    _require({d["instance"] for d in result} == set(devices) and len(result) == len(devices),
             "Mapping is not one-to-one")
    return {
        "cell_name": cell_name, "method": "netlist-assisted original-mask connectivity; not full LVS",
        "gds_path": str(Path(gds_path).resolve()), "netlist_path": str(Path(netlist_path).resolve()),
        "gds_sha256": hashlib.sha256(Path(gds_path).read_bytes()).hexdigest(),
        "netlist_sha256": hashlib.sha256(Path(netlist_path).read_bytes()).hexdigest(),
        "contracts": [
            "Two locally disconnected access/poly stripes are the same logical WL in the bundled netlist; only the bottom stripe has a local WL label.",
            "Each 25 nm PMOS edge overlap touches one storage diffusion; the bundled extracted netlist assigns that same net to both D and S.",
            "Drain/source ordering is taken from the netlist; geometric terminal matching uses the unordered pair.",
            "PMOS channels lie wholly in nwell; NMOS channels lie outside nwell in the implicit p-body. The (64,44) VNB port marker is not a full pwell mask.",
        ],
        "precision_um": PRECISION_UM, "adjacency_probe_um": ADJACENCY_UM,
        "ports": port_evidence, "contact_bridges": bridges,
        "conductor_nodes": [
            {"node": i, "layer": layer_name, "component": root(i),
             "logical_net": names.get(root(i)), "bbox_um": _bbox(polygon),
             "polygon_um": polygon.points.tolist()}
            for i, (layer_name, polygon) in enumerate(nodes)
        ],
        "physical_conductor_component_count": len({root(i) for i in range(len(nodes))}),
        "access_gate_components": sorted(access_gate_components),
        "channels": sorted(result, key=lambda d: int(d["instance"][1:])),
    }


def check_manifest(report, manifest_path):
    """Compare a fresh connectivity derivation with the committed correspondence."""
    import sys

    sys.path.insert(0, str(ROOT))
    from gds.bitcell_mapping import geometry_fingerprint

    manifest = json.loads(Path(manifest_path).read_text())
    _require(manifest["cell_name"] == report["cell_name"], "Manifest cell name changed")
    # The full-source digest also protects label text/positions, which are not
    # contained in a polygon-only geometry fingerprint.
    _require(manifest["source_gds_sha256"] == report["gds_sha256"],
             "Original GDS file changed, including possible label/port changes")
    lib = gdstk.read_gds(report["gds_path"])
    cell = next(c for c in lib.cells if c.name == report["cell_name"])
    by_layer = {}
    for polygon in cell.get_polygons():
        by_layer.setdefault((polygon.layer, polygon.datatype), []).append(polygon)
    _require(geometry_fingerprint(by_layer) == manifest["geometry_sha256"],
             "Manifest original-mask geometry fingerprint changed")
    for relative_path, expected in manifest["netlist_sha256"].items():
        actual = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
        _require(actual == expected, f"Manifest netlist hash mismatch: {relative_path}")
    _require(report["netlist_sha256"] in manifest["netlist_sha256"].values(),
             "Audit used a netlist not covered by the manifest")
    saved = {record["instance"]: record for record in manifest["channels"]}
    fresh = {record["instance"]: record for record in report["channels"]}
    _require(len(saved) == len(manifest["channels"]) == 8 and set(saved) == set(fresh),
             "Manifest instance list is not the independently verified bijection")
    for instance in sorted(fresh):
        for key in ("bbox_um", "centroid_um", "device_class", "gate", "drain", "source",
                    "body", "model", "w_um", "l_um", "short_ds_from_netlist_contract"):
            _require(saved[instance].get(key) == fresh[instance][key],
                     f"Manifest correspondence differs for {instance}.{key}")
        _require(sorted(saved[instance]["adjacent_sd_nets"]) == sorted(fresh[instance]["adjacent_sd_nets"]),
                 f"Manifest exact-mask diffusion adjacency differs for {instance}")
    report["manifest_check"] = {"path": str(Path(manifest_path).resolve()), "passed": True}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gds", type=Path, default=ROOT / "data/sram22_64x22m4w22.gds")
    parser.add_argument("--netlist", type=Path, default=ROOT / "data/sram_sp_cell.spice")
    parser.add_argument("--cell", default="sram_sp_cell")
    parser.add_argument("--output", type=Path, help="Write the evidence report as JSON; otherwise print JSON")
    parser.add_argument("--check-manifest", type=Path, nargs="?",
                        const=ROOT / "data/sram_sp_cell_channel_map.json",
                        help="Check the saved correspondence and fingerprints (optional alternate JSON path)")
    args = parser.parse_args()
    report = verify(args.gds, args.netlist, args.cell)
    if args.check_manifest is not None:
        check_manifest(report, args.check_manifest)
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
        print(f"Verified all {len(report['channels'])} instances; wrote {args.output}")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
