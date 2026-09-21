"""Cell-specific distributed interconnect resistance, not foundry/signoff PEX.

Original masks and the independently verified terminal correspondence establish
connectivity. A clipped Cartesian finite-volume sheet network retains bends,
branches, contacts, and separate local transistor terminals. All geometries are
in original local GDS micrometres. No electrical capacitance is invented here:
the compact-device capacitances and the existing external bitline load remain
the capacitance model. In particular no extra CV² budget is added as heat.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import gdstk
import numpy as np

from cases.verify_bitcell_mapping import verify, check_manifest

ROOT = Path(__file__).resolve().parents[1]
REVISION = "sky130-bitcell-clipped-sheet-network-v1"
PRECISION_UM = 1e-6
AREA_EPS = 1e-11
# SKY130 rcx resistance table reports sheet/contact values in milliohms;
# convert the listed 48200/12800/125 and 15000/152000/4500 to ohms.
RESISTANCE_SOURCE = "https://skywater-pdk.readthedocs.io/en/main/rules/rcx.html#resistance-values"
SHEET_OHM = {"poly": 48.2, "li1": 12.8, "met1": 0.125, "met2": 0.125}
CONTACT_OHM = {"licon1": 15.0, "mcon": 152.0, "via": 4.5}
CONTACT_SIZE_UM = {"licon1": 0.17, "mcon": 0.17, "via": 0.15}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _intersection(a, b):
    return gdstk.boolean(a, b, "and", precision=PRECISION_UM)


def _area(polygons):
    return sum(p.area() for p in polygons)


def _center(polygon):
    """Area centroid, including nonrectangular clipped sheet cells."""
    xy = np.asarray(polygon.points)
    following = np.roll(xy, -1, axis=0)
    cross = xy[:, 0] * following[:, 1] - following[:, 0] * xy[:, 1]
    return np.sum((xy + following) * cross[:, None], axis=0) / (3 * cross.sum())


def _line_intervals(polygon, axis, value):
    """Positive-length polygon edges on a Cartesian tile interface."""
    points = polygon.points
    result = []
    for p, q in zip(points, np.roll(points, -1, axis=0)):
        if abs(p[axis] - value) < 2e-6 and abs(q[axis] - value) < 2e-6:
            a, b = sorted((p[1-axis], q[1-axis]))
            if b-a > 1e-9:
                result.append((a, b))
    return result


def _shared_length(a, b, axis, value):
    return sum(max(0.0, min(j, l)-max(i, k))
               for i, j in _line_intervals(a, axis, value)
               for k, l in _line_intervals(b, axis, value))


def sheet_network(polygon, sheet_ohm, step_um, *, origin_um=None):
    """Return clipped tiles and (i,j,R) edges for one sheet polygon.

    The two-point finite-volume approximation uses normal centroid separation
    divided by shared face length. It is exact for rectangular, aligned sheets;
    diagonal boundaries require refinement (the grid is not a field extractor).
    Polygon clipping, not bounding boxes, controls both area and connectivity.
    """
    _require(math.isfinite(step_um) and 0 < step_um <= 0.2,
             "Interconnect grid step must be finite and in (0, 0.2] um")
    _require(math.isfinite(sheet_ohm) and sheet_ohm > 0, "Sheet resistance must be positive")
    lo, hi = np.asarray(polygon.bounding_box())
    origin = np.asarray(origin_um if origin_um is not None else lo)
    start = np.floor((lo-origin)/step_um + 1e-8).astype(int)
    stop = np.ceil((hi-origin)/step_um - 1e-8).astype(int)
    tiles, grid = [], defaultdict(list)
    for i in range(start[0], stop[0]):
        for j in range(start[1], stop[1]):
            a = np.round(origin + np.array([i, j])*step_um, 6)
            b = np.round(origin + np.array([i+1, j+1])*step_um, 6)
            for part in _intersection(polygon, gdstk.rectangle(a, b)):
                if part.area() <= AREA_EPS:
                    continue
                grid[i, j].append(len(tiles))
                tiles.append({"polygon": part, "center": _center(part), "grid": (i, j)})
    _require(abs(sum(t["polygon"].area() for t in tiles)-polygon.area())
             < max(1e-8, polygon.area()*2e-5), "Sheet tiling lost polygon area")
    edges = []
    for (i, j), indices in grid.items():
        for axis, other in ((0, (i+1, j)), (1, (i, j+1))):
            coordinate = round(origin[axis] + (other[axis])*step_um, 6)
            for a in indices:
                for b in grid.get(other, []):
                    width = _shared_length(tiles[a]["polygon"], tiles[b]["polygon"], axis, coordinate)
                    if width <= 1e-9:
                        continue
                    distance = abs(tiles[b]["center"][axis]-tiles[a]["center"][axis])
                    _require(distance > 1e-9, "Degenerate finite-volume centroid spacing")
                    edges.append((a, b, float(sheet_ohm*distance/width)))
    return tiles, edges


class _Union:
    def __init__(self):
        self.parent = {}

    def root(self, node):
        self.parent.setdefault(node, node)
        if self.parent[node] != node:
            self.parent[node] = self.root(self.parent[node])
        return self.parent[node]

    def join(self, a, b):
        self.parent[self.root(b)] = self.root(a)


def network_fingerprint(network):
    """Content identity; safe to use for archived electrical/thermal provenance."""
    payload = {k: v for k, v in network.items() if k != "fingerprint"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def build_interconnect_network(step_um=0.05, gds_path=None, netlist_path=None,
                               cell_name="sram_sp_cell"):
    """Build an auditable original-mask resistor network for the bundled cell.

    ``port_nodes`` lists attachments that SPICE must alias to each external pin.
    ``logical_nodes`` gives Q/QB representatives for state checks, *not* ideal
    aliases across physically separated storage-node locations. Each resistor's
    heat_regions weights sum to one. Their physical overlap is intentional:
    several resistor losses can be deposited in the same thermal volume.
    """
    gds_path = Path(gds_path or ROOT/"data/sram22_64x22m4w22.gds")
    netlist_path = Path(netlist_path or ROOT/"data/sram_sp_cell.spice")
    report = verify(gds_path, netlist_path, cell_name)
    check_manifest(report, ROOT/"data/sram_sp_cell_channel_map.json")
    _require(math.isfinite(step_um) and .0125 <= step_um <= .2,
             "Bitcell interconnect grid step must be finite and in [0.0125, 0.2] um")
    conductors = {c["node"]: c for c in report["conductor_nodes"]}
    polygons = {i: gdstk.Polygon(c["polygon_um"]) for i, c in conductors.items()}
    nodes, resistors, tiles_by_conductor = [], [], {}
    aliases = _Union()
    excluded = []

    def add_node(layer, net, polygon=None, center=None, conductor_id=None):
        name = f"n{len(nodes):05d}"
        record = {"name": name, "layer": layer, "logical_net": net,
                  "conductor_id": conductor_id}
        if polygon is not None:
            record["polygon_um"] = polygon.points.tolist()
        if center is not None:
            record["center_um"] = [float(v) for v in center]
        nodes.append(record)
        aliases.root(name)
        return name

    def region(layer, polygon, weight=1.0, **extra):
        return {"layer": layer, "polygon_um": polygon.points.tolist(),
                "weight": weight, **extra}

    def resistor(a, b, resistance, heat, **extra):
        _require(math.isfinite(resistance) and resistance > 0, "Nonpositive resistor")
        resistors.append({"name": f"Rint{len(resistors):05d}", "node_a": a,
                          "node_b": b, "resistance_ohm": resistance,
                          "heat_regions": heat, **extra})

    for i, conductor in conductors.items():
        layer, net, polygon = conductor["layer"], conductor["logical_net"], polygons[i]
        if net is None:
            excluded.append({"conductor_id": i, "layer": layer,
                             "polygon_um": polygon.points.tolist(),
                             "reason": "No local contact or named terminal; external connectivity unknown"})
            continue
        if layer == "sd":
            name = add_node(layer, net, polygon, _center(polygon), i)
            tiles_by_conductor[i] = [{"name": name, "polygon": polygon,
                                      "center": _center(polygon), "grid": None}]
            continue
        tiles, edges = sheet_network(polygon, SHEET_OHM[layer], step_um)
        for tile in tiles:
            tile["name"] = add_node(layer, net, tile["polygon"], tile["center"], i)
        tiles_by_conductor[i] = tiles
        for a, b, resistance in edges:
            resistor(tiles[a]["name"], tiles[b]["name"], resistance,
                     [region(layer, tiles[a]["polygon"], .5),
                      region(layer, tiles[b]["polygon"], .5)], kind="sheet", layer=layer)

    # Contact cuts may occur twice in GDS. A coincident duplicate is one physical
    # contact, never an extra parallel path. The conducting core matches the
    # thermal whole-cut/exclusive-landing policy: cut AND lower AND upper.
    seen, contact_audit = set(), []
    for bridge in report["contact_bridges"]:
        layer = bridge["layer"]
        key = (layer, tuple(bridge["bbox_um"]))
        if key in seen:
            continue
        seen.add(key)
        lo_ids, hi_ids = bridge["lower_nodes"], bridge["upper_nodes"]
        _require(len(lo_ids) == len(hi_ids) == 1, "Ambiguous contact landing")
        lower, upper = lo_ids[0], hi_ids[0]
        net = conductors[lower]["logical_net"]
        _require(net is not None and net == conductors[upper]["logical_net"],
                 "Contact crosses logical nets")
        cut = gdstk.rectangle(bridge["bbox_um"][:2], bridge["bbox_um"][2:])
        core = _intersection(_intersection(cut, polygons[lower]), polygons[upper])
        core_area = _area(core)
        _require(core_area > AREA_EPS, "Empty conducting contact core")
        resistance = CONTACT_OHM[layer]*CONTACT_SIZE_UM[layer]**2/core_area
        hub = add_node(layer, net, cut, _center(cut))
        heat = [region(layer, p, p.area()/core_area,
                       lower_layer=conductors[lower]["layer"]) for p in core]
        # Each half of the vertical contact has total conductance 2/R. Area
        # weighted parallel branches connect the distributed landing to an
        # equipotential middle plane; shorting either landing yields exactly R.
        for side, conductor_id in (("lower", lower), ("upper", upper)):
            overlaps = [(tile, _area(_intersection(core, tile["polygon"])))
                        for tile in tiles_by_conductor[conductor_id]]
            overlaps = [(tile, area) for tile, area in overlaps if area > AREA_EPS]
            coverage = sum(area for _, area in overlaps)
            _require(abs(coverage-core_area) < max(1e-8, core_area*2e-5),
                     "Contact tile coverage is incomplete")
            for tile, area in overlaps:
                resistor(tile["name"], hub, resistance*coverage/(2*area), heat,
                         kind="contact", layer=layer, contact_index=len(contact_audit), side=side)
        contact_audit.append({"layer": layer, "bbox_um": bridge["bbox_um"],
                              "core_area_um2": core_area, "resistance_ohm": resistance,
                              "lower_layer": conductors[lower]["layer"]})

    def attach_cross_section(conductor_id, axis, coordinate):
        tiles = tiles_by_conductor[conductor_id]
        # Choose a finite-width cross section, not a point injection whose sheet
        # spreading resistance would diverge with grid refinement. Its width
        # tends to zero with step_um. Compact gates are terminals at gate centre.
        index = min(tiles, key=lambda t: abs(t["center"][axis]-coordinate))["grid"][axis]
        selected = [t["name"] for t in tiles if t["grid"][axis] == index]
        _require(selected, "Empty terminal cross section")
        for other in selected[1:]:
            aliases.join(selected[0], other)
        return selected[0]

    terminals = {}
    for device in report["channels"]:
        adjacent = {conductors[i]["logical_net"]: tiles_by_conductor[i][0]["name"]
                    for i in device["adjacent_sd_nodes"]}
        gate_ids = [i for i, c in conductors.items() if c["layer"] == "poly"
                    and c["component"] == device["gate_component"]]
        _require(len(gate_ids) == 1, "Ambiguous gate conductor")
        gate = attach_cross_section(gate_ids[0], 0, device["centroid_um"][0])
        terminals[device["instance"]] = {
            "drain": adjacent[device["drain"]], "gate": gate,
            "source": adjacent[device["source"]], "body": device["body"]}

    ports = {"VNB": ["VNB"], "VPB": ["VPB"]}
    for port in report["ports"]:
        if port["port"] == "WL":
            continue
        candidates = [i for i, c in conductors.items()
                      if c["layer"] == "met1" and c["component"] == port["component"]
                      and gdstk.inside([port["position_um"]], [polygons[i]])[0]]
        _require(len(candidates) == 1, "Named port does not identify one metal conductor")
        ports[port["port"]] = [attach_cross_section(candidates[0], 1, port["position_um"][1])]
    wl_x = next(p["position_um"][0] for p in report["ports"] if p["port"] == "WL")
    ports["WL"] = [attach_cross_section(i, 0, wl_x) for i, c in conductors.items()
                   if c["layer"] == "poly" and c["logical_net"] == "WL"]
    _require(len(ports["WL"]) == 2, "Expected the two explicit external WL attachments")
    for device in terminals.values():
        for key in ("drain", "gate", "source"):
            device[key] = aliases.root(device[key])
    ports = {key: sorted({aliases.root(n) for n in values}) for key, values in ports.items()}
    actual_resistors = []
    for r in resistors:
        r["node_a"], r["node_b"] = aliases.root(r["node_a"]), aliases.root(r["node_b"])
        if r["node_a"] != r["node_b"]:
            actual_resistors.append(r)
    nodes = [n for n in nodes if aliases.root(n["name"]) == n["name"]]
    net_by_name = {n["name"]: n["logical_net"] for n in nodes}
    graph = _Union()
    for r in actual_resistors:
        _require(net_by_name[r["node_a"]] == net_by_name[r["node_b"]], "Resistor shorts different logical nets")
        graph.join(r["node_a"], r["node_b"])
        _require(abs(sum(p["weight"] for p in r["heat_regions"])-1) < 1e-12,
                 "Resistor heat weights do not sum to one")
    for attachments in ports.values():
        for other in attachments[1:]:
            graph.join(attachments[0], other)
    roots = defaultdict(set)
    for node in nodes:
        roots[node["logical_net"]].add(graph.root(node["name"]))
    _require(all(len(group) == 1 for group in roots.values()),
             f"Disconnected distributed conductor network: {dict(roots)}")
    logical = {name: next(terminals[d["instance"]][role] for d in report["channels"]
                         for role in ("drain", "source") if d[role] == name)
               for name in ("Q", "QB")}
    network = {
        "revision": REVISION, "cell_name": cell_name, "step_um": float(step_um),
        "source_gds_sha256": report["gds_sha256"], "source_netlist_sha256": report["netlist_sha256"],
        "resistance_source": RESISTANCE_SOURCE, "sheet_resistance_ohm_per_square": SHEET_OHM.copy(),
        "nominal_contact_resistance_ohm": CONTACT_OHM.copy(),
        "nodes": nodes, "resistors": actual_resistors, "device_terminals": terminals,
        "port_nodes": ports, "logical_nodes": logical,
        "assumptions": report["contracts"] + [
            "Approximate local nominal resistance extraction, not foundry/signoff RC PEX or an array electrical solve.",
            "Sheet resistance uses the public SKY130 nominal table; specialised SRAM contact values are not separately calibrated.",
            "Clipped Cartesian two-point finite-volume sheets; refine step_um to assess diagonal-edge and terminal-location error.",
            "Drain/source diffusion is ideal locally; diffusion/well/substrate and compact-device internal series losses are not additionally extracted.",
            "Poly compact gates are equipotential cross sections at the channel x-centre; their finite numerical width shrinks with the grid.",
            "External metal pins attach to full conductor cross sections at the GDS label y-coordinate; external routing/driver loss is not local heat.",
            "Both disconnected WL stripes attach at the labelled stripe's x-coordinate through the explicit external WL contract.",
            "Contact resistance scales inversely with cut AND lower AND upper core area; half cuts retain only their local-cell share.",
            "Each contact uses an equipotential middle plane and area-weighted parallel half-resistances; no contact lateral spreading is resolved.",
            "Coincident duplicate contact polygons count once. Unconnected/unlabelled metal is excluded electrically but remains a thermal conductor.",
            "No layout capacitance extraction: compact capacitances and the existing external bitline load are retained; no extra CV-squared heat is added.",
            "Resistances are temperature independent; no electrothermal iteration is performed.",
        ],
        "audit": {"node_count": len(nodes), "resistor_count": len(actual_resistors),
                  "contact_count": len(contact_audit), "duplicate_contacts_removed": len(report["contact_bridges"])-len(contact_audit),
                  "contacts": contact_audit, "excluded_conductors": excluded,
                  "connected_logical_nets": sorted(roots)},
    }
    network["fingerprint"] = network_fingerprint(network)
    return network
