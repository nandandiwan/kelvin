"""Electrical geometry checks independent of ngspice and the thermal solver."""

import math
from collections import defaultdict

import gdstk
import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from gds.interconnect import (CONTACT_OHM, CONTACT_SIZE_UM, SHEET_OHM,
                             build_interconnect_network, network_fingerprint,
                             sheet_network)


def effective_resistance(edges, high, low):
    """Two-terminal Dirichlet solve; other logical networks are excluded."""
    high, low = set(high), set(low)
    adjacency = defaultdict(set)
    for a, b, _ in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    connected, pending = set(high), list(high)
    while pending:
        for other in adjacency[pending.pop()]:
            if other not in connected:
                connected.add(other)
                pending.append(other)
    assert low <= connected
    unknown = {n: i for i, n in enumerate(sorted(connected-high-low))}
    rows, cols, data = [], [], []
    rhs = np.zeros(len(unknown))
    for a, b, resistance in edges:
        if a not in connected:
            continue
        conductance = 1/resistance
        for n, other in ((a, b), (b, a)):
            if n not in unknown:
                continue
            rows.append(unknown[n]); cols.append(unknown[n]); data.append(conductance)
            if other in unknown:
                rows.append(unknown[n]); cols.append(unknown[other]); data.append(-conductance)
            elif other in high:
                rhs[unknown[n]] += conductance
    matrix = coo_matrix((data, (rows, cols)), shape=(len(unknown), len(unknown))).tocsr()
    solution = spsolve(matrix, rhs) if unknown else []
    voltage = {**{n: 1.0 for n in high}, **{n: 0.0 for n in low},
               **{n: solution[i] for n, i in unknown.items()}}
    current = sum((voltage[a]-voltage[b])/r if a in high and b not in high
                  else (voltage[b]-voltage[a])/r if b in high and a not in high else 0
                  for a, b, r in edges if a in connected)
    return 1/current


@pytest.fixture(scope="module")
def network():
    return build_interconnect_network()


@pytest.mark.parametrize("step", [.1, .05, .025])
def test_rectangular_sheet_has_exact_known_resistance(step):
    tiles, edges = sheet_network(gdstk.rectangle((0, 0), (1, .2)), 12.8, step)
    left = [i for i, t in enumerate(tiles) if t["grid"][0] == 0]
    right = [i for i, t in enumerate(tiles) if t["grid"][0] == round(1/step)-1]
    # Terminal planes pass through first/last cell centres, not the outer edge.
    expected = 12.8*(1-step)/.2
    assert effective_resistance(edges, left, right) == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize("step", [0, -1, math.nan, math.inf, .3])
def test_invalid_grid_rejected(step):
    with pytest.raises(ValueError, match="grid step"):
        sheet_network(gdstk.rectangle((0, 0), (1, .2)), 1, step)


def test_nonconvex_clipping_preserves_area_not_bounding_box():
    polygon = gdstk.Polygon([(0, 0), (1, 0), (1, .2), (.2, .2), (.2, 1), (0, 1)])
    tiles, edges = sheet_network(polygon, 12.8, .05)
    assert sum(t["polygon"].area() for t in tiles) == pytest.approx(.36)
    assert all(r > 0 and np.isfinite(r) for _, _, r in edges)
    # All cells form one graph even at the elbow; no diagonal-corner shortcut.
    assert np.isfinite(effective_resistance(edges, [0], [len(tiles)-1]))


def test_nominal_values_and_unit_conversion():
    assert SHEET_OHM == {"poly": 48.2, "li1": 12.8, "met1": .125, "met2": .125}
    assert CONTACT_OHM == {"licon1": 15, "mcon": 152, "via": 4.5}


def test_network_is_connected_named_and_instance_preserving(network):
    assert set(network["device_terminals"]) == {f"X{i}" for i in range(8)}
    assert set(network["port_nodes"]) == {"BL", "BR", "VDD", "VSS", "WL", "VNB", "VPB"}
    assert len(network["port_nodes"]["WL"]) == 2
    assert network["audit"]["connected_logical_nets"] == ["BL", "BR", "Q", "QB", "VDD", "VSS", "WL"]
    assert network["device_terminals"]["X1"]["drain"] != network["device_terminals"]["X6"]["drain"]
    assert network["device_terminals"]["X3"]["drain"] == network["device_terminals"]["X3"]["source"]
    known = {n["name"] for n in network["nodes"]} | {"VNB", "VPB"}
    assert all(n in known for terminal in network["device_terminals"].values() for n in terminal.values())
    assert all(n in known for attachment in network["port_nodes"].values() for n in attachment)


def test_positive_resistors_and_exact_heat_partition(network):
    assert len({r["name"] for r in network["resistors"]}) == len(network["resistors"])
    for r in network["resistors"]:
        assert r["node_a"] != r["node_b"]
        assert math.isfinite(r["resistance_ohm"]) and r["resistance_ohm"] > 0
        assert sum(h["weight"] for h in r["heat_regions"]) == pytest.approx(1)
        assert all(gdstk.Polygon(h["polygon_um"]).area() > 0 for h in r["heat_regions"])


def test_contact_half_resistors_recover_nominal_core_resistance(network):
    for i, cut in enumerate(network["audit"]["contacts"]):
        layer = cut["layer"]
        expected = CONTACT_OHM[layer]*CONTACT_SIZE_UM[layer]**2/cut["core_area_um2"]
        assert cut["resistance_ohm"] == pytest.approx(expected)
        halves = []
        for side in ("lower", "upper"):
            conductance = sum(1/r["resistance_ohm"] for r in network["resistors"]
                              if r.get("contact_index") == i and r["side"] == side)
            halves.append(1/conductance)
        assert sum(halves) == pytest.approx(expected)
    assert network["audit"]["duplicate_contacts_removed"] == 1
    assert network["audit"]["contact_count"] == 16


def test_floating_metal_is_explicitly_excluded(network):
    excluded = network["audit"]["excluded_conductors"]
    assert len(excluded) == 1 and excluded[0]["layer"] == "met2"
    assert all(n["conductor_id"] != excluded[0]["conductor_id"] for n in network["nodes"])


def test_fingerprint_covers_resistance_and_geometry(network):
    assert network_fingerprint(network) == network["fingerprint"]
    changed = dict(network, step_um=.025)
    assert network_fingerprint(changed) != network["fingerprint"]


def test_grid_refinement_preserves_bitline_and_diagonal_storage_paths(network):
    refined = build_interconnect_network(step_um=.025)
    values = []
    for n in (network, refined):
        edges = [(r["node_a"], r["node_b"], r["resistance_ohm"]) for r in n["resistors"]]
        d = n["device_terminals"]
        values.append([
            effective_resistance(edges, n["port_nodes"]["BL"], [d["X2"]["drain"]]),
            effective_resistance(edges, [d["X1"]["drain"]], [d["X6"]["drain"]]),
            effective_resistance(edges, [d["X0"]["drain"]], [d["X5"]["source"]]),
        ])
    assert values[0] == pytest.approx(values[1], rel=.005)


@pytest.mark.parametrize("step", [.001, .21, float("nan")])
def test_bitcell_grid_resource_guard(step):
    with pytest.raises(ValueError, match="grid step"):
        build_interconnect_network(step)
