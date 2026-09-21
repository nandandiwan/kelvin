"""Exact subcell wire heat projection, conservation, and actual-field audits."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("dolfinx")
from dolfinx import mesh as dmesh
from dolfinx.fem import Function
from mpi4py import MPI

from mesh.boxes import Region
from mesh.gds_import import ImportedHeatSource
from physics.device_sources import DEVICE_NAMES
from physics.interconnect_sources import (
    CombinedHeatSources, InterconnectHeatSources, _convex_parts, _polygon_area,
    _prism, _tetra_prism_volume,
)


def rectangle(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def resistor(name, polygon, layer="li1", **extras):
    return {"name": name, "resistance_ohm": 10, "node_a": "a", "node_b": "b",
            "heat_regions": [{"layer": layer, "polygon_um": polygon, "weight": 1, **extras}]}


def registry_with(regions):
    return SimpleNamespace(all=lambda: list(regions))


@pytest.fixture(scope="module")
def fixture():
    # Eight independent channel tags, one wire tag, and one unused background.
    mesh = dmesh.create_box(MPI.COMM_SELF, [[0, 0, 0], [10e-6, 1e-6, 1e-6]], [10, 1, 1])
    cells = np.arange(mesh.topology.index_map(3).size_local, dtype=np.int32)
    midpoints = dmesh.compute_midpoints(mesh, 3, cells)
    tags = np.floor(midpoints[:, 0] / 1e-6).astype(np.int32) + 1
    mesh_data = SimpleNamespace(mesh=mesh, cell_tags=dmesh.meshtags(mesh, 3, cells, tags))
    regions = [Region(i + 1, f"channel_X{i}", "Si_channel",
                      ImportedHeatSource(f"channel_X{i}", 0, volume_m3=1e-18,
                                         intended_power_w=0, device=f"X{i}", kind="channel"))
               for i in range(8)]
    regions.extend((Region(9, "li1", "TiN"), Region(10, "background", "Si_bulk")))
    manifest = {"regions": [{"name": "li1", "role": "conductor", "physical_tag": 9,
                              "z_min_um": 0, "z_max_um": 1}]}
    network = {"resistors": [resistor("R_left", rectangle(8, 0, 8.15, 1)),
                              resistor("R_right", rectangle(8.8, 0, 9, 1)),
                              resistor("R_overlap", rectangle(8, 0, 8.15, 1))]}
    return mesh_data, registry_with(regions), manifest, network


@pytest.fixture
def source(fixture):
    return InterconnectHeatSources(*fixture)


def devices(scale=1e-7):
    return {name: (i + 1) * scale for i, name in enumerate(DEVICE_NAMES)}


def test_prism_intersection_known_volumes_and_touching():
    tet = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    whole = _prism(np.array(rectangle(-1, -1, 2, 2)), -1, 2)
    assert _tetra_prism_volume(tet, whole) == pytest.approx(1 / 6, rel=1e-12)
    half_x = _prism(np.array(rectangle(.5, -1, 2, 2)), -1, 2)
    assert _tetra_prism_volume(tet, half_x) == pytest.approx(1 / 48, rel=1e-12)
    outside = _prism(np.array(rectangle(1, -1, 2, 2)), -1, 2)
    assert _tetra_prism_volume(tet, outside) == 0


def test_concave_polygon_triangulates_without_bbox_filling():
    polygon = [[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]]
    parts = _convex_parts(polygon)
    assert len(parts) > 1
    assert sum(_polygon_area(p) for p in parts) == pytest.approx(3)
    assert sum(_polygon_area(p) for p in _convex_parts(polygon[::-1])) == pytest.approx(3)


def test_subcell_and_overlapping_resistor_powers_conserved(source):
    field = source.q
    values = {"R_left": 2e-7, "R_right": 5e-7, "R_overlap": 9e-7}
    assert source.set_powers(values) is field
    actual = source.integrated_powers()
    assert actual["total_w"] == pytest.approx(sum(values.values()), rel=1e-12)
    assert actual["per_layer"]["li1"] == pytest.approx(sum(values.values()), rel=1e-12)
    assert len(source.volume_audit) == 2  # Reused overlap support is projected only once.
    assert np.all(field.x.array[source._cell_dofs[source._tags != 9]] == 0)
    assert all(v["ratio"] == pytest.approx(1, rel=1e-12) for v in source.volume_audit.values())
    source.set_powers(dict.fromkeys(values, 0))
    assert not np.any(field.x.array)


def test_small_polygon_without_any_mesh_centroid_still_deposits(fixture):
    data, registry, manifest, _ = fixture
    tiny = {"resistors": [resistor("R_tiny", rectangle(8.001, .001, 8.002, .002))]}
    source = InterconnectHeatSources(data, registry, manifest, tiny)
    source.set_powers({"R_tiny": 1e-9})
    assert source.integrated_powers()["total_w"] == pytest.approx(1e-9, rel=1e-9)
    assert np.count_nonzero(source.q.x.array) > 0


def test_actual_energy_audit_matches_each_timestep_and_detects_spatial_tampering(source):
    qdt = Function(source.q.function_space)
    expected = dict.fromkeys(source.resistor_names, 0.)
    for dt, values in ((2e-12, {"R_left": 2e-5, "R_right": 1e-6, "R_overlap": 0}),
                       (7e-11, {"R_left": 2e-7, "R_right": 0, "R_overlap": 9e-7})):
        qdt.x.array[:] += source.set_powers(values).x.array * dt
        for name, power in values.items():
            expected[name] += power * dt
    audit = source.audit_energy(qdt, expected)
    assert audit["deposited_total_j"] == pytest.approx(sum(expected.values()), rel=1e-12)
    assert "projected_j" in audit["per_interconnect"]["R_left"]
    # Alter location while keeping total joules exactly unchanged.
    wire_cells = np.flatnonzero(source._tags == 9)
    a, b = wire_cells[:2]
    energy = 1e-18
    qdt.x.array[source._cell_dofs[a]] += energy / source._cell_volumes[a]
    qdt.x.array[source._cell_dofs[b]] -= energy / source._cell_volumes[b]
    with pytest.raises(ValueError, match="spatially projected|nonnegative"):
        source.audit_energy(qdt, expected)


def test_combined_field_audit_and_power_units(fixture):
    source = CombinedHeatSources(*fixture)
    wire_power = {"R_left": 2e-7, "R_right": 5e-7, "R_overlap": 9e-7}
    channel_power = devices()
    field = source.set_powers(channel_power, wire_power)
    report = source.audit_powers(field, channel_power, wire_power)
    total = sum(channel_power.values()) + sum(wire_power.values())
    assert report["meshed_total_w"] == pytest.approx(total, rel=1e-12)
    assert report["electrical_total_w"] == pytest.approx(total, rel=1e-12)
    assert report["per_device"]["X3"]["meshed_w"] == pytest.approx(channel_power["X3"])
    qdt = Function(field.function_space)
    qdt.x.array[:] = field.x.array * 1e-9
    report = source.audit_energy(qdt, {k: v * 1e-9 for k, v in channel_power.items()},
                                 {k: v * 1e-9 for k, v in wire_power.items()})
    assert report["deposited_total_j"] == pytest.approx(total * 1e-9, rel=1e-12)
    assert report["interconnects"]["deposited_total_j"] == pytest.approx(sum(wire_power.values()) * 1e-9)
    source.set_powers(devices(0), dict.fromkeys(wire_power, 0))
    assert not np.any(source.q.x.array)


def test_combined_audit_rejects_wire_energy_deposited_in_background(fixture):
    source = CombinedHeatSources(*fixture)
    wire_power = {"R_left": 2e-7, "R_right": 5e-7, "R_overlap": 9e-7}
    source.set_powers(devices(), wire_power)
    wrong = Function(source.q.function_space)
    wrong.x.array[:] = source.q.x.array
    background = source.interconnects._cell_dofs[source.interconnects._tags == 10]
    wrong.x.array[background] = 1e9
    with pytest.raises(ValueError, match="spatially projected"):
        source.audit_energy(wrong, devices(), wire_power)


def test_small_wire_budget_cannot_hide_under_large_channel_budget(fixture):
    source = CombinedHeatSources(*fixture)
    channel_power = devices(1.)
    wire_power = {"R_left": 1e-12, "R_right": 2e-12, "R_overlap": 3e-12}
    source.set_powers(channel_power, wire_power)
    wrong = Function(source.q.function_space)
    wrong.x.array[:] = source.devices.q.x.array  # All wire watts disappeared.
    with pytest.raises(ValueError, match="spatially projected"):
        source.audit_powers(wrong, channel_power, wire_power)


@pytest.mark.parametrize("bad", [-1., float("nan"), float("inf")])
def test_invalid_powers_do_not_change_source(source, bad):
    valid = {"R_left": 2e-7, "R_right": 5e-7, "R_overlap": 9e-7}
    source.set_powers(valid)
    previous = source.q.x.array.copy()
    valid["R_left"] = bad
    with pytest.raises(ValueError, match="finite and nonnegative"):
        source.set_powers(valid)
    np.testing.assert_array_equal(previous, source.q.x.array)


def test_incomplete_power_dictionary_rejected(source):
    with pytest.raises(ValueError, match="exactly the network resistor names"):
        source.set_powers({"R_left": 1e-8})


@pytest.mark.parametrize("change,match", [
    ("wrong_layer", "Missing or ambiguous"),
    ("outside", "does not match nominal"),
    ("wrong_weights", "sum to one"),
    ("negative_resistance", "finite and positive"),
    ("duplicate", "distinct"),
    ("missing_licon_landing", "explicit poly"),
])
def test_invalid_networks_fail_closed(fixture, change, match):
    data, registry, manifest, original = fixture
    network = deepcopy(original)
    item = network["resistors"][0]
    if change == "wrong_layer":
        item["heat_regions"][0]["layer"] = "met99"
    elif change == "outside":
        item["heat_regions"][0]["polygon_um"] = rectangle(8.9, 0, 9.1, 1)
    elif change == "wrong_weights":
        item["heat_regions"][0]["weight"] = .5
    elif change == "negative_resistance":
        item["resistance_ohm"] = -1
    elif change == "duplicate":
        network["resistors"][1]["name"] = item["name"]
    else:
        item["heat_regions"][0]["layer"] = "licon1"
    with pytest.raises(ValueError, match=match):
        InterconnectHeatSources(data, registry, manifest, network)


def test_updates_do_not_intersect_or_compile_forms(source, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Waveform updates must only multiply the cached sparse projection")
    monkeypatch.setattr("physics.interconnect_sources._tetra_prism_volume", unexpected)
    monkeypatch.setattr("physics.device_sources.form", unexpected)
    source.set_powers({"R_left": 2e-7, "R_right": 5e-7, "R_overlap": 9e-7})


def test_disconnected_wires_with_same_layer_tag_are_not_smeared():
    mesh = dmesh.create_box(MPI.COMM_SELF, [[0, 0, 0], [3e-6, 1e-6, 1e-6]], [3, 1, 1])
    cells = np.arange(mesh.topology.index_map(3).size_local, dtype=np.int32)
    centers = dmesh.compute_midpoints(mesh, 3, cells)
    tags = np.where((centers[:, 0] < 1e-6) | (centers[:, 0] > 2e-6), 9, 10).astype(np.int32)
    data = SimpleNamespace(mesh=mesh, cell_tags=dmesh.meshtags(mesh, 3, cells, tags))
    registry = registry_with([Region(9, "li1", "TiN"), Region(10, "oxide", "SiO2")])
    manifest = {"regions": [{"name": "li1", "role": "conductor", "physical_tag": 9,
                              "z_min_um": 0, "z_max_um": 1}]}
    network = {"resistors": [resistor("R", rectangle(.1, .1, .2, .2))]}
    source = InterconnectHeatSources(data, registry, manifest, network)
    source.set_powers({"R": 1e-7})
    remote_dofs = source._cell_dofs[centers[:, 0] > 2e-6]
    assert len(remote_dofs)
    assert np.all(source.q.x.array[remote_dofs] == 0)
    assert source.integrated_powers()["total_w"] == pytest.approx(1e-7, rel=1e-12)


def test_concave_region_exact_volume_and_weighted_split(fixture):
    data, registry, manifest, _ = fixture
    concave = [[8, 0], [9, 0], [9, .5], [8.5, .5], [8.5, 1], [8, 1]]
    network = {"resistors": [resistor("R", concave)]}
    network["resistors"][0]["heat_regions"][0]["weight"] = .4
    network["resistors"][0]["heat_regions"].append(
        {"layer": "li1", "polygon_um": rectangle(8.8, .8, 9, 1), "weight": .6})
    source = InterconnectHeatSources(data, registry, manifest, network)
    source.set_powers({"R": 1e-7})
    assert source.volume_audit["region_0"]["nominal_m3"] == pytest.approx(.75e-18)
    assert source.integrated_powers()["total_w"] == pytest.approx(1e-7, rel=1e-12)


def test_real_notebook_mesh_projects_each_electrical_layer():
    from gds.interconnect import build_interconnect_network
    from mesh.sram import load_sram_mesh

    path = Path(__file__).resolve().parents[1] / "out/sram_notebook_mesh/sram_sp_cell_material_regions.json"
    if not path.is_file():
        pytest.skip("Generate the sealed notebook SRAM mesh to test its real source polygons")
    case = load_sram_mesh(path, devices(0))
    network = build_interconnect_network()
    selected = {}
    for item in network["resistors"]:
        region = item["heat_regions"][0]
        key = (region["layer"], region.get("lower_layer"))
        selected.setdefault(key, item)
    # This fast regression samples every process layer and both LICON heights.
    # A full2452-resistor projection is additionally checked in end-to-end runs.
    assert {key[0] for key in selected} == {"poly", "li1", "met1", "met2", "licon1", "mcon", "via"}
    source = CombinedHeatSources(case.mesh_data, case.registry, case.manifest,
                                 {"resistors": list(selected.values())})
    values = {name: (i + 1) * 1e-9 for i, name in enumerate(source.resistor_names)}
    source.set_powers(devices(), values)
    audit = source.audit_powers(source.q, devices(), values)
    assert audit["meshed_total_w"] == pytest.approx(sum(devices().values()) + sum(values.values()), rel=1e-6)
    assert len(audit["interconnects"]["per_layer"]) == 8
