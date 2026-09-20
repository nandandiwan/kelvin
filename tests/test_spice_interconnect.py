"""Layout-linked electrical dissipation, local voltages and conservative traces."""

import hashlib
import json

import numpy as np
import pytest

from gds import spice_netlist as decks
from gds import spice_transient as transient
from gds.spice_power import find_ngspice


def _minimal_network():
    devices = {
        "X0": ("QB", "WL", "BR", "VNB"), "X1": ("Q", "QB", "VSS", "VNB"),
        "X2": ("bl_local", "WL", "Q", "VNB"), "X3": ("Q", "WL", "Q", "VPB"),
        "X4": ("QB", "WL", "QB", "VPB"), "X5": ("VDD", "Q", "QB", "VPB"),
        "X6": ("Q", "QB", "VDD", "VPB"), "X7": ("VSS", "Q", "QB", "VNB"),
    }
    return {
        "revision": "test", "nodes": [], "step_um": .05,
        "port_nodes": {p: [p] for p in ("BL", "BR", "VDD", "VSS", "WL", "VNB", "VPB")},
        "logical_nodes": {"Q": "Q", "QB": "QB"},
        "device_terminals": {name: dict(zip(("drain", "gate", "source", "body"), nodes))
                             for name, nodes in devices.items()},
        "resistors": [{"name": "Rtest", "node_a": "BL", "node_b": "bl_local",
                       "resistance_ohm": 100.0,
                       "heat_regions": [{"layer": "met1", "weight": 1.0}]}],
    }


def _valid_curves():
    curves = {
        "time": np.array([0.0, .5, 1.0]),
        "v_vdd": np.full(3, 1.8), "v_vss": np.zeros(3),
        "v_q": np.full(3, 1.8), "v_qb": np.zeros(3),
        "v_wl": np.array([0.0, 1.8, 0.0]),
        "v_bl": np.full(3, 1.8), "v_br": np.array([1.8, 1.4, 1.02]),
    }
    for i in range(8):
        curves[f"l_x{i}"] = np.full(3, .025e-6 if i in (3, 4) else .15e-6)
        curves[f"w_x{i}"] = np.full(3, .21e-6 if i in (1, 7) else .14e-6)
        curves[f"id_x{i}"] = np.zeros(3)
        curves[f"ibd_x{i}"] = np.zeros(3)
        curves[f"ibs_x{i}"] = np.zeros(3)
    metadata = {
        "initial_stored_one": True, "final_stored_one": True, "operation": "read",
        "access_start_s": 0.0, "source_voltage_columns": {}, "source_current_columns": {},
        "interconnect": "layout", "interconnect_network": _minimal_network(),
        "interconnect_voltage_columns": {"Rtest": "dv_rtest"},
        "device_terminal_columns": {"X0": {"drain": "vd_local", "source": "vs_local", "body": "vb_local"}},
    }
    curves.update(vd_local=np.full(3, .1), vs_local=np.full(3, .9), vb_local=np.zeros(3),
                  dv_rtest=np.array([0.0, .5, 0.0]))
    curves["id_x0"][:] = 1e-4
    return curves, metadata


def test_channel_heat_uses_actual_local_terminal_voltages():
    curves, metadata = _valid_curves()
    result = transient.access_result_from_waveform(curves, metadata)
    np.testing.assert_allclose(result.device_channel_power_w["X0"], 8e-5)
    assert not np.allclose(result.device_channel_power_w["X0"],
                           curves["id_x0"] * abs(curves["v_qb"] - curves["v_br"]))


def test_resistor_heat_voltage_squared_and_all_source_energy_conserved():
    curves, metadata = _valid_curves()
    result = transient.access_result_from_waveform(curves, metadata)
    np.testing.assert_array_equal(result.interconnect_power_w["Rtest"], [0, .0025, 0])
    assert result.interconnect_energy_j()["Rtest"] == pytest.approx(.00125)
    edges = [0, .11, .39, .68, .99, 1]
    energy = sum(result.average_interconnect_power_w(a, b)["Rtest"] * (b - a)
                 for a, b in zip(edges[:-1], edges[1:]))
    assert energy == pytest.approx(.00125, rel=1e-14)
    assert result.report["interconnect_layer_energy_j"] == {"met1": pytest.approx(.00125)}
    assert result.report["total_local_heat_energy_j"] == pytest.approx(.00125 + 8e-5)
    assert set(result.interconnect_power_w) == {"Rtest"}  # no external RWL/RBL/RBR


def test_old_synthetic_result_constructor_remains_compatible():
    result = transient.SpiceAccessResult({}, {"time": np.array([0, 1])}, {}, {}, {}, {})
    assert result.interconnect_power_w == {}
    assert result.interconnect_energy_j() == {}


@pytest.mark.parametrize("resistance", [0, -1, np.nan, np.inf])
def test_invalid_local_resistance_rejected(resistance):
    curves, metadata = _valid_curves()
    metadata["interconnect_network"]["resistors"][0]["resistance_ohm"] = resistance
    with pytest.raises(ValueError, match="finite and positive"):
        transient.access_result_from_waveform(curves, metadata)


def test_missing_branch_voltage_is_not_silently_omitted():
    curves, metadata = _valid_curves()
    metadata["interconnect_voltage_columns"] = {}
    with pytest.raises(ValueError, match="Every local interconnect resistor"):
        transient.access_result_from_waveform(curves, metadata)


@pytest.mark.parametrize("weight", [-1, 0, .5, 2, np.nan])
def test_incomplete_or_invalid_heat_assignment_is_rejected(weight):
    curves, metadata = _valid_curves()
    metadata["interconnect_network"]["resistors"][0]["heat_regions"][0]["weight"] = weight
    with pytest.raises(ValueError, match="weights"):
        transient.access_result_from_waveform(curves, metadata)


def test_layout_failure_uses_bounded_extended_timeout_without_success_marker(tmp_path, monkeypatch):
    pytest.importorskip("gdstk")
    from gds import interconnect
    monkeypatch.setattr(interconnect, "build_interconnect_network", lambda **_: _minimal_network())
    calls = []
    def fail(deck_path, timeout_s):
        calls.append(timeout_s)
        raise RuntimeError("ngspice layout simulation aborted")
    monkeypatch.setattr(transient, "_run", fail)
    with pytest.raises(RuntimeError, match="aborted"):
        transient.run_access_transient("read", pulse_width_ns=.239519,
                                       interconnect="layout", out_dir=tmp_path)
    assert calls == [300]
    assert not (tmp_path / "summary.json").exists()
    assert not (tmp_path / "interconnect_power.npz").exists()
    assert json.loads((tmp_path / "metadata.json").read_text())["ngspice_timeout_s"] == 300


def test_cell_include_keeps_compact_geometry_and_reconnects_local_nodes(tmp_path):
    path = tmp_path / "interconnect.spice"
    terminals, resistors, logical = decks._write_interconnect_cell(path, _minimal_network())
    text = path.read_text()
    assert ".SUBCKT sram_sp_cell_layout BL BR VDD VSS WL VNB VPB" in text
    assert "X2 bl_local WL Q VNB sky130_fd_pr__special_nfet_pass l=0.150 w=0.140" in text
    assert "X3 Q WL Q VPB sky130_fd_pr__special_pfet_pass_shortl l=0.025 w=0.140" in text
    assert "Rtest BL bl_local 100" in text
    assert terminals["X2"]["drain"] == "bl_local"
    assert resistors[0]["spice_node_a"] == "BL"
    assert logical == {"Q": "Q", "QB": "QB"}


def test_distinct_port_short_is_rejected(tmp_path):
    network = _minimal_network()
    network["port_nodes"]["BR"].append("BL")
    with pytest.raises(ValueError, match="shorts distinct external ports"):
        decks._write_interconnect_cell(tmp_path / "bad.spice", network)


@pytest.mark.parametrize("mode,step", [
    ("none", .025), ("none", np.nan), ("none", np.inf),
    ("layout", .001), ("layout", .201), ("layout", -1), ("layout", 0),
    ("layout", np.nan), ("layout", np.inf), ("layout", True),
    ("layout", None), ("layout", "invalid"),
])
def test_invalid_or_ignored_interconnect_step_fails_before_any_files(tmp_path, mode, step):
    with pytest.raises(ValueError, match="interconnect_step_um"):
        decks.write_access_deck(tmp_path / "bad.sp", tmp_path / "waveform.txt",
                                interconnect=mode, interconnect_step_um=step)
    assert not list(tmp_path.iterdir())


def test_invalid_interconnect_model_fails_before_any_files(tmp_path):
    with pytest.raises(ValueError, match="interconnect"):
        decks.write_access_deck(tmp_path / "bad.sp", tmp_path / "waveform.txt",
                                interconnect="magic")
    assert not list(tmp_path.iterdir())


@pytest.fixture(scope="module")
def layout_runs(tmp_path_factory):
    pytest.importorskip("gdstk")
    try:
        find_ngspice()
    except FileNotFoundError:
        pytest.skip("ngspice is not installed")
    parent = tmp_path_factory.mktemp("layout-spice")
    return {
        (operation, initial, step): transient.run_access_transient(
            operation, stored_one=initial, pulse_width_ns=.239519, max_step_ps=step,
            interconnect="layout", out_dir=parent / f"{operation}_{int(initial)}_{step}",
        )
        for operation in ("read", "write")
        for initial in (False, True)
        for step in (1.0, .5)
    }


@pytest.mark.parametrize("operation", ["read", "write"])
@pytest.mark.parametrize("initial", [False, True])
def test_layout_access_completes_with_nonzero_local_joule_heating(layout_runs, operation, initial):
    result = layout_runs[operation, initial, 1.0]
    assert result.report["completed"] and result.report["operation_verified"]
    assert result.report["interconnect_energy_j"] > 0
    assert result.report["channel_energy_j"] > 0
    assert result.report["interconnect_energy_j"] < result.report["channel_energy_j"]
    assert set(result.interconnect_power_w) == {
        resistor["name"] for resistor in result.metadata["interconnect_network"]["resistors"]}
    assert np.isfinite(np.array(list(result.interconnect_power_w.values()))).all()
    assert (np.array(list(result.interconnect_power_w.values())) >= 0).all()
    assert sum(result.report["interconnect_layer_energy_j"].values()) == pytest.approx(
        result.report["interconnect_energy_j"], rel=1e-12)
    assert set(result.metadata["device_terminal_columns"]) == {f"X{i}" for i in range(8)}


@pytest.mark.parametrize("operation", ["read", "write"])
@pytest.mark.parametrize("initial", [False, True])
def test_layout_access_temporal_convergence(layout_runs, operation, initial):
    coarse, fine = (layout_runs[operation, initial, step] for step in (1.0, .5))
    for key in ("channel_energy_j", "interconnect_energy_j"):
        assert coarse.report[key] == pytest.approx(fine.report[key], rel=2e-3, abs=0)


def test_layout_archives_complete_electrical_and_geometry_evidence(layout_runs):
    from pathlib import Path
    result = layout_runs["read", True, 1.0]
    directory = Path(result.metadata["deck_path"]).parent
    for filename in ("interconnect.spice", "interconnect_network.json", "interconnect_power.npz"):
        assert (directory / filename).is_file()
    assert json.loads((directory / "interconnect_network.json").read_text()) == result.metadata["interconnect_network"]
    assert hashlib.sha256((directory / "interconnect.spice").read_bytes()).hexdigest() == result.metadata["interconnect_spice_file_sha256"]
    assert hashlib.sha256((directory / "interconnect_network.json").read_bytes()).hexdigest() == result.metadata["interconnect_network_file_sha256"]
    with np.load(directory / "interconnect_power.npz") as archive:
        assert set(archive.files) == {"time_s", *result.interconnect_power_w}
