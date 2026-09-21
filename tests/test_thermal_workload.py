"""SPICE event/steady-power identity and persisted provenance for heat drivers."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

import gds.thermal_workload as workload
from gds.spice_power import find_ngspice
from gds.spice_transient import SpiceAccessResult


def synthetic_trace():
    times = np.array([0.0, 0.13e-9, 0.67e-9, 0.939519e-9])
    powers = {f"X{i}": (i + 1) * np.array([1e-12, 2e-6, 3e-6, 1e-12])
              for i in range(8)}
    powers["X3"] *= 0
    powers["X4"] *= 0
    return SpiceAccessResult(
        {"assumptions": ["synthetic electrical event"]}, {"time": times}, powers,
        {}, {}, {"completed": True, "operation_verified": True},
    )


@pytest.fixture
def mocked_spice(monkeypatch, tmp_path):
    trace = synthetic_trace()
    calls = []

    def run(operation, **kwargs):
        calls.append((operation, kwargs))
        directory = kwargs["out_dir"]
        if directory is not None:
            directory = tmp_path / directory
            directory.mkdir(exist_ok=True)
            for name in workload._ARTIFACT_NAMES:
                (directory / name).write_text(f"synthetic {name}\n")
        return trace

    monkeypatch.setattr(workload, "run_access_transient", run)
    return trace, calls


@pytest.mark.parametrize("point", workload.SPICE_ACCESS_POINTS)
@pytest.mark.parametrize("rate", [0, 0.125, 1])
def test_each_device_average_is_its_own_event_energy_times_rate(mocked_spice, point, rate):
    trace, calls = mocked_spice
    result = workload.build_spice_workload(
        point, rate, period_ns=2, pulse_width_ns=0.21, spice_max_step_ps=0.5,
    )
    operation, initial = workload.SPICE_ACCESS_POINTS[point]
    assert calls[0][0] == operation
    assert calls[0][1]["stored_one"] is initial
    assert calls[0][1]["pulse_width_ns"] == 0.21
    assert calls[0][1]["max_step_ps"] == 0.5
    assert result.trace is trace
    assert result.row_hit_rate == rate
    assert result.period_s == 2e-9
    for device, expected in trace.device_energy_j().items():
        assert result.device_energy_j[device] == expected
        assert result.device_average_power_w[device] == pytest.approx(expected * rate / 2e-9)
    assert result.metadata["kind"] == "spice-transient"
    assert result.metadata["thermal_power_revision"] == workload.THERMAL_POWER_REVISION
    assert result.metadata["initial_stored_one"] is initial
    assert result.metadata["final_stored_one"] is (initial if operation == "read" else not initial)
    assert result.metadata["trace_duration_s"] == trace.time_s[-1]
    assert result.metadata["device_energy_j"] == result.device_energy_j
    assert result.metadata["device_average_power_w"] == result.device_average_power_w
    assert result.metadata["timing_lib_sha256"] is None
    assert result.metadata["spice_artifact_sha256"] == {}
    json.dumps(result.metadata, allow_nan=False)


def test_default_timing_is_mined_once_and_fingerprinted(mocked_spice, monkeypatch, tmp_path):
    lib_path = tmp_path / "timing.lib"
    lib_path.write_text("synthetic timing fixture\n")
    timing_calls = []

    def timing(path):
        timing_calls.append(path)
        return SimpleNamespace(min_period_ns=3.86681, min_pulse_width_high_ns=0.239519)

    monkeypatch.setattr(workload, "mine_access_timing", timing)
    result = workload.build_spice_workload(lib_path=lib_path)
    assert timing_calls == [lib_path]
    assert result.period_s == pytest.approx(3.86681e-9)
    assert result.metadata["pulse_width_s"] == pytest.approx(0.239519e-9)
    assert result.metadata["timing_lib_path"] == str(lib_path.resolve())
    assert result.metadata["timing_lib_sha256"] == hashlib.sha256(lib_path.read_bytes()).hexdigest()


def test_rate_and_period_change_average_not_single_event_or_waveform(mocked_spice):
    first = workload.build_spice_workload(period_ns=2, pulse_width_ns=0.21)
    second = workload.build_spice_workload(row_hit_rate=0.25, period_ns=4, pulse_width_ns=0.21)
    assert first.device_energy_j == second.device_energy_j
    assert first.metadata["power_trace_sha256"] == second.metadata["power_trace_sha256"]
    for device in first.device_energy_j:
        assert second.device_average_power_w[device] == pytest.approx(
            first.device_average_power_w[device] * 0.125,
        )


def test_power_fingerprint_detects_energy_or_instance_distribution_change(mocked_spice):
    trace, _ = mocked_spice
    first = workload.build_spice_workload(period_ns=2, pulse_width_ns=0.21)
    trace.device_channel_power_w["X0"][1] *= 2
    second = workload.build_spice_workload(period_ns=2, pulse_width_ns=0.21)
    assert first.metadata["power_trace_sha256"] != second.metadata["power_trace_sha256"]
    assert first.device_energy_j["X0"] != second.device_energy_j["X0"]


def test_layout_workload_conserves_each_resistor_and_fingerprints_geometry(mocked_spice):
    trace, calls = mocked_spice
    trace.interconnect_power_w = {"Rwire0": np.array([0, 2e-7, 1e-7, 0]),
                                  "Rcontact0": np.array([0, 7e-7, 4e-7, 0])}
    trace.metadata["interconnect_network"] = {
        "resistors": [{"name": name, "resistance_ohm": 10.0}
                      for name in trace.interconnect_power_w]}
    result = workload.build_spice_workload(period_ns=2, pulse_width_ns=0.21,
                                           interconnect="layout", row_hit_rate=0.25)
    assert calls[-1][1]["interconnect"] == "layout"
    for name, energy in trace.interconnect_energy_j().items():
        assert result.interconnect_energy_j[name] == energy
        assert result.interconnect_average_power_w[name] == pytest.approx(energy * 0.25 / 2e-9)
    assert result.metadata["total_local_heat_energy_j"] == pytest.approx(
        sum(result.device_energy_j.values()) + sum(result.interconnect_energy_j.values()))
    first_hash = result.metadata["power_trace_sha256"]
    trace.interconnect_power_w["Rwire0"][1] *= 2
    second_hash = workload._power_trace_sha256(trace)
    assert second_hash != first_hash
    trace.metadata["interconnect_network"]["resistors"][0]["resistance_ohm"] = 20.0
    assert workload._power_trace_sha256(trace) != second_hash


@pytest.mark.parametrize("fault", ["missing_network", "missing_trace", "extra_trace", "negative", "nan"])
def test_incomplete_interconnect_cannot_silently_be_omitted(mocked_spice, fault):
    trace, _ = mocked_spice
    trace.interconnect_power_w = {"Rwire0": np.array([0, 2e-7, 1e-7, 0])}
    trace.metadata["interconnect_network"] = {"resistors": [{"name": "Rwire0"}]}
    if fault == "missing_network":
        del trace.metadata["interconnect_network"]
    elif fault == "missing_trace":
        trace.interconnect_power_w.clear()
    elif fault == "extra_trace":
        trace.interconnect_power_w["Rextra"] = np.zeros(4)
    else:
        trace.interconnect_power_w["Rwire0"][1] = -1 if fault == "negative" else np.nan
    with pytest.raises(ValueError):
        workload.build_spice_workload(period_ns=2, pulse_width_ns=0.21, interconnect="layout")


@pytest.mark.parametrize("options", [{"interconnect": "invented"},
                                    {"interconnect_step_um": 0},
                                    {"interconnect_step_um": np.nan},
                                    {"interconnect_step_um": 0.02}])
def test_invalid_interconnect_options_fail_before_spice(mocked_spice, options):
    _, calls = mocked_spice
    with pytest.raises(ValueError, match="interconnect"):
        workload.build_spice_workload(period_ns=2, pulse_width_ns=0.21, **options)
    assert not calls


@pytest.mark.parametrize("point", ["hold_1", "crowbar", "write_1_settled", "write_1", "unknown"])
def test_dc_names_require_explicit_surrogate(mocked_spice, point):
    _, calls = mocked_spice
    with pytest.raises(ValueError, match="--power-model dc-surrogate"):
        workload.build_spice_workload(point, period_ns=2, pulse_width_ns=0.21)
    assert not calls


@pytest.mark.parametrize("rate", [-1, 1.01, np.nan, np.inf, None, "invalid", True])
def test_invalid_rate_is_rejected_before_electrical_run(mocked_spice, rate):
    _, calls = mocked_spice
    with pytest.raises(ValueError, match="row_hit_rate"):
        workload.build_spice_workload(row_hit_rate=rate, period_ns=2, pulse_width_ns=0.21)
    assert not calls


@pytest.mark.parametrize("name", ["period_ns", "pulse_width_ns", "spice_max_step_ps"])
@pytest.mark.parametrize("value", [0, -1, np.inf, np.nan, True, "invalid"])
def test_invalid_timing_is_rejected_before_electrical_run(mocked_spice, name, value):
    _, calls = mocked_spice
    kwargs = {"period_ns": 2, "pulse_width_ns": 0.21, "spice_max_step_ps": 1}
    kwargs[name] = value
    with pytest.raises(ValueError, match=name):
        workload.build_spice_workload(**kwargs)
    assert not calls


@pytest.mark.parametrize("rate", [0, 0.01, 1])
def test_period_cannot_overlap_completed_trace_even_at_low_activity(mocked_spice, rate):
    with pytest.raises(ValueError, match="shorter than the complete SPICE trace"):
        workload.build_spice_workload(row_hit_rate=rate, period_ns=0.5, pulse_width_ns=0.21)


def test_period_may_equal_observation_window(mocked_spice):
    trace, _ = mocked_spice
    result = workload.build_spice_workload(period_ns=trace.time_s[-1] * 1e9, pulse_width_ns=0.21)
    for device, energy in result.device_energy_j.items():
        assert result.device_average_power_w[device] == pytest.approx(energy / trace.time_s[-1])


@pytest.mark.parametrize("fault", ["failed", "unverified", "missing_device", "negative", "nonfinite"])
def test_incomplete_or_invalid_access_cannot_feed_heat_solver(mocked_spice, fault):
    trace, _ = mocked_spice
    if fault == "failed":
        trace.report["completed"] = False
    elif fault == "unverified":
        trace.report["operation_verified"] = False
    elif fault == "missing_device":
        del trace.device_channel_power_w["X4"]
    else:
        trace.device_channel_power_w["X0"][1] = -1 if fault == "negative" else np.nan
    with pytest.raises(ValueError):
        workload.build_spice_workload(period_ns=2, pulse_width_ns=0.21)


def test_archived_artifacts_are_all_fingerprinted(mocked_spice, tmp_path):
    directory = tmp_path / "spice"
    result = workload.build_spice_workload(period_ns=2, pulse_width_ns=0.21, out_dir=directory)
    assert result.metadata["spice_artifact_dir"] == str(directory)
    assert set(result.metadata["spice_artifact_sha256"]) == set(workload._ARTIFACT_NAMES)
    for name, fingerprint in result.metadata["spice_artifact_sha256"].items():
        assert fingerprint == hashlib.sha256((directory / name).read_bytes()).hexdigest()


def test_electrical_model_and_netlist_contents_are_fingerprinted(mocked_spice, monkeypatch, tmp_path):
    model, netlist = tmp_path / "model.spice", tmp_path / "cell.spice"
    model.write_text("model parameters\n")
    netlist.write_text("transistor connections\n")
    monkeypatch.setattr(workload, "_INCLUDES", [model, netlist])
    result = workload.build_spice_workload(period_ns=2, pulse_width_ns=0.21)
    assert result.metadata["spice_input_sha256"] == {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (model, netlist)
    }


@pytest.fixture(scope="module")
def real_workloads(tmp_path_factory):
    try:
        find_ngspice()
    except FileNotFoundError:
        pytest.skip("ngspice is not installed")
    directory = tmp_path_factory.mktemp("thermal-spice-workloads")
    return {
        point: workload.build_spice_workload(
            point, row_hit_rate=0.25, period_ns=3.86681, pulse_width_ns=0.239519,
            spice_max_step_ps=1.0, out_dir=directory / point,
        ) for point in workload.SPICE_ACCESS_POINTS
    }


@pytest.mark.parametrize("point", workload.SPICE_ACCESS_POINTS)
def test_real_access_preserves_each_devices_energy_identity(real_workloads, point):
    result = real_workloads[point]
    assert result.trace.report["completed"] and result.trace.report["operation_verified"]
    for device, energy in result.trace.device_energy_j().items():
        assert result.device_energy_j[device] == pytest.approx(energy, rel=1e-14, abs=0)
        assert result.device_average_power_w[device] == pytest.approx(
            energy * result.row_hit_rate / result.period_s, rel=1e-14, abs=0,
        )
    directory = result.metadata["spice_artifact_dir"]
    assert len(result.metadata["spice_artifact_sha256"]) == 6
    assert directory is not None
    assert result.metadata["channel_energy_j"] > 0
    assert result.device_energy_j["X3"] == result.device_energy_j["X4"] == 0


def test_real_stored_state_changes_spatial_distribution(real_workloads):
    one, zero = real_workloads["read_1"], real_workloads["read_0"]
    assert one.device_energy_j["X0"] > 100 * one.device_energy_j["X2"]
    assert zero.device_energy_j["X2"] > 100 * zero.device_energy_j["X0"]
    assert sum(one.device_energy_j.values()) == pytest.approx(
        sum(zero.device_energy_j.values()), rel=1e-6, abs=0,
    )


def test_real_pulse_width_override_changes_event_not_just_average(real_workloads):
    reference = real_workloads["read_1"]
    wider = workload.build_spice_workload(
        "read_1", row_hit_rate=0.25, period_ns=3.86681, pulse_width_ns=0.35,
    )
    assert wider.metadata["pulse_width_s"] == pytest.approx(0.35e-9)
    assert wider.trace.time_s[-1] > reference.trace.time_s[-1]
    assert wider.metadata["power_trace_sha256"] != reference.metadata["power_trace_sha256"]
    assert wider.metadata["channel_energy_j"] > reference.metadata["channel_energy_j"]
