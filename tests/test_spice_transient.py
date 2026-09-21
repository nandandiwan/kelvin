"""Electrical access validation, dissipation extraction and energy integration."""

import json

import numpy as np
import pytest

import gds.spice_transient as transient
from gds.spice_netlist import VDD
from gds.spice_power import find_ngspice


DEVICE_IDS = {f"X{i}" for i in range(8)}
PULSE_NS = 0.239519


@pytest.fixture(scope="module")
def electrical_runs():
    try:
        find_ngspice()
    except FileNotFoundError:
        pytest.skip("ngspice is not installed")
    return {
        (operation, initial, step): transient.run_access_transient(
            operation, stored_one=initial, pulse_width_ns=PULSE_NS, max_step_ps=step,
        )
        for operation in ("read", "write")
        for initial in (False, True)
        for step in (1.0, 0.5, 0.25)
    }


@pytest.mark.parametrize("operation", ["read", "write"])
@pytest.mark.parametrize("initial", [False, True])
def test_real_access_completes_and_preserves_device_power(electrical_runs, operation, initial):
    result = electrical_runs[operation, initial, 1.0]
    report, curves, meta = result.report, result.curves, result.metadata
    assert report["completed"] and report["operation_verified"]
    assert not report["thermal_solver_connected"]
    assert result.time_s[0] == 0
    assert result.time_s[-1] == pytest.approx(meta["end_s"], rel=1e-9, abs=0)
    assert np.all(np.diff(result.time_s) > 0)
    assert len(result.time_s) > 100
    assert (curves["v_q"][0] > VDD / 2) == initial
    assert (curves["v_qb"][0] > VDD / 2) != initial
    final = initial if operation == "read" else not initial
    assert (curves["v_q"][-1] > VDD / 2) == final
    assert (curves["v_qb"][-1] > VDD / 2) != final
    assert set(result.device_channel_power_w) == DEVICE_IDS
    assert set(result.device_junction_power_w) == DEVICE_IDS
    for powers in (result.device_channel_power_w, result.device_junction_power_w):
        for power in powers.values():
            assert power.shape == result.time_s.shape
            assert np.all(np.isfinite(power))
            assert np.all(power >= 0)
    for device in ("X3", "X4"):
        # These extracted parasitic MOS instances have D=S: gate displacement
        # current must not be misclassified as their channel heat.
        assert np.all(result.device_channel_power_w[device] == 0)
    assert report["channel_energy_j"] > 0
    assert sum(result.device_energy_j().values()) == pytest.approx(
        report["channel_energy_j"], rel=1e-14, abs=0,
    )
    if operation == "read":
        for node in ("v_bl", "v_br"):
            assert curves[node][0] == pytest.approx(VDD, rel=0.01)
            assert np.interp(meta["access_start_s"], result.time_s, curves[node]) == pytest.approx(
                VDD, rel=0.01,
            )
        assert report["read_differential_v"] >= transient.SENSE_MARGIN_V
        assert report["bitline_stored_energy_change_j"] < 0


@pytest.mark.parametrize("operation", ["read", "write"])
@pytest.mark.parametrize("initial", [False, True])
def test_real_access_timestep_convergence(electrical_runs, operation, initial):
    fine = electrical_runs[operation, initial, 0.25]
    fine_energy = fine.report["channel_energy_j"]
    fine_peak = sum(fine.device_channel_power_w.values()).max()
    for step in (1.0, 0.5):
        coarse = electrical_runs[operation, initial, step]
        assert coarse.report["channel_energy_j"] == pytest.approx(fine_energy, rel=1e-3, abs=0)
        assert sum(coarse.device_channel_power_w.values()).max() == pytest.approx(
            fine_peak, rel=5e-3, abs=0,
        )


@pytest.mark.parametrize("operation", ["read", "write"])
def test_opposite_data_states_have_equal_total_energy(electrical_runs, operation):
    zero = electrical_runs[operation, False, 0.5]
    one = electrical_runs[operation, True, 0.5]
    assert zero.report["channel_energy_j"] == pytest.approx(
        one.report["channel_energy_j"], rel=1e-6, abs=0,
    )
    # Instance identity is preserved: the active READ access device switches.
    if operation == "read":
        assert zero.device_energy_j()["X2"] > zero.device_energy_j()["X0"] * 100
        assert one.device_energy_j()["X0"] > one.device_energy_j()["X2"] * 100


def test_linear_trace_analytic_integral_with_interpolated_edges():
    times = np.array([0.0, 0.3, 1.2, 2.0])
    values = 2 * times + 1
    assert transient.integrate_trace(times, values) == pytest.approx(6.0)
    assert transient.integrate_trace(times, values, 0.2, 1.7) == pytest.approx(4.35)


def test_non_aligned_thermal_steps_conserve_each_devices_energy():
    times, pulse = np.array([0.0, 1.0, 3.0]), np.array([0.0, 2.0, 0.0])
    channels = {f"X{i}": (i + 1) * pulse for i in range(8)}
    result = transient.SpiceAccessResult({}, {"time": times}, channels, {}, {}, {})
    assert transient.integrate_trace(times, pulse, 0.3, 1.8) == pytest.approx(2.19)
    edges = [0.0, 0.11, 0.8, 1.33, 1.91, 3.0]
    delivered = dict.fromkeys(channels, 0.0)
    for start, end in zip(edges[:-1], edges[1:]):
        powers = result.average_device_power_w(start, end)
        for device in delivered:
            delivered[device] += powers[device] * (end - start)
    for device, expected in result.device_energy_j().items():
        assert expected > 0
        assert delivered[device] == pytest.approx(expected, rel=1e-14, abs=0)


@pytest.mark.parametrize("times,values,start,end", [
    ([0, 0, 1], [1, 1, 1], None, None),
    ([0, 1, 0.5], [1, 1, 1], None, None),
    ([0, 1], [1], None, None),
    ([0], [1], None, None),
    ([0, np.nan], [1, 1], None, None),
    ([0, 1], [1, np.inf], None, None),
    ([0, 1], [1, 1], -0.1, 1),
    ([0, 1], [1, 1], 0, 1.1),
    ([0, 1], [1, 1], 0.5, 0.5),
    ([0, 1], [1, 1], np.nan, 1),
])
def test_integrate_trace_rejects_invalid_trace_or_window(times, values, start, end):
    with pytest.raises(ValueError):
        transient.integrate_trace(times, values, start, end)


@pytest.mark.parametrize("contents", [
    "time other\n0 0\n1 1\n",
    "time v_q\n0\n1\n",
    "time v_q\n0 0\n0.8 1\n",  # aborted before requested final time
    "time v_q\n0.1 0\n1 1\n",  # missing initialization sample
    "time v_q\n0 0\n0.5 1\n0.5 1\n1 1\n",
    "time v_q\n0 0\n0.7 1\n0.6 1\n1 1\n",
    "time v_q\n0 0\n1 nan\n",
    "time v_q\n0 0\n1 inf\n",
    "time v_q\n0 0\n",
])
def test_reader_rejects_invalid_or_incomplete_output(tmp_path, contents):
    path = tmp_path / "waveform.txt"
    path.write_text(contents)
    with pytest.raises(ValueError):
        transient.read_access_waveform(path, {"columns": ["time", "v_q"], "end_s": 1.0})


def test_reader_accepts_complete_case_insensitive_header(tmp_path):
    path = tmp_path / "waveform.txt"
    path.write_text("TIME V_Q\n0 0\n0.4 0.2\n1 1.8\n")
    curves = transient.read_access_waveform(path, {"columns": ["time", "v_q"], "end_s": 1.0})
    np.testing.assert_array_equal(curves["time"], [0, 0.4, 1.0])
    np.testing.assert_array_equal(curves["v_q"], [0, 0.2, 1.8])


@pytest.mark.parametrize("fault,match", [
    ("geometry", "physical transistor dimension"),
    ("initial", "initialize"),
    ("final", "final complementary"),
    ("wordline", "wordline"),
    ("precharge_initial", "precharged"),
    ("precharge_at_access", "precharged"),
    ("read_signal", "differential"),
    ("read_disturb", "disturbed"),
])
def test_functional_validation_rejects_invalid_access(electrical_runs, fault, match):
    original = electrical_runs["read", True, 1.0]
    curves = {key: value.copy() for key, value in original.curves.items()}
    if fault == "geometry":
        curves["l_x4"] *= 1e6
    elif fault in {"initial", "final"}:
        index = 0 if fault == "initial" else -1
        curves["v_q"][index], curves["v_qb"][index] = 0, VDD
    elif fault == "wordline":
        curves["v_wl"][:] = 0
    elif fault == "precharge_initial":
        curves["v_bl"][0] = 0
    elif fault == "precharge_at_access":
        at_access = np.abs(curves["time"] - original.metadata["access_start_s"]) < 5e-12
        curves["v_bl"][at_access] = 0
    elif fault == "read_signal":
        curves["v_bl"][-1] = curves["v_br"][-1]
    else:
        curves["v_q"][len(curves["time"]) // 2] = 0
    with pytest.raises(ValueError, match=match):
        transient.validate_access_waveform(curves, original.metadata)


def test_negative_intrinsic_channel_current_is_rejected(electrical_runs):
    original = electrical_runs["read", True, 1.0]
    curves = {key: value.copy() for key, value in original.curves.items()}
    curves["id_x0"][10] = -1e-6
    with pytest.raises(ValueError, match="negative BSIM3 channel-current"):
        transient.access_result_from_waveform(curves, original.metadata)


def test_source_power_retains_delivery_and_absorption_sign(electrical_runs):
    original = electrical_runs["read", True, 1.0]
    curves = {key: value.copy() for key, value in original.curves.items()}
    curves["i_vvdd"][10:12] = [-1e-6, 1e-6]
    result = transient.access_result_from_waveform(curves, original.metadata)
    assert result.source_power_w["vvdd"][10] > 0
    assert result.source_power_w["vvdd"][11] < 0


def test_nonempty_output_directory_cannot_reuse_stale_success(tmp_path, monkeypatch):
    stale = tmp_path / "summary.json"
    original = json.dumps({"completed": True, "channel_energy_j": 1.0})
    stale.write_text(original)
    monkeypatch.setattr(transient, "_run", lambda _: pytest.fail("must refuse before running SPICE"))
    with pytest.raises(FileExistsError, match="not empty"):
        transient.run_access_transient("read", pulse_width_ns=PULSE_NS, out_dir=tmp_path)
    assert stale.read_text() == original


def test_failed_run_does_not_publish_success_summary(tmp_path, monkeypatch):
    def fail(_):
        raise RuntimeError("ngspice simulation aborted")

    monkeypatch.setattr(transient, "_run", fail)
    with pytest.raises(RuntimeError, match="aborted"):
        transient.run_access_transient("read", pulse_width_ns=PULSE_NS, out_dir=tmp_path)
    assert "aborted" in (tmp_path / "ngspice.log").read_text()
    assert not (tmp_path / "summary.json").exists()
    assert not (tmp_path / "device_power.npz").exists()
