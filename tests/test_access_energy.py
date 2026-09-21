"""Averaged loads and selected-access pulses use exactly the same event energy."""

from dataclasses import replace

import pytest

from gds import spice_power
from gds.power import AccessTiming


@pytest.fixture
def compact_model(monkeypatch):
    # Asymmetric device powers expose accidental redistribution or averaging.
    raw = {f"X{i}": (i + 1) ** 2 * 1e-6 for i in range(8)}
    timing = AccessTiming(
        min_period_ns=4.0,
        min_pulse_width_high_ns=0.25,
        min_pulse_width_low_ns=0.3,
        clk_to_q_access_ns=4.5,
    )
    monkeypatch.setattr(spice_power, "bias_device_power_w", lambda *args: dict(raw))
    monkeypatch.setattr(spice_power, "selfconsistent_device_power_w", lambda *args: dict(raw))
    monkeypatch.setattr(spice_power, "mine_access_timing", lambda path: timing)
    return raw, timing


@pytest.mark.parametrize("name", ["crowbar", "read_1", "write_1_settled"])
@pytest.mark.parametrize("hit_rate", [0.0, 1.0 / 64.0, 0.4, 1.0])
def test_pulse_and_average_conserve_each_devices_energy(compact_model, name, hit_rate):
    pulse = spice_power.named_access_pulse(name)
    average = spice_power.named_bias_point_power_w(name, row_hit_rate=hit_rate)
    assert set(pulse.device_energy_j) == set(pulse.device_power_w) == set(average)
    for device, energy in pulse.device_energy_j.items():
        assert pulse.device_power_w[device] * pulse.duration_s == pytest.approx(energy, rel=2e-15, abs=0)
        assert average[device] * pulse.period_s == pytest.approx(energy * hit_rate, rel=2e-15, abs=0)
    assert average == pulse.average_power_w(hit_rate)


@pytest.mark.parametrize("name", ["crowbar", "read_1", "write_1_settled"])
@pytest.mark.parametrize("access_ns", [1.5, 4.5])
def test_existing_average_values_are_preserved(compact_model, monkeypatch, name, access_ns):
    raw, timing = compact_model
    timing = replace(timing, clk_to_q_access_ns=access_ns)
    monkeypatch.setattr(spice_power, "mine_access_timing", lambda path: timing)
    rate = 0.23
    if name == "read_1":
        total_average = spice_power.read_access_energy_j() * rate / (timing.min_period_ns * 1e-9)
        expected = {device: power / sum(raw.values()) * total_average
                    for device, power in raw.items()}
    else:
        width = timing.min_pulse_width_high_ns if name == "crowbar" else access_ns
        duty = min(1.0, width / timing.min_period_ns) * rate
        expected = {device: power * duty for device, power in raw.items()}
    actual = spice_power.named_bias_point_power_w(name, row_hit_rate=rate)
    assert actual == pytest.approx(expected, rel=2e-15, abs=0)


def test_crowbar_uses_short_high_time_not_access_period(compact_model):
    raw, timing = compact_model
    pulse = spice_power.named_access_pulse("crowbar")
    assert pulse.duration_s == timing.min_pulse_width_high_ns * 1e-9
    assert pulse.device_power_w == raw
    assert pulse.duration_s < pulse.period_s
    assert "Synthetic" in pulse.description


@pytest.mark.parametrize("name", ["crowbar", "read_1", "write_1_settled"])
def test_pulse_duration_is_capped_at_period(compact_model, monkeypatch, name):
    _, timing = compact_model
    timing = replace(timing, min_pulse_width_high_ns=8.0, clk_to_q_access_ns=8.0)
    monkeypatch.setattr(spice_power, "mine_access_timing", lambda path: timing)
    pulse = spice_power.named_access_pulse(name)
    assert pulse.duration_s == pulse.period_s == timing.min_period_ns * 1e-9


def test_read_rectangle_is_charge_bounded_not_dc_peak(compact_model):
    raw, _ = compact_model
    pulse = spice_power.named_access_pulse("read_1")
    assert pulse.total_energy_j == pytest.approx(spice_power.read_access_energy_j(), rel=2e-15)
    for device in raw:
        assert pulse.device_energy_j[device] / pulse.total_energy_j == pytest.approx(
            raw[device] / sum(raw.values()), rel=2e-15)
        assert pulse.device_power_w[device] < raw[device]
    assert "energy-equivalent" in pulse.description
    assert "not the raw DC peak" in pulse.description


def test_hold_remains_sustained_and_rejects_pulse(compact_model):
    raw, _ = compact_model
    for rate in (0.0, 0.25, 1.0):
        assert spice_power.named_bias_point_power_w("hold_1", row_hit_rate=rate) == raw
    with pytest.raises(ValueError, match="sustained state"):
        spice_power.named_access_pulse("hold_1")


@pytest.mark.parametrize("rate", [-0.1, 1.1, float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("name", ["crowbar", "hold_1"])
def test_invalid_row_hit_rate_is_rejected(compact_model, name, rate):
    with pytest.raises(ValueError, match="row_hit_rate"):
        spice_power.named_bias_point_power_w(name, row_hit_rate=rate)
    with pytest.raises(ValueError, match="row_hit_rate"):
        spice_power.named_access_pulse("crowbar").average_power_w(rate)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
@pytest.mark.parametrize("name,field", [
    ("crowbar", "min_period_ns"),
    ("crowbar", "min_pulse_width_high_ns"),
    ("read_1", "clk_to_q_access_ns"),
    ("write_1_settled", "clk_to_q_access_ns"),
])
def test_invalid_event_timings_are_rejected(compact_model, monkeypatch, name, field, value):
    _, timing = compact_model
    timing = replace(timing, **{field: value})
    monkeypatch.setattr(spice_power, "mine_access_timing", lambda path: timing)
    with pytest.raises(ValueError, match=field):
        spice_power.named_access_pulse(name)


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_invalid_dc_device_power_is_rejected(compact_model, monkeypatch, value):
    raw, _ = compact_model
    monkeypatch.setattr(spice_power, "selfconsistent_device_power_w", lambda *args: dict(raw, X3=value))
    with pytest.raises(ValueError, match="finite and nonnegative"):
        spice_power.named_access_pulse("read_1")


def test_missing_device_power_is_rejected(compact_model, monkeypatch):
    raw, _ = compact_model
    monkeypatch.setattr(spice_power, "selfconsistent_device_power_w", lambda *args: {
        device: power for device, power in raw.items() if device != "X7"
    })
    with pytest.raises(ValueError, match="exactly X0 through X7"):
        spice_power.named_access_pulse("read_1")


def test_zero_read_power_cannot_silently_discard_charge_energy(compact_model, monkeypatch):
    raw, _ = compact_model
    zeros = {device: 0.0 for device in raw}
    monkeypatch.setattr(spice_power, "selfconsistent_device_power_w", lambda *args: zeros)
    with pytest.raises(ValueError, match="positive total DC power"):
        spice_power.named_access_pulse("read_1")
    with pytest.raises(ValueError, match="positive total DC power"):
        spice_power.named_bias_point_power_w("read_1")
    assert spice_power.named_access_pulse("write_1_settled").total_energy_j == 0.0


def test_caller_lib_path_reaches_timing_model(compact_model, monkeypatch):
    _, timing = compact_model
    seen = []

    def timing_from(path):
        seen.append(path)
        return timing

    monkeypatch.setattr(spice_power, "mine_access_timing", timing_from)
    spice_power.named_access_pulse("read_1", lib_path="custom.lib")
    spice_power.named_bias_point_power_w("crowbar", lib_path="custom.lib")
    assert seen == ["custom.lib", "custom.lib"]
