"""Deposited FEM source energy must match the independent electrical event."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from post.access_energy import verify_access_source_energy


@pytest.fixture
def event():
    duration = 2e-10
    powers = {f"X{i}": (i + 1) * 1e-7 for i in range(8)}
    pulse = SimpleNamespace(duration_s=duration,
                            device_energy_j={d: p * duration for d, p in powers.items()})
    audit = {"per_device": {d: {"meshed_w": p} for d, p in powers.items()},
             "meshed_total_w": sum(powers.values())}
    return pulse, audit, [duration / 8] * 8


def test_actual_timesteps_preserve_every_devices_event_energy(event):
    pulse, audit, steps = event
    report = verify_access_source_energy(pulse, audit, steps)
    assert report["active_step_count"] == 8
    assert report["deposited_total_j"] == pytest.approx(sum(pulse.device_energy_j.values()), abs=0)
    for device, values in report["per_device"].items():
        assert values["deposited_j"] == pytest.approx(pulse.device_energy_j[device], abs=0)


def test_wrong_pulse_duration_is_rejected(event):
    pulse, audit, steps = event
    with pytest.raises(ValueError, match="duration"):
        verify_access_source_energy(pulse, audit, steps * 2)


def test_swapped_devices_fail_even_when_total_energy_is_correct(event):
    pulse, audit, steps = event
    audit = deepcopy(audit)
    powers = audit["per_device"]
    powers["X0"], powers["X7"] = powers["X7"], powers["X0"]
    with pytest.raises(ValueError, match="X0.*deposited energy"):
        verify_access_source_energy(pulse, audit, steps)


def test_unaccounted_domain_source_energy_is_rejected(event):
    pulse, audit, steps = event
    audit["meshed_total_w"] *= 2
    with pytest.raises(ValueError, match="Total source"):
        verify_access_source_energy(pulse, audit, steps)


def test_missing_device_energy_is_rejected(event):
    pulse, audit, steps = event
    del audit["per_device"]["X7"]
    with pytest.raises(ValueError, match="every prescribed device"):
        verify_access_source_energy(pulse, audit, steps)


@pytest.mark.parametrize("steps", [[], [0], [-1], [float("nan")], [float("inf")]])
def test_invalid_powered_timesteps_are_rejected(event, steps):
    pulse, audit, _ = event
    with pytest.raises(ValueError, match="timesteps"):
        verify_access_source_energy(pulse, audit, steps)
