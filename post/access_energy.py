"""Audit a rectangular access against its independently prescribed event energy."""

import math


def verify_access_source_energy(pulse, source_power_audit, active_steps_s,
                                relative_tolerance=1e-6):
    """Integrate audited FEM source power over the actual powered timesteps.

    This checks the deposited source energy, not the discretized temperature
    solution's storage/cooling balance. Checking each device detects allocation
    mistakes that a total-energy check alone would miss.
    """
    steps = [float(dt) for dt in active_steps_s]
    if not steps or any(not math.isfinite(dt) or dt <= 0.0 for dt in steps):
        raise ValueError("Access timesteps must be finite and strictly positive")
    duration = math.fsum(steps)
    if not math.isclose(duration, pulse.duration_s, rel_tol=1e-12, abs_tol=0.0):
        raise ValueError("Powered timesteps do not match the prescribed access duration")
    powers = source_power_audit["per_device"]
    if set(powers) != set(pulse.device_energy_j):
        raise ValueError("Access energy audit requires every prescribed device exactly once")

    def check(name, delivered, expected):
        if (not math.isfinite(delivered) or delivered < 0.0
                or not math.isclose(delivered, expected,
                                    rel_tol=relative_tolerance, abs_tol=1e-30)):
            raise ValueError(f"{name}: deposited energy {delivered:.12g} J "
                             f"does not match prescribed access energy {expected:.12g} J")

    report = {}
    for device, expected in pulse.device_energy_j.items():
        delivered = powers[device]["meshed_w"] * duration
        check(device, delivered, expected)
        report[device] = {"expected_j": expected, "deposited_j": delivered}
    total_expected = math.fsum(pulse.device_energy_j.values())
    total_delivered = source_power_audit["meshed_total_w"] * duration
    check("Total source", total_delivered, total_expected)
    return {"active_duration_s": duration, "active_step_count": len(steps),
            "per_device": report, "expected_total_j": total_expected,
            "deposited_total_j": total_delivered}
