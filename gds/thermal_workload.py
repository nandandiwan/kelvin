"""One validated electrical event shared by averaged and transient heat models.

The access energy is the integral of each physical transistor's channel heat
and, when enabled, the cell-local wire/contact resistor losses
over the *complete* SPICE trace, including initialization dwell and settling.
Steady heating uses that same energy times the selected access rate; transient
heating must integrate the trace over each thermal time interval. No DC peak,
charge-budget normalization, or rectangular-pulse approximation is used here.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from gds.power import mine_access_timing
from gds.spice_netlist import _INCLUDES
from gds.spice_power import LIB_PATH, SPICE_MODEL_REVISION
from gds.spice_transient import (
    SPICE_TRANSIENT_REVISION, SpiceAccessResult, run_access_transient,
)


THERMAL_POWER_REVISION = "spice-channel-interconnect-conservative-v2"
SPICE_ACCESS_POINTS = {
    "read_0": ("read", False),
    "read_1": ("read", True),
    "write_0_to_1": ("write", False),
    "write_1_to_0": ("write", True),
}
_DEVICE_IDS = tuple(f"X{i}" for i in range(8))
_ARTIFACT_NAMES = (
    "access.sp", "waveform.txt", "metadata.json", "ngspice.log",
    "summary.json", "device_power.npz",
)
_INTERCONNECT_ARTIFACT_NAMES = (
    "interconnect_network.json", "interconnect.spice", "interconnect_power.npz",
)


@dataclass(frozen=True)
class SpiceThermalWorkload:
    trace: SpiceAccessResult
    device_energy_j: dict
    device_average_power_w: dict
    period_s: float
    row_hit_rate: float
    metadata: dict

    @property
    def interconnect_energy_j(self):
        return self.trace.interconnect_energy_j()

    @property
    def interconnect_average_power_w(self):
        return {name: energy * self.row_hit_rate / self.period_s
                for name, energy in self.interconnect_energy_j.items()}


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and positive")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and positive") from error
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _power_trace_sha256(trace):
    """Stable content fingerprint independent of output directory/ZIP metadata."""
    digest = hashlib.sha256()
    for name, values in [("time_s", trace.time_s), *(
            (name, trace.device_channel_power_w[name]) for name in _DEVICE_IDS),
            *sorted(getattr(trace, "interconnect_power_w", {}).items())]:
        values = np.asarray(values, dtype="<f8")
        digest.update(f"{name}:{len(values)}:".encode("ascii"))
        digest.update(values.tobytes(order="C"))
    if trace.metadata.get("interconnect_network") is not None:
        digest.update(json.dumps(trace.metadata["interconnect_network"], sort_keys=True,
                                 separators=(",", ":"), allow_nan=False).encode())
    return digest.hexdigest()


def build_spice_workload(point="read_1", row_hit_rate=1.0, period_ns=None,
                         pulse_width_ns=None, spice_max_step_ps=1.0,
                         out_dir=None, lib_path=LIB_PATH, interconnect="none",
                         interconnect_step_um=0.05):
    """Run a checked READ/WRITE event and return its common thermal contract.

    ``row_hit_rate`` changes only the *averaged* access frequency, not the
    waveform or energy of a single access. ``period_ns`` must contain the full
    electrical observation window, even at a low row-hit rate. By default the
    period and WL high plateau are borrowed from the supplied Liberty file;
    explicit overrides are supported and recorded.

    With ``out_dir``, all electrical artifacts are preserved by
    :func:`run_access_transient`, which refuses to overwrite a nonempty folder.
    ``interconnect='layout'`` adds the verified cell-local resistor losses
    from the same electrical solve. External drivers are never local heat.
    """
    if point not in SPICE_ACCESS_POINTS:
        raise ValueError(
            f"{point!r} is not a transient SPICE access; choose "
            f"{', '.join(SPICE_ACCESS_POINTS)}. For hold, crowbar, or settled DC "
            "cases, select --power-model dc-surrogate explicitly."
        )
    if isinstance(row_hit_rate, (bool, np.bool_)):
        raise ValueError("row_hit_rate must be finite and between 0 and 1")
    try:
        rate = float(row_hit_rate)
    except (TypeError, ValueError) as error:
        raise ValueError("row_hit_rate must be finite and between 0 and 1") from error
    if not math.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError("row_hit_rate must be finite and between 0 and 1")
    max_step_ps = _positive(spice_max_step_ps, "spice_max_step_ps")
    if interconnect not in {"none", "layout"}:
        raise ValueError("interconnect must be 'none' or 'layout'")
    network_step = _positive(interconnect_step_um, "interconnect_step_um")
    if interconnect == "none" and network_step != 0.05:
        raise ValueError("interconnect_step_um requires interconnect='layout'")
    period_is_default, pulse_is_default = period_ns is None, pulse_width_ns is None
    timing = mine_access_timing(lib_path) if period_is_default or pulse_is_default else None
    period = _positive(timing.min_period_ns if period_is_default else period_ns, "period_ns")
    width = _positive(timing.min_pulse_width_high_ns if pulse_is_default else pulse_width_ns,
                      "pulse_width_ns")
    period_s = period * 1e-9
    operation, initial_one = SPICE_ACCESS_POINTS[point]
    network_options = ({"interconnect": interconnect, "interconnect_step_um": network_step}
                       if interconnect == "layout" else {})
    trace = run_access_transient(
        operation, stored_one=initial_one, pulse_width_ns=width,
        max_step_ps=max_step_ps, lib_path=lib_path, out_dir=out_dir, **network_options,
    )
    if not trace.report.get("completed") or not trace.report.get("operation_verified"):
        raise ValueError("Thermal workload requires a completed, functionally validated SPICE access")
    if set(trace.device_channel_power_w) != set(_DEVICE_IDS):
        raise ValueError("Thermal workload requires per-instance channel powers for X0 through X7")
    wire_powers = getattr(trace, "interconnect_power_w", {})
    network = trace.metadata.get("interconnect_network")
    if interconnect == "layout":
        if not network or not wire_powers:
            raise ValueError("Layout interconnect heat requires the complete electrical network and traces")
        names = [r["name"] for r in network["resistors"]]
        if len(names) != len(set(names)) or set(names) != set(wire_powers):
            raise ValueError("Interconnect waveform IDs do not match the spatial resistor network")
    elif wire_powers or network:
        raise ValueError("Unexpected interconnect network in channel-only workload")
    for device, power in {**trace.device_channel_power_w, **wire_powers}.items():
        values = np.asarray(power)
        if (values.shape != trace.time_s.shape or not np.isfinite(values).all()
                or np.any(values < 0)):
            raise ValueError(f"Invalid nonnegative finite channel-power waveform for {device}")
    energy = trace.device_energy_j()  # Validates every complete integration window.
    wire_energy = trace.interconnect_energy_j() if wire_powers else {}
    if any(not math.isfinite(value) or value < 0 for value in energy.values()):
        raise ValueError("Integrated per-device SPICE energies must be finite and nonnegative")
    start_s, end_s = float(trace.time_s[0]), float(trace.time_s[-1])
    if start_s != 0.0:
        raise ValueError("Thermal workload requires a SPICE trace starting at zero")
    duration_s = end_s - start_s
    # Tolerate only rounding of a period specified in nanoseconds, not overlap.
    if period_s < duration_s and not math.isclose(period_s, duration_s, rel_tol=1e-12):
        raise ValueError(
            f"period_ns ({period:g}) is shorter than the complete SPICE trace "
            f"({duration_s * 1e9:g} ns); increase --period-ns or shorten --pulse-width-ns"
        )
    average = {device: value * rate / period_s for device, value in energy.items()}
    wire_average = {name: value * rate / period_s for name, value in wire_energy.items()}
    artifact_dir = Path(out_dir).resolve() if out_dir is not None else None
    artifacts = {}
    if artifact_dir is not None:
        for name in _ARTIFACT_NAMES + (_INTERCONNECT_ARTIFACT_NAMES if network else ()):
            path = artifact_dir / name
            if not path.is_file():
                raise ValueError(f"Completed SPICE run is missing its archived artifact: {path}")
            artifacts[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = {
        "kind": "spice-transient",
        "thermal_power_revision": THERMAL_POWER_REVISION,
        "spice_transient_revision": SPICE_TRANSIENT_REVISION,
        "spice_model_revision": SPICE_MODEL_REVISION,
        "spice_input_sha256": {
            str(Path(path).resolve()): hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in _INCLUDES
        },
        "point": point,
        "operation": operation,
        "initial_stored_one": initial_one,
        "final_stored_one": initial_one if operation == "read" else not initial_one,
        "period_s": period_s,
        "row_hit_rate": rate,
        "access_rate_hz": rate / period_s,
        "trace_start_s": start_s,
        "trace_end_s": end_s,
        "trace_duration_s": duration_s,
        "pulse_width_s": width * 1e-9,
        "spice_max_step_s": max_step_ps * 1e-12,
        "period_source": "borrowed Liberty minimum period" if period_is_default else "explicit override",
        "pulse_width_source": "borrowed Liberty minimum high pulse" if pulse_is_default else "explicit override",
        "timing_lib_path": str(Path(lib_path).resolve()) if timing is not None else None,
        "timing_lib_sha256": hashlib.sha256(Path(lib_path).read_bytes()).hexdigest()
        if timing is not None else None,
        "device_energy_j": dict(energy),
        "channel_energy_j": sum(energy.values()),
        "device_average_power_w": dict(average),
        "average_channel_power_w": sum(average.values()),
        "interconnect_model": interconnect,
        "interconnect_step_um": network_step if network else None,
        "interconnect_network": network,
        "resistor_energy_j": wire_energy,
        "interconnect_energy_j": sum(wire_energy.values()),
        "interconnect_layer_energy_j": dict(trace.report.get("interconnect_layer_energy_j", {})),
        "interconnect_average_power_w": wire_average,
        "average_interconnect_power_w": sum(wire_average.values()),
        "total_local_heat_energy_j": sum(energy.values()) + sum(wire_energy.values()),
        "average_total_local_power_w": sum(average.values()) + sum(wire_average.values()),
        "power_trace_sha256": _power_trace_sha256(trace),
        "spice_artifact_dir": str(artifact_dir) if artifact_dir is not None else None,
        "spice_artifact_sha256": artifacts,
        "heat_model": "BSIM3 intrinsic channel id * abs(Vd-Vs), separately mapped by instance"
                      + ("; cell-local resistor (Va-Vb)^2/R mapped by geometry" if network else ""),
        "energy_window": "complete SPICE observation window, including pre-access dwell and settling",
        "averaging_rule": "each channel/resistor integrated event energy * row_hit_rate / period_s",
        "transient_transfer": "piecewise-linear SPICE trace integrated over each thermal timestep",
        "omissions": [
            "holding leakage outside the electrical observation window",
            "junction, external driver, precharge, and sense-amplifier heat",
            *([] if network else ["interconnect heat"]),
            "electrothermal feedback and extracted macro parasitics",
        ],
        "electrical_assumptions": list(trace.metadata.get("assumptions", [])),
    }
    return SpiceThermalWorkload(trace, energy, average, period_s, rate, metadata)
