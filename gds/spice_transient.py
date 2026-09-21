"""Validated electrical READ/WRITE traces, separate from thermal surrogates.

The bitcell's BSIM3v3.2 ``@id`` is intrinsic channel-current magnitude, not
terminal current including charge storage. Channel heat is ``id * abs(Vds)``.
Optional layout resistors contribute ``(Va-Vb)^2/R`` at their physical volumes.
Junction loss and signed external-source power are reported separately; neither
all-source power nor VDD power alone is instantaneous heat inside the cell.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from gds.power import mine_access_timing
from gds.spice_netlist import BITLINE_CAP_F, VDD, write_access_deck
from gds.spice_power import (LIB_PATH, SENSE_MARGIN_V, SPICE_MODEL_REVISION,
                             _DEVICE_BULK, _DEVICES, _run)


SPICE_TRANSIENT_REVISION = "micron-scaled-precharged-access-v1"


def integrate_trace(time_s, values, start_s=None, end_s=None):
    """Integrate piecewise-linear samples, including interpolated window edges."""
    time_s = np.asarray(time_s, dtype=float)
    values = np.asarray(values, dtype=float)
    if (time_s.ndim != 1 or values.shape != time_s.shape or len(time_s) < 2
            or not np.isfinite(time_s).all() or not np.isfinite(values).all()
            or np.any(np.diff(time_s) <= 0)):
        raise ValueError("Trace must have finite values and strictly increasing timestamps")
    start = time_s[0] if start_s is None else float(start_s)
    end = time_s[-1] if end_s is None else float(end_s)
    if not time_s[0] <= start < end <= time_s[-1]:
        raise ValueError("Integration window must lie within the completed transient")
    interior = (time_s > start) & (time_s < end)
    times = np.r_[start, time_s[interior], end]
    samples = np.r_[np.interp(start, time_s, values), values[interior],
                    np.interp(end, time_s, values)]
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapezoid(samples, times))


def read_access_waveform(path, metadata):
    """Reject truncated, malformed, nonfinite or wrongly ordered wrdata output."""
    path = Path(path)
    with path.open() as stream:
        header = stream.readline().lower().split()
    expected = [name.lower() for name in metadata["columns"]]
    if header != expected:
        raise ValueError(f"Unexpected SPICE waveform columns in {path}: {header}")
    data = np.loadtxt(path, skiprows=1, ndmin=2)
    if data.shape[0] < 2 or data.shape[1] != len(expected) or not np.isfinite(data).all():
        raise ValueError("SPICE waveform must contain finite, complete multi-sample rows")
    time = data[:, 0]
    end = float(metadata["end_s"])
    if time[0] != 0.0 or np.any(np.diff(time) <= 0):
        raise ValueError("SPICE waveform timestamps must start at zero and strictly increase")
    if not np.isclose(time[-1], end, rtol=1e-8, atol=1e-18):
        raise ValueError(f"Incomplete SPICE transient: stopped at {time[-1]:.12g} s; "
                         f"expected {end:.12g} s")
    return {name: data[:, index] for index, name in enumerate(expected)}


def validate_access_waveform(curves, metadata):
    """Check actual dimensions, initialization, WL activity and logical outcome."""
    time = curves["time"]
    for i in range(8):
        length = 0.025e-6 if i in (3, 4) else 0.150e-6
        width = 0.210e-6 if i in (1, 7) else 0.140e-6
        for key, expected in ((f"l_x{i}", length), (f"w_x{i}", width)):
            if not np.allclose(curves[key], expected, rtol=1e-8, atol=0):
                raise ValueError(f"Incorrect physical transistor dimension {key}; check SPICE scale")
    if curves["v_wl"].max() < 0.9 * VDD:
        raise ValueError("SPICE access did not assert the wordline")

    def state_ok(one, q, qb):
        high, low = (q, qb) if one else (qb, q)
        return high > 0.8 * VDD and low < 0.2 * VDD

    initial_one, final_one = metadata["initial_stored_one"], metadata["final_stored_one"]
    if not state_ok(initial_one, curves["v_q"][0], curves["v_qb"][0]):
        raise ValueError("SPICE cell did not initialize to the requested stored state")
    if not state_ok(final_one, curves["v_q"][-1], curves["v_qb"][-1]):
        raise ValueError("SPICE access failed: final complementary storage nodes are incorrect")
    report = {
        "completed": True, "operation_verified": True, "sample_count": len(time),
        "initial_q_v": float(curves["v_q"][0]), "initial_qb_v": float(curves["v_qb"][0]),
        "final_q_v": float(curves["v_q"][-1]), "final_qb_v": float(curves["v_qb"][-1]),
        "final_bl_v": float(curves["v_bl"][-1]), "final_br_v": float(curves["v_br"][-1]),
    }
    if metadata["operation"] == "read":
        for node in ("v_bl", "v_br"):
            pre_access = np.interp(metadata["access_start_s"], time, curves[node])
            if not (abs(curves[node][0] - VDD) < 0.01 * VDD
                    and abs(pre_access - VDD) < 0.01 * VDD):
                raise ValueError("READ bitlines were not actually precharged before access")
        delta = curves["v_bl"][-1] - curves["v_br"][-1]
        if (delta if initial_one else -delta) < SENSE_MARGIN_V:
            raise ValueError("READ failed to develop the expected differential bitline signal")
        high, low = ((curves["v_q"], curves["v_qb"]) if initial_one else
                     (curves["v_qb"], curves["v_q"]))
        if np.any(high < VDD / 2) or np.any(low > VDD / 2):
            raise ValueError("READ disturbed the stored state")
        report["read_differential_v"] = float(abs(delta))
    return report


@dataclass
class SpiceAccessResult:
    metadata: dict
    curves: dict
    device_channel_power_w: dict
    device_junction_power_w: dict
    source_power_w: dict
    report: dict
    interconnect_power_w: dict = field(default_factory=dict)

    @property
    def time_s(self):
        return self.curves["time"]

    def device_energy_j(self, start_s=None, end_s=None):
        return {name: integrate_trace(self.time_s, values, start_s, end_s)
                for name, values in self.device_channel_power_w.items()}

    def average_device_power_w(self, start_s, end_s):
        """Conservative source for a thermal timestep, not point sampling."""
        return {name: energy / (end_s - start_s)
                for name, energy in self.device_energy_j(start_s, end_s).items()}

    def interconnect_energy_j(self, start_s=None, end_s=None):
        """Local sheet/contact losses only, never external driver resistors."""
        return {name: integrate_trace(self.time_s, values, start_s, end_s)
                for name, values in self.interconnect_power_w.items()}

    def average_interconnect_power_w(self, start_s, end_s):
        return {name: energy / (end_s - start_s)
                for name, energy in self.interconnect_energy_j(start_s, end_s).items()}


def access_result_from_waveform(curves, metadata):
    report = validate_access_waveform(curves, metadata)
    channels, junctions = {}, {}
    for name, (_, drain, source) in _DEVICES.items():
        suffix = name.lower()
        terminals = metadata.get("device_terminal_columns", {}).get(name)
        if terminals:
            vd, vs, vb = (curves[terminals[term]] for term in ("drain", "source", "body"))
        else:
            vd, vs = curves[f"v_{drain}"], curves[f"v_{source}"]
            vb = curves[f"v_{_DEVICE_BULK[name]}"]
        current = curves[f"id_{suffix}"]
        if np.any(current < -1e-20):
            raise ValueError(f"Unexpected negative BSIM3 channel-current magnitude for {name}")
        channels[name] = np.maximum(current, 0.0) * abs(vd - vs)
        junctions[name] = (abs(curves[f"ibd_{suffix}"] * (vb - vd))
                           + abs(curves[f"ibs_{suffix}"] * (vb - vs)))
    source_powers = {
        name: -curves[column] * curves[metadata["source_voltage_columns"][name]]
        for name, column in metadata["source_current_columns"].items()
    }
    network = metadata.get("interconnect_network")
    wire_powers = {}
    if metadata.get("interconnect", "none") == "layout" and network is None:
        raise ValueError("Layout interconnect power requires the archived physical network")
    if network is not None:
        resistors = network["resistors"]
        columns = metadata.get("interconnect_voltage_columns", {})
        if set(columns) != {resistor["name"] for resistor in resistors}:
            raise ValueError("Every local interconnect resistor must have a saved branch voltage")
        for resistor in resistors:
            resistance = float(resistor["resistance_ohm"])
            if not np.isfinite(resistance) or resistance <= 0:
                raise ValueError("Local interconnect resistance must be finite and positive")
            weights = np.asarray([region["weight"] for region in resistor["heat_regions"]], dtype=float)
            if (not len(weights) or not np.isfinite(weights).all() or np.any(weights < 0)
                    or not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-12)):
                raise ValueError("Local interconnect heat-region weights must be nonnegative and sum to one")
            voltage = curves[columns[resistor["name"]]]
            power = voltage * voltage / resistance
            if not np.isfinite(power).all() or np.any(power < 0):
                raise ValueError("Local interconnect Joule heating must be finite and nonnegative")
            wire_powers[resistor["name"]] = power
    result = SpiceAccessResult(metadata, curves, channels, junctions, source_powers, report,
                               wire_powers)
    wire_energy = result.interconnect_energy_j()
    layer_energy = {}
    for resistor in network["resistors"] if network is not None else []:
        for region in resistor["heat_regions"]:
            layer = region["layer"]
            layer_energy[layer] = layer_energy.get(layer, 0.0) + (
                wire_energy[resistor["name"]] * float(region["weight"]))
    report.update({
        "revision": SPICE_TRANSIENT_REVISION,
        "energy_window_s": [float(result.time_s[0]), float(result.time_s[-1])],
        "device_channel_energy_j": result.device_energy_j(),
        "channel_energy_j": sum(result.device_energy_j().values()),
        "interconnect": metadata.get("interconnect", "none"),
        "interconnect_revision": network.get("revision") if network is not None else None,
        "interconnect_network_fingerprint": network.get("fingerprint") if network is not None else None,
        "resistor_energy_j": wire_energy,
        "interconnect_energy_j": sum(wire_energy.values()),
        "interconnect_layer_energy_j": layer_energy,
        "interconnect_heat_scope": "Local GDS sheet/contact resistors only; external driver and precharge losses excluded",
        "total_local_heat_energy_j": sum(result.device_energy_j().values()) + sum(wire_energy.values()),
        "device_junction_energy_j": {
            name: integrate_trace(result.time_s, power) for name, power in junctions.items()},
        "signed_source_energy_j": {
            name: integrate_trace(result.time_s, power) for name, power in source_powers.items()},
        "heat_model": "BSIM3 intrinsic channel id * abs(local Vd-Vs), plus local resistor (Va-Vb)^2/R; junctions reported separately",
        "thermal_solver_connected": False,
    })
    if metadata["operation"] == "read":
        report["bitline_stored_energy_change_j"] = float(0.5 * BITLINE_CAP_F * sum(
            curves[node][-1] ** 2 - curves[node][0] ** 2 for node in ("v_bl", "v_br")))
    return result


def run_access_transient(operation, *, stored_one=True, pulse_width_ns=None,
                         max_step_ps=1.0, lib_path=LIB_PATH, out_dir=None,
                         interconnect="none", interconnect_step_um=0.05):
    """Run and validate a real electrical access; optionally preserve all artifacts.

    ``stored_one`` is the INITIAL state. WRITE flips it; READ must preserve it.
    Fixed pulse width defaults to the borrowed Liberty clock-high proxy. This
    bench has no sense-amplifier feedback; it does not stop READ at 100 mV.
    Layout mode includes nominal local interconnect/contact resistance, not
    extracted capacitance or routing losses in the rest of the SRAM array.
    """
    if pulse_width_ns is None:
        pulse_width_ns = mine_access_timing(lib_path).min_pulse_width_high_ns

    def run_in(directory):
        directory.mkdir(parents=True, exist_ok=True)
        metadata = write_access_deck(
            str(directory / "access.sp"), str(directory / "waveform.txt"),
            operation=operation, stored_one=stored_one,
            pulse_width_ns=pulse_width_ns, max_step_ps=max_step_ps,
            interconnect=interconnect, interconnect_step_um=interconnect_step_um,
        )
        metadata["revision"] = SPICE_TRANSIENT_REVISION
        metadata["spice_model_revision"] = SPICE_MODEL_REVISION
        metadata["ngspice_timeout_s"] = 300 if interconnect == "layout" else 60
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        try:
            # Fine distributed sheets contain thousands of electrical nodes;
            # their solve may legitimately exceed the compact-only 60 s cap.
            output = (_run(metadata["deck_path"], timeout_s=metadata["ngspice_timeout_s"])
                      if interconnect == "layout" else _run(metadata["deck_path"]))
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            (directory / "ngspice.log").write_text(str(error))
            raise
        (directory / "ngspice.log").write_text(output)
        curves = read_access_waveform(metadata["data_path"], metadata)
        result = access_result_from_waveform(curves, metadata)
        # This marker is written only after completion and functional checks.
        (directory / "summary.json").write_text(json.dumps(result.report, indent=2) + "\n")
        np.savez_compressed(directory / "device_power.npz", time_s=result.time_s,
                            **{name: value for name, value in result.device_channel_power_w.items()})
        if result.metadata.get("interconnect") == "layout":
            np.savez_compressed(directory / "interconnect_power.npz", time_s=result.time_s,
                                **result.interconnect_power_w)
        return result

    if out_dir is not None:
        directory = Path(out_dir).resolve()
        # Preserve existing evidence instead of leaving stale successful results
        # next to a failed rerun. Callers select a new output directory per run.
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"SPICE output directory is not empty: {directory}")
        return run_in(directory)
    with tempfile.TemporaryDirectory(prefix="kelvin-spice-access-") as directory:
        return run_in(Path(directory))
