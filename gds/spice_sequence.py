"""Continuous, stateful small-array SRAM SPICE testbenches.

Uses the bundled SKY130 cell models. Rows share wordlines; columns share
bitlines, precharge and masked write drivers. Each column has ONE explicit
external load, not one macro-sized load per cell. This is an electrical
reference with ideal routing, not layout-extracted macro electrothermals.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from gds.spice_netlist import _INCLUDES, _DEVICE_MODELS, VDD
from gds.spice_power import _run
from gds.spice_transient import integrate_trace, read_access_waveform


def validate_schedule(schedule):
    if not isinstance(schedule, dict):
        raise ValueError("Schedule must be an object")
    if set(schedule) != {"initial_bits", "operations", "bitline_cap_f"}:
        raise ValueError("Schedule requires initial_bits, operations, bitline_cap_f")
    bits = np.asarray(schedule["initial_bits"])
    if (bits.ndim != 2 or min(bits.shape, default=0) < 1 or bits.size > 256
            or bits.dtype.kind not in "biu" or not np.isin(bits, [0, 1]).all()):
        raise ValueError("initial_bits must be a rectangular binary array of at most 256 cells")
    capacitance = schedule["bitline_cap_f"]
    if isinstance(capacitance, bool) or not math.isfinite(capacitance) or capacitance <= 0:
        raise ValueError("bitline_cap_f must be finite and positive")
    operations = schedule["operations"]
    if not isinstance(operations, list) or not operations:
        raise ValueError("At least one operation is required")
    expected = bits.astype(int).copy()
    records = []
    previous = -math.inf
    for operation in operations:
        if not isinstance(operation, dict):
            raise ValueError("Each operation must be an object")
        kind = operation.get("operation")
        keys = {"start_ns", "operation", "row"} | ({"data", "mask"} if kind == "write" else set())
        if kind not in {"read", "write"} or set(operation) != keys:
            raise ValueError("Operation fields must match read or masked write schema")
        start, row = operation["start_ns"], operation["row"]
        if (isinstance(start, bool) or not math.isfinite(start) or start < .1
                or start - previous < 1.0 - 1e-12):
            raise ValueError("Operations must start after 0.1 ns and be separated by at least 1 ns")
        if isinstance(row, bool) or not isinstance(row, int) or not 0 <= row < bits.shape[0]:
            raise ValueError("Operation row is out of range")
        previous = start
        before = expected.tolist()
        if kind == "write":
            data, mask = np.asarray(operation["data"]), np.asarray(operation["mask"])
            if (data.shape != (bits.shape[1],) or mask.shape != data.shape
                    or data.dtype.kind not in "biu" or mask.dtype.kind not in "biu"
                    or not np.isin(data, [0, 1]).all() or not np.isin(mask, [0, 1]).all()):
                raise ValueError("Write data and mask must be binary vectors with one entry per column")
            expected[row, mask.astype(bool)] = data[mask.astype(bool)]
        records.append({**operation, "before": before, "after": expected.tolist(),
                        "check_time_s": (start + .9) * 1e-9})
    return bits.astype(int), records


def _pulse(starts, offset, width, end_ns, initial=0.0):
    points = [(0.0, initial)]
    for start in starts:
        begin = start + offset
        points.extend([(begin, 0.0), (begin + .05, VDD),
                       (begin + .05 + width, VDD), (begin + .1 + width, 0.0)])
    points.append((end_ns, 0.0))
    return "PWL(" + " ".join(f"{t:.15g}n {v:.15g}" for t, v in points) + ")"


def write_sequence_deck(directory, schedule, *, temperature_c=26.85, max_step_ps=1.0):
    bits, records = validate_schedule(schedule)
    if (isinstance(temperature_c, bool) or not math.isfinite(temperature_c) or temperature_c <= -273.15
            or isinstance(max_step_ps, bool) or not math.isfinite(max_step_ps) or max_step_ps <= 0):
        raise ValueError("Invalid temperature or maximum electrical timestep")
    directory = Path(directory).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Sequence evidence directory must be empty")
    if any(c.isspace() or c in "\"'" for c in str(directory)):
        raise ValueError("ngspice output path cannot contain whitespace or quotes")
    directory.mkdir(parents=True, exist_ok=True)
    end_ns = records[-1]["start_ns"] + 1.0
    lines = ["* Stateful SRAM array sequence; ideal shared routing", ".option scale=1u",
             f".temp {temperature_c:.15g}", *[f'.include "{p}"' for p in _INCLUDES],
             f"VVDD VDD 0 {VDD}", "VVSS VSS 0 0",
             ".model precharge SW(Ron=1000 Roff=1e12 Vt=0.9 Vh=0.1)",
             ".model writer SW(Ron=1 Roff=1e12 Vt=0.9 Vh=0.1)"]
    # Initially precharge at DC; release before first WL, then recharge for
    # every operation. Hold zero until the next explicit rising edge.
    starts = [item["start_ns"] for item in records]
    pre_points = [(0.0, VDD)]
    for i, start in enumerate(starts):
        if i:
            pre_points.extend([(start, 0), (start + .05, VDD)])
        pre_points.extend([(start + .20, VDD), (start + .25, 0)])
    pre_points.append((end_ns, 0))
    lines.append("VPRE pre 0 PWL(" + " ".join(f"{t:.15g}n {v}" for t, v in pre_points) + ")")
    vectors = {"v_vdd": "v(vdd)", "i_vvdd": "i(vvdd)"}
    for row in range(bits.shape[0]):
        active = [item["start_ns"] for item in records if item["row"] == row]
        lines.extend([f"VWL{row} wl_src{row} 0 {_pulse(active, .35, .25, end_ns)}",
                      f"RWL{row} wl_src{row} wl{row} 1"])
        vectors[f"v_wl{row}"] = f"v(wl{row})"
    for col in range(bits.shape[1]):
        lines.extend([f"CBL{col} bl{col} 0 {schedule['bitline_cap_f']:.15g}",
                      f"CBR{col} br{col} 0 {schedule['bitline_cap_f']:.15g}",
                      f"SPL{col} bl{col} VDD pre 0 precharge",
                      f"SPR{col} br{col} VDD pre 0 precharge"])
        for bit in [0, 1]:
            write_starts = [item["start_ns"] for item in records if item["operation"] == "write"
                            and item["mask"][col] and item["data"][col] == bit]
            control = f"write_{col}_{bit}"
            lines.extend([f"VEN{col}_{bit} {control} 0 {_pulse(write_starts, .27, .48, end_ns)}",
                          f"SWL{col}_{bit} bl{col} {'VDD' if bit else 'VSS'} {control} 0 writer",
                          f"SWR{col}_{bit} br{col} {'VSS' if bit else 'VDD'} {control} 0 writer"])
        vectors[f"v_bl{col}"] = f"v(bl{col})"
        vectors[f"v_br{col}"] = f"v(br{col})"
    device_ids = []
    for row in range(bits.shape[0]):
        for col in range(bits.shape[1]):
            cell = f"xc{row}_{col}"
            lines.append(f"{cell} bl{col} br{col} VDD VSS wl{row} VSS VDD sram_sp_cell")
            high, low = (VDD - .01, .01) if bits[row, col] else (.01, VDD - .01)
            lines.append(f".nodeset v({cell}.q)={high} v({cell}.qb)={low}")
            vectors[f"q_{row}_{col}"] = f"v({cell}.q)"
            vectors[f"qb_{row}_{col}"] = f"v({cell}.qb)"
            for instance, model in _DEVICE_MODELS:
                identity = f"r{row}_c{col}_{instance}"
                device_ids.append(identity)
                path = f"@m.{cell}.{instance}.msky130_fd_pr__special_{model}"
                for quantity in ["id", "vds", "l", "w"]:
                    expression = f"{path}[{quantity}]"
                    lines.append(f".save {expression}")
                    vectors[f"{quantity}_{identity}"] = expression
    lines.extend([".save all", f".tran {max_step_ps:.15g}p {end_ns:.15g}n 0 {max_step_ps:.15g}p",
                  ".control", "set wr_singlescale", "set wr_vecnames", "set numdgt=15", "run"])
    lines.extend(f"let {alias} = {expression}" for alias, expression in vectors.items())
    lines.extend([f"wrdata {directory / 'waveform.txt'} " + " ".join(vectors), ".endc", ".end"])
    deck = directory / "sequence.sp"
    deck.write_text("\n".join(lines) + "\n")
    metadata = {"columns": ["time", *vectors], "end_s": end_ns * 1e-9,
                "temperature_c": temperature_c, "max_step_ps": max_step_ps,
                "schedule": schedule, "checks": records, "device_ids": device_ids,
                "input_sha256": {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in _INCLUDES},
                "deck_sha256": hashlib.sha256(deck.read_bytes()).hexdigest(),
                "scope": "shared-row/column electrical tile, ideal routing, estimated external C per column",
                "excluded_heat": "external drivers/precharge, junctions and interconnect; no thermal solve"}
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def validate_sequence_waveform(curves, metadata):
    time = curves["time"]
    bits, _ = validate_schedule(metadata["schedule"])

    def check_state(at, expected):
        for row in range(bits.shape[0]):
            for col in range(bits.shape[1]):
                q, qb = [np.interp(at, time, curves[f"{node}_{row}_{col}"]) for node in ("q", "qb")]
                high, low = (q, qb) if expected[row][col] else (qb, q)
                if high < .8 * VDD or low > .2 * VDD:
                    raise ValueError(f"Incorrect state r{row}/c{col} at {at:g} s")
    check_state(0, bits)
    for record in metadata["checks"]:
        start = record["start_ns"] * 1e-9
        check_state(start, record["before"])
        check_state(record["check_time_s"], record["after"])
        row = record["row"]
        if np.interp(start + .5e-9, time, curves[f"v_wl{row}"]) < .9 * VDD:
            raise ValueError("Requested wordline was not asserted")
        window = (time >= start) & (time <= record["check_time_s"])
        for r in range(bits.shape[0]):
            for col in range(bits.shape[1]):
                changed = record["before"][r][col] != record["after"][r][col]
                if not changed:
                    q, qb = [curves[f"{node}_{r}_{col}"][window] for node in ("q", "qb")]
                    high, low = (q, qb) if record["before"][r][col] else (qb, q)
                    if np.any(high < .5 * VDD) or np.any(low > .5 * VDD):
                        raise ValueError("Read/half-selected/unselected cell disturbed")
                if r == row and record["operation"] == "read":
                    for node in ("bl", "br"):
                        if abs(np.interp(start + .2e-9, time, curves[f"v_{node}{col}"]) - VDD) > .01 * VDD:
                            raise ValueError("READ bitlines not precharged")
                    delta = np.interp(start + .69e-9, time, curves[f"v_bl{col}"] - curves[f"v_br{col}"])
                    if delta * (1 if record["before"][row][col] else -1) < .1:
                        raise ValueError("READ differential below 100 mV")
    power = {}
    for identity in metadata["device_ids"]:
        device = int(identity.rsplit("x", 1)[1])
        for quantity, expected in [("l", .025e-6 if device in (3, 4) else .15e-6),
                                   ("w", .21e-6 if device in (1, 7) else .14e-6)]:
            if not np.allclose(curves[f"{quantity}_{identity}"], expected, rtol=1e-8, atol=0):
                raise ValueError("Incorrect compact-model physical dimensions")
        current = curves[f"id_{identity}"]
        if np.any(current < -1e-20):
            raise ValueError("Invalid intrinsic channel-current magnitude")
        power[identity] = np.maximum(current, 0) * abs(curves[f"vds_{identity}"])
    return power


def run_sequence(directory, schedule, **options):
    metadata = write_sequence_deck(directory, schedule, **options)
    directory = Path(directory)
    log = _run(str(directory / "sequence.sp"), timeout_s=300)
    (directory / "ngspice.log").write_text(log)
    curves = read_access_waveform(directory / "waveform.txt", metadata)
    power = validate_sequence_waveform(curves, metadata)
    energies = {identity: integrate_trace(curves["time"], values) for identity, values in power.items()}
    np.savez_compressed(directory / "channel_power.npz", time_s=curves["time"], **power)
    report = {"completed": True, "operation_count": len(metadata["checks"]),
              "cell_count": len(power) // 8, "channel_energy_j": math.fsum(energies.values()),
              "per_device_energy_j": energies, "temperature_c": metadata["temperature_c"],
              "scope": metadata["scope"], "excluded_heat": metadata["excluded_heat"]}
    (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
