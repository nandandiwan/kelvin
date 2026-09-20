"""ngspice deck generator for the real sram_sp_cell bitcell (data/sram_sp_cell.spice
— the actual 8-device netlist extracted from sram22_sky130_macros, using real SKY130
special_nfet_pass/special_nfet_latch/special_pfet_pass BSIM3 models; see
data/sky130_fd_pr/ and PLAN.md's [[optimized-singing-creek]] plan Phase 0/1).

Three testbenches; access high times may be supplied from the macro Liberty
file, with explicitly assumed edges, bitline loading and external drivers:
  - HOLD: DC operating point, cell idle (WL=0, BL=BR=VDD precharged) -- static leakage.
  - WRITE: select a known initial DC state via .nodeset, pulse WL
    while driving BL/BR to the opposite state, verify the write actually took.
  - READ: select a known initial DC state, switch-precharge BL=BR=VDD, pulse WL,
    no forced BL/BR drive during access -- the cell's own access+pull-down current
    discharges whichever bit line matches its stored '0' through a lumped bitline
    capacitance (see BITLINE_CAP_F below for why, and what it approximates).

Results are extracted via `wrdata` to a plain text file, not console `print`/
`.measure` output -- verified empirically that ngspice's `[time=X]` vector
indexing is not valid syntax here, and that console `.measure`/`print` output
is easy to misparse; `wrdata`'s file is unambiguous column data.
"""

import math
import os
import hashlib
import json
from pathlib import Path

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_PDK_DIR = os.path.join(_DATA_DIR, "sky130_fd_pr")

_INCLUDES = [
    os.path.join(_PDK_DIR, "globals.spice"),
    os.path.join(_PDK_DIR, "specialized_cells_curated.spice"),
    os.path.join(_PDK_DIR, "cells/special_nfet_pass/sky130_fd_pr__special_nfet_pass__mismatch.corner.spice"),
    os.path.join(_PDK_DIR, "cells/special_nfet_latch/sky130_fd_pr__special_nfet_latch__mismatch.corner.spice"),
    os.path.join(_PDK_DIR, "cells/special_pfet_pass/sky130_fd_pr__special_pfet_pass__mismatch.corner.spice"),
    os.path.join(_PDK_DIR, "ngspice_fixed/special_nfet_pass.spice"),
    os.path.join(_PDK_DIR, "ngspice_fixed/special_nfet_latch.spice"),
    os.path.join(_PDK_DIR, "ngspice_fixed/special_pfet_pass.spice"),
    os.path.join(_PDK_DIR, "ngspice_fixed/special_pfet_pass_shortl.spice"),
    os.path.join(_DATA_DIR, "sram_sp_cell.spice"),
]

VDD = 1.8
TEMPERATURE_C = 25.0

# Real array dimensions for data/sram22_64x22m4w22.gds/.spice -- the macro this
# project's bitcell (data/sram_sp_cell.spice) is actually extracted from. Counted
# directly from the real netlist's net names, not assumed:
#   grep -o 'wl\[[0-9]*\]' data/sram22_64x22m4w22.spice | sort -u | wc -l  -> 16
#   grep -o 'bl\[[0-9]*\]' data/sram22_64x22m4w22.spice | sort -u | wc -l  -> 88
# A previous version of BITLINE_CAP_F hardcoded 2048 -- the row count of a
# DIFFERENT macro (sram22_2048x8m8w1) this project also uses elsewhere for its
# validated Liberty numbers, copied in by mistake. 128x too large for this one.
N_ROWS = 16
N_COLS = 88

# Real bitline capacitance isn't meaningful for an isolated single-cell testbench --
# a real access discharges the FULL shared bitline (this row's cell plus every other
# row's access-transistor junction capacitance hanging off the same line, plus real
# metal capacitance), not just this one cell's own terminals. Approximated here as
# ~1fF/cell (a commonly cited order-of-magnitude figure for advanced-node SRAM bitline
# loading per row) times this macro's REAL row count (N_ROWS, above) -- a labeled
# ESTIMATE, not a measurement, but now at least the right array's row count.
BITLINE_CAP_F = 1e-15 * N_ROWS


def _preamble(cell_include=None):
    # The extracted cell and the SKY130 wrappers specify L/W in micrometres;
    # the BSIM model-card quantities are already SI.  Without SCALE ngspice
    # silently builds 0.15-metre transistors, corrupting DC and capacitances.
    includes = _INCLUDES if cell_include is None else [*_INCLUDES[:-1], cell_include]
    return [".option scale=1u", f".temp {TEMPERATURE_C}"] + [f'.include "{p}"' for p in includes] + [
        f"VVDD VDD 0 {VDD}",
        "VVSS VSS 0 0",
    ]


def write_hold_deck(out_path: str, stored_one: bool = True) -> str:
    """DC operating point, cell idle -- static leakage power.

    Needs a .nodeset bias toward one real stable corner: verified directly
    that an UNBIASED .op on this symmetric cross-coupled latch can converge to
    Q=QB, the latch's symmetric METASTABLE point (not a real stored
    '0' or '1'), which is not what real HOLD leakage means.

    The existing DC interface uses console `print`, parsed as "name = value"
    text.  Access transients instead use the headered waveform contract below.
    """
    q0, qb0 = (VDD - 0.01, 0.01) if stored_one else (0.01, VDD - 0.01)
    lines = ["* HOLD: static leakage, cell idle"] + _preamble() + [
        "VWL WL 0 0",
        f"VBL BL 0 {VDD}",
        f"VBR BR 0 {VDD}",
        "X0 BL BR VDD VSS WL VSS VDD sram_sp_cell",
        f".nodeset v(x0.q)={q0} v(x0.qb)={qb0}",
        ".op",
        ".control",
        "run",
        "let ivdd = -i(vvdd)",
        "let pvdd = ivdd * vdd",
        "print pvdd",
        ".endc",
        ".end",
    ]
    _write(out_path, lines)
    return out_path


_EPS = 0.01  # nodeset guess kept slightly off-rail; exact-rail guesses can sit at a zero-gradient point


def _write_interconnect_cell(out_path, network):
    """Keep compact instances intact while replacing their ideal-wire topology.

    Every layout attachment of a port is an ideal boundary attachment to that
    external testbench net. No resistor outside the cell is counted as local
    wire heat. Internal source/drain and gate voltages remain distinct.
    """
    ports = ("BL", "BR", "VDD", "VSS", "WL", "VNB", "VPB")
    aliases = {}
    for port, attachments in network["port_nodes"].items():
        if port not in ports or not isinstance(attachments, list) or not attachments:
            raise ValueError(f"Invalid layout port attachment: {port}")
        for node in attachments:
            if node in aliases and aliases[node] != port:
                raise ValueError(f"Layout node {node} shorts distinct external ports")
            aliases[node] = port

    def local(node):
        node = aliases.get(node, node)
        if not isinstance(node, str) or not node or any(c.isspace() for c in node):
            raise ValueError("Invalid local interconnect node name")
        return node

    if set(network["port_nodes"]) != set(ports):
        raise ValueError("Layout network must expose every original bitcell port")
    if set(network["device_terminals"]) != {f"X{i}" for i in range(8)}:
        raise ValueError("Layout network must map all eight physical compact instances")
    lines = ["* Layout-linked local interconnect resistances; estimated external C retained",
             ".SUBCKT sram_sp_cell_layout " + " ".join(ports)]
    terminals = {}
    for line in Path(_INCLUDES[-1]).read_text().splitlines():
        parts = line.split()
        if not parts or not parts[0].upper().startswith("X"):
            continue
        instance = parts[0].upper()
        mapped = {term: local(network["device_terminals"][instance][term])
                  for term in ("drain", "gate", "source", "body")}
        terminals[instance] = mapped
        lines.append(" ".join([parts[0], *mapped.values(), *parts[5:]]))
    names = set()
    resistors = []
    for resistor in network["resistors"]:
        name, resistance = resistor["name"], float(resistor["resistance_ohm"])
        if (not isinstance(name, str) or not name.lower().startswith("r")
                or not name.isalnum() or name.lower() in names
                or not math.isfinite(resistance) or resistance <= 0):
            raise ValueError("Layout resistors need unique R names and positive finite resistance")
        names.add(name.lower())
        a, b = local(resistor["node_a"]), local(resistor["node_b"])
        lines.append(f"{name} {a} {b} {resistance:.17g}")
        resistors.append({**resistor, "spice_node_a": a, "spice_node_b": b})
    lines.append(".ENDS sram_sp_cell_layout")
    _write(os.fspath(out_path), lines)
    return terminals, resistors, {key: local(node) for key, node in network["logical_nodes"].items()}


def write_access_deck(
    out_path: str,
    data_path: str,
    operation: str = "read",
    stored_one: bool = True,
    pulse_width_ns: float = 0.239519,
    max_step_ps: float = 1.0,
    interconnect: str = "none",
    interconnect_step_um: float = 0.05,
) -> dict:
    """Write a single-access transient and return its explicit data contract.

    ``stored_one`` is the INITIAL state.  A WRITE targets its opposite; a
    READ must preserve it.  A nodeset selects the stable initial DC solution
    without forcing Q/QB during the access.  READ bitlines have real DC
    precharge paths, released before WL rises; a nodeset on a floating
    capacitor would only supply an initial Newton guess, not precharge it.

    The measured waveform is for this isolated loaded cell, not an extracted
    macro: bitline loading, driver resistance and edge times are explicit
    approximations.  No internal capacitances or model parameters are changed
    to obtain convergence.  ``pulse_width_ns`` specifies the high plateau;
    both 50-ps edges are included in ``access_end_s``.  The default high time
    is a borrowed macro-Liberty minimum-pulse-width proxy, not a calibrated
    WL waveform or sense-amplifier termination time for this SRAM.

    The whitespace-delimited output has one time column and a header.  All
    intrinsic currents are saved *during* transient analysis, not queried
    only at its final operating point.  Source currents use SPICE's passive
    sign convention (negative current for a delivering positive supply).

    ``interconnect='layout'`` adds a GDS-linked distributed resistance model
    inside the cell, preserving each compact instance. Every local resistor's
    branch voltage and the actual device terminal voltages are archived for
    heat extraction. This is nominal local R extraction, not full array RC
    extraction: the external estimated bitline capacitors remain unchanged.
    """
    if operation not in {"read", "write"}:
        raise ValueError("operation must be 'read' or 'write'")
    if not isinstance(stored_one, bool):
        raise ValueError("stored_one must be a boolean initial state")
    if interconnect not in {"none", "layout"}:
        raise ValueError("interconnect must be 'none' or 'layout'")
    try:
        network_step = float(interconnect_step_um)
    except (TypeError, ValueError) as error:
        raise ValueError("interconnect_step_um must be finite and in [0.0125, 0.2] um") from error
    if (isinstance(interconnect_step_um, bool) or not math.isfinite(network_step)
            or not 0.0125 <= network_step <= 0.2):
        raise ValueError("interconnect_step_um must be finite and in [0.0125, 0.2] um")
    if interconnect == "none" and network_step != 0.05:
        raise ValueError("interconnect_step_um requires interconnect='layout'")
    for name, value in (("pulse_width_ns", pulse_width_ns), ("max_step_ps", max_step_ps)):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    out_path, data_path = os.fspath(out_path), os.fspath(data_path)
    if any(ch.isspace() or ch in '\"\'' for ch in data_path):
        raise ValueError("ngspice wrdata requires a data_path without whitespace or quotes")

    network, cell_include, local_terminals, local_resistors = None, None, {}, []
    logical_nodes = {"Q": "q", "QB": "qb"}
    if interconnect == "layout":
        from gds.interconnect import build_interconnect_network
        network = build_interconnect_network(step_um=network_step)
        cell_include = str(Path(out_path).resolve().parent / "interconnect.spice")
        local_terminals, local_resistors, logical_nodes = _write_interconnect_cell(cell_include, network)
        network_path = Path(out_path).resolve().parent / "interconnect_network.json"
        network_path.write_text(json.dumps(network, indent=2) + "\n")

    start_ns, edge_ns, cooldown_ns = 0.1, 0.05, 0.5
    access_end_ns = start_ns + 2 * edge_ns + pulse_width_ns
    end_ns = access_end_ns + cooldown_ns
    final_one = stored_one if operation == "read" else not stored_one
    q0, qb0 = (VDD - _EPS, _EPS) if stored_one else (_EPS, VDD - _EPS)
    lines = [f"* {operation.upper()}: initial Q={int(stored_one)}, target Q={int(final_one)}"] + _preamble(cell_include) + [
        f"VWL wl_src 0 PULSE(0 {VDD} {start_ns:.15g}n {edge_ns:.15g}n {edge_ns:.15g}n {pulse_width_ns:.15g}n {2*end_ns:.15g}n)",
        "RWL wl_src WL 1",
        f"CBL BL 0 {BITLINE_CAP_F:.15g}",
        f"CBR BR 0 {BITLINE_CAP_F:.15g}",
    ]
    vectors = {
        "v_vdd": "v(vdd)", "v_vss": "v(vss)", "v_wl": "v(wl)",
        "v_bl": "v(bl)", "v_br": "v(br)",
        "v_q": f"v(x0.{logical_nodes['Q']})", "v_qb": f"v(x0.{logical_nodes['QB']})",
        "i_vvdd": "i(vvdd)", "i_vvss": "i(vvss)", "i_vwl": "i(vwl)",
        "v_wl_src": "v(wl_src)",
    }
    source_voltages = {"vvdd": "v_vdd", "vvss": "v_vss", "vwl": "v_wl_src"}
    source_currents = {"vvdd": "i_vvdd", "vvss": "i_vvss", "vwl": "i_vwl"}
    driver_resistors = {"rwl": {"resistance_ohm": 1.0, "node_a": "v_wl_src", "node_b": "v_wl"}}
    if operation == "read":
        lines += [
            "* Precharge ON at DC; OFF by 70 ps, before the 100-ps WL edge.",
            f"VPRE pre 0 PWL(0 {VDD} 50p {VDD} 70p 0)",
            "SBL BL VDD pre 0 precharge",
            "SBR BR VDD pre 0 precharge",
            ".model precharge SW(Ron=1000 Roff=1e12 Vt=0.9 Vh=0.1)",
        ]
        vectors.update({"i_vpre": "i(vpre)", "v_pre": "v(pre)"})
        source_voltages["vpre"], source_currents["vpre"] = "v_pre", "i_vpre"
    else:
        bl, br = (VDD, 0.0) if final_one else (0.0, VDD)
        lines += [f"VBL bl_src 0 {bl}", "RBL bl_src BL 1", f"VBR br_src 0 {br}", "RBR br_src BR 1"]
        vectors.update({"i_vbl": "i(vbl)", "v_bl_src": "v(bl_src)",
                        "i_vbr": "i(vbr)", "v_br_src": "v(br_src)"})
        source_voltages.update({"vbl": "v_bl_src", "vbr": "v_br_src"})
        source_currents.update({"vbl": "i_vbl", "vbr": "i_vbr"})
        driver_resistors.update({
            "rbl": {"resistance_ohm": 1.0, "node_a": "v_bl_src", "node_b": "v_bl"},
            "rbr": {"resistance_ohm": 1.0, "node_a": "v_br_src", "node_b": "v_br"},
        })
    cell_name = "sram_sp_cell" if network is None else "sram_sp_cell_layout"
    lines += [
        f"X0 BL BR VDD VSS WL VSS VDD {cell_name}",
        f".nodeset v(x0.{logical_nodes['Q']})={q0} v(x0.{logical_nodes['QB']})={qb0}",
        ".save all",
    ]
    terminal_columns, resistor_columns = {}, {}

    def node_voltage(node):
        # Subcircuit pins are flattened by ngspice to the external node.
        if node in {"BL", "BR", "VDD", "VSS", "WL"}:
            return f"v({node.lower()})"
        if node in {"VNB", "VPB"}:
            return "v(vss)" if node == "VNB" else "v(vdd)"
        return f"v(x0.{node})"

    for instance, terminals in local_terminals.items():
        terminal_columns[instance] = {}
        for term in ("drain", "source", "body"):
            alias = f"v{term[0]}_{instance.lower()}"
            vectors[alias] = node_voltage(terminals[term])
            terminal_columns[instance][term] = alias
    for resistor in local_resistors:
        alias = "dv_" + resistor["name"].lower()
        vectors[alias] = (f"{node_voltage(resistor['spice_node_a'])} - "
                          f"{node_voltage(resistor['spice_node_b'])}")
        resistor_columns[resistor["name"]] = alias
    for inst, model in _DEVICE_MODELS:
        dev = f"@m.x0.{inst}.msky130_fd_pr__special_{model}"
        for term in ("id", "ibd", "ibs", "vds", "vbs", "l", "w"):
            expression = f"{dev}[{term}]"
            lines.append(f".save {expression}")
            vectors[f"{term}_{inst}"] = expression
    lines += [
        f".tran {max_step_ps:.15g}p {end_ns:.15g}n 0 {max_step_ps:.15g}p",
        ".control", "set wr_singlescale", "set wr_vecnames", "set numdgt=15", "run",
    ]
    lines += [f"let {alias} = {expression}" for alias, expression in vectors.items()]
    lines += [f"wrdata {data_path} " + " ".join(vectors), ".endc", ".end"]
    _write(out_path, lines)
    return {
        "deck_path": out_path, "data_path": data_path,
        "columns": ["time", *vectors], "operation": operation,
        "end_s": end_ns * 1e-9, "access_start_s": start_ns * 1e-9,
        "access_end_s": access_end_ns * 1e-9,
        "pulse_width_s": pulse_width_ns * 1e-9, "edge_s": edge_ns * 1e-9,
        "max_step_s": max_step_ps * 1e-12,
        "initial_stored_one": stored_one, "final_stored_one": final_one,
        "source_voltage_columns": source_voltages,
        "source_current_columns": source_currents,
        "driver_resistances": driver_resistors,
        "precharge": {"ron_ohm": 1000.0, "roff_ohm": 1e12,
                      "release_start_s": 50e-12, "release_end_s": 70e-12}
        if operation == "read" else None,
        "temperature_c": TEMPERATURE_C, "bitline_cap_f": BITLINE_CAP_F,
        "interconnect": interconnect,
        "interconnect_step_um": network["step_um"] if network is not None else None,
        "interconnect_network": network,
        "device_terminal_columns": terminal_columns,
        "interconnect_voltage_columns": resistor_columns,
        "interconnect_network_file_sha256": hashlib.sha256(network_path.read_bytes()).hexdigest()
        if network is not None else None,
        "interconnect_spice_file_sha256": hashlib.sha256(Path(cell_include).read_bytes()).hexdigest()
        if network is not None else None,
        "assumptions": [
            "SKY130 TT compact models with instance geometry scaled from micrometres to metres",
            "25 C fixed electrical temperature; no electrothermal feedback",
            "16 fF estimated lumped load per bitline, not extracted parasitic capacitance",
            "50 ps prescribed WL rise/fall edges and 1 ohm WL series resistance",
            "Initial stored state selected by nodeset and verified from the solved waveform",
            "No extracted diffusion areas/perimeters; intrinsic BSIM capacitances retained",
            "Layout-linked local sheet/contact resistances; estimated lumped bitline C is retained, not full RC extraction"
            if network is not None else "No extracted interconnect RC",
            "READ uses ideal controlled precharge switches (Ron 1 kohm, Roff 1 Tohm), no sense amplifier or recharge"
            if operation == "read" else
            "WRITE starts with target bitlines driven through 1 ohm; preceding driver setup energy is outside this event",
        ],
    }


def write_write_deck(out_path: str, data_path: str, pulse_width_ns: float, write_one: bool = True) -> str:
    """Compatibility wrapper: WRITE the target, starting from its opposite."""
    write_access_deck(out_path, data_path, "write", not write_one, pulse_width_ns)
    return out_path


def write_read_deck(out_path: str, data_path: str, pulse_width_ns: float, stored_one: bool = True) -> str:
    """Compatibility wrapper: READ a correctly precharged, initialized cell."""
    write_access_deck(out_path, data_path, "read", stored_one, pulse_width_ns)
    return out_path


def write_bias_op_deck(out_path: str, wl: float, bl: float, br: float, q: float, qb: float) -> str:
    """DC-only (no transient) snapshot of the cell at an EXPLICITLY FORCED
    (WL, BL, BR, Q, QB) bias, on the variant with Q/QB exposed as extra ports
    (data/sram_sp_cell_exposed.spice). This deliberately is NOT a transient:
    no real circuit necessarily passes through an arbitrary (Q, QB) pair,
    so this is a forced-bias diagnostic snapshot, not a measured access.

    Covers every operating point cases/run_bitcell_gallery.py uses:
      - HOLD:  wl=0,    bl=br=VDD,        q/qb = a real stored state
      - WRITE: wl=VDD,  bl/br = write target, q/qb = the SETTLED state
               (real currents here are tiny -- the access transistors have
               already finished charging/discharging by the settled state)
      - READ:  wl=VDD,  bl=br=VDD (precharged), q/qb = the stored state
      - CROWBAR: wl=bl=br=VDD, q=qb=0.9 -- the worst-case mid-switching
               instant, both pull-down NMOS partially on simultaneously
               (real currents symmetric when q==qb and bl==br, confirmed
               empirically: X0==X2, X1==X7, X5==X6 exactly)
    """
    exposed_path = os.path.join(_DATA_DIR, "sram_sp_cell_exposed.spice")
    includes = _INCLUDES[:-1] + [exposed_path]  # swap in the Q/QB-exposed variant
    lines = [f"* forced bias snapshot (DC op only, no transient): "
             f"wl={wl} bl={bl} br={br} q={q} qb={qb}"] \
        + [".option scale=1u", f".temp {TEMPERATURE_C}"] + [f'.include "{p}"' for p in includes] + [
        f"VVDD VDD 0 {VDD}",
        "VVSS VSS 0 0",
        f"VWL WL 0 {wl}",
        f"VBL BL 0 {bl}",
        f"VBR BR 0 {br}",
        f"VQ Q 0 {q}",
        f"VQB QB 0 {qb}",
        "X0 BL BR VDD VSS WL VSS VDD Q QB sram_sp_cell_exposed",
        ".op",
        ".control",
        "run",
        "print i(vvdd)",
        "print i(vvss)",
        "print i(vwl)",
        "print i(vbl)",
        "print i(vbr)",
        "print i(vq)",
        "print i(vqb)",
    ] + _device_print_lines() + [
        ".endc",
        ".end",
    ]
    _write(out_path, lines)
    return out_path


def write_selfconsistent_op_deck(out_path: str, wl: float, bl: float, br: float, q_guess: float, qb_guess: float) -> str:
    """DC-only snapshot for a REAL, physically-sustained state (HOLD, a
    settled WRITE, or READ) -- unlike write_bias_op_deck, Q/QB are NOT
    forced by ideal sources here; `.nodeset` only supplies a starting guess
    for the Newton iteration, so the solver finds whatever Q/QB the network
    actually implies (respecting KCL at those nodes), using the standard
    (non-exposed) sram_sp_cell subckt.

    Hard-forcing Q/QB while WL and BL/BR are externally driven creates
    artificial paths between the forcing sources and the bitlines.  The
    nodeset version lets the stored nodes settle according to circuit KCL.
    Its READ case still holds bitlines at VDD indefinitely; that is a DC
    diagnostic, not the discharging-bitline access in write_access_deck.

    Only usable for bias points that describe a genuinely stable network
    state (a nodeset is a mere hint, not a lock) -- NOT for the crowbar
    snapshot, whose whole point is an intentionally-impossible instant no
    self-consistent solve would ever reach on its own; that one still needs
    write_bias_op_deck's hard forcing.
    """
    lines = [f"* self-consistent bias snapshot (DC op, nodeset-guided): "
             f"wl={wl} bl={bl} br={br}"] \
        + _preamble() + [
        f"VWL WL 0 {wl}",
        f"VBL BL 0 {bl}",
        f"VBR BR 0 {br}",
        "X0 BL BR VDD VSS WL VSS VDD sram_sp_cell",
        f".nodeset v(x0.q)={q_guess} v(x0.qb)={qb_guess}",
        ".op",
        ".control",
        "run",
        "print i(vvdd)",
        "print i(vvss)",
        "print i(vwl)",
        "print i(vbl)",
        "print i(vbr)",
    ] + _device_print_lines() + [
        "print v(x0.q)",
        "print v(x0.qb)",
        ".endc",
        ".end",
    ]
    _write(out_path, lines)
    return out_path


# (instance, ngspice model suffix) for the 8 devices in sram_sp_cell, in
# netlist order. X3/X4 are declared as *_shortl in data/sram_sp_cell.spice but
# ngspice reports them under the plain special_pfet_pass name (verified against
# real output, not assumed) -- gds/spice_power.py::_DEVICES relies on the same.
_DEVICE_MODELS = [
    ("x0", "nfet_pass"), ("x1", "nfet_latch"), ("x2", "nfet_pass"),
    ("x3", "pfet_pass"), ("x4", "pfet_pass"), ("x5", "pfet_pass"),
    ("x6", "pfet_pass"), ("x7", "nfet_latch"),
]


def _device_print_lines():
    """`id` plus the two bulk-junction currents for every device.

    `ibd`/`ibs` are NOT redundant with `id`: `id` is the CHANNEL current only,
    so `|id * vds|` misses drain-bulk and source-bulk junction terms.  These
    matter relative to tiny HOLD leakage; ngspice's junction current also
    contains its numerical gmin conductance, which must not be mistaken for
    calibrated physical leakage.  See gds/spice_power.py::device_power_breakdown_w.

    BSIM3 exposes no `ig`/`is`/`ib` instance vectors in this ngspice build
    (checked directly: "Error: no such parameter ig"), so `ibd`/`ibs` are the
    available junction terms.  Transient gate/displacement current is not
    channel heat; the access deck saves intrinsic channel current explicitly.
    """
    lines = []
    for inst, model in _DEVICE_MODELS:
        dev = f"@m.x0.{inst}.msky130_fd_pr__special_{model}"
        lines += [f"print {dev}[{term}]" for term in ("id", "ibd", "ibs")]
    return lines


def _write(path, lines):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
