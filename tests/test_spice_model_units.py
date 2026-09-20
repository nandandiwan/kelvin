"""Real-ngspice guards for SKY130 micrometre scaling and BSIM3 Id semantics."""

import re
import subprocess

import pytest

from gds.spice_netlist import (
    VDD,
    _INCLUDES,
    write_bias_op_deck,
    write_hold_deck,
    write_read_deck,
    write_selfconsistent_op_deck,
    write_write_deck,
)
from gds.spice_power import _DEVICES, find_ngspice


@pytest.fixture(scope="module")
def ngspice_binary():
    try:
        return find_ngspice()
    except FileNotFoundError:
        pytest.skip("ngspice is not installed")


def _simulate(binary, deck):
    result = subprocess.run(
        [binary, "-b", str(deck)], capture_output=True, text=True, timeout=30,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output[-4000:]
    assert "timestep too small" not in output.lower(), output[-4000:]
    assert "simulation(s) aborted" not in output.lower(), output[-4000:]
    return output


def _scalar(output, vector):
    match = re.findall(
        rf"^\s*{re.escape(vector)}\s*=\s*([-+\d.eE]+)\s*$", output, re.M,
    )
    assert match, f"Missing scalar {vector!r}:\n{output[-4000:]}"
    return float(match[-1])


@pytest.mark.parametrize("kind", ["hold", "selfconsistent", "forced", "read", "write"])
def test_every_deck_uses_physical_transistor_dimensions(tmp_path, ngspice_binary, kind):
    """Catch omission of SCALE in shared and separately assembled preambles.

    Inspect actual parsed BSIM instances, rather than just checking that a
    string occurs in a generated deck. Query an OP after the transient export
    so length/width are scalar even when those vectors were saved during TRAN.
    """
    deck, data = tmp_path / "units.sp", tmp_path / "units.txt"
    if kind == "hold":
        write_hold_deck(str(deck))
    elif kind == "selfconsistent":
        write_selfconsistent_op_deck(str(deck), VDD, VDD, VDD, VDD - 0.01, 0.01)
    elif kind == "forced":
        write_bias_op_deck(str(deck), VDD, VDD, VDD, VDD / 2, VDD / 2)
    elif kind == "read":
        write_read_deck(str(deck), str(data), 0.239519, stored_one=True)
    else:
        write_write_deck(str(deck), str(data), 0.239519, write_one=True)
    queries = ["op"]
    for path, _, _ in _DEVICES.values():
        queries.extend(f"print @{path}[{parameter}]" for parameter in ("l", "w"))
    deck.write_text(deck.read_text().replace(".endc", "\n".join(queries) + "\n.endc"))
    output = _simulate(ngspice_binary, deck)
    for instance, (path, _, _) in _DEVICES.items():
        expected_l = 0.025e-6 if instance in {"X3", "X4"} else 0.150e-6
        expected_w = 0.210e-6 if instance in {"X1", "X7"} else 0.140e-6
        assert _scalar(output, f"@{path}[l]") == pytest.approx(expected_l, rel=1e-8, abs=0)
        assert _scalar(output, f"@{path}[w]") == pytest.approx(expected_w, rel=1e-8, abs=0)


@pytest.mark.parametrize("polarity", ["nfet", "pfet"])
@pytest.mark.parametrize("reverse", [False, True])
def test_bsim3_id_is_channel_magnitude_in_both_orientations(
    tmp_path, ngspice_binary, polarity, reverse,
):
    """For these models Id is unsigned, not signed terminal drain current.

    The four cases also protect capacitance scale: a submicron transistor's
    gate capacitance cannot silently become the old tens-of-microfarads value.
    Exact calibrated current/capacitance values are deliberately not pinned.
    """
    drain, source = (("hi", "0") if polarity == "nfet" else ("0", "hi"))
    if reverse:
        drain, source = source, drain
    gate, bulk = (("hi", "0") if polarity == "nfet" else ("0", "hi"))
    path = f"@m.xdut.msky130_fd_pr__special_{polarity}_pass"
    lines = ["* BSIM3 intrinsic current convention", ".option scale=1u"]
    lines += [f'.include "{include}"' for include in _INCLUDES[:-1]]
    lines += [
        f"VHI hi 0 {VDD}",
        f"XDUT {drain} {gate} {source} {bulk} "
        f"sky130_fd_pr__special_{polarity}_pass l=.150 w=.140",
        ".op", ".control", "run",
    ]
    lines += [f"print {path}[{parameter}]" for parameter in ("id", "vds", "cgg", "l", "w")]
    lines += [".endc", ".end"]
    deck = tmp_path / "current_convention.sp"
    deck.write_text("\n".join(lines) + "\n")
    output = _simulate(ngspice_binary, deck)
    assert _scalar(output, f"{path}[id]") > 0.0
    # BSIM3's Vds is also normalized by MOS polarity, not the external Vd-Vs.
    assert _scalar(output, f"{path}[vds]") == pytest.approx(-VDD if reverse else VDD)
    assert 1e-18 < _scalar(output, f"{path}[cgg]") < 1e-14
    assert _scalar(output, f"{path}[l]") == pytest.approx(0.150e-6, rel=1e-8, abs=0)
    assert _scalar(output, f"{path}[w]") == pytest.approx(0.140e-6, rel=1e-8, abs=0)
