"""A SPICE output file or zero exit code does not establish successful analysis."""

from types import SimpleNamespace

import pytest

from gds import spice_power


@pytest.mark.parametrize("returncode,output", [
    (1, "analysis stopped"),
    (0, "doAnalyses: TRAN: Timestep too small; time = 1e-20"),
    (0, "run simulation(s) aborted"),
    (0, "Error: no such vector id_x0"),
    (0, "Fatal error: bad circuit"),
])
def test_spice_failures_are_not_accepted(monkeypatch, returncode, output):
    monkeypatch.setattr(spice_power, "find_ngspice", lambda: "ngspice")
    monkeypatch.setattr(spice_power.subprocess, "run", lambda *a, **kw:
                        SimpleNamespace(returncode=returncode, stdout=output, stderr=""))
    with pytest.raises(RuntimeError, match="ngspice failed"):
        spice_power._run("invalid.sp")


def test_successful_spice_output_is_retained(monkeypatch):
    monkeypatch.setattr(spice_power, "find_ngspice", lambda: "ngspice")
    monkeypatch.setattr(spice_power.subprocess, "run", lambda *a, **kw:
                        SimpleNamespace(returncode=0, stdout="No. of Data Rows : 500", stderr=""))
    assert "500" in spice_power._run("valid.sp")
