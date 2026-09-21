"""Executable discovery is portable and never silently ignores an override."""

from pathlib import Path

import pytest

from gds import spice_power


def test_explicit_ngspice_override_wins(monkeypatch):
    monkeypatch.setenv("KELVIN_NGSPICE", "/chosen/ngspice")
    monkeypatch.setattr(spice_power.shutil, "which", lambda name: name)
    assert spice_power.find_ngspice() == "/chosen/ngspice"


def test_invalid_override_does_not_fall_back(monkeypatch):
    monkeypatch.setenv("KELVIN_NGSPICE", "/missing/ngspice")
    monkeypatch.setattr(
        spice_power.shutil, "which", lambda name: "/path/ngspice" if name == "ngspice" else None
    )
    with pytest.raises(FileNotFoundError, match="KELVIN_NGSPICE"):
        spice_power.find_ngspice()


def test_path_precedes_local_install(monkeypatch):
    monkeypatch.delenv("KELVIN_NGSPICE", raising=False)
    monkeypatch.setattr(spice_power.shutil, "which", lambda name: "/path/ngspice")
    assert spice_power.find_ngspice() == "/path/ngspice"


def test_local_install_precedes_legacy(monkeypatch):
    monkeypatch.delenv("KELVIN_NGSPICE", raising=False)
    monkeypatch.setattr(spice_power.shutil, "which", lambda name: None)
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    monkeypatch.setattr(spice_power.os, "access", lambda path, mode: True)
    assert spice_power.find_ngspice() == str(spice_power._LOCAL_NGSPICE)


def test_missing_ngspice_has_actionable_error(monkeypatch):
    monkeypatch.delenv("KELVIN_NGSPICE", raising=False)
    monkeypatch.setattr(spice_power.shutil, "which", lambda name: None)
    monkeypatch.setattr(Path, "is_file", lambda self: False)
    with pytest.raises(FileNotFoundError, match="Install ngspice on PATH"):
        spice_power.find_ngspice()
