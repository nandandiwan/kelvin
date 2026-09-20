"""Deck contracts: physical units, initialized loading, and saved intrinsic data."""

import math

import pytest

from gds import spice_netlist as decks


@pytest.mark.parametrize("writer,args", [
    (decks.write_hold_deck, ()),
    (decks.write_selfconsistent_op_deck, (0, 1.8, 1.8, 1.79, .01)),
    (decks.write_bias_op_deck, (1.8, 1.8, 1.8, .9, .9)),
])
def test_every_dc_deck_scales_micrometre_instances(tmp_path, writer, args):
    path = tmp_path / "op.cir"
    writer(str(path), *args)
    text = path.read_text()
    assert text.count(".option scale=1u") == 1
    assert text.index(".option scale=1u") < text.index(".include")
    assert ".temp 25.0" in text


@pytest.mark.parametrize("operation", ["read", "write"])
@pytest.mark.parametrize("initial", [False, True])
def test_transient_deck_contract(tmp_path, operation, initial):
    path, data = tmp_path / "access.cir", tmp_path / "access.dat"
    metadata = decks.write_access_deck(path, data, operation, initial, .24, .5)
    text = path.read_text()
    assert metadata["initial_stored_one"] is initial
    assert metadata["final_stored_one"] is (initial if operation == "read" else not initial)
    assert metadata["end_s"] > metadata["access_end_s"] > metadata["access_start_s"]
    assert metadata["max_step_s"] == .5e-12
    assert metadata["columns"][0] == "time"
    assert "set wr_singlescale" in text and "set wr_vecnames" in text
    assert ".option scale=1u" in text
    assert "CBL BL 0 1.6e-14" in text and "CBR BR 0 1.6e-14" in text
    for instance, model in decks._DEVICE_MODELS:
        for term in ("id", "ibd", "ibs", "vds", "vbs", "l", "w"):
            assert f"{term}_{instance}" in metadata["columns"]
            assert f".save @m.x0.{instance}.msky130_fd_pr__special_{model}[{term}]" in text
    if operation == "read":
        assert "SBL BL VDD pre 0 precharge" in text
        assert "SBR BR VDD pre 0 precharge" in text
        assert metadata["precharge"]["release_end_s"] < metadata["access_start_s"]
    else:
        assert "VBL bl_src" in text and "VBR br_src" in text


@pytest.mark.parametrize("parameter", ["pulse_width_ns", "max_step_ps"])
@pytest.mark.parametrize("value", [0, -1, math.nan, math.inf, True])
def test_invalid_timing_fails_before_writing(tmp_path, parameter, value):
    path = tmp_path / "bad.cir"
    with pytest.raises(ValueError, match=parameter):
        decks.write_access_deck(path, tmp_path / "bad.dat", **{parameter: value})
    assert not path.exists()


def test_legacy_write_wrapper_keeps_target_semantics(tmp_path):
    path = tmp_path / "write.cir"
    assert decks.write_write_deck(str(path), str(tmp_path / "write.dat"), .24, True) == str(path)
    assert "* WRITE: initial Q=0, target Q=1" in path.read_text()


def test_basename_output_path_supported(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    decks.write_hold_deck("hold.cir")
    assert (tmp_path / "hold.cir").exists()
