"""Saved thermal fields must identify the corrected transistor-power map."""

import hashlib
import json

import pytest


@pytest.fixture
def renderer(monkeypatch):
    pytest.importorskip("dolfinx")
    from cases import render_transient_layers

    monkeypatch.setattr(render_transient_layers, "mapping_revision", lambda: "current-map")
    return render_transient_layers


@pytest.mark.parametrize("metadata", [{}, {"source_mapping_sha256": "old-map"}])
def test_old_or_unversioned_transient_fields_are_rejected(renderer, metadata):
    metadata = dict(metadata, spice_model_revision=renderer.SPICE_MODEL_REVISION)
    with pytest.raises(ValueError, match="rerun cases/run_bitcell_transient.py"):
        renderer._validate_mapping_metadata(metadata)


def test_current_transient_mapping_is_accepted(renderer):
    renderer._validate_mapping_metadata({"source_mapping_sha256": "current-map",
                                          "spice_model_revision": renderer.SPICE_MODEL_REVISION})


@pytest.mark.parametrize("revision", [None, "old-energy-model"])
def test_old_pulse_energy_model_is_rejected(renderer, revision):
    with pytest.raises(ValueError, match="access-energy model.*rerun"):
        renderer._validate_mapping_metadata({
            "source_mapping_sha256": "current-map", "instantaneous": True,
            "access_energy_model_revision": revision,
        })


def test_current_pulse_energy_model_is_accepted(renderer):
    renderer._validate_mapping_metadata({
        "source_mapping_sha256": "current-map", "instantaneous": True,
        "access_energy_model_revision": renderer.ACCESS_ENERGY_MODEL_REVISION,
        "spice_model_revision": renderer.SPICE_MODEL_REVISION,
    })


def test_pre_unit_correction_fields_are_rejected(renderer):
    with pytest.raises(ValueError, match="SPICE geometry-unit correction"):
        renderer._validate_mapping_metadata({"source_mapping_sha256": "current-map"})


@pytest.fixture
def spice_fields_meta(renderer, tmp_path):
    artifact = tmp_path / "waveform.txt"
    artifact.write_text("complete electrical waveform")
    workload = {
        "kind": "spice-transient", "thermal_power_revision": renderer.THERMAL_POWER_REVISION,
        "spice_artifact_dir": str(tmp_path),
        "spice_artifact_sha256": {artifact.name: hashlib.sha256(artifact.read_bytes()).hexdigest()},
    }
    path = tmp_path / "power_workload.json"
    path.write_text(json.dumps(workload))
    return {"power_model": "spice-transient", "instantaneous": True,
            "spice_model_revision": renderer.SPICE_MODEL_REVISION,
            "source_mapping_sha256": "current-map",
            "thermal_power_revision": renderer.THERMAL_POWER_REVISION,
            "power_workload_path": str(path),
            "power_workload_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_spice_fields_do_not_require_legacy_rectangle_revision(renderer, spice_fields_meta):
    renderer._validate_mapping_metadata(spice_fields_meta)


def test_obsolete_spice_transfer_is_rejected(renderer, spice_fields_meta):
    spice_fields_meta["thermal_power_revision"] = "old"
    with pytest.raises(ValueError, match="obsolete SPICE-to-thermal"):
        renderer._validate_mapping_metadata(spice_fields_meta)


def test_changed_workload_metadata_is_rejected(renderer, spice_fields_meta):
    from pathlib import Path
    Path(spice_fields_meta["power_workload_path"]).write_text("{}")
    with pytest.raises(ValueError, match="workload is missing or changed"):
        renderer._validate_mapping_metadata(spice_fields_meta)


def test_changed_electrical_waveform_is_rejected(renderer, spice_fields_meta):
    from pathlib import Path
    path = Path(spice_fields_meta["power_workload_path"]).parent / "waveform.txt"
    path.write_text("different electrical waveform")
    with pytest.raises(ValueError, match="electrical artifact is missing or changed"):
        renderer._validate_mapping_metadata(spice_fields_meta)


@pytest.mark.parametrize("history", [
    [300.0, 301.0, 302.0, 301.0, 300.0],
    [300.0, 303.0, 301.0, 304.0, 300.1],
    [300.0, 301.0, 302.0],
    [300.0, 300.0, 300.0],
])
def test_frame_selection_retains_pulse_peak_and_chronology(renderer, history):
    import numpy as np
    indices = renderer._select_uniform_dt_frames(history, 3)
    assert int(np.argmax(history)) in indices
    assert indices[0] == 0 and indices[-1] == len(history) - 1
    assert indices == sorted(set(indices))


def test_frame_selection_requires_multiple_frames(renderer):
    with pytest.raises(ValueError, match="At least two frames"):
        renderer._select_uniform_dt_frames([300, 301], 1)
