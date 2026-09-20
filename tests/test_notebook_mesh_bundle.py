"""Partial notebook reruns must not certify stale thermal material metadata."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from mesh.gds_notebook import (
    build_notebook_mesh_bundle, build_region_manifest, create_context,
    region_contract_sha256, region_layout_identity,
)
from mesh.sram import _contract_hash, prepare_sram_regions, seal_sram_manifest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def generic_gds(tmp_path):
    """Small SKY130-like cell; no separately downloaded PDK examples needed."""
    gdstk = pytest.importorskip("gdstk")
    library = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = library.new_cell("synthetic_inverter")
    for lower, upper, layer, datatype in (
        ((0, 1), (2, 2), 64, 20),       # nwell
        ((0.2, 0.2), (1.8, 0.7), 65, 20),
        ((0.2, 1.2), (1.8, 1.7), 65, 20),
        ((0.9, 0), (1.1, 2), 66, 20),   # poly crossing both diff strips
        ((0.25, 0.3), (0.42, 0.47), 66, 44),
        ((0.25, 1.3), (0.42, 1.47), 66, 44),
        ((0.2, 0.25), (0.5, 1.65), 67, 20),
        ((0.25, 0.52), (0.42, 0.69), 67, 44),
        ((0.2, 0.25), (0.5, 1.65), 68, 20),
    ):
        cell.add(gdstk.rectangle(lower, upper, layer=layer, datatype=datatype))
    path = tmp_path / "synthetic_inverter.gds"
    library.write_gds(path)
    return path


def _stub_mesher(regions, mesh, quality, brep, *_):
    """Cheap stand-in which records exactly the inputs supplied to the mesher."""
    contract = region_contract_sha256(regions)
    mesh.write_text("meshed " + contract)
    quality.write_text("validated quality")
    brep.write_text("validated geometry")
    return {
        "mesh_regions_sha256": contract,
        "mesh_sha256": hashlib.sha256(mesh.read_bytes()).hexdigest(),
        "tetrahedron_count": 1, "node_count": 4,
        "shared_interface_surface_count": 0, "regions": [],
    }


@pytest.fixture
def context(tmp_path):
    context, _ = prepare_sram_regions()
    context.update(OUTPUT_DIR=tmp_path, build_and_validate_mesh=_stub_mesher)
    return context


def _material(manifest, name="silicon_substrate"):
    return next(r["material"] for r in manifest["regions"] if r["name"] == name)


def _change_material(context):
    next(r for r in context["regions"] if r["name"] == "silicon_substrate")["material"] = "SiO2"


def _cell(cell_id):
    notebook = json.loads((ROOT / "read_gds.ipynb").read_text())
    return "".join(next(c["source"] for c in notebook["cells"] if c["id"] == cell_id))


def test_partial_rerun_publishes_current_material_and_process_not_stale_json(context):
    old, _, paths = build_notebook_mesh_bundle(context)
    context["manifest"] = old  # Deliberately retain the prior notebook variable too.
    _change_material(context)
    context["process_model"]["current_user_note"] = "new thermal assumptions"
    new, stats, new_paths = build_notebook_mesh_bundle(context)
    assert paths == new_paths
    assert _material(old) == "Si_bulk"
    assert _material(new) == "SiO2"
    assert new["process_model"]["current_user_note"] == "new thermal assumptions"
    assert new["mesh_regions_sha256"] == stats["mesh_regions_sha256"]
    assert new["mesh_manifest_contract_sha256"] == _contract_hash(new)
    assert json.loads(paths["manifest"].read_text()) == json.loads(json.dumps(new))


def test_notebook_mesh_cell_alone_rebuilds_manifest_from_current_state(context):
    old, _, paths = build_notebook_mesh_bundle(context)
    context.update(manifest=old, JSON_PATH=paths["manifest"])
    _change_material(context)
    # Execute the actual notebook cell, without its preceding manifest cell.
    exec(compile(_cell("49218f14"), "<notebook-mesh-cell>", "exec"), context)
    saved = json.loads(paths["manifest"].read_text())
    assert _material(saved) == "SiO2"
    assert saved["mesh_manifest_contract_sha256"] == _contract_hash(saved)


def test_notebook_manifest_preview_does_not_overwrite_completed_bundle(context):
    _, _, paths = build_notebook_mesh_bundle(context)
    saved = paths["manifest"].read_bytes()
    _change_material(context)
    exec(compile(_cell("cee4122c"), "<notebook-preview-cell>", "exec"), context)
    assert _material(context["manifest_preview"]) == "SiO2"
    assert paths["manifest"].read_bytes() == saved


def test_failed_mesh_does_not_modify_existing_valid_bundle(context):
    _, _, paths = build_notebook_mesh_bundle(context)
    old_files = {key: path.read_bytes() for key, path in paths.items() if path.exists()}
    _change_material(context)

    def fail_after_writing_partial_mesh(regions, mesh, quality, brep, *_):
        mesh.write_text("incomplete new mesh")
        brep.write_text("new CAD")
        raise RuntimeError("quality validation failed")

    context["build_and_validate_mesh"] = fail_after_writing_partial_mesh
    with pytest.raises(RuntimeError, match="quality validation failed"):
        build_notebook_mesh_bundle(context)
    assert {key: paths[key].read_bytes() for key in old_files} == old_files
    assert not list(context["OUTPUT_DIR"].glob(".*-build-*"))


def test_first_failed_build_does_not_publish_solver_manifest(context):
    def fail(*_):
        raise RuntimeError("meshing failed")
    context["build_and_validate_mesh"] = fail
    with pytest.raises(RuntimeError, match="meshing failed"):
        build_notebook_mesh_bundle(context)
    assert not list(context["OUTPUT_DIR"].iterdir())


def test_stale_on_disk_manifest_cannot_be_resealed_with_new_material_mesh(context):
    _, old_stats, paths = build_notebook_mesh_bundle(context)
    old_json = paths["manifest"].read_bytes()
    _change_material(context)
    new_stats = _stub_mesher(context["regions"], paths["mesh"], paths["quality"], paths["brep"])
    with pytest.raises(ValueError, match="region/material contract"):
        seal_sram_manifest(paths["manifest"], paths["mesh"], mesh_stats=new_stats)
    with pytest.raises(ValueError, match="changed after validation"):
        seal_sram_manifest(paths["manifest"], paths["mesh"], mesh_stats=old_stats)
    assert paths["manifest"].read_bytes() == old_json


def test_blind_reseal_is_rejected_without_validation_proof(context):
    _, _, paths = build_notebook_mesh_bundle(context)
    saved = paths["manifest"].read_bytes()
    with pytest.raises(ValueError, match="region/material contract"):
        seal_sram_manifest(paths["manifest"], paths["mesh"])
    assert paths["manifest"].read_bytes() == saved


def test_matching_validation_proof_can_reseal(context):
    _, stats, paths = build_notebook_mesh_bundle(context)
    assert seal_sram_manifest(paths["manifest"], paths["mesh"], mesh_stats=stats) == paths["manifest"]
    saved = json.loads(paths["manifest"].read_text())
    assert saved["mesh_manifest_contract_sha256"] == _contract_hash(saved)


def test_mesher_snapshot_is_independent_of_mutable_notebook_regions(context):
    def mutate_notebook_after_snapshot(regions, *args):
        _change_material(context)
        return _stub_mesher(regions, *args)
    context["build_and_validate_mesh"] = mutate_notebook_after_snapshot
    saved, stats, _ = build_notebook_mesh_bundle(context)
    assert _material(context) == "SiO2"
    assert _material(saved) == "Si_bulk"
    assert stats["mesh_regions_sha256"] == region_contract_sha256(saved["regions"])


@pytest.mark.parametrize("changed", ["selected_top", "layout", "missing_provenance", "bbox"])
def test_partial_case_switch_requires_region_rebuild(context, changed):
    if changed == "selected_top":
        context["TOP_CELL_NAME"] = "different_cell"
    elif changed == "layout":
        context["layout"]["geometry_hash"] = "different_layout"
    elif changed == "bbox":
        context["layout"]["bbox_um"][0] -= 0.1
    else:
        context.pop("REGION_LAYOUT_IDENTITY")
    with pytest.raises(ValueError, match="rerun .*region construction"):
        build_notebook_mesh_bundle(context)


def test_generic_notebook_bundle_remains_supported(tmp_path, generic_gds):
    context = create_context()
    path = generic_gds
    layout = context["extract_layout"](path)
    context.update(layout=layout, GDS_PATH=path, TOP_CELL_NAME=None, OUTPUT_DIR=tmp_path,
                   OUTPUT_STEM="inverter", SRAM_PIPELINE=False)
    regions, model, used = context["build_sky130_regions"]()
    context.update(regions=regions, process_model=model, used_gds_specs=used,
                   REGION_LAYOUT_IDENTITY=region_layout_identity(layout),
                   build_and_validate_mesh=_stub_mesher)
    saved, stats, paths = build_notebook_mesh_bundle(context)
    assert saved["top_cell"] == "synthetic_inverter"
    assert "source_mapping_sha256" not in saved
    assert saved["mesh_regions_sha256"] == stats["mesh_regions_sha256"]
    assert paths["manifest"].is_file()


def test_notebook_configuration_case_switch_resets_sram_only_state(monkeypatch, generic_gds):
    monkeypatch.chdir(ROOT)
    namespace = {}
    config_cell = compile(_cell("d30a3a4c"), "<notebook-config-cell>", "exec")
    monkeypatch.setenv("GDS_CASE", "sram")
    monkeypatch.delenv("GDS_INPUT", raising=False)
    monkeypatch.delenv("GDS_TOP_CELL", raising=False)
    exec(config_cell, namespace)
    assert namespace["ROOT"] == ROOT
    assert namespace["GDS_PATH"] == ROOT / "data/sram22_64x22m4w22.gds"
    assert namespace["CHANNEL_DEPTH_UM"] == 0.01
    assert namespace["ACTIVE_BACKGROUND_MATERIAL"] == "SiO2"
    namespace["REGION_LAYOUT_IDENTITY"] = {"old": "sram"}
    monkeypatch.setenv("GDS_CASE", "generic")
    monkeypatch.setenv("GDS_INPUT", str(generic_gds))
    exec(config_cell, namespace)
    assert namespace["GDS_PATH"] == generic_gds
    assert namespace["CHANNEL_DEPTH_UM"] is None
    assert namespace["ACTIVE_BACKGROUND_MATERIAL"] == "Si_bulk"
    assert namespace["TOP_PASSIVATION_THICKNESS_UM"] == 0.0
    assert "REGION_LAYOUT_IDENTITY" not in namespace


def test_notebook_generic_mode_requires_explicit_input(monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("GDS_CASE", "generic")
    monkeypatch.delenv("GDS_INPUT", raising=False)
    with pytest.raises(ValueError, match="Generic mode requires GDS_INPUT"):
        exec(compile(_cell("d30a3a4c"), "<notebook-config-cell>", "exec"), {})


def test_notebook_configuration_supports_standalone_clone(monkeypatch, tmp_path):
    clone = tmp_path / "standalone-clone"
    (clone / "data").mkdir(parents=True)
    (clone / "mesh").mkdir()
    (clone / "mesh/gds_notebook.py").touch()  # Repository discovery marker.
    (clone / "read_gds.ipynb").touch()
    (clone / "data/sram22_64x22m4w22.gds").touch()  # Config checks presence only.
    monkeypatch.chdir(clone)
    monkeypatch.syspath_prepend(str(clone))
    monkeypatch.setenv("GDS_CASE", "sram")
    monkeypatch.delenv("GDS_INPUT", raising=False)
    monkeypatch.delenv("GDS_TOP_CELL", raising=False)
    namespace = {}
    exec(compile(_cell("d30a3a4c"), "<notebook-config-cell>", "exec"), namespace)
    assert namespace["ROOT"] == clone
    assert namespace["GDS_PATH"] == clone / "data/sram22_64x22m4w22.gds"
    assert namespace["SKY130_BUNDLE_DIR"] == clone / "data/sky130"
    assert namespace["sky130_sources"] is None
    assert not (clone.parent / "data").exists()


def test_real_mesher_records_exact_region_material_contract(tmp_path):
    pytest.importorskip("gmsh")
    context = create_context(layer_profile="none", OUTPUT_STEM="one_box")
    component = {"footprint_xy_um": [[0, 0], [1, 0], [1, 1], [0, 1]],
                 "area_um2": 1.0, "bbox_um": [0, 0, 1, 1]}
    region = context["make_region"](1, "box", "Si_bulk", 0, 1, "#000000",
                                    [component], role="substrate")
    mesh = tmp_path / "box.msh"
    stats = context["build_and_validate_mesh"](
        [region], mesh, tmp_path / "quality.msh", tmp_path / "box.brep", 0.5, 0.01,
    )
    assert stats["mesh_regions_sha256"] == region_contract_sha256([region])
    assert stats["mesh_sha256"] == hashlib.sha256(mesh.read_bytes()).hexdigest()
    changed = deepcopy(region)
    changed["material"] = "SiO2"
    assert stats["mesh_regions_sha256"] != region_contract_sha256([changed])
