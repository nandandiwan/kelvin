"""Polygon geometry and instance contracts for the notebook SRAM pipeline."""
from copy import deepcopy
import json
import math

import gdstk
import pytest

from mesh.sram import _contract_hash, match_source_tags, prepare_sram_regions, thermal_context_options


@pytest.fixture(scope="module")
def prepared():
    return prepare_sram_regions()


def test_preserves_previous_thermal_parameters(prepared):
    _, manifest = prepared
    regions = manifest["regions"]
    by_name = {r["name"]: r for r in regions}
    assert min(r["z_min_um"] for r in regions) == pytest.approx(-50.3262)
    assert max(r["z_max_um"] for r in regions) == pytest.approx(3.04)
    assert by_name["passivation"]["material"] == "SiN"
    assert by_name["passivation"]["z_max_um"] - by_name["passivation"]["z_min_um"] == pytest.approx(1.0)
    background = next(r for r in regions if r["name"].startswith("active_background"))
    assert background["material"] == "SiO2"
    assert all(r["z_min_um"] == pytest.approx(-0.01) and r["z_max_um"] == 0.0
               for r in regions if r["role"] == "source")


def test_exact_mask_footprints_and_li1_components(prepared):
    context, manifest = prepared
    by_name = {r["name"]: r for r in manifest["regions"]}
    for name in ("poly", "li1", "mcon", "met1", "via", "met2"):
        before = context["layout"]["polygons_by_spec"][context["SKY130_SPECS"][name]]
        after = by_name[name]["volumes"]
        assert not gdstk.boolean([gdstk.Polygon(v["footprint_xy_um"]) for v in before],
                                 [gdstk.Polygon(v["footprint_xy_um"]) for v in after],
                                 "xor", precision=1e-6)
    assert len(by_name["li1"]["volumes"]) == 6
    assert sum(v["area_um2"] for v in by_name["li1"]["volumes"]) == pytest.approx(0.8949)


def test_material_volumes_fill_domain_without_gaps_or_overlaps(prepared):
    context, manifest = prepared
    regions = manifest["regions"]
    context["assert_no_material_overlaps"](regions, context["layout"]["boolean_grid_um"])
    volume = sum(sum(v["area_um2"] for v in r["volumes"]) * (r["z_max_um"]-r["z_min_um"])
                 for r in regions)
    assert volume == pytest.approx(1.896 * 53.3662, rel=1e-12)


def test_eight_source_ids_independent_of_region_order_and_tags(prepared):
    _, original = prepared
    manifest = deepcopy(original)
    manifest["regions"].reverse()
    for region in manifest["regions"]:
        region["physical_tag"] += 100
    expected = {tag+100: name for tag, name in match_source_tags(original).items()}
    assert match_source_tags(manifest) == expected
    assert set(expected.values()) == {f"X{i}" for i in range(8)}


def test_rejects_changed_channel_footprint(prepared):
    _, original = prepared
    manifest = deepcopy(original)
    source = next(r for r in manifest["regions"] if r["role"] == "source")
    source["volumes"][0]["footprint_xy_um"][0] = (0, 0)
    with pytest.raises(ValueError, match="polygon source"):
        match_source_tags(manifest)


@pytest.mark.parametrize("field,value", [("source_sha256", "wrong"), ("top_cell", "other"),
                                         ("source_mapping_sha256", "stale")])
def test_rejects_stale_identity(prepared, field, value):
    _, original = prepared
    manifest = dict(original, **{field: value})
    with pytest.raises(ValueError):
        match_source_tags(manifest)


@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf])
def test_bad_mesh_scale_rejected(value):
    with pytest.raises(ValueError, match="refine"):
        thermal_context_options(value)


def test_padding_adds_background_without_moving_channels(prepared):
    _, base = prepared
    _, padded = prepare_sram_regions(pad_um=0.1)
    assert match_source_tags(padded) == match_source_tags(base)
    old = base["process_model"]["domain_bbox_um"]
    new = padded["process_model"]["domain_bbox_um"]
    assert new == pytest.approx([old[0]-0.1, old[1]-0.1, old[2]+0.1, old[3]+0.1])


def test_manifest_contract_survives_json_round_trip(prepared):
    _, manifest = prepared
    assert _contract_hash(manifest) == _contract_hash(json.loads(json.dumps(manifest)))


def test_equal_volume_source_tag_swap_changes_contract(prepared):
    _, original = prepared
    manifest = deepcopy(original)
    tags = {name: tag for tag, name in match_source_tags(manifest).items()}
    x0 = next(r for r in manifest["regions"] if r["physical_tag"] == tags["X0"])
    x2 = next(r for r in manifest["regions"] if r["physical_tag"] == tags["X2"])
    x0["physical_tag"], x2["physical_tag"] = x2["physical_tag"], x0["physical_tag"]
    # Merely matching footprint names or checking equal source volumes would
    # miss this corruption of the tag correspondence to an unchanged MSH.
    assert match_source_tags(manifest) != match_source_tags(original)
    assert _contract_hash(manifest) != _contract_hash(original)


def test_import_rejects_tampered_manifest_before_reading_mesh(prepared, tmp_path):
    pytest.importorskip("dolfinx")
    from mesh.sram import load_sram_mesh

    _, original = prepared
    manifest = deepcopy(original)
    manifest["mesh_sha256"] = "unchanged-mesh-fingerprint"
    manifest["mesh_manifest_contract_sha256"] = _contract_hash(manifest)
    manifest["regions"][0]["material"] = "SiO2"
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest contract changed"):
        load_sram_mesh(path, {f"X{i}": 0.0 for i in range(8)})
