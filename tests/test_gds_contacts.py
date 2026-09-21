"""Whole-cut landing validation without discarding SRAM contact overhangs."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
gdstk = pytest.importorskip("gdstk")

from mesh.gds_contacts import SPECS, classify_sky130_licon, validate_sky130_contacts


def _component(polygon):
    (x0, y0), (x1, y1) = polygon.bounding_box()
    return {"footprint_xy_um": polygon.points.tolist(), "area_um2": polygon.area(),
            "bbox_um": [x0, y0, x1, y1]}


def _layout(**masks):
    return {"boolean_grid_um": 1e-6,
            "polygons_by_spec": {SPECS[name]: [_component(p) for p in polygons]
                                 for name, polygons in masks.items()}}


def _basic(*, partial=True, core=True):
    cut = gdstk.rectangle((0, 0), (0.17, 0.17))
    data = _layout(licon1=[cut], li1=[cut.copy()],
                   diff=[gdstk.rectangle((0, 0), (0.14 if partial else 0.17, 0.17))])
    if core:
        data["polygons_by_spec"][SPECS["memory_core"]] = [_component(cut.copy())]
    return data


def test_core_partial_contact_retains_whole_cut():
    data = _basic()
    result = classify_sky130_licon(data)
    assert len(result["diff_components"]) == 1
    assert result["poly_components"] == []
    assert result["diff_components"][0] == data["polygons_by_spec"][SPECS["licon1"]][0]
    assert result["report"]["whole_cut_area_um2"] == pytest.approx(0.0289)
    assert result["report"]["outside_lower_area_um2"] == pytest.approx(0.0051)
    assert result["report"]["core_partial_cut_count"] == 1


def test_full_contact_outside_core_remains_valid():
    result = classify_sky130_licon(_basic(partial=False, core=False))
    assert result["report"]["core_partial_cut_count"] == 0


@pytest.mark.parametrize("core_kind", ["absent", "partial"])
def test_partial_contact_requires_whole_cut_inside_core(core_kind):
    data = _basic(core=False)
    if core_kind == "partial":
        data["polygons_by_spec"][SPECS["memory_core"]] = [
            _component(gdstk.rectangle((0, 0), (0.08, 0.17)))
        ]
    with pytest.raises(ValueError, match="outside memory core"):
        classify_sky130_licon(data)


@pytest.mark.parametrize("missing", ["diff", "li1"])
def test_disconnected_cut_rejected_even_in_core(missing):
    data = _basic()
    data["polygons_by_spec"].pop(SPECS[missing])
    with pytest.raises(ValueError, match="does not (land|overlap)"):
        classify_sky130_licon(data)


def test_cut_on_poly_and_diff_is_rejected_instead_of_split():
    data = _basic()
    data["polygons_by_spec"][SPECS["poly"]] = [
        _component(gdstk.rectangle((0.14, 0), (0.17, 0.17)))
    ]
    with pytest.raises(ValueError, match="ambiguous landing"):
        classify_sky130_licon(data)


def test_floating_cut_cannot_hide_in_aggregate_coverage():
    data = _basic(partial=False)
    floating = gdstk.rectangle((0.2, 0), (0.37, 0.17))
    data["polygons_by_spec"][SPECS["licon1"]].append(_component(floating))
    data["polygons_by_spec"][SPECS["li1"]].append(_component(floating.copy()))
    data["polygons_by_spec"][SPECS["memory_core"]] = [
        _component(gdstk.rectangle((0, 0), (0.4, 0.2)))
    ]
    with pytest.raises(ValueError, match="does not land"):
        validate_sky130_contacts(data)


def test_tangent_contact_is_not_positive_area_landing():
    data = _basic()
    data["polygons_by_spec"][SPECS["diff"]] = [
        _component(gdstk.rectangle((-0.17, 0), (0, 0.17)))
    ]
    with pytest.raises(ValueError, match="does not land"):
        classify_sky130_licon(data)


def test_mcon_requires_both_landings_per_cut():
    cut = gdstk.rectangle((0, 0), (0.17, 0.17))
    data = _layout(mcon=[cut], li1=[cut.copy()], memory_core=[cut.copy()])
    with pytest.raises(ValueError, match="does not overlap both"):
        validate_sky130_contacts(data)


def test_bundled_sram_contacts_preserve_exact_masks():
    gds_path = Path(__file__).resolve().parents[1] / "data/sram22_64x22m4w22.gds"
    library = gdstk.read_gds(str(gds_path))
    cell = next(c for c in library.cells if c.name == "sram_sp_cell")
    by = {}
    for polygon in cell.get_polygons():
        by.setdefault((polygon.layer, polygon.datatype), []).append(polygon)
    data = {"boolean_grid_um": 1e-6, "polygons_by_spec": {
        spec: [_component(p) for p in gdstk.boolean(polygons, [], "or", precision=1e-6)]
        for spec, polygons in by.items()
    }}
    result = classify_sky130_licon(data)
    assert result["report"]["cut_count"] == 10
    assert result["report"]["poly_cut_count"] == 2
    assert result["report"]["diff_or_tap_cut_count"] == 8
    assert result["report"]["whole_cut_area_um2"] == pytest.approx(0.2312)
    assert result["report"]["outside_lower_area_um2"] == pytest.approx(0.02975)
    preserved = [gdstk.Polygon(c["footprint_xy_um"])
                 for c in result["poly_components"] + result["diff_components"]]
    assert not gdstk.boolean(preserved, by[SPECS["licon1"]], "xor", precision=1e-6)
    rows = validate_sky130_contacts(data)
    assert {tuple(row["connector_spec"]) for row in rows} == {(66, 44), (67, 44), (68, 44)}
    assert all(row["validated"] for row in rows)
    assert all(c["lower_overlap_um2"] > 0 and c["upper_overlap_um2"] > 0
               for row in rows for c in row["contacts"])
