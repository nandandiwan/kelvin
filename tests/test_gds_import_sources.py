"""Power/identity accounting for arbitrary polygon-prism imports.

The fixture is a tiny conformal rectangular domain split into an L-shaped
source and its square complement. It exercises the public MSH import path,
not a mocked registry or a bounding-box substitute for the source footprint.
"""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip("dolfinx")
gmsh = pytest.importorskip("gmsh")

from mesh.gds_import import ImportedHeatSource, load_tagged_msh
from physics.coeffs import build_coeffs
from post.budget import power_balance, print_power_balance, verify_device_source_powers
from solve.steady import solve_steady
from spec.chip import BoundaryConditions


POWER_BY_TAG = {11: 1.5e-6, 19: 2.0e-6}
DEVICE_BY_TAG = {11: "X0", 19: "X7"}
POWER_BY_DEVICE = {"X0": 1.5e-6, "X7": 2.0e-6}


@pytest.fixture(scope="module")
def imported_mesh_files(tmp_path_factory):
    directory = tmp_path_factory.mktemp("polygon_source_import")
    msh_path = directory / "l_source.msh"
    manifest_path = directory / "l_source.json"
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("l_source")
        occ = gmsh.model.occ
        footprint = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
        points = [occ.addPoint(x, y, 0) for x, y in footprint]
        lines = [occ.addLine(a, b) for a, b in zip(points, points[1:] + points[:1])]
        surface = occ.addPlaneSurface([occ.addCurveLoop(lines)])
        extruded = occ.extrude([(2, surface)], 0, 0, 1)
        l_volume = next(tag for dim, tag in extruded if dim == 3)
        square_volume = occ.addBox(1, 1, 0, 1, 1, 1)
        _, fragments = occ.fragment([(3, l_volume)], [(3, square_volume)])
        occ.synchronize()
        for physical_tag, name, entities in zip((11, 19), ("channel_l", "channel_square"), fragments):
            volumes = [tag for dim, tag in entities if dim == 3]
            assert len(volumes) == 1
            gmsh.model.addPhysicalGroup(3, volumes, physical_tag, name)
            boundary = gmsh.model.getBoundary([(3, volumes[0])], oriented=False)
            for suffix, z, offset in (("bottom", 0, 10000), ("top", 1, 20000)):
                surfaces = [tag for dim, tag in boundary if dim == 2
                            and abs(occ.getCenterOfMass(2, tag)[2] - z) < 1e-9]
                assert surfaces
                gmsh.model.addPhysicalGroup(2, surfaces, offset + physical_tag,
                                            f"{name}_external_{suffix}")
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.3)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.4)
        gmsh.model.mesh.generate(3)
        gmsh.write(str(msh_path))
    finally:
        gmsh.finalize()

    regions = []
    for tag, name, area in ((11, "channel_l", 3.0), (19, "channel_square", 1.0)):
        regions.append({
            "physical_tag": tag, "name": name, "material": "Si_channel",
            "gds_layer": 65, "datatype": 20, "z_min_um": 0.0, "z_max_um": 1.0,
            "role": "source", "thermal_source_candidate": True,
            "volumes": [{"area_um2": area}],
        })
    manifest = {
        "schema": "gds-material-regions-v2", "geometry_length_scale_to_m": 1e-6,
        "boolean_grid_um": 1e-6, "thermal_source_candidate_tags": [11, 19],
        "regions": regions,
    }
    manifest_path.write_text(json.dumps(manifest))
    return msh_path, manifest_path


def _load_power_case(paths, **kwargs):
    return load_tagged_msh(*paths, source_power_by_tag=POWER_BY_TAG,
                           source_device_by_tag=DEVICE_BY_TAG, **kwargs)


def test_nonrectangular_source_power_identity_and_periodic_solve(imported_mesh_files):
    pytest.importorskip("dolfinx_mpc")
    case = _load_power_case(imported_mesh_files,
                           bcs=BoundaryConditions(periodic_x=True, periodic_y=True))
    case.assert_solver_ready()
    source_l = next(region.source for region in case.registry.all() if region.tag_id == 11)
    assert source_l.volume_m3 == pytest.approx(3e-18, rel=1e-12, abs=0)
    # The L's bounding box would be 4 um^3: the denominator must be 3 um^3.
    assert source_l.power_density_w_per_m3 == pytest.approx(5e11, rel=1e-12)
    assert source_l.device == "X0" and source_l.kind == "channel"
    assert source_l.power_uw == pytest.approx(1.5)
    T, k, q = solve_steady(case.mesh_data, case.registry, case.chip)
    device_report = verify_device_source_powers(case.mesh_data, case.registry, q, POWER_BY_DEVICE)
    assert device_report["meshed_total_w"] == pytest.approx(3.5e-6, rel=1e-9, abs=0)
    balance = power_balance(case.mesh_data, case.registry, case.chip, T, k, q)
    print_power_balance(balance)
    assert balance["p_gen_vs_intended"] == pytest.approx(1.0, rel=1e-10)
    assert balance["robin_rel_err"] < 1e-8
    assert balance["per_tag"][11]["nominal_m3"] == pytest.approx(3e-18, rel=1e-12, abs=0)
    assert all(entry["ratio"] == pytest.approx(1.0, rel=1e-10)
               for entry in case.report["source_volumes"].values())


def test_density_only_api_retains_backward_compatible_behavior(imported_mesh_files):
    density = {11: 5e11, 19: 2e12}
    case = load_tagged_msh(*imported_mesh_files, source_density_by_tag=density)
    case.assert_solver_ready()
    sources = [region.source for region in case.registry.all()]
    assert all(source.kind == "imported" for source in sources)
    assert {source.device for source in sources} == {"channel_l", "channel_square"}
    assert sum(source.power_uw for source in sources) == pytest.approx(3.5)
    assert case.report["source_power_input_tags"] == []
    assert case.report["source_density_input_tags"] == [11, 19]
    T, k, q = solve_steady(case.mesh_data, case.registry, case.chip)
    balance = power_balance(case.mesh_data, case.registry, case.chip, T, k, q)
    print_power_balance(balance)
    assert balance["p_intended_w"] == pytest.approx(3.5e-6, rel=1e-12, abs=0)


def test_density_and_power_can_target_different_tags(imported_mesh_files):
    case = load_tagged_msh(*imported_mesh_files, source_power_by_tag={11: 1.5e-6},
                           source_density_by_tag={19: 2e12}, source_device_by_tag=DEVICE_BY_TAG)
    _, _, q = build_coeffs(case.mesh_data.mesh, case.mesh_data.cell_tags, case.registry)
    verify_device_source_powers(case.mesh_data, case.registry, q, POWER_BY_DEVICE)
    assert case.report["source_power_input_tags"] == [11]
    assert case.report["source_density_input_tags"] == [19]


def test_kind_override_and_zero_power_are_supported(imported_mesh_files):
    case = load_tagged_msh(*imported_mesh_files, source_power_by_tag={11: 0.0},
                           source_device_by_tag={11: "wire0"}, source_kind_by_tag={11: "interconnect"})
    source = next(region.source for region in case.registry.all() if region.source is not None)
    assert source.kind == "interconnect"
    assert source.power_uw == source.power_density_w_per_m3 == 0.0


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), -float("inf")])
def test_invalid_power_is_rejected_before_meshing(imported_mesh_files, value):
    with pytest.raises(ValueError, match="powers must be finite and non-negative"):
        load_tagged_msh(*imported_mesh_files, source_power_by_tag={11: value})


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_invalid_density_is_rejected(imported_mesh_files, value):
    with pytest.raises(ValueError, match="power density must be finite and non-negative"):
        load_tagged_msh(*imported_mesh_files, source_density_by_tag={11: value})


def test_density_and_power_cannot_both_assign_one_tag(imported_mesh_files):
    with pytest.raises(ValueError, match="both density and power"):
        load_tagged_msh(*imported_mesh_files, source_power_by_tag={11: 1e-6},
                        source_density_by_tag={11: 1e12})


@pytest.mark.parametrize("devices", [{11: "X0"}, {11: "X0", 19: "X7", 99: "X8"},
                                     {11: "X0", 19: "X0"}, {11: "X0", 19: ""}])
def test_missing_extra_duplicate_or_empty_device_ids_are_rejected(imported_mesh_files, devices):
    with pytest.raises(ValueError, match="source_device_by_tag|unique|nonempty"):
        load_tagged_msh(*imported_mesh_files, source_power_by_tag=POWER_BY_TAG,
                        source_device_by_tag=devices)


def test_unknown_source_tag_is_rejected(imported_mesh_files):
    with pytest.raises(KeyError, match="absent from manifest"):
        load_tagged_msh(*imported_mesh_files, source_power_by_tag={99: 1e-6})


def _changed_manifest(imported_mesh_files, tmp_path, change):
    mesh_path, manifest_path = imported_mesh_files
    manifest = json.loads(manifest_path.read_text())
    change(manifest)
    path = tmp_path / "changed_manifest.json"
    path.write_text(json.dumps(manifest))
    return mesh_path, path


def test_power_sources_obey_candidate_policy(imported_mesh_files, tmp_path):
    def remove_candidates(manifest):
        manifest["thermal_source_candidate_tags"] = []
        for region in manifest["regions"]:
            region["thermal_source_candidate"] = False

    paths = _changed_manifest(imported_mesh_files, tmp_path, remove_candidates)
    with pytest.raises(ValueError, match="non-candidate"):
        _load_power_case(paths)
    case = _load_power_case(paths, allow_non_candidate_sources=True)
    assert case.report["source_region_tags"] == [11, 19]


@pytest.mark.parametrize("area", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_manifest_source_area_is_rejected(imported_mesh_files, tmp_path, area):
    def change(manifest):
        manifest["regions"][0]["volumes"][0]["area_um2"] = area

    paths = _changed_manifest(imported_mesh_files, tmp_path, change)
    with pytest.raises(ValueError, match="area must be finite and positive"):
        _load_power_case(paths)


def test_manifest_volume_error_is_not_hidden_by_power_normalization(imported_mesh_files, tmp_path):
    def use_bbox_area(manifest):
        manifest["regions"][0]["volumes"][0]["area_um2"] = 4.0

    paths = _changed_manifest(imported_mesh_files, tmp_path, use_bbox_area)
    with pytest.raises(ValueError, match="meshed volume.*manifest volume"):
        _load_power_case(paths)


def test_per_device_audit_detects_swapped_power_with_unchanged_total(imported_mesh_files):
    import ufl
    from dolfinx.fem import Function, assemble_scalar, form

    case = _load_power_case(imported_mesh_files)
    _, _, q = build_coeffs(case.mesh_data.mesh, case.mesh_data.cell_tags, case.registry)
    changed = Function(q.function_space)
    changed.x.array[:] = q.x.array
    for region in case.registry.all():
        other_power = POWER_BY_TAG[19 if region.tag_id == 11 else 11]
        cells = case.mesh_data.cell_tags.find(region.tag_id)
        for cell in cells:
            changed.x.array[changed.function_space.dofmap.cell_dofs(int(cell))] = other_power / region.source.volume_m3
    changed.x.scatter_forward()
    dx = ufl.Measure("dx", domain=case.mesh_data.mesh)
    assert assemble_scalar(form(changed * dx)) == pytest.approx(assemble_scalar(form(q * dx)), rel=1e-10, abs=0)
    with pytest.raises(ValueError, match="X[07]: meshed power"):
        verify_device_source_powers(case.mesh_data, case.registry, changed, POWER_BY_DEVICE)


def test_imported_source_budget_rejects_2d_depth_override(imported_mesh_files):
    case = _load_power_case(imported_mesh_files)
    T, k, q = solve_steady(case.mesh_data, case.registry, case.chip)
    with pytest.raises(ValueError, match="Imported 3D source volumes"):
        power_balance(case.mesh_data, case.registry, case.chip, T, k, q, source_depth_m=1e-6)


def test_legacy_source_constructor_and_explicit_power_consistency():
    legacy = ImportedHeatSource("heater", 2e12)
    assert legacy.power_density_w_per_m3 == 2e12
    assert legacy.device == "heater"
    with pytest.raises(ValueError, match="volume is required"):
        _ = legacy.power_uw
    with pytest.raises(ValueError, match="density times volume"):
        ImportedHeatSource("X0", 2e12, volume_m3=1e-18, intended_power_w=3e-6)
