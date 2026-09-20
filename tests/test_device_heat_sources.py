"""Time-varying source mapping: individual watts and integrated joules."""

from dataclasses import replace
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("dolfinx")
from dolfinx import mesh as dmesh
from dolfinx.fem import Function
from mpi4py import MPI

from mesh.boxes import Region
from mesh.gds_import import ImportedHeatSource
from physics.device_sources import DEVICE_NAMES, DeviceHeatSources


def registry_with(regions):
    return SimpleNamespace(all=lambda: list(regions))


@pytest.fixture(scope="module")
def channel_mesh():
    mesh = dmesh.create_box(MPI.COMM_SELF, [[0, 0, 0], [9e-6, 1e-6, 1e-6]], [9, 1, 1])
    cells = np.arange(mesh.topology.index_map(3).size_local, dtype=np.int32)
    midpoints = dmesh.compute_midpoints(mesh, 3, cells)
    tags = np.floor(midpoints[:, 0] / 1e-6).astype(np.int32) + 1
    mesh_data = SimpleNamespace(mesh=mesh, cell_tags=dmesh.meshtags(mesh, 3, cells, tags))
    regions = [Region(i + 1, f"channel_X{i}", "Si_channel",
                      ImportedHeatSource(f"channel_X{i}", 0, volume_m3=1e-18,
                                         intended_power_w=0, device=f"X{i}", kind="channel"))
               for i in range(8)]
    regions.append(Region(9, "background", "Si_bulk"))
    return mesh_data, regions


@pytest.fixture
def sources(channel_mesh):
    mesh_data, regions = channel_mesh
    return DeviceHeatSources(mesh_data, registry_with(regions))


def powers(scale=1e-7):
    return {name: (i + 1) * scale for i, name in enumerate(DEVICE_NAMES)}


def assert_device_values(actual, expected):
    assert set(actual) == set(expected)
    for name in expected:
        assert actual[name] == pytest.approx(expected[name], rel=1e-10, abs=1e-30)


def test_updates_preserve_every_instance_and_allow_zero(sources, channel_mesh):
    original_q = sources.q
    assert not np.any(original_q.x.array)
    for prescribed in (powers(), dict(zip(DEVICE_NAMES, reversed(list(powers().values())))), powers(0)):
        assert sources.set_powers(prescribed) is original_q
        assert_device_values(sources.integrated_powers(), prescribed)
    assert not np.any(original_q.x.array)
    assert all(region.source.power_uw == 0 for region in channel_mesh[1] if region.source)
    assert all(entry["ratio"] == pytest.approx(1, rel=1e-10)
               for entry in sources.volume_audit.values())


def test_updates_never_compile_new_forms(sources, monkeypatch):
    def unexpected_compile(*args, **kwargs):
        pytest.fail("A waveform update must not JIT new forms")
    monkeypatch.setattr("physics.device_sources.form", unexpected_compile)
    for scale in (1e-7, 2e-7, 0):
        sources.set_powers(powers(scale))
        assert_device_values(sources.integrated_powers(), powers(scale))


def test_legacy_rectangular_channel_sources_use_the_same_normalization(channel_mesh):
    from spec.chip import SourceBox
    mesh_data, original_regions = channel_mesh
    regions = [replace(region, source=SourceBox(
        device=region.source.device, kind="channel", x_um=i + 0.5, y_um=0.5,
        z0_um=0, w_um=1, l_um=1, t_um=1, power_uw=0))
        for i, region in enumerate(original_regions[:8])]
    regions.append(original_regions[-1])
    mapper = DeviceHeatSources(mesh_data, registry_with(regions))
    mapper.set_powers(powers())
    assert_device_values(mapper.integrated_powers(), powers())


def test_independent_energy_audit_uses_actual_timestep_loads(sources):
    qdt = Function(sources.q.function_space)
    expected = powers(0)
    for dt, scale in ((2e-12, 2e-5), (7e-11, 3e-7), (1e-8, 0)):
        prescribed = powers(scale)
        sources.set_powers(prescribed)
        qdt.x.array[:] += dt * sources.q.x.array
        for name, value in prescribed.items():
            expected[name] += dt * value
    assert_device_values(sources.integrated_energy(qdt), expected)
    report = sources.audit_energy(qdt, expected)
    assert report["deposited_total_j"] == pytest.approx(sum(expected.values()), rel=1e-10, abs=0)
    assert report["expected_total_j"] == sum(expected.values())
    for name in DEVICE_NAMES:
        assert report["per_device"][name]["expected_j"] == expected[name]


def test_energy_audit_rejects_swapped_instances_even_with_correct_total(sources):
    qdt = Function(sources.q.function_space)
    qdt.x.array[:] = sources.set_powers(powers()).x.array * 1e-9
    expected = powers(1e-16)
    expected["X0"], expected["X7"] = expected["X7"], expected["X0"]
    with pytest.raises(ValueError, match="X0.*deposited energy"):
        sources.audit_energy(qdt, expected)


def test_energy_audit_rejects_nonchannel_heat(sources):
    qdt = Function(sources.q.function_space)
    qdt.x.array[:] = sources.set_powers(powers()).x.array * 1e-9
    channel_dofs = np.concatenate(list(sources._dofs.values()))
    background = np.setdiff1d(np.arange(len(qdt.x.array)), channel_dofs)
    assert len(background)
    qdt.x.array[background] = 1e5
    with pytest.raises(ValueError, match="Total source energy"):
        sources.audit_energy(qdt, powers(1e-16))


@pytest.mark.parametrize("bad", [-1, float("inf"), float("nan")])
def test_invalid_power_does_not_damage_previous_valid_field(sources, bad):
    sources.set_powers(powers())
    previous = sources.q.x.array.copy()
    prescribed = powers()
    prescribed["X3"] = bad
    with pytest.raises(ValueError, match="finite and nonnegative"):
        sources.set_powers(prescribed)
    np.testing.assert_array_equal(sources.q.x.array, previous)


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_exact_power_device_names_required(sources, change):
    prescribed = powers()
    if change == "missing":
        del prescribed["X0"]
    else:
        prescribed["X8"] = 0
    with pytest.raises(ValueError, match="exactly X0 through X7"):
        sources.set_powers(prescribed)


@pytest.mark.parametrize("change, message", [
    ("missing", "exactly X0 through X7"),
    ("duplicate_device", "Duplicate channel device"),
    ("duplicate_tag", "Duplicate physical tags"),
    ("bad_volume", "meshed volume.*nominal channel volume"),
    ("missing_tag", "meshed volume.*nominal channel volume"),
    ("other_heat", "Nonchannel source.*nonzero heat"),
])
def test_invalid_registry_fails_closed(channel_mesh, change, message):
    mesh_data, original_regions = channel_mesh
    regions = list(original_regions)
    if change == "missing":
        regions.pop(0)
    elif change == "duplicate_device":
        regions[0] = replace(regions[0], source=replace(regions[0].source, device="X1"))
    elif change == "duplicate_tag":
        regions[0] = replace(regions[0], tag_id=2)
    elif change == "bad_volume":
        regions[0] = replace(regions[0], source=replace(regions[0].source, volume_m3=2e-18))
    elif change == "missing_tag":
        regions[0] = replace(regions[0], tag_id=999)
    elif change == "other_heat":
        regions[-1] = replace(regions[-1], source=ImportedHeatSource(
            "background_heat", 1e10, volume_m3=1e-18, intended_power_w=1e-8))
    with pytest.raises(ValueError, match=message):
        DeviceHeatSources(mesh_data, registry_with(regions))


def test_two_dimensional_source_mesh_is_rejected(channel_mesh):
    mesh = dmesh.create_unit_square(MPI.COMM_SELF, 2, 2)
    with pytest.raises(ValueError, match="3D thermal mesh"):
        DeviceHeatSources(SimpleNamespace(mesh=mesh), registry_with(channel_mesh[1]))


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf")])
def test_invalid_accumulated_energy_rejected(sources, bad):
    qdt = Function(sources.q.function_space)
    qdt.x.array[0] = bad
    with pytest.raises(ValueError, match="finite and nonnegative"):
        sources.audit_energy(qdt, powers(0))


def test_real_notebook_mesh_conserves_individual_waveform_energies():
    from mesh.sram import load_sram_mesh
    root = Path(__file__).resolve().parents[1]
    explicit_path = os.environ.get("KELVIN_SRAM_TEST_MANIFEST")
    path = Path(explicit_path) if explicit_path else root / (
        "out/sram_notebook_mesh/sram_sp_cell_material_regions.json")
    if not path.is_file():
        if explicit_path:
            pytest.fail(f"KELVIN_SRAM_TEST_MANIFEST does not exist: {path}")
        pytest.skip("Generate the sealed notebook SRAM mesh or set KELVIN_SRAM_TEST_MANIFEST")
    case = load_sram_mesh(path, powers(0))
    sources = DeviceHeatSources(case.mesh_data, case.registry)
    qdt = Function(sources.q.function_space)
    expected = powers(0)
    for dt, values in ((3e-12, powers(1e-5)), (8e-11, powers(2e-7)), (1e-8, powers(0))):
        qdt.x.array[:] += sources.set_powers(values).x.array * dt
        assert_device_values(sources.integrated_powers(), values)
        for name, value in values.items():
            expected[name] += value * dt
    report = sources.audit_energy(qdt, expected)
    assert report["deposited_total_j"] == pytest.approx(sum(expected.values()), rel=1e-6, abs=0)
    assert set(report["per_device"]) == set(DEVICE_NAMES)
