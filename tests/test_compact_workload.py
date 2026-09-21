"""The steady bitcell driver uses integrated SPICE energy without class averaging."""

from types import SimpleNamespace

import pytest


@pytest.fixture
def driver():
    pytest.importorskip("dolfinx")
    from cases import run_bitcell_compact
    return run_bitcell_compact


def _options(tmp_path, **overrides):
    result = dict(point="read_1", power_model="spice-transient", row_hit_rate=0.25,
                  period_ns=4.0, pulse_width_ns=0.24, spice_step_ps=0.5,
                  out_dir=tmp_path)
    result.update(overrides)
    return result


def test_default_cli_selects_real_read_not_synthetic_crowbar(driver):
    args = driver._argument_parser().parse_args([])
    assert args.power_model == "spice-transient"
    assert args.point == "read_1"
    assert args.row_hit_rate == 1.0
    assert args.spice_step_ps == 1.0
    assert args.period_ns is args.pulse_width_ns is None


@pytest.mark.parametrize("point", ["read_0", "read_1", "write_0_to_1", "write_1_to_0"])
def test_real_access_preserves_per_instance_energy_and_cli_options(driver, monkeypatch, tmp_path, point):
    energies = {f"X{i}": (i + 1) ** 2 * 1e-16 for i in range(8)}
    powers = {device: energy * 0.25 / 4e-9 for device, energy in energies.items()}
    calls = []

    def build(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(device_energy_j=energies, device_average_power_w=powers,
                               period_s=4e-9, row_hit_rate=0.25,
                               metadata={"power_trace_sha256": "spice-trace-hash"})

    monkeypatch.setattr(driver, "build_spice_workload", build)
    actual, metadata = driver._build_power_workload(**_options(tmp_path, point=point))
    assert actual == powers
    assert metadata["device_energy_j"] == energies
    assert metadata["power_trace_sha256"] == "spice-trace-hash"
    assert metadata["thermal_power_revision"] == driver.THERMAL_POWER_REVISION
    assert metadata["thermal_solver_connected"] is True
    assert metadata["application"] == "steady-average"
    assert calls == [dict(point=point, row_hit_rate=0.25, period_ns=4.0,
                          pulse_width_ns=0.24, spice_max_step_ps=0.5,
                          out_dir=tmp_path / "spice", lib_path=driver.LIB_PATH)]
    source_audit = {"per_device": {device: {"meshed_w": power}
                                  for device, power in powers.items()},
                    "meshed_total_w": sum(powers.values())}
    audit = driver._audit_averaged_energy(source_audit, metadata)
    assert audit["event_energy_j"] == sum(energies.values())
    assert set(audit["per_device"]) == set(energies)
    source_audit["per_device"]["X1"]["meshed_w"] *= 1.01
    with pytest.raises(ValueError, match="energy audit failed for X1"):
        driver._audit_averaged_energy(source_audit, metadata)


@pytest.mark.parametrize("point", ["crowbar", "hold_1", "write_1_settled"])
def test_legacy_points_are_never_silently_reinterpreted_as_switching(driver, tmp_path, point):
    with pytest.raises(ValueError, match="not a transient-SPICE access"):
        driver._build_power_workload(**_options(tmp_path, point=point))


def test_explicit_legacy_model_retains_named_bias_and_row_rate(driver, monkeypatch, tmp_path):
    calls = []

    def bias(point, **kwargs):
        calls.append((point, kwargs))
        return {"X0": 1e-8}

    monkeypatch.setattr(driver, "named_bias_point_power_w", bias)
    powers, meta = driver._build_power_workload(**_options(
        tmp_path, point="crowbar", power_model="dc-surrogate", period_ns=None,
        pulse_width_ns=None, spice_step_ps=1.0))
    assert powers == {"X0": 1e-8}
    assert meta["power_model"] == "dc-surrogate"
    assert calls == [("crowbar", {"row_hit_rate": 0.25, "lib_path": driver.LIB_PATH})]
    assert driver._audit_averaged_energy({}, meta) is None


@pytest.mark.parametrize("override", [{"period_ns": 4.0}, {"pulse_width_ns": 0.24},
                                       {"spice_step_ps": 0.5}])
def test_legacy_model_rejects_ignored_electrical_controls(driver, tmp_path, override):
    options = _options(tmp_path, point="crowbar", power_model="dc-surrogate",
                       period_ns=None, pulse_width_ns=None, spice_step_ps=1.0)
    options.update(override)
    with pytest.raises(ValueError, match="only supported"):
        driver._build_power_workload(**options)


@pytest.mark.parametrize("artifact", ["simulation_summary.json", "power_workload.json", "spice/access.sp"])
def test_existing_results_are_preserved(driver, tmp_path, artifact):
    path = tmp_path / artifact
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("existing results")
    with pytest.raises(ValueError, match="choose a new --out-dir"):
        driver._check_output_directory(tmp_path)
    assert path.read_text() == "existing results"


def test_geometry_only_directory_can_be_reused(driver, tmp_path):
    (tmp_path / "material_regions.msh").write_text("existing mesh")
    (tmp_path / "spice").mkdir()
    driver._check_output_directory(tmp_path)
