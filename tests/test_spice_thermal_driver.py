"""Driver wiring tests; electrical physics and FEM are tested separately.

The fake-FEM exercise deliberately inspects the actual load consumed by the
solver, so replacing source-energy auditing with prescribed electrical values
cannot make a corrupted heat input pass.
"""

import hashlib
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("dolfinx")
import cases.run_bitcell_transient as driver
from gds.spice_transient import SpiceAccessResult


@pytest.mark.parametrize("duration,maximum_ps", [(0.939519e-9, 5), (5.5e-12, 2), (1e-12, 10)])
def test_thermal_grid_spans_whole_event_with_reusable_uniform_timestep(duration, maximum_ps):
    trace = SimpleNamespace(time_s=np.array([0, duration / 3, duration]))
    edges, dt = driver.spice_thermal_time_grid(trace, maximum_ps)
    assert edges[0] == 0
    assert edges[-1] == duration
    assert 0 < dt <= maximum_ps * 1e-12
    assert len(edges) == math.ceil(duration / (maximum_ps * 1e-12)) + 1
    np.testing.assert_allclose(np.diff(edges), dt, rtol=1e-12, atol=0)
    assert (len(edges) - 1) * dt == pytest.approx(duration, rel=1e-14, abs=0)


@pytest.mark.parametrize("maximum_ps", [0, -1, np.nan, np.inf])
def test_thermal_grid_rejects_invalid_step(maximum_ps):
    with pytest.raises(ValueError, match="thermal-step-ps"):
        driver.spice_thermal_time_grid(SimpleNamespace(time_s=np.array([0, 1e-9])), maximum_ps)


@pytest.mark.parametrize("times", [[1e-12, 1e-9], [0, 0], [0, -1e-9], [0, np.inf], [0, np.nan]])
def test_thermal_grid_requires_complete_positive_zero_origin_event(times):
    with pytest.raises(ValueError, match="starting at zero"):
        driver.spice_thermal_time_grid(SimpleNamespace(time_s=np.array(times)), 5)


def test_thermal_grid_caps_excessive_field_output():
    with pytest.raises(ValueError, match="exceed 100000"):
        driver.spice_thermal_time_grid(SimpleNamespace(time_s=np.array([0, 1e-9])), 0.001)


@pytest.mark.parametrize("arguments,message", [
    (["--point", "crowbar"], "dc-surrogate"),
    (["--point", "hold_1"], "dc-surrogate"),
    (["--point", "write_1_settled"], "dc-surrogate"),
    (["--power-model", "dc-surrogate", "--point", "write_0_to_1"], "requires --power-model spice-transient"),
    (["--power-model", "dc-surrogate", "--period-ns", "4"], "SPICE timing controls"),
    (["--power-model", "dc-surrogate", "--pulse-width-ns", "0.2"], "SPICE timing controls"),
    (["--power-model", "dc-surrogate", "--spice-step-ps", "0.5"], "SPICE timing controls"),
    (["--power-model", "dc-surrogate", "--thermal-step-ps", "2"], "SPICE timing controls"),
    (["--power-model", "dc-surrogate", "--point", "hold_1", "--instantaneous"], "sustained leakage"),
    (["--instantaneous", "--row-hit-rate", "0.5"], "only to sustained"),
    (["--row-hit-rate", "nan"], "between 0 and 1"),
    (["--row-hit-rate", "-1"], "between 0 and 1"),
    (["--row-hit-rate", "2"], "between 0 and 1"),
    (["--thermal-step-ps", "nan"], "finite and positive"),
    (["--thermal-step-ps", "0"], "finite and positive"),
])
def test_cli_rejects_ambiguous_or_invalid_power_models_before_running(
        monkeypatch, tmp_path, capsys, arguments, message):
    out_dir = tmp_path / "must-not-be-created"
    monkeypatch.setattr(driver.sys, "argv", ["run_bitcell_transient.py", *arguments,
                                            "--out-dir", str(out_dir)])
    monkeypatch.setattr(driver, "build_spice_workload", lambda *a, **k: pytest.fail("SPICE must not run"))
    with pytest.raises(SystemExit) as error:
        driver.main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert not out_dir.exists()


@pytest.mark.parametrize("prior_kind", ["fields", "fields-file", "spice", "spice-file", "workload"])
def test_cli_preserves_previous_run_artifacts(monkeypatch, tmp_path, capsys, prior_kind):
    if prior_kind in {"fields", "spice"}:
        directory = tmp_path / prior_kind
        directory.mkdir()
        prior = directory / "evidence.txt"
    else:
        name = {"fields-file": "fields", "spice-file": "spice", "workload": "power_workload.json"}[prior_kind]
        prior = tmp_path / name
    prior.write_text("preserve this prior result\n")
    monkeypatch.setattr(driver.sys, "argv", ["run_bitcell_transient.py", "--out-dir", str(tmp_path)])
    monkeypatch.setattr(driver, "build_spice_workload", lambda *a, **k: pytest.fail("SPICE must not run"))
    with pytest.raises(SystemExit) as error:
        driver.main()
    assert error.value.code == 2
    assert "prior run" in capsys.readouterr().err
    assert prior.read_text() == "preserve this prior result\n"


@pytest.fixture
def fake_pipeline(monkeypatch, tmp_path):
    import dolfinx.fem

    device_ids = [f"X{i}" for i in range(8)]
    times = np.array([0.0, 0.7e-12, 2.0e-12, 5.5e-12])
    powers = {name: (index + 1) * np.array([1e-12, 3e-6, 1e-6, 1e-12])
              for index, name in enumerate(device_ids)}
    powers["X3"] *= 0
    powers["X4"] *= 0
    trace = SpiceAccessResult({}, {"time": times}, powers, {}, {},
                              {"completed": True, "operation_verified": True})
    energies = trace.device_energy_j()
    period = 10e-12
    workload = SimpleNamespace(
        trace=trace, device_energy_j=energies, period_s=period, row_hit_rate=1.0,
        device_average_power_w={name: energy / period for name, energy in energies.items()},
        metadata={"kind": "spice-transient", "thermal_power_revision": driver.THERMAL_POWER_REVISION,
                  "device_energy_j": energies, "period_s": period},
    )
    state = SimpleNamespace(workload=workload, workload_calls=[], mesh_calls=[],
                            power_audits=[], source_calls=[], solver_calls=[], energy_audits=[],
                            consumed_power_scale=1.0)
    mesh_data = SimpleNamespace(mesh=SimpleNamespace(comm=SimpleNamespace(rank=0)), cell_tags=object())
    registry = object()
    space = SimpleNamespace(
        tabulate_dof_coordinates=lambda: np.zeros((8, 3)),
        dofmap=SimpleNamespace(list=np.array([[0, 1, 2, 3], [4, 5, 6, 7]])),
    )

    class Function:
        def __init__(self, function_space=space):
            self.function_space = function_space
            self.x = SimpleNamespace(array=np.zeros(8))

    def build_workload(*args, **kwargs):
        state.workload_calls.append((args, kwargs))
        return workload

    def build_mesh(powers, out_dir, **kwargs):
        state.mesh_calls.append((dict(powers), out_dir, kwargs))
        return mesh_data, registry, (0, 1, 0, 1)

    def build_coeffs(*args, **kwargs):
        source = Function()
        source.x.array[:] = [state.mesh_calls[-1][0][name] for name in device_ids]
        return object(), object(), source

    def audit_powers(md, reg, source, intended):
        assert md is mesh_data and reg is registry
        np.testing.assert_array_equal(source.x.array, [intended[name] for name in device_ids])
        state.power_audits.append(dict(intended))
        return {"per_device": {name: {"meshed_w": intended[name]} for name in device_ids},
                "meshed_total_w": sum(intended.values())}

    class DeviceSources:
        def __init__(self, md, reg):
            assert md is mesh_data and reg is registry
            self.q = Function()

        def set_powers(self, selected):
            state.source_calls.append(dict(selected))
            self.q.x.array[:] = [selected[name] for name in device_ids]
            return self.q

        def audit_energy(self, deposited, expected):
            state.energy_audits.append(deposited.x.array.copy())
            if not np.allclose(deposited.x.array, [expected[name] for name in device_ids],
                               rtol=1e-12, atol=1e-30):
                raise ValueError("actual solver load does not match prescribed SPICE energy")
            return {"expected_total_j": sum(expected.values()),
                    "deposited_total_j": float(deposited.x.array.sum())}

    class Solver:
        def __init__(self, *args, dt, T0):
            self.V = space
            self.T_prev, self.q = Function(), Function()
            self.T_prev.x.array[:] = T0

        def step(self, source, dt):
            self.q.x.array[:] = np.asarray(source) * state.consumed_power_scale
            state.solver_calls.append((self.q.x.array.copy(), dt))
            self.T_prev.x.array[:] += self.q.x.array * dt * 1e15
            return self.T_prev

    monkeypatch.setattr(driver, "build_spice_workload", build_workload)
    monkeypatch.setattr(driver, "build_bitcell_mesh", build_mesh)
    monkeypatch.setattr(driver, "build_coeffs", build_coeffs)
    monkeypatch.setattr(driver, "verify_device_source_powers", audit_powers)
    monkeypatch.setattr(driver, "DeviceHeatSources", DeviceSources)
    monkeypatch.setattr(driver, "TransientHeatSolver", Solver)
    monkeypatch.setattr(driver, "tmax", lambda field: (float(field.x.array.max()), np.zeros(3)))
    monkeypatch.setattr(dolfinx.fem, "Function", Function)
    monkeypatch.setattr(driver, "N_IDLE_STEPS", 2)
    monkeypatch.setattr(driver, "mapping_revision", lambda: "synthetic-source-map")
    manifest = tmp_path / "material_regions.json"
    manifest.write_text('{"synthetic": true}\n')
    state.out_dir = tmp_path / "run"
    state.manifest = manifest
    state.argv = ["run_bitcell_transient.py", "--instantaneous", "--no-renders",
                  "--thermal-step-ps", "2", "--mesh-manifest", str(manifest),
                  "--out-dir", str(state.out_dir)]
    monkeypatch.setattr(driver.sys, "argv", state.argv)
    return state


@pytest.mark.parametrize("point", ["read_0", "write_0_to_1"])
def test_single_access_main_transfers_waveform_energy_and_saves_complete_evidence(fake_pipeline, point):
    state = fake_pipeline
    state.argv.extend(["--point", point])
    driver.main()
    trace = state.workload.trace
    edges, dt = driver.spice_thermal_time_grid(trace, 2)
    count = len(edges) - 1
    assert len(state.solver_calls) == count + 2
    assert len(state.source_calls) == count
    assert len(state.power_audits) == 1
    assert state.workload_calls[0][0] == (point,)
    assert state.workload_calls[0][1]["out_dir"] == state.out_dir / "spice"
    assert state.mesh_calls[0][2]["mesh_manifest"] == str(state.manifest)
    assert len({call[1] for call in state.solver_calls[:count]}) == 1
    # A sampled peak or one constant event-mean rectangle would fail this.
    assert not np.allclose(state.solver_calls[0][0], state.solver_calls[count - 1][0], atol=0)
    for index, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
        expected = trace.average_device_power_w(start, end)
        np.testing.assert_allclose(state.solver_calls[index][0],
                                   [expected[f"X{i}"] for i in range(8)], rtol=1e-14, atol=0)
    for source, _ in state.solver_calls[count:]:
        np.testing.assert_array_equal(source, np.zeros(8))
    np.testing.assert_allclose(state.energy_audits[0],
                               [state.workload.device_energy_j[f"X{i}"] for i in range(8)],
                               rtol=1e-14, atol=1e-30)

    fields = state.out_dir / "fields"
    times = np.load(fields / "times_s.npy")
    assert len(times) == count + 3
    np.testing.assert_array_equal(times[:count + 1], edges)
    assert len(list(fields.glob("T_*.npy"))) == len(times)
    assert (fields / "tmin_hist.npy").is_file()
    metadata = json.loads((fields / "meta.json").read_text())
    assert metadata["active_end_s"] == trace.time_s[-1]
    assert metadata["n_steps"] == len(times)
    assert metadata["power_model"] == "spice-transient"
    assert metadata["thermal_dt_s"] == dt
    assert metadata["row_hit_rate"] is None
    assert metadata["spice_event"]["thermal_solver_connected"] is True
    assert metadata["spice_event"]["application"] == "single-access"
    assert metadata["mesh_manifest_sha256"] == hashlib.sha256(state.manifest.read_bytes()).hexdigest()
    assert metadata["power_workload_sha256"] == hashlib.sha256(
        (state.out_dir / "power_workload.json").read_bytes()).hexdigest()
    audit = json.loads((state.out_dir / "source_energy_audit.json").read_text())
    assert audit["deposited_total_j"] == pytest.approx(sum(state.workload.device_energy_j.values()),
                                                       rel=1e-14, abs=0)
    assert audit["active_step_count"] == count
    source_record = np.load(state.out_dir / "thermal_source_steps.npz")
    np.testing.assert_array_equal(source_record["edges_s"], edges)
    np.testing.assert_array_equal(source_record["device_ids"], [f"X{i}" for i in range(8)])
    np.testing.assert_allclose(source_record["device_power_w"],
                               [source for source, _ in state.solver_calls[:count]], rtol=0, atol=0)


def test_single_access_energy_audit_uses_actual_consumed_solver_load(fake_pipeline):
    state = fake_pipeline
    state.consumed_power_scale = 1.1
    with pytest.raises(ValueError, match="actual solver load"):
        driver.main()
    assert len(state.energy_audits) == 1
    assert not (state.out_dir / "fields" / "meta.json").exists()
    assert not (state.out_dir / "source_energy_audit.json").exists()
    # It must fail before cooldown or publishing successful field metadata.
    assert len(state.solver_calls) == 3


def test_sustained_mode_uses_same_spice_event_average_then_zero_cooldown(fake_pipeline, monkeypatch):
    state = fake_pipeline
    state.argv.remove("--instantaneous")
    monkeypatch.setattr(driver, "N_ACTIVE_STEPS", 3)
    monkeypatch.setattr(driver, "power_balance", lambda *args, **kwargs: {})
    monkeypatch.setattr(driver, "print_power_balance", lambda *args, **kwargs: None)
    driver.main()
    expected = [state.workload.device_energy_j[f"X{i}"] / state.workload.period_s for i in range(8)]
    assert len(state.solver_calls) == 5
    assert not state.source_calls  # Uses the averaged spatial source, not a waveform.
    for source, _ in state.solver_calls[:3]:
        np.testing.assert_array_equal(source, expected)
    for source, _ in state.solver_calls[3:]:
        np.testing.assert_array_equal(source, np.zeros(8))
    assert len(state.power_audits) == 1
    assert state.power_audits[0] == state.workload.device_average_power_w
    metadata = json.loads((state.out_dir / "fields" / "meta.json").read_text())
    assert metadata["instantaneous"] is False
    assert metadata["spice_event"]["application"] == "sustained-average"
    assert metadata["device_power_active_w"] == state.workload.device_average_power_w
    assert metadata["n_steps"] == 6
