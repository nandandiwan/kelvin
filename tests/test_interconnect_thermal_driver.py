"""Driver regression: local resistor waveforms reach the actual FEM load."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("dolfinx")
import cases.run_bitcell_compact as compact
import cases.run_bitcell_transient as driver

# Reuse the existing lightweight SPICE/mesh pipeline, extending its source and
# solver vectors with two independently observable interconnect components.
from test_spice_thermal_driver import fake_pipeline  # noqa: F401


@pytest.fixture
def layout_pipeline(fake_pipeline, monkeypatch):
    import dolfinx.fem
    import physics.interconnect_sources

    state = fake_pipeline
    trace = state.workload.trace
    # Deliberately insert names in reverse order: the archive must identify its
    # sorted columns instead of relying on dictionary construction order.
    trace.interconnect_power_w = {
        "R_z": np.array([0, 1e-6, 9e-6, 0]),
        "R_a": np.array([3e-7, 7e-6, 2e-6, 1e-8]),
    }
    network = {"schema": "synthetic-layout-network", "resistors": [
        {"name": name, "resistance_ohm": index + 1}
        for index, name in enumerate(trace.interconnect_power_w)]}
    trace.metadata["interconnect_network"] = network
    state.workload.interconnect_energy_j = trace.interconnect_energy_j()
    state.workload.interconnect_average_power_w = {
        name: energy / state.workload.period_s
        for name, energy in state.workload.interconnect_energy_j.items()}
    state.workload.metadata.update(
        resistor_energy_j=state.workload.interconnect_energy_j, interconnect_model="layout")
    state.combined_constructors = []
    state.combined_source_calls = []
    state.combined_power_audits = []
    state.combined_energy_audits = []
    state.balance_calls = []
    state.wire_consumed_scale = 1.0
    state.swap_wire_loads = False
    state.wire_ids = sorted(trace.interconnect_power_w)
    state.argv.extend(["--interconnect", "layout"])
    ids = [f"X{i}" for i in range(8)]
    space = SimpleNamespace(
        tabulate_dof_coordinates=lambda: np.zeros((10, 3)),
        dofmap=SimpleNamespace(list=np.array([[0, 1, 2, 3], [4, 5, 6, 7]])))

    class Function:
        def __init__(self, function_space=space):
            self.function_space = function_space
            self.x = SimpleNamespace(array=np.zeros(10))

    def vector(device_values, wire_values):
        assert set(device_values) == set(ids)
        assert set(wire_values) == set(state.wire_ids)
        return np.array([device_values[name] for name in ids]
                        + [wire_values[name] for name in state.wire_ids])

    class Combined:
        def __init__(self, mesh_data, registry, manifest, selected_network):
            state.combined_constructors.append((mesh_data, registry, manifest, selected_network))
            assert manifest == {"synthetic": True}
            assert selected_network is network
            self.q = Function()
            self.projection_report = {"method": "synthetic-spatial-test", "resistor_count": 2}

        def set_powers(self, devices, wires):
            state.combined_source_calls.append((dict(devices), dict(wires)))
            self.q.x.array[:] = vector(devices, wires)
            return self.q

        def audit_powers(self, actual, devices, wires):
            np.testing.assert_array_equal(actual.x.array, vector(devices, wires))
            state.combined_power_audits.append((dict(devices), dict(wires)))
            total = float(actual.x.array.sum())
            return {"per_device": {name: {"meshed_w": devices[name]} for name in ids},
                    "electrical_total_w": total, "meshed_total_w": total}

        def audit_energy(self, actual, device_energy, wire_energy):
            state.combined_energy_audits.append(
                (actual.x.array.copy(), dict(device_energy), dict(wire_energy)))
            expected = vector(device_energy, wire_energy)
            if not np.allclose(actual.x.array, expected, rtol=1e-12, atol=1e-30):
                raise ValueError("actual combined solver load does not match SPICE energies")
            return {"expected_total_j": float(expected.sum()),
                    "deposited_total_j": float(actual.x.array.sum()),
                    "interconnects": {"expected_total_j": sum(wire_energy.values())}}

    class Solver:
        def __init__(self, *args, dt, T0):
            self.V = space
            self.T_prev, self.q = Function(), Function()
            self.T_prev.x.array[:] = T0

        def step(self, source, dt):
            self.q.x.array[:] = np.asarray(source) * state.consumed_power_scale
            self.q.x.array[8:] *= state.wire_consumed_scale
            if state.swap_wire_loads:
                self.q.x.array[8:] = self.q.x.array[8:][::-1]
            state.solver_calls.append((self.q.x.array.copy(), dt))
            self.T_prev.x.array[:] += self.q.x.array * dt * 1e15
            return self.T_prev

    def balance(*args, **kwargs):
        state.balance_calls.append((args, kwargs))
        return {}

    monkeypatch.setattr(physics.interconnect_sources, "CombinedHeatSources", Combined)
    monkeypatch.setattr(dolfinx.fem, "Function", Function)
    monkeypatch.setattr(driver, "TransientHeatSolver", Solver)
    monkeypatch.setattr(driver, "power_balance", balance)
    monkeypatch.setattr(driver, "print_power_balance", lambda *args, **kwargs: None)
    return state


@pytest.mark.parametrize("point", ["read_1", "write_0_to_1"])
def test_layout_waveforms_reach_every_step_and_archive_resistor_identity(layout_pipeline, point):
    state = layout_pipeline
    state.argv.extend(["--point", point])
    driver.main()
    trace = state.workload.trace
    edges, dt = driver.spice_thermal_time_grid(trace, 2)
    count = len(edges) - 1
    assert len(state.combined_constructors) == 1
    assert len(state.combined_power_audits) == 1
    assert len(state.combined_source_calls) == count + 1  # Initial reference plus actual steps.
    assert not state.source_calls  # Channel-only adapter must not be used.
    assert state.workload_calls[0][1]["interconnect"] == "layout"
    assert state.workload_calls[0][1]["interconnect_step_um"] == .05
    expected_mean = {name: energy / trace.time_s[-1]
                     for name, energy in state.workload.interconnect_energy_j.items()}
    assert state.combined_source_calls[0][1] == expected_mean
    for i, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
        expected_devices = trace.average_device_power_w(start, end)
        expected_wires = trace.average_interconnect_power_w(start, end)
        selected_devices, selected_wires = state.combined_source_calls[i + 1]
        assert selected_devices == expected_devices
        assert selected_wires == expected_wires
        expected = [expected_devices[f"X{j}"] for j in range(8)]
        expected += [expected_wires[name] for name in state.wire_ids]
        np.testing.assert_allclose(state.solver_calls[i][0], expected, rtol=1e-14, atol=0)
        assert state.solver_calls[i][1] == dt
    assert len(state.solver_calls) == count + 2
    for actual, _ in state.solver_calls[count:]:
        np.testing.assert_array_equal(actual, np.zeros(10))
    assert len(state.combined_energy_audits) == 1
    actual_energy, device_energy, wire_energy = state.combined_energy_audits[0]
    assert device_energy == state.workload.device_energy_j
    assert wire_energy == state.workload.interconnect_energy_j
    np.testing.assert_allclose(actual_energy[8:], [wire_energy[name] for name in state.wire_ids],
                               rtol=1e-14, atol=1e-30)

    archive = np.load(state.out_dir / "thermal_source_steps.npz")
    np.testing.assert_array_equal(archive["resistor_ids"], state.wire_ids)
    np.testing.assert_array_equal(archive["device_ids"], [f"X{i}" for i in range(8)])
    np.testing.assert_array_equal(archive["edges_s"], edges)
    np.testing.assert_allclose(archive["interconnect_power_w"],
                               [actual[8:] for actual, _ in state.solver_calls[:count]], rtol=0, atol=0)
    np.testing.assert_allclose(archive["device_power_w"],
                               [actual[:8] for actual, _ in state.solver_calls[:count]], rtol=0, atol=0)
    audit = json.loads((state.out_dir / "source_energy_audit.json").read_text())
    assert audit["deposited_total_j"] == pytest.approx(sum(device_energy.values()) + sum(wire_energy.values()),
                                                       rel=1e-14)
    metadata = json.loads((state.out_dir / "fields/meta.json").read_text())
    assert metadata["interconnect_model"] == "layout"
    assert metadata["spice_event"]["resistor_energy_j"] == wire_energy
    assert json.loads((state.out_dir / "interconnect_projection.json").read_text())["resistor_count"] == 2


@pytest.mark.parametrize("corruption", ["scale_all", "drop_wires", "swap_wires"])
def test_layout_energy_audit_uses_actual_consumed_source_not_prescribed_values(layout_pipeline, corruption):
    state = layout_pipeline
    if corruption == "scale_all":
        state.consumed_power_scale = 1.1
    elif corruption == "drop_wires":
        state.wire_consumed_scale = 0
    else:
        state.swap_wire_loads = True  # Same total watts, wrong resistor location.
    with pytest.raises(ValueError, match="actual combined solver load"):
        driver.main()
    assert len(state.combined_energy_audits) == 1
    assert len(state.solver_calls) == 3  # Failed before cooldown and successful metadata.
    assert not (state.out_dir / "source_energy_audit.json").exists()
    assert not (state.out_dir / "fields/meta.json").exists()
    assert not (state.out_dir / "thermal_source_steps.npz").exists()


def test_sustained_layout_source_combines_event_averages_and_zeroes_cooldown(layout_pipeline, monkeypatch):
    state = layout_pipeline
    state.argv.remove("--instantaneous")
    monkeypatch.setattr(driver, "N_ACTIVE_STEPS", 3)
    driver.main()
    expected_devices = state.workload.device_average_power_w
    expected_wires = state.workload.interconnect_average_power_w
    assert state.combined_source_calls == [(expected_devices, expected_wires)]
    assert state.combined_power_audits == [(expected_devices, expected_wires)]
    expected = [expected_devices[f"X{i}"] for i in range(8)]
    expected += [expected_wires[name] for name in state.wire_ids]
    assert len(state.solver_calls) == 5
    for actual, _ in state.solver_calls[:3]:
        np.testing.assert_array_equal(actual, expected)
    for actual, _ in state.solver_calls[3:]:
        np.testing.assert_array_equal(actual, np.zeros(10))
    assert len(state.balance_calls) == 1
    assert state.balance_calls[0][1]["intended_total_w"] == pytest.approx(sum(expected))
    assert not state.combined_energy_audits  # Sustained mode audits averaged watts.
    metadata = json.loads((state.out_dir / "fields/meta.json").read_text())
    assert metadata["spice_event"]["application"] == "sustained-average"
    assert metadata["interconnect_model"] == "layout"
    assert metadata["spice_event"]["resistor_energy_j"] == state.workload.interconnect_energy_j


def test_compact_average_audit_includes_every_wire_event_energy():
    energies = {f"X{i}": (i + 1) * 1e-15 for i in range(8)}
    wires = {"R_a": 3e-15, "R_z": 7e-15}
    period, rate = 4e-9, .25
    expected = (sum(energies.values()) + sum(wires.values())) * rate / period
    metadata = {"power_model": "spice-transient", "period_s": period, "row_hit_rate": rate,
                "device_energy_j": energies, "resistor_energy_j": wires}
    source_audit = {"per_device": {name: {"meshed_w": energy * rate / period}
                                   for name, energy in energies.items()}, "meshed_total_w": expected}
    report = compact._audit_averaged_energy(source_audit, metadata)
    assert report["event_energy_j"] == pytest.approx(sum(energies.values()) + sum(wires.values()))
    assert report["interconnect_energy_j"] == sum(wires.values())
    assert report["meshed_average_w"] == expected
    source_audit["meshed_total_w"] = sum(energies.values()) * rate / period
    with pytest.raises(ValueError, match="transistor plus resistor"):
        compact._audit_averaged_energy(source_audit, metadata)
