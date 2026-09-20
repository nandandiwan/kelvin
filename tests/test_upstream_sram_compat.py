"""Legacy upstream callers retain per-instance powers after the mapping merge.

Exercise real GDS source geometry, but stop each CLI at mesh construction:
these compatibility checks need neither ngspice nor an expensive thermal run.
They deliberately do not upgrade the callers' prescribed DC power model.
"""

import importlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip("gdstk")

from gds import tile_array as array


REPO = Path(__file__).resolve().parents[1]
UNEQUAL_POWERS = {f"X{i}": (i + 1) * 1e-9 for i in range(8)}


def test_real_array_tile_still_extracts_channels(monkeypatch):
    """The strap-aware upstream tiler still needs extract_channels imported."""
    monkeypatch.setattr(array, "GDS_PATH", str(REPO / array.GDS_PATH))
    _, window, channels, contacts = array.tile_array_real(1, 1)
    assert window == pytest.approx((0.0, 6.1, 0.0, 7.9))
    assert len([box for box, _ in channels if not box.device.startswith("strap")]) == 128
    assert len([box for box, _ in channels if box.device.startswith("strap")]) == 9
    assert contacts
    assert all(box.power_uw == 0.0 for box, _ in channels + contacts)


@pytest.mark.parametrize("module_name, extra_args, scale, active_rows", [
    ("run_deep", ["--freq-mhz", "200", "--pattern", "single-row"], 0.2, {1}),
    ("run_deep", ["--freq-mhz", "200", "--pattern", "all"], 0.2, {0, 1}),
    ("run_deep", ["--freq-mhz", "200", "--pattern", "row-walk"], 0.2, {0, 1}),
    ("run_slow_clock", ["--active-row", "1"], 1.0, {1}),
    ("run_steady_rise", ["--active-row", "1"], 1.0, {1}),
    ("run_steady_rise", ["--active-row", "1", "--freq-mhz", "200"], 0.2, {1}),
])
@pytest.mark.parametrize("mode", ["cell", "array"])
def test_legacy_driver_sources_keep_asymmetric_instance_powers(
        module_name, extra_args, scale, active_rows, mode, monkeypatch, tmp_path):
    pytest.importorskip("dolfinx")
    module = importlib.import_module(f"cases.{module_name}")
    from cases import run_bitcell_gallery as gallery
    from gds import power

    # Resolve the real geometry before changing the output working directory.
    monkeypatch.chdir(REPO)
    cell_geometry = gallery.build_bitcell_geometry()
    array_geometry = array.tile_array(2, 2)
    monkeypatch.setattr(module, "build_bitcell_geometry", lambda: cell_geometry)
    monkeypatch.setattr(module, "tile_array", lambda rows, cols: array_geometry)
    monkeypatch.setattr(module, "bias_device_power_w", lambda *args: dict(UNEQUAL_POWERS))
    timing = SimpleNamespace(min_period_ns=5.0, min_pulse_width_high_ns=1.0)
    monkeypatch.setattr(power, "mine_access_timing", lambda *args: timing)
    if hasattr(module, "mine_access_timing"):
        monkeypatch.setattr(module, "mine_access_timing", lambda *args: timing)

    class MeshReached(Exception):
        pass

    captured = {}

    def capture_sources(by_layer, window, total_power_w, **kwargs):
        captured.update(kwargs, total_power_w=total_power_w)
        raise MeshReached

    monkeypatch.setattr(module, "build_gds_3d_mesh", capture_sources)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [module_name, "--mode", mode,
                                     "--rows", "2", "--cols", "2", *extra_args])
    with pytest.raises(MeshReached):
        module.main()

    sources = captured["channel_sources"]
    assert len(sources) == (8 if mode == "cell" else 32)
    assert len({box.device for box, _ in sources}) == len(sources)
    for box, _ in sources:
        if mode == "cell":
            instance, enabled = box.device, True
        else:
            tile, instance = box.device.rsplit("_", 1)
            row = int(tile.split("c")[0][1:])
            enabled = row in active_rows
        expected = UNEQUAL_POWERS[instance] * scale if enabled else 0.0
        assert box.power_uw * 1e-6 == pytest.approx(expected, rel=1e-13, abs=1e-22)
    active_cells = 1 if mode == "cell" else 2 * len(active_rows)
    assert captured["total_power_w"] == pytest.approx(
        sum(UNEQUAL_POWERS.values()) * scale * active_cells)
