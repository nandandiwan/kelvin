"""Pure-Python tests for spec/ — no dolfinx dependency."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spec import MATERIALS, get_material, lod1_stack, default_layout, default_chip
from spec.layout import ARRAY_COLS, ARRAY_ROWS, HOT_POWER_UW, COLD_POWER_UW


def test_materials_resolve():
    for name in MATERIALS:
        assert get_material(name).k > 0


def test_si_k_texp_drops_with_temperature():
    si = get_material("Si_bulk")
    assert si.k_at(300.0) == si.k
    assert si.k_at(400.0) < si.k_at(300.0)


def test_stack_z_bounds_are_contiguous_and_positive_thickness():
    stack = lod1_stack()
    z = 0.0
    for z0, z1, layer in stack.z_bounds():
        assert z0 == z
        assert z1 > z0
        z = z1
    assert math.isclose(z, stack.total_thickness_um)


def test_stack_layer_at_z_matches_plan_table():
    stack = lod1_stack()
    assert stack.layer_at_z(25.0).name == "Si_substrate"
    assert stack.layer_at_z(50.01).name == "STI_SD"
    assert stack.layer_at_z(50.035).name == "STI_channel"
    assert stack.layer_at_z(50.12).name == "silicide"
    assert stack.layer_at_z(50.20).name == "MOL_contacts"
    assert stack.layer_at_z(50.30).name == "M1"


def test_layout_device_count_and_power():
    layout = default_layout()
    assert len(layout.devices) == ARRAY_COLS * ARRAY_ROWS == 30
    n_hot = sum(1 for d in layout.devices if d.is_hot)
    assert n_hot == 6
    expected_power = 6 * HOT_POWER_UW + 24 * COLD_POWER_UW
    assert math.isclose(layout.total_power_uw, expected_power)


def test_layout_flux_matches_plan_ballpark():
    layout = default_layout()
    # PLAN.md: ~168 uW over 576 um^2 ~= 29 W/cm^2
    assert math.isclose(layout.total_power_uw, 168.0, rel_tol=0.05)
    assert math.isclose(layout.flux_w_per_cm2, 29.0, rel_tol=0.05)


def test_via_stack_split_is_asymmetric():
    layout = default_layout()
    has_via = sum(1 for d in layout.devices if d.has_via_stack)
    assert 0 < has_via < len(layout.devices)


def test_default_chip_validates_and_sources_match_devices():
    chip = default_chip()
    # two sources (channel + contact) per device
    assert len(chip.sources) == 2 * len(chip.layout.devices)
    device_names = {d.name for d in chip.layout.devices}
    for s in chip.sources:
        assert s.device in device_names
        assert s.kind in ("channel", "contact")
        assert s.power_density_w_per_m3 > 0


def test_channel_and_contact_power_split_sums_to_device_power():
    chip = default_chip()
    by_device = {}
    for s in chip.sources:
        by_device.setdefault(s.device, []).append(s)
    for d in chip.layout.devices:
        channel, contact = by_device[d.name]
        assert math.isclose(channel.power_uw + contact.power_uw, d.power_uw)
        assert channel.kind == "channel" and channel.power_uw > contact.power_uw


def test_source_power_density_order_of_magnitude():
    chip = default_chip()
    hot_channel = next(
        s for s in chip.sources if s.kind == "channel" and s.power_uw == HOT_POWER_UW * 0.7
    )
    assert 1e16 < hot_channel.power_density_w_per_m3 < 1e18
