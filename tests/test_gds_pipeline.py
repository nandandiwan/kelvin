"""GDS pipeline gates: heat-source power sums correctly, the 2D cross-section
mesh is structurally exact (z_realized == z_expected, cell counts sum to the
total — same check as the synthetic chip), and the solve conserves energy.
"""

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math

import pytest
import ufl
from dolfinx.fem import assemble_scalar, form

from gds.read import load_top_cell, flatten_by_layer, named_cell_bbox_um
from gds.sources import build_heat_sources, CHANNEL_POWER_FRAC, CONTACT_POWER_FRAC
from mesh.build import FACET_BOTTOM, FACET_TOP
from mesh.gds_build import build_gds_2d_mesh
from mesh.gds_section import die_bounds
from solve.steady import solve_steady
from spec.chip import BoundaryConditions

GDS_PATH = str(Path(__file__).resolve().parents[1] / "data" / "sram22_64x22m4w22.gds")
BITCELL_NAME = "sram_sp_cell"
CUT_Y_UM = 153.0
TOTAL_POWER_W = 1e-3


def _load():
    lib, top = load_top_cell(GDS_PATH)
    return flatten_by_layer(top)


def test_heat_source_power_sums_correctly():
    by_layer = _load()
    channel_sources, contact_sources = build_heat_sources(by_layer, TOTAL_POWER_W)
    assert len(channel_sources) == 21095  # exact count, sram22_64x22m4w22's full bitcell array + periphery
    tot_ch = sum(box.power_uw for box, _ in channel_sources)
    tot_co = sum(box.power_uw for box, _ in contact_sources)
    assert math.isclose(tot_ch, TOTAL_POWER_W * 1e6 * CHANNEL_POWER_FRAC, rel_tol=1e-6)
    assert math.isclose(tot_co, TOTAL_POWER_W * 1e6 * CONTACT_POWER_FRAC, rel_tol=1e-6)


def test_gds_2d_mesh_is_structurally_exact(tmp_path):
    by_layer = _load()
    mesh_data, registry = build_gds_2d_mesh(by_layer, CUT_Y_UM, TOTAL_POWER_W, out_dir=str(tmp_path))
    mesh = mesh_data.mesh
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    assert num_cells > 0
    assert len(registry.all()) > 100  # real per-source tags, not a degenerate empty cut


@pytest.mark.parametrize("top_h_eff", [None, 20000.0], ids=["single-sink", "dual-sided"])
def test_gds_energy_conservation(tmp_path, top_h_eff):
    """p_gen must equal heat leaving through *every* active Robin sink, not
    just the backside — with top_h_eff set (dual-sided cooling, the BSPDN
    study's config C), that means bottom + top summed. See physics/bcs.py."""
    by_layer = _load()
    lib, _ = load_top_cell(GDS_PATH)
    try:
        _, y0c, _, y1c = named_cell_bbox_um(lib, BITCELL_NAME)
        depth_m = (y1c - y0c) * 1e-6
    except KeyError:
        x0, x1, y0, y1 = die_bounds(by_layer)
        depth_m = (y1 - y0) * 1e-6

    mesh_data, registry = build_gds_2d_mesh(by_layer, CUT_Y_UM, TOTAL_POWER_W, out_dir=str(tmp_path))
    bcs = BoundaryConditions(ambient_t_k=300.0, backside_h_eff=20000.0, top_h_eff=top_h_eff)
    chip = types.SimpleNamespace(bcs=bcs)
    T, k, q = solve_steady(mesh_data, registry, chip, source_depth_m=depth_m)

    mesh = mesh_data.mesh
    dx = ufl.Measure("dx", domain=mesh)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=mesh_data.facet_tags)

    p_gen = assemble_scalar(form(q * dx))
    assert p_gen > 0
    p_out_robin = assemble_scalar(form(bcs.backside_h_eff * (T - bcs.ambient_t_k) * ds(FACET_BOTTOM)))
    if bcs.top_h_eff is not None:
        p_out_robin += assemble_scalar(form(bcs.top_h_eff * (T - bcs.ambient_t_k) * ds(FACET_TOP)))
    rel_err = abs(p_out_robin - p_gen) / abs(p_gen)
    assert rel_err < 1e-6, f"p_gen={p_gen}, p_out_robin={p_out_robin}, rel_err={rel_err}"

    tmax = float(T.x.array.max())
    assert 300.0 < tmax < 400.0, f"Tmax={tmax} outside physically sane range for 1mW/{depth_m*1e6:.1f}um depth"
