"""PLAN.md's primary correctness gate: sum(source power) must equal
sum(q.n) over all exterior facets, to solver tolerance. Runs the real
row-0 chip mesh end to end (build -> solve), not a synthetic case.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ufl
from dolfinx.fem import assemble_scalar, form

from mesh.build import build_2d_mesh
from post.budget import power_balance
from solve.steady import solve_steady
from spec import default_chip
from spec.layout import PITCH_Y_UM


def test_energy_conservation_row0(tmp_path):
    chip = default_chip()
    mesh_data, registry = build_2d_mesh(chip, row=0, out_dir=str(tmp_path))
    T, k, q = solve_steady(mesh_data, registry, chip, source_depth_m=PITCH_Y_UM * 1e-6)

    mesh = mesh_data.mesh
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=mesh_data.facet_tags)
    n = ufl.FacetNormal(mesh)

    balance = power_balance(mesh_data, registry, chip, T, k, q, source_depth_m=PITCH_Y_UM * 1e-6)
    p_gen = balance["p_gen_w"]
    assert p_gen > 0

    # Primary gate: taking v=1 (representable exactly in the P1 trial space)
    # in the discrete weak form a(T,v)=L(v) collapses the grad-grad term
    # (grad(1)=0) to exactly h*int(T-T_amb)ds(bottom) = int(q)dx — this is
    # the FEM solve's own discrete energy balance, so it should hold to
    # solver tolerance, not just mesh-dependent accuracy.
    assert balance["robin_rel_err"] < 1e-6, balance
    # Real audit finding (optimized-singing-creek.md F3): the check above
    # only proves p_gen == p_out_robin, which holds BY CONSTRUCTION for any
    # q -- it never proved p_gen equals the INTENDED source power. This does.
    assert abs(balance["p_gen_vs_intended"] - 1.0) < 1e-3, balance

    # Secondary diagnostic: reconstructing flux from grad(T) (one order less
    # accurate than T itself for P1 elements) over all exterior facets.
    # Expected to carry real mesh-dependent error, especially given the
    # >100x conductivity contrasts and highly graded mesh here — not a
    # correctness bug, just a coarser check. Loose tolerance, informational.
    # Same depth_scale as power_balance (see its docstring): this is a 2D
    # cross-section, so the raw flux integral is W per metre of unmodeled
    # depth, not real watts -- must match p_gen's own real-watt scaling.
    p_out_flux = assemble_scalar(form(-k * ufl.dot(ufl.grad(T), n) * ds)) * (PITCH_Y_UM * 1e-6)
    rel_err_flux = abs(p_out_flux - p_gen) / abs(p_gen)
    print(f"[energy] p_gen={p_gen:.4g} p_out_robin={balance['p_out_robin_w']:.4g} "
          f"p_out_flux={p_out_flux:.4g} rel_err_flux={rel_err_flux:.3%}")
    assert rel_err_flux < 0.15, f"flux reconstruction way off: {rel_err_flux:.3%}"
