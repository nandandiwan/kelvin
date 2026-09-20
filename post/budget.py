"""Power-balance / energy-conservation checks, factored out of what were
three independent copy-pasted `assemble_scalar(form(...))` blocks
(`tests/test_energy_conservation.py`, `tests/test_gds_pipeline.py`,
`cases/run_gds_coarse_solve.py`) so the audit exercises one code path
instead of trusting three copies agree.

Critically, `p_gen == p_out_robin` (the Galerkin identity at v=1, checked
by all three prior copies) holds BY CONSTRUCTION for any `q` — taking v=1
in the discrete weak form collapses the grad-grad term (grad(1)=0) to
exactly the Robin/source balance, regardless of whether `q` itself is the
INTENDED power. It is a solver self-consistency check, not a check that the
right amount of power was ever injected. `power_balance` adds the check
that was actually missing: `p_gen` against the intended `sum(SourceBox.
power_uw)`, and per-source-tag, the true meshed cell volume against the
nominal box volume `physics/coeffs.py` assumed when computing `q` — a
mismatch there (real GDS channel polygons are not guaranteed rectangular)
silently injects the wrong power while every existing test still passes.
"""

import ufl
from dolfinx.fem import assemble_scalar, form

from mesh.build import (
    FACET_BOTTOM, FACET_TOP, FACET_X0, FACET_X1, FACET_Y0, FACET_Y1,
)


def verify_device_source_powers(mesh_data, registry, q, device_power_w,
                               relative_tolerance=1e-6):
    """Check each 3D channel against the ORIGINAL electrical power dictionary.

    Unlike power_balance's source-box self-consistency check, the expected
    values must come directly from the electrical workload, before mapping.
    This detects lost, duplicated, swapped and incorrectly normalized powers.
    Also supports array-qualified instance names (e.g. r0c0_X7).
    """
    import math
    from mpi4py import MPI
    from gds.bitcell_mapping import validate_device_powers

    mesh = mesh_data.mesh
    if mesh.topology.dim != 3:
        raise ValueError("Per-device source audit requires a 3D thermal mesh")
    sources = [r for r in registry.all()
               if r.source is not None and r.source.kind == "channel"]
    names = [r.source.device for r in sources]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate channel instance names in the source registry")
    expected = validate_device_powers(device_power_w, names)
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=mesh_data.cell_tags)

    def integral(expression):
        return float(mesh.comm.allreduce(assemble_scalar(form(expression)), op=MPI.SUM))

    report = {}
    for region in sources:
        name = region.source.device
        wanted = expected[name]
        assigned = region.source.power_uw * 1e-6
        delivered = integral(q * dx(region.tag_id))
        for label, value in (("assigned", assigned), ("meshed", delivered)):
            if not math.isclose(value, wanted, rel_tol=relative_tolerance, abs_tol=1e-24):
                raise ValueError(f"{name}: {label} power {value:.12g} W "
                                 f"does not match electrical power {wanted:.12g} W")
        report[name] = {"electrical_w": wanted, "assigned_w": assigned,
                        "meshed_w": delivered, "physical_tag": region.tag_id}
    total_electrical = math.fsum(expected.values())
    total_meshed = integral(q * ufl.Measure("dx", domain=mesh))
    if not math.isclose(total_meshed, total_electrical,
                        rel_tol=relative_tolerance, abs_tol=1e-24):
        raise ValueError("Total mesh heat does not match the electrical channel-power budget")
    return {"per_device": report, "electrical_total_w": total_electrical,
            "meshed_total_w": total_meshed}


def power_balance(mesh_data, registry, chip, T, k, q, source_depth_m=None, *, intended_total_w=None):
    """Returns a dict of the full power-balance picture for one solved case:

        p_gen_w          -- what the solve actually saw, in REAL WATTS
        p_intended_w      -- sum(SourceBox.power_uw)*1e-6, what was meant
        p_gen_vs_intended -- p_gen_w / p_intended_w (want ~1.0)
        p_out_robin_w     -- heat leaving through every ACTIVE Robin sink
        robin_rel_err      -- |p_out_robin - p_gen| / |p_gen| (the old check,
                              in the SOLVER's own internal units)
        per_tag           -- {tag_id: {"nominal_m3", "meshed_m3", "ratio"}}
                              for every region with a source; ratio far from
                              1.0 means the meshed region's real size doesn't
                              match what `q` was computed from.

    `source_depth_m`: MUST match whatever was passed to `build_coeffs` for
    this solve. On a 2D cross-section, `assemble_scalar(form(q*dx))`
    integrates a W/m^3 density over an AREA (dx is 2D), giving W PER METRE
    of the unmodeled depth, not real watts -- confirmed directly: a first
    version of this function compared that raw W/m number against
    `sum(power_uw)` (real W) and got a 333,333x "failure" that was purely a
    units bug in the check, not the solver (the underlying p_gen==p_out_robin
    Galerkin identity held fine throughout, in W/m on both sides, because it
    never leaves the solver's own internal units). Multiplying by
    `source_depth_m` converts back to real watts for the external
    comparison; the per-tag nominal/meshed comparison needs the same factor
    for the same reason.

    Imported polygon-prism sources expose ``volume_m3`` and ``power_uw``
    instead of box dimensions. Their nominal volume comes from independently
    declared polygon areas and thickness, not from a bounding-box estimate.
    They are 3D sources and must not use a 2D depth override.
    """
    mesh = mesh_data.mesh
    if source_depth_m is not None and any(
            getattr(region.source, "volume_m3", None) is not None
            for region in registry.all() if region.source is not None):
        raise ValueError("Imported 3D source volumes cannot use a 2D source_depth_m")
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=mesh_data.cell_tags)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=mesh_data.facet_tags)
    depth_scale = 1.0 if source_depth_m is None else source_depth_m

    p_gen_w = assemble_scalar(form(q * ufl.Measure("dx", domain=mesh))) * depth_scale

    p_intended_w = sum(
        r.source.power_uw for r in registry.all() if r.source is not None
    ) * 1e-6
    if intended_total_w is not None:
        # Spatially projected sources need not have one registry tag per
        # resistor. The caller supplies their independently audited budget.
        import math
        if not math.isfinite(intended_total_w) or intended_total_w < 0:
            raise ValueError("intended_total_w must be finite and nonnegative")
        p_intended_w = float(intended_total_w)

    h_bottom = chip.bcs.backside_h_eff
    t_amb = chip.bcs.ambient_t_k
    p_out_robin_w = assemble_scalar(form(h_bottom * (T - t_amb) * ds(FACET_BOTTOM))) * depth_scale
    if getattr(chip.bcs, "top_h_eff", None) is not None:
        p_out_robin_w += assemble_scalar(form(chip.bcs.top_h_eff * (T - t_amb) * ds(FACET_TOP))) * depth_scale
    # The lateral radiation BC (physics/bcs.py::lateral_radiation_terms) is a
    # Robin sink like any other and MUST be counted here, or the balance looks
    # broken precisely when it is working: with it active most of the heat can
    # legitimately leave sideways (measured: the 200um-Si reference sends
    # essentially all of it out laterally, since the cut is 2.5um away and the
    # backside sink is 200um away).
    r_m = getattr(chip.bcs, "lateral_radiation_r_m", None)
    if r_m:
        for tag in (FACET_X0, FACET_X1, FACET_Y0, FACET_Y1):
            p_out_robin_w += assemble_scalar(
                form((k / r_m) * (T - t_amb) * ds(tag))) * depth_scale
    robin_rel_err = abs(p_out_robin_w - p_gen_w) / abs(p_gen_w) if p_gen_w else float("nan")

    per_tag = {}
    for r in registry.all():
        if r.source is None:
            continue
        meshed_m3 = assemble_scalar(form(1.0 * dx(r.tag_id))) * depth_scale
        if getattr(r.source, "volume_m3", None) is not None:
            nominal_m3 = r.source.volume_m3
        elif source_depth_m is None:
            nominal_m3 = (r.source.w_um * r.source.l_um * r.source.t_um) * 1e-18
        else:
            nominal_m3 = (r.source.w_um * 1e-6) * (r.source.t_um * 1e-6) * source_depth_m
        per_tag[r.tag_id] = {
            "label": r.label,
            "nominal_m3": nominal_m3,
            "meshed_m3": meshed_m3,
            "ratio": meshed_m3 / nominal_m3 if nominal_m3 else float("nan"),
        }

    return {
        "p_gen_w": p_gen_w,
        "p_intended_w": p_intended_w,
        "p_gen_vs_intended": p_gen_w / p_intended_w if p_intended_w else float("nan"),
        "p_out_robin_w": p_out_robin_w,
        "robin_rel_err": robin_rel_err,
        "per_tag": per_tag,
    }


def print_power_balance(balance: dict, tol=1e-6, noise_floor_w=1e-12, check_robin=True):
    """Pretty-print + assert a `power_balance()` result. Raises AssertionError
    with the full picture on failure rather than a bare relative-error number,
    since a failure here should point straight at which tag is wrong.

    `noise_floor_w`: below this absolute p_gen, the Robin-identity RELATIVE
    tolerance is not meaningful and is skipped (reported, not asserted).
    Confirmed directly on the HOLD/leakage case (p_gen ~1pW): the discrete
    RHS `h*T_amb*v*ds` is a "large" reference term (h=20000, T_amb=300)
    present regardless of how small q is, so the KSP's relative tolerance
    (1e-10 on the linear system's own scale) leaves an ABSOLUTE residual
    that is completely negligible physically (this case's self-heating is
    23 microkelvin) but can be many orders of magnitude larger than the
    genuinely-tiny p_gen signal itself in RELATIVE terms -- rel_err=2.3e-4
    was observed here, failing a 1e-6 bar for reasons having nothing to do
    with correctness. The `p_gen_vs_intended` check has no such issue (both
    sides are the tiny power itself, not measured against a large
    background) and is NOT floored -- it stays a hard check at every scale.

    `check_robin=False`: the Robin identity `p_gen == p_out_robin` is a
    STEADY-STATE identity only -- it comes from taking v=1 in the discrete
    weak form, which kills the stiffness term but NOT a transient mass term
    `rho_cp/dt*(T^(n+1)-T^n)*v*dx` when T is still changing. Confirmed
    directly: calling this on a transient solver's last active-phase step
    (not yet fully at steady state) gave rel_err=4.6e-4 -- real, expected
    physics (some injected power is still raising stored thermal energy,
    not yet leaving through the boundary), not a bug. Callers on a
    transient step should pass `check_robin=False` and rely on
    `p_gen_vs_intended` alone, which remains valid at any timestep.
    """
    b = balance
    print(f"[budget] p_gen={b['p_gen_w']*1e6:.6f}uW  p_intended={b['p_intended_w']*1e6:.6f}uW  "
          f"ratio={b['p_gen_vs_intended']:.6f}")
    print(f"[budget] p_out_robin={b['p_out_robin_w']*1e6:.6f}uW  robin_rel_err={b['robin_rel_err']:.3e}")
    bad = [(tag, d) for tag, d in b["per_tag"].items() if abs(d["ratio"] - 1.0) > tol]
    if bad:
        for tag, d in bad:
            print(f"[budget]   TAG {tag} ({d['label']}): meshed/nominal volume ratio = {d['ratio']:.6f}")
    assert abs(b["p_gen_vs_intended"] - 1.0) < 1e-3, (
        f"delivered power {b['p_gen_w']*1e6:.4f}uW != intended {b['p_intended_w']*1e6:.4f}uW "
        f"(ratio {b['p_gen_vs_intended']:.6f}); per-tag volume ratios: "
        f"{ {t: round(d['ratio'], 4) for t, d in b['per_tag'].items()} }"
    )
    if not check_robin:
        print("[budget]   check_robin=False (transient/non-equilibrium step) -- "
              "Robin identity not asserted, only reported")
        return
    if b["p_gen_w"] < noise_floor_w:
        print(f"[budget]   p_gen < {noise_floor_w*1e12:.1f}pW noise floor -- "
              f"Robin-identity relative tolerance not meaningful, skipping assertion")
        return
    assert b["robin_rel_err"] < tol, f"Galerkin identity failed: rel_err={b['robin_rel_err']:.3e}"
