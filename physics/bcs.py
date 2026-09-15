"""Facet-tag-driven BCs. The backside always gets an explicit Robin term
(lumped TIM+spreader+sink); the top gets a second one only if
chip.bcs.top_h_eff is set (dual-sided cooling — the BSPDN study's config C),
otherwise it's natural (zero-flux), i.e. adiabatic. Left/right are always
natural: they're the artificial cut through a repeating tile array, not real
boundaries (PLAN.md: "the remaining Si acts as a guard band"). An explicit
periodic BC is a further, later, additive change here, not to forms.py.
"""

from mesh.build import (
    FACET_BOTTOM, FACET_TOP, FACET_X0, FACET_X1, FACET_Y0, FACET_Y1,
)


def robin_terms(u, v, ds, chip, k=None):
    """(bilinear_term, linear_term) summed over every active Robin sink —
    just the backside by default, backside+top under dual-sided cooling.
    Energy conservation must be checked against this same sum (see
    tests/test_gds_pipeline.py): with two sinks, p_gen == flux(bottom) +
    flux(top), not flux(bottom) alone.

    `chip.bcs.lateral_radiation_r_m` additionally opens the four LATERAL cut
    faces as an asymptotic radiation ("absorbing") boundary — see
    `lateral_radiation_terms`. `k` (the DG0 conductivity field) is required
    for that and ignored otherwise.
    """
    t_amb = chip.bcs.ambient_t_k
    h_bottom = chip.bcs.backside_h_eff
    a_robin = h_bottom * u * v * ds(FACET_BOTTOM)
    l_robin = h_bottom * t_amb * v * ds(FACET_BOTTOM)

    if chip.bcs.top_h_eff is not None:
        h_top = chip.bcs.top_h_eff
        a_robin += h_top * u * v * ds(FACET_TOP)
        l_robin += h_top * t_amb * v * ds(FACET_TOP)

    r_m = getattr(chip.bcs, "lateral_radiation_r_m", None)
    if r_m:
        if k is None:
            raise ValueError("lateral_radiation_r_m needs the conductivity field k")
        a_lat, l_lat = lateral_radiation_terms(u, v, ds, k, r_m, t_amb)
        a_robin += a_lat
        l_robin += l_lat

    return a_robin, l_robin


def lateral_radiation_terms(u, v, ds, k, r_m, t_amb):
    """Asymptotic radiation BC on the four lateral cut faces: the elliptic
    (heat-conduction) analogue of an absorbing/PML boundary.

    A PML proper is a WAVE construct — it damps outgoing waves so they don't
    reflect. The steady heat equation is elliptic: nothing propagates and
    nothing reflects, so there is no PML to build. The real problem a
    truncated lateral face causes is different: leaving it natural imposes
    ZERO flux, i.e. "heat may never leave sideways", which for a small source
    in a wide medium is badly wrong and makes the domain act like a box the
    heat is trapped in.

    The fix is to impose the far-field behaviour analytically instead of
    meshing out to it. Away from a small source the field is asymptotically
    radial, T - T_amb ~ C/r, so

        dT/dr = -(T - T_amb)/r      =>      -k dT/dn = (k/r) (T - T_amb)

    which is exactly a Robin term with h = k/r. It represents "the medium
    continues to infinity beyond this face" in closed form, using the LOCAL
    k of whatever layer each facet sits in (k is the DG0 field, so a metal
    band radiates with metal's conductivity and an oxide band with oxide's).

    Validity is not assumed: it holds once r is large compared with the
    source, and the check is that the answer stops depending on the domain
    size r. If shrinking the domain changes the result, the BC is being
    applied too close in.
    """
    h = k / r_m
    lateral = (FACET_X0, FACET_X1, FACET_Y0, FACET_Y1)
    a = sum(h * u * v * ds(tag) for tag in lateral)
    l = sum(h * t_amb * v * ds(tag) for tag in lateral)
    return a, l
