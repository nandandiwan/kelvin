"""Facet-tag-driven BCs. The backside always gets an explicit Robin term
(lumped TIM+spreader+sink); the top gets a second one only if
chip.bcs.top_h_eff is set (dual-sided cooling — the BSPDN study's config C),
otherwise it's natural (zero-flux), i.e. adiabatic. Left/right are always
natural: they're the artificial cut through a repeating tile array, not real
boundaries (PLAN.md: "the remaining Si acts as a guard band"). An explicit
periodic BC is a further, later, additive change here, not to forms.py.
"""

from mesh.build import FACET_BOTTOM, FACET_TOP


def robin_terms(u, v, ds, chip):
    """(bilinear_term, linear_term) summed over every active Robin sink —
    just the backside by default, backside+top under dual-sided cooling.
    Energy conservation must be checked against this same sum (see
    tests/test_gds_pipeline.py): with two sinks, p_gen == flux(bottom) +
    flux(top), not flux(bottom) alone.
    """
    t_amb = chip.bcs.ambient_t_k
    h_bottom = chip.bcs.backside_h_eff
    a_robin = h_bottom * u * v * ds(FACET_BOTTOM)
    l_robin = h_bottom * t_amb * v * ds(FACET_BOTTOM)

    if chip.bcs.top_h_eff is not None:
        h_top = chip.bcs.top_h_eff
        a_robin += h_top * u * v * ds(FACET_TOP)
        l_robin += h_top * t_amb * v * ds(FACET_TOP)

    return a_robin, l_robin
