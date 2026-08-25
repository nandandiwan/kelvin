"""Facet-tag-driven BCs. Only the backside gets an explicit term (Robin,
lumped TIM+spreader+sink); top/left/right are natural (zero-flux) in this
weak form — top because chip.bcs.top_face defaults to adiabatic, left/right
because they're the artificial cut through a repeating tile array, not real
boundaries (PLAN.md: "the remaining Si acts as a guard band"). A C4 flip-chip
top or an explicit periodic BC are both additive, later changes here, not to
forms.py.
"""

from mesh.build import FACET_BOTTOM


def robin_terms(u, v, ds, chip):
    """(bilinear_term, linear_term) for the backside Robin BC h_eff*(T-T_amb)."""
    if chip.bcs.top_face != "adiabatic":
        raise NotImplementedError(f"top_face={chip.bcs.top_face!r} not implemented yet")
    h = chip.bcs.backside_h_eff
    t_amb = chip.bcs.ambient_t_k
    a_robin = h * u * v * ds(FACET_BOTTOM)
    l_robin = h * t_amb * v * ds(FACET_BOTTOM)
    return a_robin, l_robin
