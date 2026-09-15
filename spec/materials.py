"""Material property registry — pure data, no dolfinx dependency.

k in W/m/K, rho in kg/m^3, cp in J/kg/K, all at 300 K reference.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Material:
    name: str
    k: float
    rho: float
    cp: float
    k_tensor: Optional[Tuple[float, float, float]] = None
    k_texp: Optional[float] = None

    def k_at(self, T: float) -> float:
        """Isotropic k(T) = k * (T / 300)^k_texp; constant if k_texp is None."""
        if self.k_texp is None:
            return self.k
        return self.k * (T / 300.0) ** self.k_texp

    @property
    def rho_cp(self) -> float:
        return self.rho * self.cp


def k_si_thin_film_w_mk(thickness_um: float) -> float:
    """Room-temperature (300K) lateral thermal conductivity of a thin
    single-crystal Si film, reduced from bulk by phonon-boundary scattering
    — the effect Oprins et al. (imec, iTherm 2022, "Package level thermal
    analysis of backside power delivery network configurations") name as the
    #1 physical reason a BSPDN's thinned Si spreads heat worse than a bulk
    wafer: "the Si thermal conductivity reduces by almost a factor 2 [at
    500nm], which has an additional impact of 50% on the temperature
    increase, on top of the thermal spreading resistance due to substrate
    thinning."

    This function is a 2-parameter fit k(d) = k_bulk / (1 + a*d_nm^-p),
    constrained to pass through two *measured* points (Liu & Asheghi,
    "Thermal Conductivity Measurements of Ultra-Thin Single Crystal Silicon
    Layers," J. Heat Transfer 128, 75 (2006) — electrical-resistance
    thermometry on suspended SOI bridges, not simulation):

        d = 20nm  -> k = 22 W/m/K   (measured, their Fig. 7/8)
        d = 100nm -> k = 60 W/m/K   (measured, their Fig. 7/8)
        d -> inf  -> k = 148 W/m/K  (bulk Si, this file's Si_bulk)

    giving a=72.30, p=0.8465 exactly (solved, not fit numerically — see
    scratch derivation in SRAM_THERMAL_REPORT.md's BSPDN section).

    Honest range at the thicknesses this project actually uses: at 500nm
    this measured-data extrapolation gives k=108 W/m/K (27% below bulk) —
    milder than Oprins' own Monte-Carlo-BTE estimate of ~factor-2 reduction
    (~74 W/m/K) at the same thickness. That gap is real and explainable, not
    a bug: Oprins' MC model explicitly assumes "diffuse surfaces (worst case
    for boundary scattering)," while Liu & Asheghi measured real fabricated
    SOI wafers, whose surfaces are neither perfectly diffuse nor perfectly
    specular. Report both as a bracket, not a single point — see the BSPDN
    section's k-sensitivity table.
    """
    d_nm = thickness_um * 1000.0
    k_bulk = 148.0  # matches Si_bulk's own k below — kept a literal, not a
    # MATERIALS lookup, since this function is called while MATERIALS itself
    # is still being constructed (registering the thin-film materials it feeds).
    a, p = 72.30277619973681, 0.8464117756967743
    return k_bulk / (1.0 + a * d_nm ** -p)


MATERIALS = {
    "Si_bulk":      Material("Si_bulk",      k=148.0, rho=2330.0,  cp=712.0, k_texp=-1.3),
    # Thin-film-corrected Si for the two BSPDN reference thicknesses this
    # project uses: 300nm (ITherm 2024's modeled value) and 500nm (Oprins et
    # al.'s reference case). Registered as named materials (each a plain
    # precomputed k, matching every other entry here) rather than a dynamic
    # per-region lookup, so physics/coeffs.py needs no changes — see
    # k_si_thin_film_w_mk's docstring above for the real measured anchors.
    "Si_thin_300nm": Material("Si_thin_300nm", k=k_si_thin_film_w_mk(0.3), rho=2330.0, cp=712.0),
    "Si_thin_500nm": Material("Si_thin_500nm", k=k_si_thin_film_w_mk(0.5), rho=2330.0, cp=712.0),

    # --- [Oprins] iTherm 2022's OWN published effective properties ---------
    # Used ONLY to reproduce that paper's results with that paper's inputs,
    # which is what validates the SOLVER (same inputs -> same answer?) as
    # opposed to validating this project's own independent material choices.
    # Keep them separate from the entries above rather than overwriting:
    # Si_thin_* above is fitted to Liu & Asheghi's MEASURED thin-film data,
    # while these come from Oprins' dedicated nanoscale Monte Carlo BTE
    # simulations (their Fig. 9) -- different kinds of evidence, and the gap
    # between them is itself a result worth reporting (107.6 vs 84.0 W/m-K at
    # 500nm, i.e. the measured-data fit is 28% more conductive than the BTE
    # value at this thickness).
    #   Si 500nm : 84 W/m-K, stated as 56% of bulk           ([Oprins] Fig. 9)
    #   Cu 250nm-wide backside metal: 330 W/m-K, 84% of bulk ([Oprins] Fig. 9)
    #   Ru 30nm buried power rail:    110 W/m-K, 90% of bulk ([Oprins] Fig. 9)
    "Si_oprins_500nm": Material("Si_oprins_500nm", k=84.0,  rho=2330.0, cp=712.0),
    "Cu_oprins":       Material("Cu_oprins",       k=330.0, rho=8940.0, cp=385.0),   # Table 2
    "Ru_oprins":       Material("Ru_oprins",       k=110.0, rho=12450.0, cp=238.0),
    # Degenerately-doped source/drain silicon: ionized-impurity phonon scattering
    # knocks k down from bulk even though it's still a ~30nm-thick crystalline film.
    "Si_SD_doped":  Material("Si_SD_doped",  k=70.0,  rho=2330.0,  cp=712.0, k_texp=-1.3),
    # Channel/inversion layer: only a few nm thick, so boundary/surface-roughness
    # phonon scattering dominates over bulk Umklapp scattering -> k drops further.
    "Si_channel":   Material("Si_channel",   k=20.0,  rho=2330.0,  cp=712.0, k_texp=-1.3),
    "SiO2":         Material("SiO2",         k=1.4,   rho=2200.0,  cp=740.0),
    "low_k":        Material("low_k",        k=0.4,   rho=1400.0,  cp=700.0),
    "HfO2":         Material("HfO2",         k=1.1,   rho=9680.0,  cp=120.0),
    "TiN":          Material("TiN",          k=30.0,  rho=5220.0,  cp=545.0),
    # Heavily-doped polysilicon gate (this GDS has no separate gate-metal
    # layer, so it's a poly-gate node): grain-boundary phonon scattering
    # knocks k down from single-crystal Si despite the heavy doping.
    "PolySi":       Material("PolySi",        k=30.0,  rho=2330.0,  cp=712.0),
    "W":            Material("W",            k=170.0, rho=19300.0, cp=134.0),
    # Self-aligned NiSi contact silicide, capping S/D and poly-gate before the
    # W plugs. Silicides conduct worse than either parent (Ni or Si) due to
    # intermetallic phonon-electron scattering.
    "NiSi":         Material("NiSi",         k=35.0,  rho=5900.0,  cp=430.0),
    "Cu_fine":      Material("Cu_fine",      k=350.0, rho=8960.0,  cp=385.0),
    "Cu_thick":     Material("Cu_thick",     k=400.0, rho=8960.0,  cp=385.0),
    # SKY130's interconnect is aluminum
    # Thin-film Al runs below bulk Al's 237 W/m/K from grain-boundary and
    # surface scattering.
    "Al":           Material("Al",           k=220.0, rho=2700.0,  cp=897.0),
    "TaN":          Material("TaN",          k=10.0,  rho=14300.0, cp=210.0),
    "SiN":          Material("SiN",          k=30.0,  rho=3170.0,  cp=700.0),
}


def get(name: str) -> Material:
    try:
        return MATERIALS[name]
    except KeyError:
        raise KeyError(f"unknown material {name!r}; known: {sorted(MATERIALS)}")
