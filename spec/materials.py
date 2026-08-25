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


MATERIALS = {
    "Si_bulk":      Material("Si_bulk",      k=148.0, rho=2330.0,  cp=712.0, k_texp=-1.3),
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
    "W":            Material("W",            k=170.0, rho=19300.0, cp=134.0),
    # Self-aligned NiSi contact silicide, capping S/D and poly-gate before the
    # W plugs. Silicides conduct worse than either parent (Ni or Si) due to
    # intermetallic phonon-electron scattering.
    "NiSi":         Material("NiSi",         k=35.0,  rho=5900.0,  cp=430.0),
    "Cu_fine":      Material("Cu_fine",      k=350.0, rho=8960.0,  cp=385.0),
    "Cu_thick":     Material("Cu_thick",     k=400.0, rho=8960.0,  cp=385.0),
    "TaN":          Material("TaN",          k=10.0,  rho=14300.0, cp=210.0),
    "SiN":          Material("SiN",          k=30.0,  rho=3170.0,  cp=700.0),
}


def get(name: str) -> Material:
    try:
        return MATERIALS[name]
    except KeyError:
        raise KeyError(f"unknown material {name!r}; known: {sorted(MATERIALS)}")
