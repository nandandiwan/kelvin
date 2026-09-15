"""Emit MATERIALS.md -- every thermal property this solver uses, with a
PROVENANCE column saying where each number came from.

Generated, not hand-written, so the table cannot drift from spec/materials.py.
The k/rho/cp values are read live from the MATERIALS registry; only the
provenance/use annotations live here, keyed by material name (a material in
the registry with no annotation is an error, not a blank row -- an
unexplained material property in a validation deck is exactly the thing a
reviewer should catch, so this fails loudly).

    python cases/dump_materials_table.py

Provenance tiers, strongest first:
    MEASURED   direct experimental measurement, cited
    PUBLISHED  another paper's simulated/derived value, cited (used to
               reproduce THAT paper with ITS OWN inputs)
    HANDBOOK   standard bulk room-temperature reference value
    ESTIMATE   a stated engineering assumption of this project -- no primary
               source; listed so it can be argued with
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spec.materials import MATERIALS, k_si_thin_film_w_mk

OUT = Path("MATERIALS.md")

# name -> (tier, source, where it is used in the model)
PROVENANCE = {
    "Si_bulk": (
        "HANDBOOK",
        "Bulk single-crystal Si at 300 K, 148 W/m-K. k(T) exponent -1.3 is the "
        "standard Umklapp-scattering power law for Si over 300-400 K.",
        "Substrate everywhere; the reference point every thin-film k is reduced from.",
    ),
    "Si_thin_300nm": (
        "MEASURED",
        "k_si_thin_film_w_mk(0.3) -- 2-parameter fit k(d)=k_bulk/(1+a*d_nm^-p) "
        "pinned to Liu & Asheghi, J. Heat Transfer 128, 75 (2006), "
        "electrical-resistance thermometry on suspended SOI bridges "
        "(measured anchors: 22 W/m-K at 20 nm, 60 W/m-K at 100 nm).",
        "BSPDN stacks at the ITherm-2024 300 nm reference thickness.",
    ),
    "Si_thin_500nm": (
        "MEASURED",
        "Same Liu & Asheghi fit, evaluated at 500 nm.",
        "BSPDN stacks at Oprins' 500 nm reference thickness -- THIS project's "
        "own k choice, as opposed to Si_oprins_500nm below.",
    ),
    "Si_oprins_500nm": (
        "PUBLISHED",
        "Oprins et al. (imec), iTherm 2022, Fig. 9: 84 W/m-K at 500 nm, stated "
        "as 56% of bulk, from their own nanoscale Monte-Carlo BTE simulations "
        "assuming diffuse surfaces (their stated worst case for boundary scattering).",
        "ONLY for reproducing that paper with that paper's inputs -- the "
        "same-inputs-same-answer solver check. Never used as this project's own k.",
    ),
    "Cu_oprins": (
        "PUBLISHED",
        "Oprins et al., iTherm 2022, Table 2 / Fig. 9: 330 W/m-K for 250 nm-wide "
        "backside Cu metal, stated as 84% of bulk.",
        "Backside power-delivery M1 lines in the Oprins reproduction stack.",
    ),
    "Ru_oprins": (
        "PUBLISHED",
        "Oprins et al., iTherm 2022, Fig. 9: 110 W/m-K for a 30 nm Ru buried "
        "power rail, stated as 90% of bulk.",
        "Buried power rails (BPR) in the Oprins reproduction stack.",
    ),
    "Si_SD_doped": (
        "ESTIMATE",
        "Degenerately-doped source/drain Si. Ionized-impurity phonon scattering "
        "knocks k below bulk even in a ~30 nm crystalline film; 70 W/m-K is a "
        "stated assumption of this project, no primary source.",
        "Source/drain diffusion regions in the FEOL band.",
    ),
    "Si_channel": (
        "ESTIMATE",
        "Channel/inversion layer, only a few nm thick, so boundary and "
        "surface-roughness phonon scattering dominate bulk Umklapp. "
        "20 W/m-K is a stated assumption, no primary source.",
        "The heat-source volume itself -- every device channel in the bitcell.",
    ),
    "SiO2": (
        "HANDBOOK",
        "Thermal (not deposited) SiO2 at 300 K, 1.4 W/m-K.",
        "ILD/field oxide background in most bands; the low-k barrier in the "
        "layered solver-verification stack.",
    ),
    "low_k": (
        "ESTIMATE",
        "Generic porous low-k BEOL dielectric, 0.4 W/m-K. Representative of the "
        "SiO2:C / porous-organosilicate class; no specific SKY130 measurement.",
        "Upper BEOL inter-metal dielectric.",
    ),
    "HfO2": (
        "HANDBOOK",
        "High-k gate dielectric, 1.1 W/m-K thin-film value.",
        "Gate dielectric band.",
    ),
    "TiN": (
        "HANDBOOK",
        "TiN barrier/metal-gate film, 30 W/m-K.",
        "Barrier layers.",
    ),
    "PolySi": (
        "ESTIMATE",
        "Heavily-doped polysilicon gate. Grain-boundary phonon scattering puts it "
        "well below single-crystal Si despite the doping; 30 W/m-K is a stated "
        "assumption. This GDS has no separate gate-metal layer, so poly IS the gate.",
        "Gate electrodes (SKY130 poly layer).",
    ),
    "W": (
        "HANDBOOK",
        "Bulk tungsten, 170 W/m-K.",
        "Contact/via plugs (licon1) and the backside uTSVs in the Oprins stack.",
    ),
    "NiSi": (
        "ESTIMATE",
        "Self-aligned NiSi contact silicide, 35 W/m-K. Silicides conduct worse than "
        "either parent from intermetallic phonon-electron scattering; stated assumption.",
        "Silicide caps on source/drain and poly gate.",
    ),
    "Cu_fine": (
        "HANDBOOK",
        "Fine-pitch Cu interconnect, 350 W/m-K -- below bulk Cu's 401 from "
        "grain-boundary and surface scattering at small line width.",
        "Lower Cu metal levels (non-SKY130 stacks).",
    ),
    "Cu_thick": (
        "HANDBOOK",
        "Thick/upper-level Cu, 400 W/m-K, essentially bulk (wide lines, "
        "little size effect).",
        "Top metal and the default BSPDN backside metal.",
    ),
    "Al": (
        "HANDBOOK",
        "Thin-film Al, 220 W/m-K, below bulk Al's 237 from grain-boundary and "
        "surface scattering. SKY130's interconnect is aluminum, not copper.",
        "met1-met5 in the SKY130 frontside stack -- i.e. the real bitcell runs.",
    ),
    "TaN": (
        "HANDBOOK",
        "TaN diffusion barrier, 10 W/m-K.",
        "Barrier liner.",
    ),
    "SiN": (
        "HANDBOOK",
        "Silicon nitride passivation/etch-stop, 30 W/m-K.",
        "Passivation and etch-stop layers.",
    ),
}

TIER_ORDER = {"MEASURED": 0, "PUBLISHED": 1, "HANDBOOK": 2, "ESTIMATE": 3}


def main():
    missing = set(MATERIALS) - set(PROVENANCE)
    if missing:
        raise SystemExit(
            f"materials with no provenance annotation: {sorted(missing)}\n"
            "Add them to PROVENANCE in this file -- an unexplained material "
            "property must not reach a validation deck.")

    lines = [
        "# Material properties and where every number comes from",
        "",
        "Generated by `cases/dump_materials_table.py` from `spec/materials.py`.",
        "Do not edit by hand -- regenerate.",
        "",
        "All values are at a 300 K reference. `k` in W/m/K, `rho` in kg/m^3, "
        "`cp` in J/kg/K.",
        "",
        "Provenance tiers, strongest evidence first:",
        "",
        "| Tier | Meaning |",
        "|---|---|",
        "| `MEASURED` | Direct experimental measurement, cited. |",
        "| `PUBLISHED` | Another paper's simulated/derived value, cited. Used to "
        "reproduce *that* paper with *its own* inputs. |",
        "| `HANDBOOK` | Standard bulk room-temperature reference value. |",
        "| `ESTIMATE` | A stated engineering assumption of this project. No primary "
        "source -- listed explicitly so it can be argued with. |",
        "",
        "## Registry",
        "",
        "| Material | k | rho | cp | rho*cp (J/m^3/K) | k(T) exp | Tier | Source | Used for |",
        "|---|---:|---:|---:|---:|:---:|:---:|---|---|",
    ]

    def sort_key(name):
        tier, _, _ = PROVENANCE[name]
        return (TIER_ORDER[tier], -MATERIALS[name].k)

    for name in sorted(MATERIALS, key=sort_key):
        m = MATERIALS[name]
        tier, source, use = PROVENANCE[name]
        texp = f"{m.k_texp:g}" if m.k_texp is not None else "--"
        lines.append(
            f"| `{name}` | {m.k:.1f} | {m.rho:.0f} | {m.cp:.0f} | {m.rho_cp:.3e} | "
            f"{texp} | `{tier}` | {source} | {use} |")

    lines += [
        "",
        "## Thin-film silicon conductivity k(d)",
        "",
        "`spec/materials.py::k_si_thin_film_w_mk` implements",
        "",
        "```",
        "k(d) = k_bulk / (1 + a * d_nm^-p),   k_bulk = 148,  a = 72.30278,  p = 0.84641",
        "```",
        "",
        "`a` and `p` are *solved exactly*, not numerically fitted, from two "
        "**measured** anchors in Liu & Asheghi (2006) -- k = 22 W/m/K at d = 20 nm "
        "and k = 60 W/m/K at d = 100 nm -- plus the bulk limit k -> 148 as d -> inf.",
        "",
        "| d (nm) | k (W/m/K) | fraction of bulk |",
        "|---:|---:|---:|",
    ]
    for d_nm in (20, 50, 100, 200, 300, 500, 1000, 2500, 10000, 50000):
        k = k_si_thin_film_w_mk(d_nm / 1000.0)
        lines.append(f"| {d_nm} | {k:.1f} | {k/148.0:.3f} |")

    lines += [
        "",
        "> The 20 nm and 100 nm rows are the measured anchors themselves "
        "(22 and 60 W/m/K); every other row is the fit.",
        "",
        "### The one disagreement worth stating out loud",
        "",
        "At 500 nm this project's measured-data fit gives **107.6 W/m/K** (27% below "
        "bulk), while Oprins et al. report **84 W/m/K** (~44% below bulk) at the same "
        "thickness. The gap is real and explainable, not a bug:",
        "",
        "- Liu & Asheghi *measured* real fabricated SOI wafers, whose surfaces are "
        "neither perfectly diffuse nor perfectly specular.",
        "- Oprins' value comes from a Monte-Carlo BTE model that explicitly assumes "
        "**diffuse surfaces**, which they state is the worst case for boundary scattering.",
        "",
        "Both are carried as separate registry entries (`Si_thin_500nm` vs "
        "`Si_oprins_500nm`) rather than one being overwritten, so the Oprins "
        "reproduction can run on Oprins' own inputs while this project's own "
        "predictions run on measured data. Report the pair as a bracket, not a point.",
        "",
    ]

    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT} ({len(MATERIALS)} materials)")


if __name__ == "__main__":
    main()
