"""GDS (layer, datatype) -> z-band + material, for the SKY130 open PDK. GDS
carries no z information at all, so the z-stack comes from outside the layout:
here, from **open_pdks' magic extraction tech file**
(`sky130/magic/sky130.tech` @ RTimothyEdwards/open_pdks), whose `height`
block gives each layer's real bottom-z and thickness in µm — the numbers magic
uses for SKY130 parasitic RC extraction. These are real process heights, not
the estimates this file previously carried. Layer *materials* are likewise
pinned to what the same file's sheet resistances imply (aluminum metals, TiN
local interconnect — see the notes on individual bands below), not assumed
from a more advanced node's Cu/low-k conventions.

Layer numbers are SKY130's real, public GDS layer map (not reverse-engineered
by evidence like the old proprietary-PDK pipeline had to do) — taken from
open_pdks' klayout layer-properties file
(sky130/klayout/sky130.lyp @ RTimothyEdwards/open_pdks), and cross-checked
against actual layer usage in data/sram22_64x22m4w22.gds (a real, placed
SRAM macro from github.com/ucb-substrate/sram22_sky130_macros, BSD-3): every
layer below actually appears in that file, and this macro draws no met3/
met4/met5/via2/via3/via4 at all (it's a small macro, internal routing tops
out at met2) — the stack stops there rather than inventing metals that
aren't really used.

Ignored on purpose (present in the GDS, not physical drawn geometry): implant
markers (psdm/nsdm/hvtp/lvtn/npc — doping flavor, not a distinct z-band),
`.pin`-datatype layers (label/LVS markers coincident with real drawn shapes,
would double-count area if treated as geometry), `maskAdd`/`maskDrop` helper
layers (cp1m/cli1m/cfom — DRC/LVS-deck bookkeeping, never physically drawn),
`areaid.*` (floorplan markers), `OUTLINE` (cell boundary), and `dnwell`
(deep n-well isolation, folded into the substrate at LOD1 rather than given
its own band).
"""

from dataclasses import dataclass
from typing import Tuple

from spec.materials import get as get_material

# (layer, datatype) keys, named for readability at the call site. Names and
# numbers match SKY130's own layer names exactly (nwell, diff, poly, ...),
# not a renaming convention borrowed from elsewhere.
NWELL = (64, 20)
PWELL = (64, 44)
DIFF = (65, 20)
TAP = (65, 44)
POLY = (66, 20)
LICON1 = (66, 44)
LI1 = (67, 20)
MCON = (67, 44)
MET1 = (68, 20)
VIA = (68, 44)
MET2 = (69, 20)

# Every raw GDS layer this pipeline actually reads — pass to
# gds.read.union_merge as its `layers` allowlist so union-merge isn't spent
# on the pin/marker/mask-processing layers this file ignores (see module
# docstring).
RELEVANT_LAYERS = (NWELL, PWELL, DIFF, TAP, POLY, LICON1, LI1, MCON, MET1, VIA, MET2)


# Sentinel `drawn` key meaning "not read from the GDS at all — generate a
# periodic grid instead" (paired with a band's `synthetic_grid`). Used for
# backside power delivery geometry: SKY130 has no BSPDN process, so nano-TSVs
# and backside rails don't exist in any GDS layer to read — they're
# synthesized directly as a repeating pattern, reusing the same
# insert_interval machinery real layers use (see mesh/gds_section.py).
SYNTHETIC = None


@dataclass(frozen=True)
class GdsLayerBand:
    name: str
    thickness_um: float
    background: str                 # ILD/field material filling the die footprint
    # ((layer, datatype), material) pairs drawn into this band, applied in
    # order (later wins on overlap). A tuple rather than one material because
    # the real stack has layers that coexist at the same height: a licon
    # contact plug runs *past* the poly gate (both occupy z=0.3262-0.5062),
    # side by side in x — they are not stacked one above the other. A
    # (SYNTHETIC, material) entry means "generate a grid" rather than "read
    # this GDS layer" — see `synthetic_grid`.
    drawn: Tuple[Tuple[Tuple[int, int], str], ...]
    mesh_size_um: float
    # (pitch_um, feature_width_um) for a SYNTHETIC-drawn band: a feature of
    # that width is placed every pitch_um across the die, background material
    # filling the gaps. None for every real (GDS-sourced) band.
    synthetic_grid: "Tuple[float, float] | None" = None

    def __post_init__(self):
        get_material(self.background)
        for key, material in self.drawn:
            get_material(material)
            if key is SYNTHETIC and self.synthetic_grid is None:
                raise ValueError(f"{self.name!r}: SYNTHETIC drawn entry needs synthetic_grid set")


# Bulk silicon below the SKY130 stack proper. The real stack's own well layer
# (0 -> 0.2062um) is folded in here: it's silicon either way, and giving it its
# own band would only add a mesh region with the substrate's own material.
SUBSTRATE_THICKNESS_UM = 50.0 + 0.2062

# Real SKY130 layer heights, from open_pdks' magic extraction tech file
# (sky130/magic/sky130.tech, the `height <layer> <z_bottom> <thickness>` block
# magic uses for parasitic RC extraction), standard non-RERAM variant:
#
#   nwell/pwell  0.0000  0.2062      li1    0.9361  0.10
#   diff         0.2062  0.12        mcon   1.0361  0.34
#   poly         0.3262  0.18        met1   1.3761  0.36
#   licon(diff)  0.3262  0.61        v1     1.7361  0.27
#   licon(poly)  0.5062  0.43        met2   2.0061  0.36
#
# Two structural facts that fall out of those real numbers, which the earlier
# estimated stack had wrong:
#   1. A licon contact plug spans 0.3262 -> 0.9362, which *overlaps* poly's own
#      0.3262 -> 0.5062 band. Contacts run alongside gates at the same height,
#      not stacked above them — hence the poly band below draws both PolySi
#      (where poly is) and W (where licon is).
#   2. There is no meshable gate-dielectric band: magic puts poly's bottom
#      exactly at diff's top. SKY130's real gate oxide is ~4.15nm, far too thin
#      to mesh across a 300um-wide die, and thermally near-transparent anyway
#      (t/k ~ 3e-9 m^2K/W). It is omitted, and the channel heat source now sits
#      where it physically belongs — in the top ~10nm of the *silicon*, the
#      inversion layer, not in the oxide above it.
# The device + BEOL bands are shared verbatim between every stack profile
# (frontside PDN and BSPDN alike) — backside power delivery changes what's
# *below* the devices, not the devices or their own interconnect. Defined
# once here so there is exactly one copy to keep in sync with real SKY130
# data, referenced by both FRONTSIDE_STACK and BSPDN_STACK below.
_DEVICE_AND_BEOL_BANDS = (
    # diff, split so the inversion layer at its top surface can carry the
    # channel heat source (see gds/sources.py). 0.12 total = 0.11 + 0.01.
    GdsLayerBand("diff",    0.11, "SiO2", ((DIFF, "Si_SD_doped"),), 0.02),
    GdsLayerBand("channel", 0.01, "SiO2", ((DIFF, "Si_SD_doped"),), 0.01),
    # poly and the lower part of the licon plugs share this height range.
    GdsLayerBand("poly",    0.18, "SiO2", ((POLY, "PolySi"), (LICON1, "W")), 0.03),
    # licon only, above poly's top up to li1's bottom.
    GdsLayerBand("licon1",  0.43, "SiO2", ((LICON1, "W"),), 0.04),
    # li1 is SKY130's local interconnect: 12.8 Ohm/sq over a 0.10um film
    # (~1.3e-6 Ohm.m) is far too resistive for a bulk metal — it's TiN, not Cu.
    GdsLayerBand("li1",     0.10, "SiO2", ((LI1, "TiN"),),  0.03),
    GdsLayerBand("mcon",    0.34, "SiO2", ((MCON, "W"),),   0.04),
    # met1/met2 are aluminum (see spec/materials.py's Al note), in a
    # conventional SiO2-based ILD — a 130nm aluminum process, not an
    # advanced-node Cu/ultra-low-k one.
    GdsLayerBand("met1",    0.36, "SiO2", ((MET1, "Al"),),  0.04),
    GdsLayerBand("via",     0.27, "SiO2", ((VIA, "W"),),    0.04),
    GdsLayerBand("met2",    0.36, "SiO2", ((MET2, "Al"),),  0.04),
    # Not in the extraction tech file (it stops at top metal) — nominal.
    GdsLayerBand("passivation", 1.00, "SiN", (), 0.20),
)


@dataclass(frozen=True)
class StackProfile:
    """A named, ordered set of z-bands from substrate to passivation. Two
    profiles exist (FRONTSIDE_STACK, BSPDN_STACK) so the BSPDN thermal study
    can solve the identical devices+BEOL against two different backside
    configurations — see SRAM_THERMAL_REPORT.md's BSPDN section.
    """
    name: str
    bands: Tuple[GdsLayerBand, ...]

    def z_bounds(self):
        """[(z0, z1, band), ...] cumulative from z=0 at the stack's base."""
        z = 0.0
        out = []
        for band in self.bands:
            out.append((z, z + band.thickness_um, band))
            z += band.thickness_um
        return out

    def total_thickness_um(self) -> float:
        return sum(b.thickness_um for b in self.bands)

    def find(self, name: str) -> GdsLayerBand:
        for b in self.bands:
            if b.name == name:
                return b
        raise KeyError(f"no band named {name!r} in stack {self.name!r}")


FRONTSIDE_STACK = StackProfile("frontside", (
    GdsLayerBand("Si_substrate", SUBSTRATE_THICKNESS_UM, "Si_bulk", (), 5.0),
) + _DEVICE_AND_BEOL_BANDS)

# --- BSPDN (backside power delivery network) ---
# SKY130 has no BSPDN process; there is nothing in any GDS to read for the
# backside stack, so it is synthesized — but every dimension below is now a
# **real published number**, not a nominal guess, from:
#   [Oprins]  H. Oprins et al., "Package level thermal analysis of backside
#             power delivery network (BS-PDN) configurations," iTherm 2022 —
#             the primary source: real via-last BS-PDN test-case dimensions,
#             SiO2 k=1.4 W/m-K (exactly matches this file's own SiO2 entry —
#             an independent cross-check, not assumed to match), and BPR(Ru)/
#             backside-metal(Cu)/nTSV(W) materials from dedicated Monte Carlo
#             BTE simulations.
#   [ITherm]  Xie, Lyu, Wei (Purdue), ITherm 2024 — die-level package stack
#             (0.5um SiO2 bonding, 300nm Si), corroborates Oprins' range.
#   [Veloso]  A. Veloso et al. (imec), "Backside Power Delivery: Game Changer
#             ...," IEDM 2023 — real *fabricated* devices thinned to 210nm,
#             110nm, and (extreme) ~20nm under STI, confirming 300-500nm is a
#             conservative, already-demonstrated choice, not a stretch.
#
# One structural fix this real data forced, beyond just better numbers: a
# real via-last nano-TSV is **etched through the thinned Si itself** to land
# on the buried power rail at the FEOL side (Oprins section III.A) — not a
# separate layer stacked below the Si, which was this file's first-pass
# approximation. The Si_substrate band below now carries a SYNTHETIC W-filled
# grid alongside its background material, at Oprins' own µTSV/M1 pitch,
# instead of a separate "ntsv" band.
_BSPDN_CARRIER_UM = 50.0          # placeholder: same truncated-with-Robin-sink
                                   # convention as FRONTSIDE_STACK's own
                                   # Si_substrate (see that band above) — no
                                   # stated carrier thickness in [Oprins]/
                                   # [ITherm] (ITherm's schematic is explicitly
                                   # "not to scale"), so this asserts the same
                                   # modeling choice already used elsewhere in
                                   # this file, not a specific real wafer spec.
_BSPDN_BOND_UM = 0.5              # [ITherm]: "SiO2 0.5um bonding"
_BSPDN_METAL_UM = 0.210           # [Oprins] Table 1: "Backside metal M1 height 210nm" (reference case)
_BSPDN_METAL_PITCH_UM = 0.5       # [Oprins] Table 1: "Backside metal M1 pitch 500nm"
_BSPDN_METAL_WIDTH_UM = 0.25      # [Oprins] gives pitch, not width, for M1 specifically
                                   # (unlike BPR, which states width=30nm) — assumed
                                   # half-pitch damascene fill, a documented assumption.
_BSPDN_SI_UM = 0.3                # [ITherm]'s modeled value; within [Oprins]'s
                                   # stated "300-500nm" via-last range and
                                   # [Veloso]'s demonstrated 210-370nm hardware.
_BSPDN_NTSV_PITCH_UM = _BSPDN_METAL_PITCH_UM  # [Oprins]: nTSV pitch = M1 pitch (one via per M1 line)
_BSPDN_NTSV_WIDTH_UM = 0.25        # [Oprins] Table 1: µTSV "180nm x 250nm" — larger
                                    # of the two in-plane dimensions, W-filled.

def bspdn_stack(name: str, si_um: float, si_material: str,
                 with_backside_metal: bool = True,
                 carrier_um: float = _BSPDN_CARRIER_UM) -> StackProfile:
    """Build a BSPDN backside stack + the shared real SKY130 devices/BEOL —
    the general form of BSPDN_STACK below, parametrized so
    SRAM_THERMAL_REPORT.md's validation benchmark can build the exact
    variants [Oprins] compares (with vs. without backside metal, at their
    reference 500nm Si) without duplicating the device+BEOL band list.
    `si_material` lets the caller pick bulk Si_bulk vs. a thin-film-corrected
    entry (spec/materials.py) — comparing the two in isolation is exactly
    [Oprins]' own headline finding (thin-film k alone contributes ~50% of
    the total temperature increase at 500nm).
    """
    bands = [
        GdsLayerBand("Si_carrier", carrier_um, "Si_bulk", (), 5.0),
        GdsLayerBand("bonding_interface", _BSPDN_BOND_UM, "SiO2", (), 0.10),
    ]
    if with_backside_metal:
        bands.append(GdsLayerBand("backside_metal", _BSPDN_METAL_UM, "SiO2",
                                   ((SYNTHETIC, "Cu_thick"),), 0.03,
                                   synthetic_grid=(_BSPDN_METAL_PITCH_UM, _BSPDN_METAL_WIDTH_UM)))
        # nTSVs only exist to land on a backside metal/BPR — no metal, no via.
        si_drawn = ((SYNTHETIC, "W"),)
        si_grid = (_BSPDN_NTSV_PITCH_UM, _BSPDN_NTSV_WIDTH_UM)
    else:
        si_drawn, si_grid = (), None
    bands.append(GdsLayerBand("Si_substrate", si_um, si_material, si_drawn, 0.02, synthetic_grid=si_grid))
    return StackProfile(name, tuple(bands) + _DEVICE_AND_BEOL_BANDS)


# The real-layout stack (cases/run_bspdn_study.py, the SRAM macro): [ITherm]'s
# 300nm Si, thin-film-corrected, with backside metal + through-Si nTSVs.
BSPDN_STACK = bspdn_stack("bspdn", _BSPDN_SI_UM, "Si_thin_300nm", with_backside_metal=True)

STACKS = {"frontside": FRONTSIDE_STACK, "bspdn": BSPDN_STACK}

# Backward-compat: every existing caller uses these as bare module-level
# functions/constants meaning "the frontside stack" (`techmap.z_bounds()`,
# `techmap.find("...")`, `techmap.GDS_STACK`) — kept working unchanged, now
# with an optional `stack=` for callers that want the BSPDN profile instead.
GDS_STACK = FRONTSIDE_STACK.bands

CHANNEL_THICKNESS_UM = FRONTSIDE_STACK.find("channel").thickness_um
LICON1_THICKNESS_UM = FRONTSIDE_STACK.find("licon1").thickness_um


def z_bounds(stack: "StackProfile | None" = None):
    """[(z0, z1, band), ...] cumulative from z=0 at the stack's base."""
    return (stack or FRONTSIDE_STACK).z_bounds()


def total_thickness_um(stack: "StackProfile | None" = None) -> float:
    return (stack or FRONTSIDE_STACK).total_thickness_um()


def find(name: str, stack: "StackProfile | None" = None) -> GdsLayerBand:
    return (stack or FRONTSIDE_STACK).find(name)
