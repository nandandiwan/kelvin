# SRAM contact geometry: whole cuts and validated landings

Checked against `data/sram22_64x22m4w22.gds / sram_sp_cell` on 2026-09-16.

## Why the old notebook check failed

The bitcell has ten distinct LICON cuts: eight contact diffusion and two contact
polysilicon. Every cut overlaps its intended lower conductor and LI1. However,
the cuts are not wholly enclosed by those drawn landing masks:

| Measured quantity | Area or count |
| --- | ---: |
| Complete LICON mask union | 0.2312 µm² |
| Area overlapping the selected lower mask | 0.20145 µm² |
| Area outside the lower mask | 0.02975 µm² |
| Area outside the LI1 mask | 0.00705 µm² |
| Diffusion-contact cuts | 8 |
| Poly-contact cuts | 2 |

The old region builder intersected each contact with poly/diff and demanded
that their union recover the entire LICON mask. Its error therefore identified
**partial enclosure**, not a floating contact. Keeping only those intersections
would discard 12.87% of the drawn contact area and alter the contact geometry.

## The classification used now

`mesh/gds_contacts.py` classifies each complete cut before extrusion:

1. It must overlap exactly one lower category: poly, or diffusion/tap.
2. It must have positive-area overlap with LI1 above it.
3. Its entire drawn footprint is preserved; it is not clipped to the landing.
4. Partial lower/upper coverage is accepted only when the entire cut lies inside
   the explicit memory-core marker, `(81, 2)`. Outside that marker, the former
   full-coverage requirement remains in force.
5. A cut with no lower landing, no upper landing, or both poly and diffusion/tap
   landings raises an error. An unrelated well-covered cut cannot hide a
   disconnected cut through an aggregate coverage percentage.

This distinction has process-documentation support. SKY130 rule `licon.4`
requires overlap with the lower conductor and LI1; `licon.17` prohibits a
mixed poly/diffusion landing. The diffusion/poly enclosure rules carry the
periphery-only flag, which excludes the memory core. See the
[SKY130 contact rules and flag definitions](https://skywater-pdk.readthedocs.io/en/main/rules/periphery.html).
The [SKY130 layer reference](https://skywater-pdk.readthedocs.io/en/main/rules/layers.html)
identifies `(81, 2)` as the SRAM memory-core marker; this particular cell's marker
covers its complete 1.20 × 1.58 µm footprint.

`validate_sky130_contacts()` applies per-cut lower/upper landing checks to MCON
and the via levels too. The actual SRAM has four distinct MCON cuts after
deduplicating one repeated polygon, and two VIA cuts. Both levels have 100%
lower and upper coverage. Thus their validation does not rely on the
memory-core partial-coverage provision.

## What is, and is not, established

The shared notebook/backend uses full-footprint tungsten prisms. Diffusion/tap
contacts start at the active-silicon top; poly contacts start at the poly top;
both end at LI1's bottom. Conductor/dielectric partitions are rebuilt around
those complete prisms. The classification report is retained under
`process_model.licon_contact_classification`, including each cut's measured
coverage, area and acceptance reason. Coverage figures remain honest
measurements, rather than being forced to 100% to pass a check.

This is a validated **drawn-mask thermal abstraction**, not a foundry DRC
certificate or reconstruction of the fabricated contact profile. The model
still assumes a uniform vertical prism and the configured lower landing plane,
including where a cut overhangs a drawn lower mask. It does not infer etch
profiles, silicide, field-oxide topography, mask bias or optical corrections.
Mask-add/drop purposes are not independent thermal films, and this model does
not reconstruct the final manufacturing masks from them. Those limitations
remain relevant to absolute-temperature accuracy.

## Verification

`tests/test_gds_contacts.py` verifies:

- Original SRAM contact footprints are preserved exactly by a polygon XOR
  comparison; all ten cuts pass both landing checks and reproduce the measured
  0.02975 µm² overhang.
- Partial cuts outside the core, including cuts only partly covered by the core
  marker, are rejected.
- Missing landings, edge-only tangency and mixed poly/diff landings are rejected.
- Floating contacts cannot hide in aggregate statistics.
- Fully covered contacts outside the core remain valid.

The initial focused run passed **11 tests**. A shared-backend region build also
produced all eight channel-source regions and passed its material-overlap check.
These checks establish geometry bookkeeping and landing classification; full
mesh, source-power and thermal-solver verification are separate pipeline checks.
