"""Vector isometric of the [Oprins] iTherm 2022 BS-PDN stack -- the same
renderer used for the SKY130 bitcell (mesh/gds_svg_viz.py), driven from the
real emitters instead of from GDS masks.

None of this geometry exists in any mask layer: SKY130 has no BS-PDN process,
so the backside metal tracks, the uTSVs and the buried power rails are
SYNTHETIC grids generated from pitch/width rules (gds.techmap.SYNTHETIC). This
script builds the renderer's `regions` list directly from those same rules --
gds.techmap.oprins_stack for the band table and z-heights,
mesh.gds_volume._synthetic_grid_rects for the features -- so the picture is the
geometry that actually gets meshed and solved, not a redrawing of it.

    python cases/render_oprins_svg.py [--window-um 2.4] [--full-stack]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cases.run_bspdn_benchmark_3d import CENTER_UM
from cases.run_oprins_reproduction import thinned_stack
from gds import techmap
from mesh.gds_svg_viz import render_regions_geometry_svg
from mesh.gds_volume import _synthetic_grid_rects

OUT = Path("out/presentation")

# Palette echoing [Oprins] Fig. 5 so the two can be read side by side.
BAND_STYLE = {
    # Cu track and W via must be visibly different: they sit directly on top
    # of each other and are the two halves of the mechanism being modelled.
    "backside_metal":    ("backside M1 (Cu)", "#b3261e"),
    "p_substrate":       ("&#181;TSV / p-sub (W)", "#f7a7b5"),
    "well":              ("&#181;TSV / well (W)", "#f07f96"),
    "buried_power_rail": ("buried rail (Ru)", "#7b2fbe"),
    "diff":              ("diffusion", "#7fc08a"),
    "poly":              ("poly gate", "#c9a0dc"),
    "licon1":            ("contact (W)", "#2b2b2b"),
    "li1":               ("local interconnect", "#d1342a"),
    "met1":              ("met1", "#e0a840"),
    "met2":              ("met2", "#3d6fd6"),
}
# Bands that are solid slabs rather than patterned grids. Drawn only when the
# caller asks for the full stack -- a solid carrier slab painted over the whole
# window hides every feature underneath it in an isometric view.
SOLID_BANDS = {"Si_carrier": ("Si carrier", "#cfd6db"),
               "bonding_interface": ("bonding oxide", "#eef4f8")}


def build_oprins_regions(stack, window, include_solid=False, include_beol=True,
                          context=True):
    """One region per band that has visible geometry, in the renderer's schema.

    `context=True` adds a translucent slab spanning the thinned silicon. Without
    it the uTSVs and rails hang in empty space and the figure does not convey
    that the vias run THROUGH silicon -- which is the whole point of the
    structure. It is drawn at low opacity so the features stay readable.
    """
    x0, x1, y0, y1 = window
    anchor = (CENTER_UM, CENTER_UM)
    regions, tag = [], 0

    if context:
        zb = {b.name: (a, c) for a, c, b in techmap.z_bounds(stack)}
        si_z0 = zb["p_substrate"][0]
        si_z1 = zb["buried_power_rail"][1] if "buried_power_rail" in zb else zb["well"][1]
        regions.append({
            "physical_tag": tag, "name": "thinned Si (body)", "role": "substrate",
            "color": "#9fb3c8", "z_min_um": round(si_z0, 4), "z_max_um": round(si_z1, 4),
            "volumes": [{"footprint_xy_um": [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                         "area_um2": (x1 - x0) * (y1 - y0)}],
            "source_polygon_count": 1, "placeholder": False, "material": "Si",
            "opacity": 0.30,
        })
        tag += 1

    for z0, z1, band in techmap.z_bounds(stack):
        if band.name in SOLID_BANDS:
            if not include_solid:
                continue
            label, color = SOLID_BANDS[band.name]
            rects = [(x0, y0, x1, y1)]
        elif band.synthetic_grid is not None:
            label, color = BAND_STYLE.get(band.name, (band.name, "#999999"))
            rects = _synthetic_grid_rects(
                x0, y0, x1, y1, *band.synthetic_grid, band.synthetic_shape,
                band.synthetic_width_y_um, anchor, band.synthetic_pitch_y_um,
                band.synthetic_stagger_y_um)
        else:
            # A real GDS-backed band (the FEOL/BEOL on top). Those carry no
            # geometry in this synthetic test case -- the Oprins cell is a bare
            # stack with a single analytic heat source, not a laid-out cell --
            # so there is nothing to draw and they are skipped rather than
            # faked as full-window slabs.
            continue

        if not include_beol and band.name in ("diff", "poly", "licon1", "li1", "met1", "met2"):
            continue
        if not rects:
            continue
        regions.append({
            "physical_tag": tag, "name": label, "role": "conductor", "color": color,
            "z_min_um": round(z0, 4), "z_max_um": round(z1, 4),
            "volumes": [{"footprint_xy_um": [(rx0, ry0), (rx1, ry0), (rx1, ry1), (rx0, ry1)],
                         "area_um2": (rx1 - rx0) * (ry1 - ry0)}
                        for rx0, ry0, rx1, ry1 in rects],
            "source_polygon_count": len(rects), "placeholder": False,
            "material": band.background,
        })
        tag += 1
    return regions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--window-um", type=float, default=2.4,
                   help="lateral extent drawn; a few uTSV pitches reads best")
    p.add_argument("--si-um", type=float, default=0.5)
    p.add_argument("--full-stack", action="store_true",
                   help="also draw the solid carrier and bonding oxide")
    p.add_argument("--no-context", action="store_true",
                   help="omit the translucent silicon body the vias run through")
    p.add_argument("--z-exaggeration", type=float, default=3.0,
                   help="the stack is 0.86um tall over a several-um window; the "
                        "renderer's automatic rule leaves it too flat to read")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    stack = thinned_stack(args.si_um, with_metal=True,
                          si_material="Si_oprins_500nm", with_bpr=True)
    half = args.window_um / 2
    window = (CENTER_UM - half, CENTER_UM + half, CENTER_UM - half, CENTER_UM + half)
    regions = build_oprins_regions(stack, window, include_solid=args.full_stack,
                                    context=not args.no_context)

    # Dimensions pulled from the band table itself, not retyped -- if the
    # geometry changes, the caption changes with it.
    bands = {b.name: (z0, z1, b) for z0, z1, b in techmap.z_bounds(stack)}
    notes = []
    if "backside_metal" in bands:
        z0, z1, b = bands["backside_metal"]
        notes.append(f"backside M1 (Cu): {b.synthetic_grid[1]*1000:.0f} nm wide tracks on a "
                     f"{b.synthetic_grid[0]*1000:.0f} nm pitch, {(z1-z0)*1000:.0f} nm thick")
    if "well" in bands:
        z0, z1, b = bands["well"]
        zp0, zp1, _ = bands["p_substrate"]
        if b.synthetic_grid:
            notes.append(f"&#181;TSV (W): {b.synthetic_grid[1]*1000:.0f} &#215; "
                         f"{b.synthetic_width_y_um*1000:.0f} nm posts, "
                         f"{b.synthetic_grid[0]*1000:.0f} nm across the track &#215; "
                         f"{b.synthetic_pitch_y_um*1000:.0f} nm along it, staggered by "
                         f"{b.synthetic_stagger_y_um*1000:.0f} nm; through all "
                         f"{(z1-zp0)*1000:.0f} nm of Si")
    if "buried_power_rail" in bands:
        z0, z1, b = bands["buried_power_rail"]
        notes.append(f"buried power rail (Ru): {b.synthetic_grid[1]*1000:.0f} nm wide on a "
                     f"{b.synthetic_grid[0]*1000:.0f} nm pitch, {(z1-z0)*1000:.0f} nm tall, "
                     f"running perpendicular to the M1 tracks")

    n_shapes = sum(len(r["volumes"]) for r in regions)
    svg = render_regions_geometry_svg(
        regions,
        title="BS-PDN unit cell &#8212; backside metal, &#181;TSVs and buried power rails",
        subtitle=f"{len(regions)} bands &#183; {n_shapes} shapes &#183; "
                 f"{args.si_um*1000:.0f}nm thinned Si &#183; {args.window_um:g}&#181;m window",
        banner="GEOMETRY MATCHED TO [Oprins] iTherm 2022 Fig. 5 "
               "(synthetic grids -- SKY130 has no BS-PDN process)",
        plan_label="top view &#8212; tracks, vias and rails",
        z_exaggeration=args.z_exaggeration, notes=notes)
    (OUT / "fig4_oprins_stack_3d.svg").write_text(svg)
    print(f"wrote {OUT}/fig4_oprins_stack_3d.svg")
    for r in regions:
        print(f"  {r['name']:28s} z {r['z_min_um']:8.4f}..{r['z_max_um']:8.4f}  "
              f"{len(r['volumes']):4d} shapes")

    try:
        import cairosvg
        cairosvg.svg2png(bytestring=svg.encode(),
                         write_to=str(OUT / "fig4_oprins_stack_3d.png"), scale=1.8)
        print(f"wrote {OUT}/fig4_oprins_stack_3d.png")
    except ImportError:
        print("  (cairosvg missing -- SVG only)")


if __name__ == "__main__":
    main()
