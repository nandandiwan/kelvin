"""Preserve SKY130 contact cuts while validating their landing relationships.

Contact identity is a whole cut, not its clipped intersection with a landing
mask. SKY130 licon.4 requires lower/upper overlap and licon.17 prohibits a cut
on both poly and diffusion/tap. Periphery enclosure constraints do not apply
unchanged inside the areaid.ce memory core. We retain full-coverage checks
outside that marker and permit/report partial coverage only inside it.

This is a thermal-geometry validation policy, not a foundry DRC or mask/lithography
reconstruction. The raw drawn masks and prescribed vertical stack are retained.
See SRAM_CONTACT_GEOMETRY.md for the evidence and remaining approximations.
"""

import math

import gdstk


SPECS = {
    "diff": (65, 20), "tap": (65, 44), "poly": (66, 20),
    "licon1": (66, 44), "li1": (67, 20), "mcon": (67, 44),
    "met1": (68, 20), "via": (68, 44), "met2": (69, 20),
    "via2": (69, 44), "met3": (70, 20), "via3": (70, 44),
    "met4": (71, 20), "via4": (71, 44), "met5": (72, 20),
    "memory_core": (81, 2),
}


def _components(layout, name):
    return list(layout["polygons_by_spec"].get(SPECS[name], []))


def _polygons(components):
    return [gdstk.Polygon(c["footprint_xy_um"]) for c in components]


def _area(polygons):
    return math.fsum(float(p.area()) for p in polygons)


def _intersection_area(left, right, precision):
    if not left or not right:
        return 0.0
    return _area(gdstk.boolean(left, right, "and", precision=precision))


def _precision(layout):
    precision = float(layout["boolean_grid_um"])
    if not math.isfinite(precision) or precision <= 0:
        raise ValueError("Contact validation requires a positive finite boolean grid")
    return precision, max(precision**2, 1e-12)


def _cut_report(polygon, lower, upper, core, precision, tolerance, *, name, index):
    area = float(polygon.area())
    if area <= tolerance:
        raise ValueError(f"{name} cut {index}: zero/sub-grid contact area")
    lower_area = _intersection_area([polygon], lower, precision)
    upper_area = _intersection_area([polygon], upper, precision)
    core_area = _intersection_area([polygon], core, precision)
    lower_outside = max(0.0, area - lower_area)
    upper_outside = max(0.0, area - upper_area)
    wholly_in_core = area - core_area <= tolerance
    (x0, y0), (x1, y1) = polygon.bounding_box()
    if lower_area <= tolerance or upper_area <= tolerance:
        raise ValueError(
            f"{name} cut {index} at {(x0, y0, x1, y1)} does not overlap both landing layers "
            f"(lower={lower_area:g}, upper={upper_area:g} um^2)"
        )
    partial = max(lower_outside, upper_outside) > tolerance
    if partial and not wholly_in_core:
        raise ValueError(
            f"{name} cut {index}: incomplete landing coverage outside memory core; "
            f"lower={lower_area / area:.6g}, upper={upper_area / area:.6g}"
        )
    return {
        "cut_index": index,
        "bbox_um": [float(v) for v in (x0, y0, x1, y1)],
        "area_um2": area,
        "lower_overlap_um2": lower_area, "upper_overlap_um2": upper_area,
        "lower_coverage": lower_area / area, "upper_coverage": upper_area / area,
        "outside_lower_area_um2": lower_outside, "outside_upper_area_um2": upper_outside,
        "wholly_in_memory_core": wholly_in_core,
        "partial_coverage_accepted_in_core": partial,
        "whole_cut_preserved": True,
    }


def classify_sky130_licon(layout):
    """Return whole ``poly_components``/``diff_components`` and a cut audit.

    The latter class includes taps, which have the same surface landing height.
    A cut with zero lower/upper overlap, a mixed poly/diff landing, or partial
    coverage outside the explicit memory-core marker raises instead of being
    clipped, guessed, or silently dropped.
    """
    precision, tolerance = _precision(layout)
    cuts = _components(layout, "licon1")
    poly = _polygons(_components(layout, "poly"))
    diff = _polygons(_components(layout, "diff") + _components(layout, "tap"))
    upper = _polygons(_components(layout, "li1"))
    core = _polygons(_components(layout, "memory_core"))
    poly_components, diff_components, rows = [], [], []
    for index, component in enumerate(cuts):
        polygon = gdstk.Polygon(component["footprint_xy_um"])
        poly_area = _intersection_area([polygon], poly, precision)
        diff_area = _intersection_area([polygon], diff, precision)
        if poly_area > tolerance and diff_area > tolerance:
            raise ValueError(f"licon1 cut {index}: ambiguous landing on both poly and diffusion/tap")
        if poly_area > tolerance:
            landing, lower, destination = "poly", poly, poly_components
        elif diff_area > tolerance:
            landing, lower, destination = "diff_or_tap", diff, diff_components
        else:
            raise ValueError(f"licon1 cut {index}: does not land on poly/diff/tap")
        row = _cut_report(polygon, lower, upper, core, precision, tolerance,
                          name="licon1", index=index)
        row["landing"] = landing
        destination.append(dict(component))
        rows.append(row)
    report = {
        "policy": "whole-cut exclusive lower landing; core-only partial coverage",
        "cut_count": len(rows), "poly_cut_count": len(poly_components),
        "diff_or_tap_cut_count": len(diff_components), "contacts": rows,
        "whole_cut_area_um2": math.fsum(row["area_um2"] for row in rows),
        "outside_lower_area_um2": math.fsum(row["outside_lower_area_um2"] for row in rows),
        "outside_upper_area_um2": math.fsum(row["outside_upper_area_um2"] for row in rows),
        "core_partial_cut_count": sum(row["partial_coverage_accepted_in_core"] for row in rows),
    }
    return {"poly_components": poly_components, "diff_components": diff_components, "report": report}


def validate_sky130_contacts(layout):
    """Per-cut validation plus aggregate rows compatible with notebook tables.

    Coverage remains a reported measurement, not a disguised pass flag. A
    passing row means every individual cut passed the documented policy.
    """
    precision, tolerance = _precision(layout)
    licon = classify_sky130_licon(layout)
    core = _polygons(_components(layout, "memory_core"))
    rules = [
        ("licon1", None, "li1", "(diff OR tap OR poly) → licon1 → li1"),
        ("mcon", "li1", "met1", "li1 → mcon → met1"),
        ("via", "met1", "met2", "met1 → via → met2"),
        ("via2", "met2", "met3", "met2 → via2 → met3"),
        ("via3", "met3", "met4", "met3 → via3 → met4"),
        ("via4", "met4", "met5", "met4 → via4 → met5"),
    ]
    result = []
    for name, lower_name, upper_name, path in rules:
        cuts = _components(layout, name)
        if not cuts:
            continue
        if name == "licon1":
            rows = licon["report"]["contacts"]
        else:
            lower = _polygons(_components(layout, lower_name))
            upper = _polygons(_components(layout, upper_name))
            rows = [_cut_report(gdstk.Polygon(component["footprint_xy_um"]), lower, upper, core,
                                precision, tolerance, name=name, index=index)
                    for index, component in enumerate(cuts)]
        area = math.fsum(row["area_um2"] for row in rows)
        lower_area = math.fsum(row["lower_overlap_um2"] for row in rows)
        upper_area = math.fsum(row["upper_overlap_um2"] for row in rows)
        result.append({
            "path": path, "connector_layer": SPECS[name][0], "connector_spec": list(SPECS[name]),
            "connector_area_um2": area, "connector_count": len(rows),
            "lower_overlap_um2": lower_area, "upper_overlap_um2": upper_area,
            "lower_coverage": lower_area / area, "upper_coverage": upper_area / area,
            "outside_lower_area_um2": math.fsum(row["outside_lower_area_um2"] for row in rows),
            "outside_upper_area_um2": math.fsum(row["outside_upper_area_um2"] for row in rows),
            "core_partial_cut_count": sum(row["partial_coverage_accepted_in_core"] for row in rows),
            "validation_policy": "whole-cut overlap; exclusive LICON lower category; core-only partial coverage",
            "validated": True, "contacts": rows,
        })
    return result
