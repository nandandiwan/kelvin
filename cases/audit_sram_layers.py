"""Inventory every polygon and text specification in the requested SRAM cell.

Names/purposes come from the pinned local SKY130 GDS CSV. OUTLINE and the
explicitly unresolved UNKNOWN3 name come from the pinned Magic GDS technology
file. Model treatment is a description of the current Kelvin/notebook code,
not an assertion that unmodeled process masks are physically unimportant.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import colorsys
import csv
import hashlib
import json
from pathlib import Path

import gdstk


ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data/sky130/pdk/gds_layers.csv"
MAGIC_PATH = ROOT / "data/sky130/pdk/sky130gds.tech"
NOTEBOOK = ROOT / "read_gds.ipynb"
SOLID_NAMES = {"diff", "poly", "licon1", "li1", "mcon", "met1", "via", "met2"}
OLD_FIGURE_NAMES = {"diff", "li1", "met1"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_names():
    names = {}
    with CSV_PATH.open(newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                spec = tuple(map(int, row["GDS layer:datatype"].split(":")))
            except ValueError:
                continue
            if len(spec) != 2:
                continue
            names[spec] = {
                "name": row["Layer name"], "purpose": row["Purpose"],
                "description": row["Description"], "mapping_source": str(CSV_PATH),
                "mapping_status": "pinned_gds_layer_table",
            }
    magic_text = MAGIC_PATH.read_text()
    for spec, name, purpose, description in (
        ((64, 44), "UNKNOWN3", "unmapped", "Not named in pinned GDS CSV; Magic calls it UNKNOWN3. Kelvin's PWELL alias is not established by these sources."),
        ((236, 0), "OUTLINE", "boundary", "Cell outline; not a material slab."),
    ):
        statement = f"calma {name} {spec[0]} {spec[1]}"
        if statement not in " ".join(magic_text.split()):
            raise ValueError(f"Pinned Magic mapping changed: {statement}")
        names[spec] = {
            "name": name, "purpose": purpose, "description": description,
            "mapping_source": str(MAGIC_PATH),
            "mapping_status": "explicitly_unmapped" if name == "UNKNOWN3" else "pinned_magic_mapping",
        }
    return names


def notebook_colors():
    notebook = json.loads(NOTEBOOK.read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "SKY130_LAYER_COLORS" for target in node.targets):
                return ast.literal_eval(node.value)
    raise ValueError("Notebook SKY130_LAYER_COLORS not found")


def color_for(spec, name, colors):
    if name in colors:
        return colors[name]
    hue = ((spec[0] * 97 + spec[1] * 31) % 360) / 360
    rgb = colorsys.hsv_to_rgb(hue, 0.48, 0.72)
    return "#" + "".join(f"{round(channel * 255):02X}" for channel in rgb)


def treatment(name, purpose):
    if name in SOLID_NAMES and "pin" not in purpose:
        new = "Exact polygon material/source regions in intended notebook thermal construction; complete SRAM construction currently fails contact validation."
        old = "Bounding-box-derived solids in the legacy thermal mesh."
        preview = "Exact polygon prism mesh, visualization-only."
        if name == "diff":
            old += " Split into 110 nm diffusion and 10 nm channel bands."
            new += " 120 nm diffusion split laterally into source/drain and channel-source regions."
        if name == "poly":
            old += " Licon tungsten takes precedence where contact and poly boxes overlap."
        if name == "licon1":
            old += " Tungsten spans poly/lower-contact and upper-contact bands."
            preview = "Exact 2D footprint only; unresolved landing means no guessed 3D contact extrusion."
        return "modeled_physical_mask", old, new, preview
    if name == "nwell" and purpose == "drawing":
        return (
            "well_region", "Not a separate material solid; folded into bulk substrate silicon.",
            "Nwell mask partitions silicon well regions and classifies PMOS channel candidates; wells use bulk-silicon thermal material.",
            "Well-region silicon prism preview (same bulk-silicon material, not an extra layer atop substrate).",
        )
    if name == "dnwell":
        return (
            "well_region_unresolved_depth", "Folded into bulk silicon; no distinct deep-well solid.",
            "No explicit deep-well region; existing bulk-silicon substrate approximation remains.",
            "GDS context only; no independent prism or invented depth.",
        )
    if name == "UNKNOWN3":
        return (
            "unmapped", "Alias PWELL=(64,44) exists in techmap but no stack band extrudes it.",
            "Unmapped; excluded from thermal material construction.",
            "GDS-only; explicitly unmapped, not guessed to be pwell.",
        )
    if "pin" in purpose:
        return "pin_annotation", "Not a material solid.", "Not a material solid.", "GDS-only pin polygon; no extra material slab."
    if purpose in {"mask add", "mask drop"}:
        return (
            "mask_edit", "Not applied by the current simplified solid builder.",
            "Not applied by the current simplified solid builder.",
            "GDS-only mask-generation edit; not an independent stacked material, and not asserted physically irrelevant.",
        )
    if name in {"hvtp", "ncm", "nsdm", "psdm"}:
        return (
            "implant_mask", "No distinct implant slab or calibrated implant-dependent property field.",
            "No distinct implant slab; thermal materials use engineering defaults.",
            "GDS-only implant context; modifies semiconductor processing, not an added film.",
        )
    if name == "npc":
        return (
            "process_cut_mask", "No explicit nitride-cut process geometry.",
            "No explicit nitride-cut process geometry.",
            "GDS-only process-cut context; not an independent material slab.",
        )
    if purpose in {"identifier", "boundary"}:
        return "layout_annotation", "Not a material solid.", "Not a material solid.", "GDS-only layout/context marker."
    return "unmapped", "Not explicitly classified in this audit.", "Not explicitly classified in this audit.", "GDS-only; requires process interpretation."


def markdown(audit):
    lines = [
        "# SRAM bitcell layer inventory", "",
        f"Source: `{audit['source_gds']}`; cell `{audit['cell']}`; SHA-256 `{audit['source_sha256']}`.", "",
        f"The cell has **{audit['polygon_count']} raw polygons on {audit['polygon_spec_count']} layer/datatype pairs**, plus **{audit['label_count']} text labels on {audit['label_spec_count']} layer/texttype pairs**.", "",
        "The earlier figure displayed only DIFF, LI1, and M1 to highlight bounding-box distortion. That was a display selection, not the complete GDS or thermal model. POLY, LICON1, MCON, VIA, and M2 were omitted from that figure. NWELL supplies well-region context.", "",
        "The expanded preview includes the modeled physical masks, while the all-layer GDS panel/inventory retains process edits, implants, pins, identifiers, and the unresolved layer. These are not all independent material slabs. In particular, ignoring a mask-generation edit is a model approximation, not evidence that it is physically unimportant.", "",
        "## Every polygon specification", "",
        "Counts below are raw polygons / union components. New-model status refers to the intended region builder; the complete SRAM thermal stack is still blocked by contact-coverage checks.", "",
        "| GDS | Name / purpose | Count | Area (µm²) | Role |",
        "|---|---|---:|---:|---|",
    ]
    for layer in audit["layers"]:
        spec = "/".join(map(str, layer["spec"]))
        lines.append(f"| {spec} | {layer['name']} / {layer['purpose']} | {layer['raw_polygon_count']} / {layer['union_component_count']} | {layer['union_area_um2']:.6f} | {layer['role'].replace('_', ' ')} |")
    lines += ["", "## Old / new geometry treatment", ""]
    for name in ("diff", "poly", "licon1", "li1", "mcon", "met1", "via", "met2", "nwell", "dnwell", "UNKNOWN3"):
        layer = next(item for item in audit["layers"] if item["name"] == name and item["role"] != "pin_annotation")
        lines.append(f"- **{name}:** old — {layer['old_status']} New — {layer['new_status']} Preview — {layer['preview_status']}")
    lines += ["", "Implants, process cuts, and mask edits are retained as GDS context but are not separately reconstructed thermal solids. Pins, the SRAM identifier, and OUTLINE are annotations, not extra matter.", "", "## Text specifications", "", "| GDS text spec | Name / purpose | Count | Text |", "|---|---|---:|---|"]
    for group in audit["labels"]:
        texts = ", ".join(entry["text"] for entry in group["entries"])
        lines.append(f"| {'/'.join(map(str, group['spec']))} | {group['name']} / {group['purpose']} | {group['count']} | {texts} |")
    lines += ["", "## Mapping sources and limitations", "", f"- Pinned names/purposes: `{CSV_PATH}`.", f"- `236/0` = OUTLINE and `64/44` = UNKNOWN3: `{MAGIC_PATH}`.", "- `64/44` is absent from the pinned CSV and explicitly remains unmapped. Kelvin's existing `PWELL` alias is not used as authoritative evidence.", "- Polygon and text namespaces are reported separately; text labels are not triangulated.", "- No metal 3–5 or via 2–4 polygons exist in this selected bitcell.", ""]
    return "\n".join(lines)


def assert_complete_inventory(polygons, labels, audit):
    """Check identity/multiplicity, not just aggregate counts."""
    source_polygons = Counter(
        (spec, tuple(map(tuple, polygon.points.tolist())))
        for spec, group in polygons.items() for polygon in group
    )
    recorded_polygons = Counter(
        (tuple(layer["spec"]), tuple(map(tuple, ring)))
        for layer in audit["layers"] for ring in layer["raw_rings"]
    )
    assert source_polygons == recorded_polygons, "Polygon inventory changed shape/spec/multiplicity"
    source_labels = Counter(
        (spec, label.text, tuple(label.origin), label.rotation, label.magnification, label.x_reflection)
        for spec, group in labels.items() for label in group
    )
    recorded_labels = Counter(
        (tuple(group["spec"]), label["text"], tuple(label["origin_xy_um"]),
         label["rotation"], label["magnification"], label["x_reflection"])
        for group in audit["labels"] for label in group["entries"]
    )
    assert source_labels == recorded_labels, "Label inventory changed content/spec/multiplicity"
    assert len(audit["layers"]) == len({tuple(layer["spec"]) for layer in audit["layers"]}) == len(polygons)
    assert len(audit["labels"]) == len({tuple(group["spec"]) for group in audit["labels"]}) == len(labels)
    assert sum(layer["raw_polygon_count"] for layer in audit["layers"]) == sum(source_polygons.values())
    assert sum(group["count"] for group in audit["labels"]) == sum(source_labels.values())
    audit["inventory_validation"] = {
        "raw_polygon_shape_spec_and_multiplicity_exact": True,
        "label_text_spec_transform_and_multiplicity_exact": True,
        "one_entry_per_observed_spec": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gds", type=Path, default=ROOT / "kelvin/data/sram22_64x22m4w22.gds")
    parser.add_argument("--cell", default="sram_sp_cell")
    parser.add_argument("--out", type=Path, default=ROOT / "kelvin/out/sram_mesh_comparison")
    args = parser.parse_args()
    names, colors = local_names(), notebook_colors()
    original_unit, original_precision = gdstk.gds_units(args.gds)
    precision_um = original_precision * 1e6
    library = gdstk.read_gds(args.gds, unit=1e-6)
    cell = next(cell for cell in library.cells if cell.name == args.cell)
    polygons, labels = defaultdict(list), defaultdict(list)
    for polygon in cell.get_polygons(apply_repetitions=True, include_paths=True, depth=None):
        polygons[(polygon.layer, polygon.datatype)].append(polygon)
    for label in cell.get_labels(apply_repetitions=True, depth=None):
        labels[(label.layer, label.texttype)].append(label)
    bounds = cell.bounding_box()
    audit = {
        "source_gds": str(args.gds.resolve()), "source_sha256": sha256(args.gds),
        "cell": args.cell, "bbox_um": [*bounds[0], *bounds[1]],
        "original_gds_unit_m": original_unit, "original_gds_precision_m": original_precision,
        "polygon_count": sum(map(len, polygons.values())), "polygon_spec_count": len(polygons),
        "label_count": sum(map(len, labels.values())), "label_spec_count": len(labels),
        "mapping_sources": {str(CSV_PATH): sha256(CSV_PATH), str(MAGIC_PATH): sha256(MAGIC_PATH)},
        "new_full_thermal_mesh_status": "Blocked by licon contact-coverage validation; separate mask meshes are visualization-only.",
        "layers": [], "labels": [],
    }
    for spec, raw in sorted(polygons.items()):
        mapping = names.get(spec, {"name": f"L{spec[0]}_D{spec[1]}", "purpose": "unmapped", "description": "", "mapping_source": None, "mapping_status": "unmapped"})
        union = gdstk.boolean(raw, [], "or", precision=precision_um)
        role, old_status, new_status, preview_status = treatment(mapping["name"], mapping["purpose"])
        audit["layers"].append({
            "spec": list(spec), **mapping, "role": role,
            "raw_polygon_count": len(raw), "union_component_count": len(union),
            "union_area_um2": sum(polygon.area() for polygon in union),
            "rings": [polygon.points.tolist() for polygon in union],
            "raw_rings": [polygon.points.tolist() for polygon in raw],
            "color": color_for(spec, mapping["name"], colors),
            "old_status": old_status, "new_status": new_status, "preview_status": preview_status,
            "included_in_original_three_layer_figure": mapping["name"] in OLD_FIGURE_NAMES and role == "modeled_physical_mask",
        })
    for spec, entries in sorted(labels.items()):
        mapping = names.get(spec, {"name": f"L{spec[0]}_T{spec[1]}", "purpose": "unmapped", "description": "", "mapping_source": None, "mapping_status": "unmapped"})
        audit["labels"].append({
            "spec": list(spec), **mapping, "count": len(entries),
            "entries": [{"text": entry.text, "origin_xy_um": list(entry.origin), "rotation": entry.rotation,
                         "magnification": entry.magnification, "x_reflection": entry.x_reflection} for entry in entries],
        })
    assert_complete_inventory(polygons, labels, audit)
    args.out.mkdir(parents=True, exist_ok=True)
    json_path, md_path = args.out / "sram_all_layer_audit.json", args.out / "sram_all_layer_audit.md"
    json_path.write_text(json.dumps(audit, indent=2) + "\n")
    md_path.write_text(markdown(audit))
    print(f"{audit['polygon_count']} polygons / {audit['polygon_spec_count']} specs; {audit['label_count']} labels / {audit['label_spec_count']} specs")
    print(json_path)
    print(md_path)


if __name__ == "__main__":
    main()
