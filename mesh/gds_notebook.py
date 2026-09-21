"""Shared exact-polygon GDS geometry, mesh validation, and SVG helpers.

Extracted from read_gds.ipynb so production solvers import Python, not notebook
JSON/AST. The notebook and simulation adapters bind these same functions to an
isolated configuration dictionary. This preserves the notebook's staged cells
without sharing mutable run state between simulations.

Use ``context = create_context(SUBSTRATE_DEPTH_UM=...)``, set
``context["layout"] = context["extract_layout"](gds_path, cell_name)``, then call
``context["build_sky130_regions"]()``. ``install_helpers(globals())`` is the
notebook counterpart: subsequent notebook assignments remain visible to helpers.
Geometry coordinates are micrometres; the Kelvin importer converts them to SI.
"""

from __future__ import annotations

from dataclasses import dataclass

from html import escape

from pathlib import Path

from collections import defaultdict

import colorsys

import csv

import hashlib

import json

import math

import os

import re

import gdstk

from mesh.gds_contacts import classify_sky130_licon, validate_sky130_contacts

import numpy as np

from IPython.display import HTML, SVG, display

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"

SKY130_BUNDLE_DIR = DATA_DIR / "sky130"

SKY130_PDK_DIR = SKY130_BUNDLE_DIR / "pdk"

SKY130_SOURCE_MANIFEST_PATH = SKY130_BUNDLE_DIR / "sources.json"

INPUT_GDS = os.environ.get(
    "GDS_INPUT", "sky130/gds/sky130_fd_sc_hd__inv_1.gds"
)

TOP_CELL_NAME = os.environ.get("GDS_TOP_CELL") or None

MESH_SIZE_UM = float(os.environ.get("GDS_MESH_SIZE_UM", "0.40"))

FINE_MESH_SIZE_UM = float(os.environ.get("GDS_FINE_MESH_SIZE_UM", "0.10"))

REFINEMENT_DISTANCE_UM = float(os.environ.get("GDS_REFINEMENT_DISTANCE_UM", "0.35"))

MIN_TET_QUALITY = float(os.environ.get("GDS_MIN_TET_QUALITY", "0.025"))

SUBSTRATE_DEPTH_UM = float(os.environ.get("GDS_SUBSTRATE_DEPTH_UM", "2.0"))

PLACEHOLDER_LAYER_THICKNESS_UM = 1.0

MAX_FLATTENED_POLYGONS = 250_000

IGNORED_GDS_SPECS = set()

LAYER_MAP_PROFILE = os.environ.get("GDS_LAYER_PROFILE", "auto").strip().lower()

if LAYER_MAP_PROFILE not in {"auto", "none", "scmos", "sky130"}:
    raise ValueError("GDS_LAYER_PROFILE must be one of: auto, none, scmos, sky130")

if not (0 < FINE_MESH_SIZE_UM <= MESH_SIZE_UM):
    raise ValueError("GDS_FINE_MESH_SIZE_UM must be positive and <= GDS_MESH_SIZE_UM")

if SUBSTRATE_DEPTH_UM <= 0.3262:
    raise ValueError("substrate depth must exceed the SKY130 well bottom at 0.3262 um")

PROCESS_STACK_RECORDS = []

SCMOS_LAYER_NAMES = {
    41: "p-well", 42: "n-well", 43: "active", 44: "p+ select",
    45: "n+ select", 46: "poly", 47: "poly contact",
    48: "active contact", 49: "metal 1", 50: "via 1", 51: "metal 2",
}

SCMOS_LAYER_COLORS = {
    41: "#C8A97E", 42: "#E6B566", 43: "#58A65C", 44: "#E88AB8",
    45: "#7CC7E8", 46: "#D85C4A", 47: "#585858", 48: "#7A7A7A",
    49: "#4D78C4", 50: "#2F2F2F", 51: "#B45AC9",
}

SCMOS_SIGNATURE = frozenset(SCMOS_LAYER_NAMES)

SCMOS_CONNECTOR_RULES = (
    (46, 47, 49, "poly → poly contact → metal 1"),
    (43, 48, 49, "active → active contact → metal 1"),
    (49, 50, 51, "metal 1 → via 1 → metal 2"),
)

SKY130_SPECS = {
    "nwell": (64, 20), "diff": (65, 20), "poly": (66, 20),
    "licon1": (66, 44), "li1": (67, 20), "mcon": (67, 44),
    "met1": (68, 20), "via": (68, 44), "met2": (69, 20),
    "via2": (69, 44), "met3": (70, 20), "via3": (70, 44),
    "met4": (71, 20), "via4": (71, 44), "met5": (72, 20),
    "areaid_sc": (81, 4),
}

SKY130_SIGNATURE = frozenset({
    SKY130_SPECS[name]
    for name in ("nwell", "diff", "poly", "licon1", "li1", "mcon", "met1")
})

SKY130_LAYER_COLORS = {
    "nwell": "#8ECAE6", "diff": "#7fc08a", "poly": "#c9a0dc",
    "licon1": "#2b2b2b", "li1": "#d1342a", "mcon": "#2b2b2b",
    "met1": "#e0a840", "via": "#2b2b2b", "met2": "#3d6fd6",
    "via2": "#E11D48", "met3": "#0891B2", "via3": "#FB923C",
    "met4": "#C026D3", "via4": "#FBBF24", "met5": "#059669",
}

SKY130_SOLID_Z_UM = {
    "well": (-0.3262, -0.1200),
    "diff": (-0.1200, 0.0000),
    "poly": (0.0000, 0.1800),
    "li1": (0.6099, 0.7099),
    "mcon": (0.7099, 1.0499),
    "met1": (1.0499, 1.4099),
    "via": (1.4099, 1.6799),
    "met2": (1.6799, 2.0399),
    "via2": (2.0399, 2.4599),
    "met3": (2.4599, 3.3049),
    "via3": (3.3049, 3.6949),
    "met4": (3.6949, 4.5399),
    "via4": (4.5399, 5.0449),
    "met5": (5.0449, 6.3049),
}

SKY130_DIELECTRIC_CEILINGS_UM = (0.6099, 1.0499, 1.6799, 2.4599, 3.6949, 5.0449, 11.5572)

# New options are opt-in; the historical inverter notebook defaults are kept.
CHANNEL_DEPTH_UM = None
ACTIVE_BACKGROUND_MATERIAL = "Si_bulk"
TOP_PASSIVATION_THICKNESS_UM = 0.0
TOP_PASSIVATION_MATERIAL = "SiN"
layer_profile = "sky130"
explicit_user_stack = False
OUTPUT_STEM = "gds_material_regions"
sky130_sources = None


def signed_area(points):
    return 0.5 * sum(
        x0 * y1 - x1 * y0
        for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1])
    )


def polygon_bbox(points):
    xs, ys = zip(*points)
    return [min(xs), min(ys), max(xs), max(ys)]


def clean_ring(points, tolerance_um):
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all():
        raise ValueError("Invalid polygon coordinate array")
    clean = []
    for x, y in array:
        point = (float(x), float(y))
        if not clean or math.dist(point, clean[-1]) > tolerance_um:
            clean.append(point)
    if len(clean) > 1 and math.dist(clean[0], clean[-1]) <= tolerance_um:
        clean.pop()
    if len(clean) < 3 or abs(signed_area(clean)) <= tolerance_um**2:
        raise ValueError("Degenerate polygon after boolean cleanup")
    return clean if signed_area(clean) > 0 else list(reversed(clean))


def repeated_grid_vertex(points, grid_um):
    keys = [(round(x / grid_um), round(y / grid_um)) for x, y in points]
    return len(keys) != len(set(keys))


def canonical_ring(points, digits=12):
    rounded = [(round(x, digits), round(y, digits)) for x, y in points]
    start = min(range(len(rounded)), key=rounded.__getitem__)
    return rounded[start:] + rounded[:start]


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_component(value):
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return cleaned or "unnamed"


def choose_top_cell(library, requested_name=None):
    cells = {cell.name: cell for cell in library.cells}
    if requested_name is not None:
        if requested_name not in cells:
            raise KeyError(f"Top cell {requested_name!r} not found; available cells: {sorted(cells)}")
        return cells[requested_name]
    tops = sorted(library.top_level(), key=lambda cell: cell.name)
    if len(tops) != 1:
        raise ValueError(
            f"Expected exactly one top cell, found {[cell.name for cell in tops]}; set TOP_CELL_NAME"
        )
    return tops[0]


def extract_layout(path: Path, requested_top=None):
    original_unit_m, original_precision_m = gdstk.gds_units(path)
    library = gdstk.read_gds(path, unit=1e-6)
    top = choose_top_cell(library, requested_top)
    raw_polygons = top.get_polygons(apply_repetitions=True, include_paths=True, depth=None)
    if not raw_polygons:
        raise ValueError(f"Top cell {top.name!r} has no area geometry")
    if len(raw_polygons) > MAX_FLATTENED_POLYGONS:
        raise ValueError(
            f"Flattened hierarchy has {len(raw_polygons):,} polygons; apply a layer/ROI filter "
            f"or raise MAX_FLATTENED_POLYGONS ({MAX_FLATTENED_POLYGONS:,})"
        )

    grid_um = original_precision_m * 1e6
    clean_tolerance_um = max(grid_um * 1e-6, 1e-12)
    raw_by_spec = defaultdict(list)
    for polygon in raw_polygons:
        raw_by_spec[(int(polygon.layer), int(polygon.datatype))].append(polygon)

    polygons_by_spec = {}
    mask_stats = {}
    for spec, polygons in sorted(raw_by_spec.items()):
        unioned = gdstk.boolean(
            polygons,
            [],
            "or",
            precision=grid_um,
            layer=spec[0],
            datatype=spec[1],
        )
        simple_polygons = []
        fractured_holes = 0
        for polygon in unioned:
            points = clean_ring(polygon.points, clean_tolerance_um)
            if repeated_grid_vertex(points, grid_um):
                pieces = polygon.fracture(max_points=5, precision=grid_um)
                fractured_holes += 1
            else:
                pieces = [polygon]
            for piece in pieces:
                ring = clean_ring(piece.points, clean_tolerance_um)
                if repeated_grid_vertex(ring, grid_um):
                    raise ValueError(
                        f"L{spec[0]}/D{spec[1]} contains a bridged hole that could not be fractured"
                    )
                simple_polygons.append({
                    "footprint_xy_um": ring,
                    "area_um2": abs(signed_area(ring)),
                    "bbox_um": polygon_bbox(ring),
                })
        if not simple_polygons:
            raise ValueError(f"Boolean union removed all geometry on L{spec[0]}/D{spec[1]}")
        polygons_by_spec[spec] = simple_polygons
        mask_stats[spec] = {
            "raw_polygon_count": len(polygons),
            "union_component_count": len(simple_polygons),
            "raw_area_sum_um2": sum(float(p.area()) for p in polygons),
            "union_area_um2": sum(p["area_um2"] for p in simple_polygons),
            "fractured_hole_count": fractured_holes,
        }

    all_points = [point for polygons in polygons_by_spec.values() for p in polygons for point in p["footprint_xy_um"]]
    geometry_records = [
        [list(spec), canonical_ring(p["footprint_xy_um"])]
        for spec in sorted(polygons_by_spec)
        for p in polygons_by_spec[spec]
    ]
    geometry_records.sort(key=lambda item: json.dumps(item, separators=(",", ":")))
    geometry_hash = hashlib.sha256(
        json.dumps(geometry_records, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "source_path": path,
        "source_sha256": sha256_file(path),
        "library_name": library.name,
        "library_cell_count": len(library.cells),
        "top_cell": top.name,
        "dependency_cells": sorted(cell.name for cell in top.dependencies(True)),
        "direct_reference_count": len(top.references),
        "flattened_label_count": len(top.get_labels(apply_repetitions=True, depth=None)),
        "original_unit_m": original_unit_m,
        "original_precision_m": original_precision_m,
        "boolean_grid_um": grid_um,
        "flattened_polygon_count": len(raw_polygons),
        "union_component_count": sum(len(p) for p in polygons_by_spec.values()),
        "observed_specs": sorted(polygons_by_spec),
        "bbox_um": polygon_bbox(all_points),
        "geometry_hash": geometry_hash,
        "mask_stats": mask_stats,
        "polygons_by_spec": polygons_by_spec,
    }


def discover_and_audit(data_dir: Path):
    layouts, rows = {}, []
    for path in sorted(data_dir.rglob("*")):
        relative = path.relative_to(data_dir).as_posix()
        if path.is_file() and path.suffix.lower() in {".gds", ".gds2"}:
            try:
                layout = extract_layout(path)
                layouts[relative] = layout
                rows.append({
                    "file": relative,
                    "status": "readable",
                    "top": layout["top_cell"],
                    "cells": layout["library_cell_count"],
                    "dependencies": len(layout["dependency_cells"]),
                    "flattened": layout["flattened_polygon_count"],
                    "unioned": layout["union_component_count"],
                    "specs": len(layout["observed_specs"]),
                    "hash": layout["geometry_hash"][:12],
                })
            except Exception as exc:
                rows.append({"file": relative, "status": f"error: {exc}"})
        elif path.is_file() and path.name.lower().endswith((".gds.gpg", ".gds2.gpg")):
            rows.append({"file": relative, "status": "encrypted / skipped"})
    return layouts, rows


def audit_html(rows):
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td><code>{escape(row['file'])}</code></td><td>{escape(row['status'])}</td>"
            f"<td>{escape(str(row.get('top', '-')))}</td><td>{row.get('cells', '-')}</td>"
            f"<td>{row.get('dependencies', '-')}</td><td>{row.get('flattened', '-')}</td>"
            f"<td>{row.get('unioned', '-')}</td><td>{row.get('specs', '-')}</td>"
            f"<td><code>{row.get('hash', '-')}</code></td></tr>"
        )
    return (
        "<table><thead><tr><th>file</th><th>status</th><th>top</th><th>cells</th>"
        "<th>dependencies</th><th>flattened polygons</th><th>union components</th>"
        "<th>specs</th><th>geometry hash</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def verify_sky130_bundle(bundle_dir, source_manifest):
    if source_manifest is None:
        return []
    failures = []
    for artifact in source_manifest["artifacts"]:
        path = bundle_dir / artifact["path"]
        if not path.is_file():
            failures.append(f"missing {artifact['path']}")
        elif sha256_file(path) != artifact["sha256"]:
            failures.append(f"checksum mismatch {artifact['path']}")
    if failures:
        raise RuntimeError("Pinned SKY130 bundle verification failed: " + "; ".join(failures))
    return source_manifest["artifacts"]


def choose_layer_profile(layout, requested_profile):
    observed_specs = set(layout["observed_specs"])
    observed_layer_zero = {layer for layer, datatype in observed_specs if datatype == 0}
    if requested_profile == "none":
        return None
    if requested_profile == "scmos":
        if not SCMOS_SIGNATURE.issubset(observed_layer_zero):
            raise ValueError("Forced SCMOS profile does not match the selected layer signature")
        return "scmos"
    if requested_profile == "sky130":
        if not SKY130_SIGNATURE.issubset(observed_specs):
            raise ValueError("Forced SKY130 profile does not match the selected layer signature")
        return "sky130"
    if SKY130_SIGNATURE.issubset(observed_specs):
        return "sky130"
    return "scmos" if SCMOS_SIGNATURE.issubset(observed_layer_zero) else None


def polygon_objects(layout, spec):
    return [
        gdstk.Polygon(component["footprint_xy_um"])
        for component in layout["polygons_by_spec"].get(spec, [])
    ]


def polygon_union_for_specs(layout, specs):
    polygons = [polygon for spec in specs for polygon in polygon_objects(layout, spec)]
    if not polygons:
        return []
    return gdstk.boolean(polygons, [], "or", precision=layout["boolean_grid_um"])


def polygon_area(polygons):
    return sum(float(polygon.area()) for polygon in polygons)


def overlap_area(polygons_a, polygons_b, precision_um):
    if not polygons_a or not polygons_b:
        return 0.0
    return polygon_area(gdstk.boolean(polygons_a, polygons_b, "and", precision=precision_um))


def scmos_contact_audit(layout):
    rows = []
    for lower, connector, upper, path in SCMOS_CONNECTOR_RULES:
        specs = ((lower, 0), (connector, 0), (upper, 0))
        if any(spec not in layout["polygons_by_spec"] for spec in specs):
            continue
        connector_polygons = polygon_objects(layout, specs[1])
        area = polygon_area(connector_polygons)
        rows.append({
            "path": path,
            "connector_layer": connector,
            "connector_spec": list(specs[1]),
            "connector_area_um2": area,
            "lower_overlap_um2": overlap_area(polygon_objects(layout, specs[0]), connector_polygons, layout["boolean_grid_um"]),
            "upper_overlap_um2": overlap_area(connector_polygons, polygon_objects(layout, specs[2]), layout["boolean_grid_um"]),
        })
    for row in rows:
        area = row["connector_area_um2"]
        row["lower_coverage"] = row["lower_overlap_um2"] / area if area else 0.0
        row["upper_coverage"] = row["upper_overlap_um2"] / area if area else 0.0
    return rows


def sky130_contact_audit(layout):
    """Validate every connector, including justified memory-core overhangs."""
    return validate_sky130_contacts(layout)


def contact_audit_html(profile, rows, ignored_specs):
    if profile not in {"scmos", "sky130"}:
        return "<p>No built-in layer profile was inferred; supply a process map for connector semantics.</p>"
    body = "".join(
        "<tr>"
        f"<td>{escape(row['path'])}</td><td>L{row['connector_spec'][0]}/D{row['connector_spec'][1]}</td>"
        f"<td>{row['connector_area_um2']:.4f}</td>"
        f"<td>{100 * row['lower_coverage']:.1f}%</td>"
        f"<td>{100 * row['upper_coverage']:.1f}%</td></tr>"
        for row in rows
    )
    ignored = ", ".join(f"L{layer}/D{datatype}" for layer, datatype in sorted(ignored_specs)) or "none"
    title = "SKY130" if profile == "sky130" else "legacy SCMOS"
    return (
        '<div style="border-left:4px solid #2a7f62;padding:8px 12px;background:#eef9f4">'
        f'<b>{title} layer profile.</b> Connector coverage is measured from the flattened GDS. '
        f'Non-solid/annotation masks excluded from material volumes: {ignored}.</div>'
        '<table><thead><tr><th>expected connection</th><th>connector</th><th>area (µm²)</th>'
        '<th>covered by lower mask</th><th>covered by upper mask</th></tr></thead>'
        f'<tbody>{body}</tbody></table>'
    )


def sky130_gallery_svg(layout, title, width=330, height=250):
    xmin, ymin, xmax, ymax = layout["bbox_um"]
    xspan, yspan = xmax - xmin, ymax - ymin
    margin = 22
    scale = min((width - 2 * margin) / xspan, (height - 2 * margin - 24) / yspan)
    def point(xy):
        x, y = xy
        return margin + (x - xmin) * scale, 28 + margin + (ymax - y) * scale
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" style="width:100%;height:auto">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="12" y="19" font-family="sans-serif" font-size="13" font-weight="700">{escape(title)}</text>',
    ]
    for name in ("diff", "poly", "licon1", "li1", "mcon", "met1"):
        color = SKY130_LAYER_COLORS[name]
        for component in layout["polygons_by_spec"].get(SKY130_SPECS[name], []):
            points = " ".join(f"{x:.2f},{y:.2f}" for x, y in map(point, component["footprint_xy_um"]))
            out.append(
                f'<polygon points="{points}" fill="{color}" fill-opacity="0.34" '
                f'stroke="{color}" stroke-width="0.65"/>'
            )
    out.append("</svg>")
    return "".join(out)


def cell_power_sidecars(example_keys):
    rows = []
    for key in example_keys:
        stem = Path(key).stem
        spice_path = SKY130_BUNDLE_DIR / "cells" / f"{stem}.spice"
        liberty_path = SKY130_BUNDLE_DIR / "cells" / f"{stem}__tt_025C_1v80.lib.json"
        if not (spice_path.is_file() and liberty_path.is_file()):
            continue
        spice = spice_path.read_text(encoding="utf-8")
        liberty = json.loads(liberty_path.read_text(encoding="utf-8"))
        rows.append({
            "cell": stem,
            "transistor_count": sum(line.lstrip().startswith("X") for line in spice.splitlines()),
            "area_um2": float(liberty["area"]),
            "leakage_nw": float(liberty["cell_leakage_power"]),
            "spice": spice_path.relative_to(ROOT).as_posix(),
            "liberty": liberty_path.relative_to(ROOT).as_posix(),
        })
    return rows


@dataclass(frozen=True)
class LayerSpec:
    layer: int
    datatype: int
    name: str
    material: str
    z_min_um: float
    z_max_um: float
    physical_tag: int
    color: str
    placeholder: bool = False

    def __post_init__(self):
        if self.z_max_um <= self.z_min_um:
            raise ValueError(f"{self.name}: z_max_um must be greater than z_min_um")
        if self.physical_tag <= 0:
            raise ValueError(f"{self.name}: physical_tag must be positive")


def distinct_color(index):
    hue = (0.61 * index + 0.03) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.62, 0.86)
    return f"#{round(red * 255):02X}{round(green * 255):02X}{round(blue * 255):02X}"


def normalized_components(polygons, precision_um):
    tolerance = max(precision_um * 1e-6, 1e-12)
    components = []
    for polygon in polygons:
        points = clean_ring(polygon.points, tolerance)
        pieces = (
            polygon.fracture(max_points=5, precision=precision_um)
            if repeated_grid_vertex(points, precision_um) else [polygon]
        )
        for piece in pieces:
            ring = clean_ring(piece.points, tolerance)
            if repeated_grid_vertex(ring, precision_um):
                raise ValueError("Boolean result contains an unfractured bridged hole")
            components.append({
                "footprint_xy_um": ring,
                "area_um2": abs(signed_area(ring)),
                "bbox_um": polygon_bbox(ring),
            })
    return components


def component_polygons(components):
    return [gdstk.Polygon(component["footprint_xy_um"]) for component in components]


def boolean_components(left, right, operation, precision_um, allow_empty=False):
    left_polygons = component_polygons(left) if left and isinstance(left[0], dict) else list(left)
    right_polygons = component_polygons(right) if right and isinstance(right[0], dict) else list(right)
    if not left_polygons:
        if allow_empty:
            return []
        raise ValueError(f"Boolean {operation} has an empty left operand")
    if not right_polygons and operation == "not":
        result = left_polygons
    elif not right_polygons and operation == "and":
        result = []
    else:
        result = gdstk.boolean(left_polygons, right_polygons, operation, precision=precision_um)
    components = normalized_components(result, precision_um)
    if not components and not allow_empty:
        raise ValueError(f"Boolean {operation} produced no area")
    return components


def build_process_stack(observed_specs, records, placeholder_thickness_um, layer_names=None, layer_colors=None):
    layer_names = layer_names or {}
    layer_colors = layer_colors or {}
    if not records:
        return [
            LayerSpec(
                layer=layer,
                datatype=datatype,
                name=(f"{layer_names[layer]} (L{layer}/D{datatype})" if layer in layer_names else f"L{layer}_D{datatype}"),
                material=(f"UNSPECIFIED_{layer_names[layer].upper().replace(' ', '_')}" if layer in layer_names else f"UNSPECIFIED_L{layer}_D{datatype}"),
                z_min_um=index * placeholder_thickness_um,
                z_max_um=(index + 1) * placeholder_thickness_um,
                physical_tag=index + 1,
                color=layer_colors.get(layer, distinct_color(index)),
                placeholder=True,
            )
            for index, (layer, datatype) in enumerate(observed_specs)
        ]
    stack = []
    for index, record in enumerate(records):
        layer, datatype = int(record["layer"]), int(record.get("datatype", 0))
        stack.append(LayerSpec(
            layer=layer,
            datatype=datatype,
            name=record.get("name", f"L{layer}_D{datatype}"),
            material=record["material"],
            z_min_um=float(record["z_min_um"]),
            z_max_um=float(record["z_max_um"]),
            physical_tag=int(record.get("physical_tag", index + 1)),
            color=record.get("color", distinct_color(index)),
            placeholder=False,
        ))
    mapped = {(spec.layer, spec.datatype) for spec in stack}
    missing = sorted(set(observed_specs) - mapped)
    extra = sorted(mapped - set(observed_specs))
    if missing or extra:
        raise ValueError(f"Process-stack coverage mismatch; missing={missing}, unused={extra}")
    return stack


def make_region(tag, name, material, z_min_um, z_max_um, color, components, *, role, source_specs=(), placeholder=False, provenance="user"):
    if not components:
        raise ValueError(f"{name}: no footprint components")
    first_spec = tuple(source_specs[0]) if len(source_specs) == 1 else (-tag, 0)
    return {
        "physical_tag": int(tag),
        "name": name,
        "material": material,
        "gds_layer": int(first_spec[0]),
        "datatype": int(first_spec[1]),
        "source_gds_specs": [list(map(int, spec)) for spec in source_specs],
        "z_min_um": float(z_min_um),
        "z_max_um": float(z_max_um),
        "z_min_m": float(z_min_um) * 1e-6,
        "z_max_m": float(z_max_um) * 1e-6,
        "color": color,
        "role": role,
        "thermal_source_candidate": role == "source",
        "placeholder": bool(placeholder),
        "provenance": provenance,
        "source_polygon_count": sum(
            layout["mask_stats"].get(tuple(spec), {}).get("raw_polygon_count", 0)
            for spec in source_specs
        ),
        "volumes": [dict(component) for component in components],
    }


def generic_regions(active_specs):
    stack = build_process_stack(
        active_specs,
        PROCESS_STACK_RECORDS,
        PLACEHOLDER_LAYER_THICKNESS_UM,
        SCMOS_LAYER_NAMES if layer_profile == "scmos" else None,
        SCMOS_LAYER_COLORS if layer_profile == "scmos" else None,
    )
    return [
        make_region(
            spec.physical_tag, spec.name, spec.material,
            spec.z_min_um, spec.z_max_um, spec.color,
            layout["polygons_by_spec"][(spec.layer, spec.datatype)],
            role="placeholder" if spec.placeholder else "conductor",
            source_specs=[(spec.layer, spec.datatype)],
            placeholder=spec.placeholder,
            provenance="placeholder" if spec.placeholder else "PROCESS_STACK_RECORDS",
        )
        for spec in stack
    ]


def dielectric_name(z_mid):
    if z_mid < 0.6099:
        return "PSG_FEOL"
    if z_mid < 1.0499:
        return "LINT"
    if z_mid < 1.6799:
        return "NILD2"
    if z_mid < 2.4599:
        return "NILD3"
    if z_mid < 3.6949:
        return "NILD4"
    if z_mid < 5.0449:
        return "NILD5"
    return "NILD6_TOPOX"


def build_sky130_regions():
    precision = layout["boolean_grid_um"]
    active_bottom = SKY130_SOLID_Z_UM["diff"][0]
    channel_depth_um = -active_bottom if CHANNEL_DEPTH_UM is None else float(CHANNEL_DEPTH_UM)
    if not math.isfinite(channel_depth_um) or not 0 < channel_depth_um <= -active_bottom:
        raise ValueError("CHANNEL_DEPTH_UM must be positive and no greater than the active depth")
    if not math.isfinite(SUBSTRATE_DEPTH_UM) or SUBSTRATE_DEPTH_UM <= -SKY130_SOLID_Z_UM["well"][0]:
        raise ValueError("SUBSTRATE_DEPTH_UM must extend below the well bottom")
    passivation_um = float(TOP_PASSIVATION_THICKNESS_UM)
    if not math.isfinite(passivation_um) or passivation_um < 0:
        raise ValueError("TOP_PASSIVATION_THICKNESS_UM must be finite and nonnegative")
    channel_bottom = -channel_depth_um
    connector_audit = validate_sky130_contacts(layout)
    xmin, ymin, xmax, ymax = layout["bbox_um"]
    domain = normalized_components([gdstk.rectangle((xmin, ymin), (xmax, ymax))], precision)

    def mask(name):
        return [dict(component) for component in layout["polygons_by_spec"].get(SKY130_SPECS[name], [])]

    poly, diff, nwell = mask("poly"), mask("diff"), mask("nwell")
    channels = boolean_components(poly, diff, "and", precision)
    pmos_channels = boolean_components(channels, nwell, "and", precision, allow_empty=True)
    nmos_channels = boolean_components(channels, nwell, "not", precision, allow_empty=True)
    channel_union = boolean_components(
        component_polygons(nmos_channels + pmos_channels), [], "or", precision
    )
    source_drains = boolean_components(diff, channel_union, "not", precision, allow_empty=True)
    active_background = boolean_components(domain, diff, "not", precision)
    nwell_silicon = boolean_components(domain, nwell, "and", precision, allow_empty=True)
    pwell_silicon = boolean_components(domain, nwell, "not", precision, allow_empty=True)

    conductor_defs = []
    def conductor(name, components, material, z_pair=None, source_specs=None, color=None):
        if not components:
            return
        z0, z1 = z_pair or SKY130_SOLID_Z_UM[name]
        conductor_defs.append({
            "name": name,
            "components": components,
            "material": material,
            "z_min_um": z0,
            "z_max_um": z1,
            "source_specs": source_specs or [SKY130_SPECS[name]],
            "color": color or SKY130_LAYER_COLORS.get(name, "#59636E"),
        })

    conductor("poly", poly, "PolySi")

    licon_classification = classify_sky130_licon(layout)
    licon_on_poly = licon_classification["poly_components"]
    licon_on_diff = licon_classification["diff_components"]
    conductor(
        "licon_to_diff", licon_on_diff, "W", (0.0, SKY130_SOLID_Z_UM["li1"][0]),
        [SKY130_SPECS["licon1"], SKY130_SPECS["diff"]], "#495057",
    )
    conductor(
        "licon_to_poly", licon_on_poly, "W",
        (SKY130_SOLID_Z_UM["poly"][1], SKY130_SOLID_Z_UM["li1"][0]),
        [SKY130_SPECS["licon1"], SKY130_SPECS["poly"]], "#6C757D",
    )

    conductor("li1", mask("li1"), "TiN")
    conductor("mcon", mask("mcon"), "W")
    for name in ("met1", "via", "met2", "via2", "met3", "via3", "met4", "via4", "met5"):
        material = "W" if name.startswith("via") else "Al"
        conductor(name, mask(name), material)

    highest_top = max(item["z_max_um"] for item in conductor_defs)
    dielectric_top_z = highest_top if passivation_um else next(
        (ceiling for ceiling in SKY130_DIELECTRIC_CEILINGS_UM if ceiling > highest_top + precision),
        highest_top,
    )
    top_z = dielectric_top_z + passivation_um
    z_breaks = {0.0, dielectric_top_z}
    for item in conductor_defs:
        if 0 <= item["z_min_um"] <= dielectric_top_z:
            z_breaks.add(item["z_min_um"])
        if 0 <= item["z_max_um"] <= dielectric_top_z:
            z_breaks.add(item["z_max_um"])
    z_breaks = sorted(z_breaks)

    dielectric_defs = []
    slab_conservation = []
    domain_area = sum(component["area_um2"] for component in domain)
    for index, (z0, z1) in enumerate(zip(z_breaks, z_breaks[1:])):
        z_mid = 0.5 * (z0 + z1)
        occupants = [
            component
            for item in conductor_defs
            if item["z_min_um"] <= z_mid < item["z_max_um"]
            for component in item["components"]
        ]
        occupant_union = (
            boolean_components(component_polygons(occupants), [], "or", precision)
            if occupants else []
        )
        dielectric = boolean_components(domain, occupant_union, "not", precision)
        occupied_area = sum(component["area_um2"] for component in occupant_union)
        dielectric_area = sum(component["area_um2"] for component in dielectric)
        if not math.isclose(occupied_area + dielectric_area, domain_area, rel_tol=1e-9, abs_tol=precision**2):
            raise AssertionError("SKY130 conductor/dielectric slab area is not conservative")
        dielectric_defs.append({
            "name": f"dielectric_{index:02d}_{dielectric_name(z_mid)}",
            "components": dielectric,
            "z_min_um": z0,
            "z_max_um": z1,
            "color": "#DCEAF2" if index % 2 == 0 else "#CFE0EA",
        })
        slab_conservation.append({
            "z_min_um": z0, "z_max_um": z1,
            "domain_area_um2": domain_area,
            "conductor_area_um2": occupied_area,
            "dielectric_area_um2": dielectric_area,
            "relative_error": abs(domain_area - occupied_area - dielectric_area) / domain_area,
        })

    result = []
    def add(name, material, z0, z1, color, components, role, source_specs=(), provenance="thermal assumption"):
        tag = len(result) + 1
        result.append(make_region(
            tag, name, material, z0, z1, color, components,
            role=role, source_specs=source_specs, placeholder=False, provenance=provenance,
        ))

    well_bottom, active_bottom = SKY130_SOLID_Z_UM["well"][0], SKY130_SOLID_Z_UM["diff"][0]
    add("silicon_substrate", "Si_bulk", -SUBSTRATE_DEPTH_UM, well_bottom, "#8D999F", domain, "substrate")
    if pwell_silicon:
        add("pwell_silicon", "Si_bulk", well_bottom, active_bottom, "#A7B3B9", pwell_silicon, "substrate", provenance="SKY130 nwell complement; thermal material assumption")
    if nwell_silicon:
        add("nwell_silicon", "Si_bulk", well_bottom, active_bottom, "#8ECAE6", nwell_silicon, "substrate", [SKY130_SPECS["nwell"]], "SKY130 nwell mask; thermal material assumption")
    background_name = "active_background_silicon" if ACTIVE_BACKGROUND_MATERIAL == "Si_bulk" else "active_background"
    background_role = "substrate" if ACTIVE_BACKGROUND_MATERIAL == "Si_bulk" else "dielectric"
    add(background_name, ACTIVE_BACKGROUND_MATERIAL, active_bottom, 0.0, "#B0BEC5", active_background, background_role, provenance="SKY130 diffusion complement; explicit thermal material assumption")
    if source_drains:
        add("source_drain_silicon", "Si_SD_doped", active_bottom, 0.0, "#6ABF69", source_drains, "semiconductor", [SKY130_SPECS["diff"]], "SKY130 diffusion mask; thermal material assumption")
    if channel_bottom > active_bottom:
        add("channel_underlayer_silicon", "Si_SD_doped", active_bottom, channel_bottom, "#6ABF69", channel_union, "semiconductor", [SKY130_SPECS["poly"], SKY130_SPECS["diff"]], "Silicon below the explicitly configured heat-deposition depth")
    # Preserve one physical source tag per connected channel footprint.
    # Compact-model devices may still group several fingers, but this is
    # strictly more useful than coalescing every NMOS/PMOS in the cell.
    for index, component in enumerate(nmos_channels, start=1):
        add(f"nmos_channel_source_{index:03d}", "Si_channel", channel_bottom, 0.0, "#F94144", [component], "source", [SKY130_SPECS["poly"], SKY130_SPECS["diff"]])
    for index, component in enumerate(pmos_channels, start=1):
        add(f"pmos_channel_source_{index:03d}", "Si_channel", channel_bottom, 0.0, "#F3722C", [component], "source", [SKY130_SPECS["poly"], SKY130_SPECS["diff"], SKY130_SPECS["nwell"]])
    for item in dielectric_defs:
        add(item["name"], "SiO2", item["z_min_um"], item["z_max_um"], item["color"], item["components"], "dielectric", provenance="SKY130 stack geometry; thermal material assumption")
    if passivation_um:
        add("passivation", TOP_PASSIVATION_MATERIAL, dielectric_top_z, top_z, "#DCEAF2", domain, "dielectric", provenance="Explicit nominal thermal passivation; not inferred from GDS")
    for item in conductor_defs:
        add(item["name"], item["material"], item["z_min_um"], item["z_max_um"], item["color"], item["components"], "conductor", item["source_specs"], "SKY130 GDS + public stack; thermal material assumption")

    model = {
        "domain_bbox_um": [xmin, ymin, xmax, ymax],
        "domain_area_um2": domain_area,
        "substrate_depth_um": SUBSTRATE_DEPTH_UM,
        "well_z_um": list(SKY130_SOLID_Z_UM["well"]),
        "diffusion_z_um": list(SKY130_SOLID_Z_UM["diff"]),
        "channel_source_z_um": [channel_bottom, 0.0],
        "active_background_material": ACTIVE_BACKGROUND_MATERIAL,
        "dielectric_top_um": dielectric_top_z,
        "domain_top_um": top_z,
        "passivation_thickness_um": passivation_um,
        "passivation_material": TOP_PASSIVATION_MATERIAL if passivation_um else None,
        "licon_contact_classification": licon_classification["report"],
        "connector_overlap_audit": connector_audit,
        "slab_conservation": slab_conservation,
        "thermal_material_assumptions": {
            "wells/background": "Si_bulk", "source/drain": "Si_SD_doped",
            "poly": "PolySi", "licon/mcon/vias": "W", "li1": "TiN",
            "routing_metals": "Al", "dielectrics": "SiO2",
        },
        "z_coordinate_source": "open_pdks sky130.tech height entries, shifted by -0.3262 um",
        "known_nominal_difference": "Magic height table Metal-1=0.36 um; TLEF Metal-1=0.35 um",
    }
    used_specs = {
        tuple(spec)
        for region in result
        for spec in region["source_gds_specs"]
        if spec[0] >= 0
    }
    return result, model, used_specs


def intervals_overlap(a0, a1, b0, b1, eps=1e-12):
    return min(a1, b1) - max(a0, b0) > eps


def assert_no_material_overlaps(regions, precision_um):
    for index, region_a in enumerate(regions):
        for region_b in regions[index + 1:]:
            if not intervals_overlap(
                region_a["z_min_um"], region_a["z_max_um"],
                region_b["z_min_um"], region_b["z_max_um"],
            ):
                continue
            polygons_a = component_polygons(region_a["volumes"])
            polygons_b = component_polygons(region_b["volumes"])
            intersection = gdstk.boolean(polygons_a, polygons_b, "and", precision=precision_um)
            overlap_area = sum(float(p.area()) for p in intersection)
            if overlap_area > precision_um**2:
                raise ValueError(
                    f"3D material overlap between {region_a['name']} and {region_b['name']}: "
                    f"XY intersection={overlap_area:g} um^2"
                )


def region_contract_sha256(regions):
    """Fingerprint the complete tag/geometry/material/source meshing input."""
    return hashlib.sha256(json.dumps(
        regions, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def region_layout_identity(layout):
    """Provenance retained when regions are constructed in a staged notebook."""
    return json.loads(json.dumps({key: layout[key] for key in
                                 ("source_sha256", "geometry_hash", "top_cell", "bbox_um")}))


def build_region_manifest(context):
    """Snapshot current regions and metadata, never an earlier notebook JSON.

    Changing a material in ``regions`` is supported. Changing the selected GDS,
    top cell or extracted layout requires rebuilding regions first; otherwise a
    partial cell rerun would silently describe geometry from a different input.
    """
    layout = context["layout"]
    gds_path = Path(context.get("GDS_PATH", layout["source_path"])).resolve()
    if sha256_file(gds_path) != layout["source_sha256"]:
        raise ValueError("Selected GDS changed; rerun layout extraction and region construction")
    selected_top = context.get("TOP_CELL_NAME")
    if selected_top is not None and selected_top != layout["top_cell"]:
        raise ValueError("Selected top cell changed; rerun layout extraction and region construction")
    if context.get("REGION_LAYOUT_IDENTITY") != region_layout_identity(layout):
        raise ValueError("Layout changed or region provenance is missing; rerun region construction")
    regions = context["regions"]
    if not regions:
        raise ValueError("No regions to mesh")
    if len({r["physical_tag"] for r in regions}) != len(regions):
        raise ValueError("Physical volume tags must be unique")
    if len({r["name"] for r in regions}) != len(regions):
        raise ValueError("Physical region names must be unique")
    assert_no_material_overlaps(regions, layout["boolean_grid_um"])
    placeholder = all(r["placeholder"] for r in regions)
    sky130 = context["layer_profile"] == "sky130" and not context["explicit_user_stack"]
    if sky130:
        stack_kind = "sky130-open-pdks-magic-working-geometry"
        warnings = [
            "Substrate depth is a finite-domain modeling assumption, not a GDS/PDK wafer dimension.",
            "Thermal material mappings are engineering defaults, not foundry-qualified properties.",
            "Magic and TLEF nominal thicknesses differ; process_model records the working geometry.",
            "GDS and compact/Liberty models do not supply workload activity or spatial heat magnitudes.",
        ]
    elif placeholder:
        stack_kind = "placeholder"
        warnings = [
            "Placeholder materials/z-order are visualization-only; provide PROCESS_STACK_RECORDS for physics.",
            "Mask overlap alone does not define connector spans, substrate, or dielectric fill.",
        ]
    else:
        stack_kind, warnings = "explicit-user-stack", []
    used_specs = {tuple(spec) for spec in context["used_gds_specs"]}
    sky130_sources = context.get("sky130_sources")
    manifest = {
        "schema": "gds-material-regions-v2", "source_gds": str(gds_path),
        "source_sha256": layout["source_sha256"], "geometry_sha256": layout["geometry_hash"],
        "library": layout["library_name"], "top_cell": layout["top_cell"],
        "gdstk_version": gdstk.__version__, "original_gds_unit_m": layout["original_unit_m"],
        "original_gds_precision_m": layout["original_precision_m"],
        "geometry_length_unit": "um", "geometry_length_scale_to_m": 1e-6,
        "boolean_grid_um": layout["boolean_grid_um"], "layer_map_profile": context["layer_profile"],
        "process_stack_kind": stack_kind, "process_model": context["process_model"],
        "ignored_gds_specs": [list(spec) for spec in sorted(set(layout["observed_specs"]) - used_specs)],
        "used_gds_specs": [list(spec) for spec in sorted(used_specs)],
        "connector_overlap_audit": context.get("contact_audit", []),
        "thermal_source_candidate_tags": [r["physical_tag"] for r in regions
                                          if r["thermal_source_candidate"]],
        "pdk_source_manifest": str(context["SKY130_SOURCE_MANIFEST_PATH"]) if sky130_sources else None,
        "pdk_revisions": sky130_sources.get("repositories") if sky130_sources else None,
        "matched_power_sidecars": context.get("power_sidecars", []),
        "warnings": warnings, "regions": regions,
    }
    # JSON round-trip gives the mesher and writer one independent immutable-by-
    # convention snapshot, rather than aliases to mutable notebook globals.
    manifest = json.loads(json.dumps(manifest, allow_nan=False))
    if context.get("SRAM_PIPELINE", False):
        from mesh.sram import prepare_sram_manifest
        manifest = prepare_sram_manifest(manifest)
    return manifest


def build_notebook_mesh_bundle(context, out_dir=None, *, renders=False):
    """Build/validate/publish mesh and current manifest in one operation.

    All expensive work is staged first. A failed mesh leaves a previous valid
    bundle untouched. The JSON is replaced last, so interrupted publication is
    rejected by mesh hashes rather than accepted as a mixed old/new bundle.
    """
    from tempfile import TemporaryDirectory

    manifest = build_region_manifest(context)
    out_dir = Path(out_dir or context["OUTPUT_DIR"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_component(context["OUTPUT_STEM"])
    paths = {
        "manifest": out_dir / f"{stem}_material_regions.json",
        "mesh": out_dir / f"{stem}_material_regions.msh",
        "quality": out_dir / f"{stem}_material_regions_quality.msh",
        "brep": out_dir / f"{stem}_material_regions.brep",
        "stats": out_dir / f"{stem}_mesh_stats.json",
        "svg": out_dir / f"{stem}_geometry.svg",
    }
    with TemporaryDirectory(prefix=f".{stem}-build-", dir=out_dir) as staging_dir:
        staged = {key: Path(staging_dir) / path.name for key, path in paths.items()}
        stats = context["build_and_validate_mesh"](
            manifest["regions"], staged["mesh"], staged["quality"], staged["brep"],
            context["MESH_SIZE_UM"], context["MIN_TET_QUALITY"],
            context["FINE_MESH_SIZE_UM"], context["REFINEMENT_DISTANCE_UM"],
        )
        manifest["mesh_stats_file"] = str(paths["stats"])
        if context.get("SRAM_PIPELINE", False):
            from mesh.sram import validated_sram_manifest
            manifest = validated_sram_manifest(
                manifest, staged["mesh"], mesh_stats=stats, mesh_file=paths["mesh"],
            )
        else:
            if stats.get("mesh_regions_sha256") != region_contract_sha256(manifest["regions"]):
                raise ValueError("Mesh validation does not match the current region/material contract")
            if stats.get("mesh_sha256") != sha256_file(staged["mesh"]):
                raise ValueError("Mesh changed after validation")
            manifest["mesh_file"] = str(paths["mesh"])
            manifest["mesh_sha256"] = stats["mesh_sha256"]
            manifest["mesh_regions_sha256"] = stats["mesh_regions_sha256"]
        staged["stats"].write_text(json.dumps(stats, indent=2) + "\n")
        staged["manifest"].write_text(json.dumps(manifest, indent=2) + "\n")
        artifacts = ["mesh", "quality", "brep", "stats"]
        if renders:
            staged["svg"].write_text(context["make_svg"](
                manifest["regions"], context["layout"], Path(manifest["source_gds"]).name,
                all(r["placeholder"] for r in manifest["regions"]),
            ))
            artifacts.append("svg")
        for key in artifacts + ["manifest"]:
            staged[key].replace(paths[key])
    return manifest, stats, paths


def tetrahedron_volume(a, b, c, d):
    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    ad = (d[0] - a[0], d[1] - a[1], d[2] - a[2])
    cross = (
        ac[1] * ad[2] - ac[2] * ad[1],
        ac[2] * ad[0] - ac[0] * ad[2],
        ac[0] * ad[1] - ac[1] * ad[0],
    )
    return abs(ab[0] * cross[0] + ab[1] * cross[1] + ab[2] * cross[2]) / 6.0


def add_polygon_prism(gmsh, points, z_min, z_max, mesh_size):
    point_tags = [gmsh.model.occ.addPoint(x, y, z_min) for x, y in points]
    line_tags = [
        gmsh.model.occ.addLine(a, b)
        for a, b in zip(point_tags, point_tags[1:] + point_tags[:1])
    ]
    loop = gmsh.model.occ.addCurveLoop(line_tags)
    surface = gmsh.model.occ.addPlaneSurface([loop])
    extrusion = gmsh.model.occ.extrude([(2, surface)], 0, 0, z_max - z_min)
    volumes = [tag for dim, tag in extrusion if dim == 3]
    if len(volumes) != 1:
        raise RuntimeError(f"Expected one extruded volume, got {volumes}")
    return volumes[0]


def build_and_validate_mesh(
    regions, msh_path, quality_msh_path, brep_path, mesh_size_um, min_tet_quality,
    fine_mesh_size_um=None, refinement_distance_um=None,
):
    import gmsh

    # Record exactly the geometry, tag and material semantics actually meshed.
    regions = json.loads(json.dumps(regions, allow_nan=False))
    input_regions_sha256 = region_contract_sha256(regions)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMin", min(mesh_size_um / 2, fine_mesh_size_um or mesh_size_um / 2))
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_um)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.RandomSeed", 1)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.model.add(OUTPUT_STEM)

        input_dimtags = []
        input_region_tags = []
        for region in regions:
            for volume in region["volumes"]:
                entity = add_polygon_prism(
                    gmsh,
                    volume["footprint_xy_um"],
                    region["z_min_um"],
                    region["z_max_um"],
                    mesh_size_um,
                )
                input_dimtags.append((3, entity))
                input_region_tags.append(region["physical_tag"])

        if len(input_dimtags) > 1:
            output_dimtags, output_map = gmsh.model.occ.fragment(
                input_dimtags[:1], input_dimtags[1:], removeObject=True, removeTool=True
            )
        else:
            output_dimtags, output_map = input_dimtags, [[input_dimtags[0]]]
        gmsh.model.occ.synchronize()
        if len(output_map) != len(input_dimtags):
            raise AssertionError("Gmsh fragment map does not cover every input prism")

        region_entities = defaultdict(set)
        entity_owner = {}
        for region_tag, descendants in zip(input_region_tags, output_map):
            for dim, entity in descendants:
                if dim != 3:
                    continue
                previous = entity_owner.get(entity)
                if previous is not None and previous != region_tag:
                    raise ValueError(
                        f"Fragmented volume {entity} belongs to material tags {previous} and {region_tag}"
                    )
                entity_owner[entity] = region_tag
                region_entities[region_tag].add(entity)

        model_volume_entities = {tag for dim, tag in gmsh.model.getEntities(3)}
        if set(entity_owner) != model_volume_entities:
            raise AssertionError("Fragment mapping did not assign every output volume")

        boundary_candidates = {}
        boundary_surfaces = {}
        surface_owners = defaultdict(set)
        tolerance = 1e-5
        expected_surface_tags = set()
        for region in regions:
            tag = region["physical_tag"]
            entities = sorted(region_entities[tag])
            if not entities:
                raise AssertionError(f"Region {region['name']} has no post-fragment volumes")
            gmsh.model.addPhysicalGroup(3, entities, tag=tag)
            gmsh.model.setPhysicalName(3, tag, region["name"])

            boundary = gmsh.model.getBoundary(
                [(3, entity) for entity in entities], combined=True, oriented=False, recursive=False
            )
            bottom, top = set(), set()
            for dim, surface in boundary:
                if dim != 2:
                    continue
                surface_owners[surface].add(tag)
                _, _, z0, _, _, z1 = gmsh.model.getBoundingBox(2, surface)
                z_mid = 0.5 * (z0 + z1)
                if z1 - z0 <= 2 * tolerance and abs(z_mid - region["z_min_um"]) <= tolerance:
                    bottom.add(surface)
                if z1 - z0 <= 2 * tolerance and abs(z_mid - region["z_max_um"]) <= tolerance:
                    top.add(surface)
            if not bottom or not top:
                raise AssertionError(f"Could not identify top/bottom faces for {region['name']}")
            boundary_candidates[tag] = {"bottom": bottom, "top": top}

        # Facet tags must be single-valued for DOLFINx. Tag exposed top/bottom
        # faces per region, and give each shared interface its own unique tag.
        for region in regions:
            tag = region["physical_tag"]
            external_bottom = sorted(
                surface for surface in boundary_candidates[tag]["bottom"]
                if len(surface_owners[surface]) == 1
            )
            external_top = sorted(
                surface for surface in boundary_candidates[tag]["top"]
                if len(surface_owners[surface]) == 1
            )
            if external_bottom:
                bottom_tag = 10000 + tag
                gmsh.model.addPhysicalGroup(2, external_bottom, tag=bottom_tag)
                gmsh.model.setPhysicalName(2, bottom_tag, f"{region['name']}_external_bottom")
                expected_surface_tags.add(bottom_tag)
            if external_top:
                top_tag = 20000 + tag
                gmsh.model.addPhysicalGroup(2, external_top, tag=top_tag)
                gmsh.model.setPhysicalName(2, top_tag, f"{region['name']}_external_top")
                expected_surface_tags.add(top_tag)
            boundary_surfaces[tag] = {"bottom": external_bottom, "top": external_top}

        interfaces_by_owner = defaultdict(list)
        for surface, owners in surface_owners.items():
            if len(owners) > 1:
                interfaces_by_owner[tuple(sorted(owners))].append(surface)
        interface_groups = []
        region_name_by_tag = {region["physical_tag"]: region["name"] for region in regions}
        for index, (owners, surfaces) in enumerate(sorted(interfaces_by_owner.items()), start=1):
            interface_tag = 30000 + index
            gmsh.model.addPhysicalGroup(2, sorted(surfaces), tag=interface_tag)
            interface_name = "interface__" + "__".join(region_name_by_tag[tag] for tag in owners)
            gmsh.model.setPhysicalName(2, interface_tag, interface_name)
            expected_surface_tags.add(interface_tag)
            interface_groups.append({
                "physical_tag": interface_tag,
                "name": interface_name,
                "region_tags": list(owners),
                "surface_count": len(surfaces),
            })

        # Validate connectivity at the post-fragment OCC entity level.  This is
        # independent of region labels and catches genuinely floating solids.
        parent = {entity: entity for entity in model_volume_entities}
        def find(entity):
            while parent[entity] != entity:
                parent[entity] = parent[parent[entity]]
                entity = parent[entity]
            return entity
        def union(a, b):
            a, b = find(a), find(b)
            if a != b:
                parent[b] = a
        for _, surface in gmsh.model.getEntities(2):
            upward, _ = gmsh.model.getAdjacencies(2, surface)
            volumes = [int(entity) for entity in upward if int(entity) in parent]
            for other in volumes[1:]:
                union(volumes[0], other)
        cad_connected_component_count = len({find(entity) for entity in parent})
        if layer_profile == "sky130" and not explicit_user_stack and cad_connected_component_count != 1:
            raise AssertionError(
                f"SKY130 substrate/dielectric domain has {cad_connected_component_count} CAD components"
            )

        refinement_surfaces = set()
        refined_region_tags = []
        if fine_mesh_size_um is not None and refinement_distance_um is not None:
            for region in regions:
                if region.get("role") not in {"conductor", "source"}:
                    continue
                refined_region_tags.append(region["physical_tag"])
                boundary = gmsh.model.getBoundary(
                    [(3, entity) for entity in region_entities[region["physical_tag"]]],
                    combined=True, oriented=False, recursive=False,
                )
                refinement_surfaces.update(tag for dim, tag in boundary if dim == 2)
            if refinement_surfaces:
                distance_field = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(distance_field, "FacesList", sorted(refinement_surfaces))
                gmsh.model.mesh.field.setNumber(distance_field, "Sampling", 80)
                threshold_field = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(threshold_field, "InField", distance_field)
                gmsh.model.mesh.field.setNumber(threshold_field, "SizeMin", fine_mesh_size_um)
                gmsh.model.mesh.field.setNumber(threshold_field, "SizeMax", mesh_size_um)
                gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", refinement_distance_um / 3)
                gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", refinement_distance_um)
                gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)

        gmsh.write(str(brep_path))
        gmsh.model.mesh.generate(3)

        node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
        xyz = {
            int(tag): tuple(float(v) for v in coordinates[3 * i:3 * i + 3])
            for i, tag in enumerate(node_tags)
        }
        stats = {
            "gmsh_version": gmsh.option.getString("General.Version"),
            "length_unit": "um",
            "length_scale_to_m": 1e-6,
            "target_mesh_size_um": mesh_size_um,
            "mesh_size_min_um": min(mesh_size_um / 2, fine_mesh_size_um or mesh_size_um / 2),
            "fine_mesh_size_um": fine_mesh_size_um,
            "refinement_distance_um": refinement_distance_um,
            "refined_region_tags": sorted(refined_region_tags),
            "refinement_surface_count": len(refinement_surfaces),
            "cad_connected_component_count": cad_connected_component_count,
            "mesh_size_max_um": mesh_size_um,
            "element_family": "first-order 4-node tetrahedron",
            "surface_algorithm": "Frontal-Delaunay (6)",
            "volume_algorithm": "Delaunay (1)",
            "optimization": ["Gmsh", "Netgen"],
            "random_seed": 1,
            "required_minimum_minSICN_quality": min_tet_quality,
            "node_count": len(node_tags),
            "input_prism_count": len(input_dimtags),
            "fragmented_volume_entity_count": len(model_volume_entities),
            "shared_interface_surface_count": sum(group["surface_count"] for group in interface_groups),
            "interface_groups": interface_groups,
            "physical_surface_tags": sorted(expected_surface_tags),
            "regions": [],
        }

        actual_volume_tags = {tag for dim, tag in gmsh.model.getPhysicalGroups(3)}
        expected_volume_tags = {region["physical_tag"] for region in regions}
        if actual_volume_tags != expected_volume_tags:
            raise AssertionError(f"Volume tags {actual_volume_tags} != expected {expected_volume_tags}")
        actual_surface_tags = {tag for dim, tag in gmsh.model.getPhysicalGroups(2)}
        if actual_surface_tags != expected_surface_tags:
            raise AssertionError(f"Surface tags {actual_surface_tags} != expected {expected_surface_tags}")

        owned_elements = set()
        all_qualities = []
        quality_element_tags = []
        all_edge_keys = set()
        all_equivalent_h_um = []
        for region in regions:
            tag = region["physical_tag"]
            element_count = 0
            mesh_volume_um3 = 0.0
            all_element_tags = []
            element_types_seen = set()
            for entity in region_entities[tag]:
                element_types, element_tags, element_nodes = gmsh.model.mesh.getElements(3, entity)
                for element_type, tags, nodes in zip(element_types, element_tags, element_nodes):
                    name, dim, order, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(int(element_type))
                    if dim != 3 or not name.startswith("Tetrahedron") or num_nodes != 4:
                        raise AssertionError(f"Unexpected 3D element type: {name}")
                    element_types_seen.add(name)
                    element_count += len(tags)
                    all_element_tags.extend(int(element_tag) for element_tag in tags)
                    for i in range(0, len(nodes), 4):
                        tet_nodes = tuple(int(node) for node in nodes[i:i + 4])
                        a, b, c, d = (xyz[node] for node in tet_nodes)
                        volume_um3 = tetrahedron_volume(a, b, c, d)
                        mesh_volume_um3 += volume_um3
                        all_equivalent_h_um.append((6 * volume_um3) ** (1 / 3))
                        for u, v in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
                            all_edge_keys.add(tuple(sorted((tet_nodes[u], tet_nodes[v]))))

            region_element_tags = set(all_element_tags)
            if not region_element_tags or owned_elements & region_element_tags:
                raise AssertionError(f"Missing or multiply-owned cells for physical tag {tag}")
            owned_elements.update(region_element_tags)
            qualities = gmsh.model.mesh.getElementQualities(all_element_tags, "minSICN")
            quality_element_tags.extend(all_element_tags)
            all_qualities.extend(float(quality) for quality in qualities)
            minimum_quality = min(float(quality) for quality in qualities)
            if minimum_quality < min_tet_quality:
                raise AssertionError(
                    f"Physical tag {tag} minimum minSICN={minimum_quality:g} is below "
                    f"the required {min_tet_quality:g}; refine or adjust the geometry"
                )
            expected_volume_um3 = sum(v["area_um2"] for v in region["volumes"]) * (
                region["z_max_um"] - region["z_min_um"]
            )
            if not math.isclose(mesh_volume_um3, expected_volume_um3, rel_tol=1e-7, abs_tol=1e-7):
                raise AssertionError(
                    f"Tag {tag}: mesh volume {mesh_volume_um3} != expected {expected_volume_um3} um^3"
                )
            stats["regions"].append({
                "physical_tag": tag,
                "name": region["name"],
                "volume_entity_count": len(region_entities[tag]),
                "element_types": sorted(element_types_seen),
                "tetrahedron_count": element_count,
                "minimum_minSICN_quality": minimum_quality,
                "expected_volume_um3": expected_volume_um3,
                "mesh_volume_um3": mesh_volume_um3,
                "bottom_surface_count": len(boundary_surfaces[tag]["bottom"]),
                "top_surface_count": len(boundary_surfaces[tag]["top"]),
            })

        _, all_element_tags, _ = gmsh.model.mesh.getElements(3)
        all_model_elements = {int(tag) for tags in all_element_tags for tag in tags}
        if owned_elements != all_model_elements:
            raise AssertionError("Not every tetrahedron belongs to exactly one material tag")
        def summarize(values):
            values = np.asarray(values, dtype=float)
            return {
                "minimum": float(values.min()),
                "p05": float(np.percentile(values, 5)),
                "median": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
                "maximum": float(values.max()),
            }

        edge_lengths_um = [math.dist(xyz[a], xyz[b]) for a, b in all_edge_keys]
        stats["tetrahedron_count"] = len(all_model_elements)
        stats["quality_summary_minSICN"] = summarize(all_qualities)
        stats["realized_edge_length_um"] = summarize(edge_lengths_um)
        stats["realized_equivalent_h_um"] = summarize(all_equivalent_h_um)
        gmsh.write(str(msh_path))
        quality_view = gmsh.view.add("Tetrahedron minSICN")
        gmsh.view.addHomogeneousModelData(
            quality_view,
            0,
            gmsh.model.getCurrent(),
            "ElementData",
            quality_element_tags,
            all_qualities,
        )
        gmsh.view.write(quality_view, str(quality_msh_path))
        stats["mesh_regions_sha256"] = input_regions_sha256
        stats["mesh_sha256"] = sha256_file(msh_path)
        return stats
    finally:
        gmsh.finalize()


def make_svg(regions, layout, source_name, placeholder_stack, width=1180):
    all_xy = [point for region in regions for volume in region["volumes"] for point in volume["footprint_xy_um"]]
    xs, ys = zip(*all_xy)
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    xspan, yspan = max(xmax - xmin, 1e-12), max(ymax - ymin, 1e-12)
    lateral = max(xspan, yspan)
    zmin = min(region["z_min_um"] for region in regions)
    zmax = max(region["z_max_um"] for region in regions)
    zspan = max(zmax - zmin, 1e-12)
    visible = [region for region in regions if region.get("role") in {"conductor", "source", "placeholder"}]
    crop_bulk_substrate = (
        bool(visible) and zspan > 2 * lateral
        and any(region.get("role") == "substrate" for region in regions)
    )
    if crop_bulk_substrate:
        # A deep thermal substrate is part of the mesh, but placing its whole
        # thickness in the device cutaway would hide the interconnect detail.
        # Only this visualization is cropped; the geometry/mesh stays intact.
        zmin = min(0.0, min(region["z_min_um"] for region in visible))
        zspan = max(zmax - zmin, 1e-12)
    z_exaggeration = min(3.0, max(1.0, 0.42 * lateral / zspan))
    meshed_component_count = sum(len(region["volumes"]) for region in regions)

    panel_y, panel_h, panel_w, margin = 98, 470, 520, 55
    legend_columns = 3
    legend_rows = math.ceil(len(regions) / legend_columns)
    height = 625 + legend_rows * 25
    plan_scale = min((panel_w - 2 * margin) / xspan, (panel_h - 2 * margin) / yspan)
    def plan(point):
        x, y = point
        return (35 + margin + (x - xmin) * plan_scale, panel_y + margin + (ymax - y) * plan_scale)
    def iso(point, z):
        x = (point[0] - xmin) / lateral
        y = (point[1] - ymin) / lateral
        zz = (z - zmin) * z_exaggeration / lateral
        if crop_bulk_substrate:
            raw_width = 215 * (xspan + yspan) / lateral
            raw_height = 105 * (xspan + yspan) / lateral + 205 * zspan * z_exaggeration / lateral
            fit_scale = min((560 - 2 * margin) / raw_width, (panel_h - 2 * margin) / raw_height)
            raw_center_x = 107.5 * (xspan - yspan) / lateral
            raw_min_y = -205 * zspan * z_exaggeration / lateral
            return (
                865 + fit_scale * (215 * (x - y) - raw_center_x),
                panel_y + margin + fit_scale * (105 * (x + y) - 205 * zz - raw_min_y),
            )
        return (875 + 215 * (x - y), 415 + 105 * (x + y) - 205 * zz)
    def points_attribute(points):
        return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)

    stack_label = (
        "PLACEHOLDER STACK — NOT FOR THERMAL PHYSICS" if placeholder_stack else
        "SKY130 PUBLIC STACK + EXPLICIT THERMAL ASSUMPTIONS" if layer_profile == "sky130" else
        "EXPLICIT USER PROCESS STACK"
    )
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" style="max-width:100%;height:auto">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Inter,Arial,sans-serif;fill:#263238}.title{font-size:19px;font-weight:700}.sub{font-size:12px;fill:#546e7a}.lab{font-size:12px;font-weight:600}.legend{font-size:11px}.warn{font-size:12px;font-weight:700;fill:#a85110}</style>',
        f'<text class="title" x="35" y="32">{escape(source_name)} / {escape(layout["top_cell"])} — physical-region construction</text>',
        f'<text class="sub" x="35" y="54">{layout["flattened_polygon_count"]} source polygons · {len(regions)} tagged 3D regions · {meshed_component_count} prism components · z ×{z_exaggeration:.1f}</text>',
        f'<text class="warn" x="35" y="76">{escape(stack_label)}</text>',
        f'<rect x="35" y="{panel_y}" width="{panel_w}" height="{panel_h}" rx="8" fill="#fafafa" stroke="#cfd8dc"/>',
        f'<rect x="585" y="{panel_y}" width="560" height="{panel_h}" rx="8" fill="#fafafa" stroke="#cfd8dc"/>',
        f'<text class="lab" x="53" y="{panel_y + 27}">2D masks used by material/source solids</text>',
        f'<text class="lab" x="603" y="{panel_y + 27}">{"3D frontside cutaway — bulk substrate omitted" if crop_bulk_substrate else "3D cutaway — dielectric is wireframe"}</text>',
    ]
    domain_plan = [plan((xmin, ymin)), plan((xmax, ymin)), plan((xmax, ymax)), plan((xmin, ymax))]
    out.append(f'<polygon points="{points_attribute(domain_plan)}" fill="#eaf2f6" fill-opacity="0.35" stroke="#90a4ae" stroke-width="1"/>')
    for region in visible:
        for volume in region["volumes"]:
            polygon = [plan(point) for point in volume["footprint_xy_um"]]
            opacity = 0.48 if region.get("role") == "source" else 0.34
            out.append(f'<polygon points="{points_attribute(polygon)}" fill="{region["color"]}" fill-opacity="{opacity}" stroke="{region["color"]}" stroke-width="1.15"/>')

    faces, top_faces = [], []
    for region in visible:
        z0, z1 = region["z_min_um"], region["z_max_um"]
        for volume in region["volumes"]:
            points = volume["footprint_xy_um"]
            for p0, p1 in zip(points, points[1:] + points[:1]):
                face = [iso(p0, z0), iso(p1, z0), iso(p1, z1), iso(p0, z1)]
                faces.append((p0[0] + p0[1] + p1[0] + p1[1] + z1, region["color"], face))
            top_faces.append((z1, region["color"], [iso(point, z1) for point in points]))
    for _, color, face in sorted(faces, reverse=True):
        out.append(f'<polygon points="{points_attribute(face)}" fill="{color}" fill-opacity="0.62" stroke="#455a64" stroke-width="0.45"/>')
    for _, color, face in sorted(top_faces):
        out.append(f'<polygon points="{points_attribute(face)}" fill="{color}" fill-opacity="0.82" stroke="#37474f" stroke-width="0.60"/>')

    # Full thermal fill is represented by a restrained wireframe envelope.
    box = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    bottom = [iso(point, 0.0) for point in box]
    top = [iso(point, zmax) for point in box]
    out.append(f'<polygon points="{points_attribute(bottom)}" fill="#dceaf2" fill-opacity="0.07" stroke="#78909c" stroke-width="0.8" stroke-dasharray="4 3"/>')
    out.append(f'<polygon points="{points_attribute(top)}" fill="none" stroke="#78909c" stroke-width="0.8" stroke-dasharray="4 3"/>')
    for a, b in zip(bottom, top):
        out.append(f'<line x1="{a[0]:.2f}" y1="{a[1]:.2f}" x2="{b[0]:.2f}" y2="{b[1]:.2f}" stroke="#78909c" stroke-width="0.8" stroke-dasharray="4 3"/>')

    legend_y = panel_y + panel_h + 28
    column_width = 380
    for index, region in enumerate(regions):
        column, row = index % legend_columns, index // legend_columns
        x, y = 45 + column * column_width, legend_y + row * 25
        out.append(f'<rect x="{x}" y="{y - 12}" width="14" height="14" fill="{region["color"]}"/>')
        label = f'tag {region["physical_tag"]} · {region["name"]} · {region["role"]} · z {region["z_min_um"]:g}..{region["z_max_um"]:g}'
        out.append(f'<text class="legend" x="{x + 20}" y="{y}">{escape(label)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def read_gmsh_mesh_for_preview(msh_path):
    """Read the actual saved tetrahedra and their physical-region ownership."""
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(msh_path))
        node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
        coordinates = np.asarray(coordinates, dtype=float).reshape(-1, 3)
        xyz = {int(tag): tuple(coordinates[index]) for index, tag in enumerate(node_tags)}

        entity_owner = {}
        for dim, physical_tag in gmsh.model.getPhysicalGroups(3):
            for entity in gmsh.model.getEntitiesForPhysicalGroup(dim, physical_tag):
                if entity in entity_owner and entity_owner[entity] != physical_tag:
                    raise AssertionError(f"Volume entity {entity} belongs to multiple physical groups")
                entity_owner[int(entity)] = int(physical_tag)

        tetrahedra, element_tags = [], []
        for entity, physical_tag in sorted(entity_owner.items()):
            types, tags_by_type, nodes_by_type = gmsh.model.mesh.getElements(3, entity)
            for element_type, tags, nodes in zip(types, tags_by_type, nodes_by_type):
                name, dim, order, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(
                    int(element_type)
                )
                if dim != 3 or num_nodes != 4 or not name.startswith("Tetrahedron"):
                    raise AssertionError(f"Unexpected preview element type: {name}")
                element_tags.extend(int(tag) for tag in tags)
                for offset in range(0, len(nodes), 4):
                    tetrahedra.append((physical_tag, tuple(int(node) for node in nodes[offset:offset + 4])))

        qualities = np.asarray(
            gmsh.model.mesh.getElementQualities(element_tags, "minSICN"), dtype=float
        )
        if len(tetrahedra) != len(qualities):
            raise AssertionError("Preview tetrahedron and quality counts differ")
        return {"xyz": xyz, "tetrahedra": tetrahedra, "qualities": qualities}
    finally:
        gmsh.finalize()


def tetrahedron_plane_section(points, cut_axis, cut_value, tolerance):
    """Intersect one tetrahedron with x=cut or y=cut; return its ordered 2D polygon."""
    section_axes = (1, 2) if cut_axis == 0 else (0, 2)
    intersections = []

    def add(point):
        projected = (float(point[section_axes[0]]), float(point[section_axes[1]]))
        if not any(math.dist(projected, existing) <= tolerance for existing in intersections):
            intersections.append(projected)

    for point in points:
        if abs(point[cut_axis] - cut_value) <= tolerance:
            add(point)
    for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        a, b = points[i], points[j]
        da, db = a[cut_axis] - cut_value, b[cut_axis] - cut_value
        if da * db < -(tolerance**2):
            fraction = da / (da - db)
            add(tuple(a[k] + fraction * (b[k] - a[k]) for k in range(3)))
    if len(intersections) < 3:
        return None
    center = tuple(sum(point[k] for point in intersections) / len(intersections) for k in range(2))
    return sorted(
        intersections,
        key=lambda point: math.atan2(point[1] - center[1], point[0] - center[0]),
    )


def make_mesh_diagnostics_svg(preview, regions, mesh_stats):
    """Make a notebook-safe envelope, one true tetrahedral cut, and mesh statistics."""
    xyz, tetrahedra, qualities = preview["xyz"], preview["tetrahedra"], preview["qualities"]
    colors = {region["physical_tag"]: region["color"] for region in regions}
    roles = {region["physical_tag"]: region.get("role", "material") for region in regions}
    all_coordinates = np.asarray(list(xyz.values()))
    lower, upper = all_coordinates.min(axis=0), all_coordinates.max(axis=0)
    spans = np.maximum(upper - lower, 1e-12)
    lateral_span = max(spans[0], spans[1])
    z_exaggeration = min(3.0, max(1.0, 0.3 * lateral_span / spans[2]))

    tetra_points = [np.asarray([xyz[node] for node in nodes]) for _, nodes in tetrahedra]
    # Pick a useful audit plane, not a claim about whole-device connectivity:
    # maximize represented physical tags, then intersected cells.
    best_cut = None
    for candidate_axis in (0, 1):
        candidates = np.linspace(
            lower[candidate_axis] + 0.1 * spans[candidate_axis],
            upper[candidate_axis] - 0.1 * spans[candidate_axis],
            17,
        )
        for candidate_value in candidates:
            hit_indices = [
                index for index, points in enumerate(tetra_points)
                if points[:, candidate_axis].min() < candidate_value < points[:, candidate_axis].max()
            ]
            hit_tags = {tetrahedra[index][0] for index in hit_indices}
            score = (len(hit_tags), len(hit_indices))
            if best_cut is None or score > best_cut[0]:
                best_cut = (score, candidate_axis, float(candidate_value))
    (_, _), cut_axis, cut_value = best_cut
    horizontal_axis = 1 - cut_axis
    tolerance = max(spans) * 1e-10
    sections = []
    for (physical_tag, _), points in zip(tetrahedra, tetra_points):
        if points[:, cut_axis].min() <= cut_value <= points[:, cut_axis].max():
            polygon = tetrahedron_plane_section(points, cut_axis, cut_value, tolerance)
            if polygon is not None:
                sections.append((physical_tag, polygon))
    section_tag_count = len({physical_tag for physical_tag, _ in sections})

    width = 1180
    panel_y, panel_h = 112, 445
    left = (25, panel_y, 550, panel_h)
    right = (600, panel_y, 555, panel_h)
    stats_panel = (25, 582, 1130, 238)
    legend_columns = 3
    legend_rows = math.ceil(len(regions) / legend_columns)
    legend_y = 856
    height = legend_y + legend_rows * 29 + 24

    center = 0.5 * (lower + upper)

    def raw_view(point):
        x = (point[0] - center[0]) / lateral_span
        y = (point[1] - center[1]) / lateral_span
        z = (point[2] - center[2]) * z_exaggeration / lateral_span
        return (
            0.90 * x - 0.48 * y,
            0.30 * x + 0.52 * y - z,
            0.42 * x + 0.80 * y + 0.24 * z,
        )

    prism_faces = []
    for region in regions:
        for volume in region["volumes"]:
            ring = volume["footprint_xy_um"]
            bottom = [(x, y, region["z_min_um"]) for x, y in ring]
            top = [(x, y, region["z_max_um"]) for x, y in ring]
            prism_faces.append((region["physical_tag"], bottom))
            prism_faces.append((region["physical_tag"], top))
            for index in range(len(ring)):
                nxt = (index + 1) % len(ring)
                prism_faces.append((
                    region["physical_tag"],
                    [bottom[index], bottom[nxt], top[nxt], top[index]],
                ))

    cut_plane = (
        [
            (cut_value, lower[1], lower[2]), (cut_value, upper[1], lower[2]),
            (cut_value, upper[1], upper[2]), (cut_value, lower[1], upper[2]),
        ]
        if cut_axis == 0 else
        [
            (lower[0], cut_value, lower[2]), (upper[0], cut_value, lower[2]),
            (upper[0], cut_value, upper[2]), (lower[0], cut_value, upper[2]),
        ]
    )
    projection_points = [point for _, face in prism_faces for point in face] + cut_plane
    raw_points = [raw_view(point) for point in projection_points]
    raw_u = [point[0] for point in raw_points]
    raw_v = [point[1] for point in raw_points]
    u0, u1, v0, v1 = min(raw_u), max(raw_u), min(raw_v), max(raw_v)
    draw_x, draw_y = left[0] + 22, left[1] + 76
    draw_w, draw_h = left[2] - 44, left[3] - 102
    projection_scale = min(
        draw_w / max(u1 - u0, 1e-12),
        draw_h / max(v1 - v0, 1e-12),
    )
    projection_offset_x = draw_x + 0.5 * (draw_w - (u1 - u0) * projection_scale)
    projection_offset_y = draw_y + 0.5 * (draw_h - (v1 - v0) * projection_scale)

    def project_world(point):
        u, v, _ = raw_view(point)
        return (
            projection_offset_x + (u - u0) * projection_scale,
            projection_offset_y + (v - v0) * projection_scale,
        )

    section_h0, section_h1 = lower[horizontal_axis], upper[horizontal_axis]
    section_z0, section_z1 = lower[2], upper[2]
    section_draw_x, section_draw_y = right[0] + 28, right[1] + 82
    section_draw_w, section_draw_h = right[2] - 56, right[3] - 112
    section_scale = min(
        section_draw_w / max(section_h1 - section_h0, 1e-12),
        section_draw_h / max((section_z1 - section_z0) * z_exaggeration, 1e-12),
    )
    used_section_w = (section_h1 - section_h0) * section_scale
    used_section_h = (section_z1 - section_z0) * z_exaggeration * section_scale
    section_offset_x = section_draw_x + 0.5 * (section_draw_w - used_section_w)
    section_offset_y = section_draw_y + 0.5 * (section_draw_h - used_section_h)

    def project_section(point):
        return (
            section_offset_x + (point[0] - section_h0) * section_scale,
            section_offset_y + used_section_h
            - (point[1] - section_z0) * z_exaggeration * section_scale,
        )

    def points_attribute(points):
        return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)

    placeholder_stack = all(region["placeholder"] for region in regions)
    envelope_label = (
        "Meshed mask-region envelope" if placeholder_stack else "Meshed physical-region envelope"
    )
    envelope_detail = (
        f"opaque mask prisms · profile colors · z ×{z_exaggeration:.1f}"
        if placeholder_stack else f"transparent substrate/dielectric · opaque conductors · z ×{z_exaggeration:.1f}"
    )
    warning = (
        "MASK-EXTRUSION PLACEHOLDER — disconnected solids are not a thermal-domain result"
        if placeholder_stack else
        "Connected thermal fill; material values remain explicit assumptions"
    )
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" style="max-width:100%;height:auto">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Inter,Arial,sans-serif;fill:#263238}.title{font-size:19px;font-weight:700}.sub{font-size:12px;fill:#546e7a}.head{font-size:13px;font-weight:700}.small{font-size:11px}.metric{font-size:12px;font-weight:600}.warn{fill:#a85110;font-size:12px;font-weight:700}</style>',
        '<text class="title" x="25" y="31">Gmsh geometry and finite-element mesh</text>',
        f'<text class="sub" x="25" y="53">{len(xyz):,} nodes · {len(tetrahedra):,} linear tetrahedra · conformal shared interfaces · coordinates in µm</text>',
        f'<text class="warn" x="25" y="75">{escape(warning)}</text>',
        '<text class="sub" x="25" y="95">The left panel is a clean region envelope; tetrahedral edges appear only on the actual cut at right. One cut cannot prove whole-device connectivity.</text>',
    ]
    for x, y, w, h in (left, right, stats_panel):
        out.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" '
            'fill="#fafafa" stroke="#cfd8dc"/>'
        )
    out.extend([
        f'<text class="head" x="{left[0] + 15}" y="{left[1] + 25}">{envelope_label}</text>',
        f'<text class="small" x="{left[0] + 15}" y="{left[1] + 45}">{envelope_detail}</text>',
        f'<text class="small" x="{left[0] + 15}" y="{left[1] + 62}">blue dashed plane is the section shown at right</text>',
        f'<text class="head" x="{right[0] + 15}" y="{right[1] + 25}">Actual tetrahedral section</text>',
        f'<text class="small" x="{right[0] + 15}" y="{right[1] + 45}">{"x" if cut_axis == 0 else "y"} = {cut_value:.3f} µm · {len(sections):,} cut cells · {section_tag_count}/{len(regions)} tags</text>',
        f'<text class="small" x="{right[0] + 15}" y="{right[1] + 62}">physical lateral scale · explicit z ×{z_exaggeration:.1f} · slice only</text>',
        f'<text class="head" x="{stats_panel[0] + 15}" y="{stats_panel[1] + 26}">Quality distribution and meshing choices</text>',
    ])

    plane_projected = [project_world(point) for point in cut_plane]
    out.append(
        f'<polygon points="{points_attribute(plane_projected)}" fill="#1976d2" '
        'fill-opacity="0.08" stroke="none"/>'
    )
    sorted_faces = sorted(
        prism_faces,
        key=lambda tagged_face: sum(raw_view(point)[2] for point in tagged_face[1]) / len(tagged_face[1]),
    )
    for physical_tag, face in sorted_faces:
        projected = [project_world(point) for point in face]
        color = colors.get(physical_tag, "#607d8b")
        out.append(
            f'<polygon points="{points_attribute(projected)}" fill="{color}" '
            f'fill-opacity="0.92" stroke="{color}" stroke-width="0.45"/>'
        )
    out.append(
        f'<polygon points="{points_attribute(plane_projected)}" fill="none" '
        'stroke="#1565c0" stroke-width="1.4" stroke-dasharray="5 4"/>'
    )

    sections_by_tag = defaultdict(list)
    for physical_tag, polygon in sections:
        sections_by_tag[physical_tag].append(polygon)
    for physical_tag, polygons in sorted(sections_by_tag.items()):
        color = colors.get(physical_tag, "#607d8b")
        for polygon in polygons:
            out.append(
                f'<polygon points="{points_attribute([project_section(point) for point in polygon])}" '
                f'fill="{color}" fill-opacity="0.40" stroke="#263238" '
                'stroke-opacity="0.72" stroke-width="0.32"/>'
            )
    out.extend([
        f'<line x1="{section_offset_x:.2f}" y1="{section_offset_y + used_section_h:.2f}" '
        f'x2="{section_offset_x + used_section_w:.2f}" y2="{section_offset_y + used_section_h:.2f}" stroke="#78909c"/>',
        f'<text class="small" x="{section_offset_x:.2f}" y="{section_offset_y + used_section_h + 18:.2f}">{section_h0:g}</text>',
        f'<text class="small" x="{section_offset_x + used_section_w - 24:.2f}" y="{section_offset_y + used_section_h + 18:.2f}">{section_h1:g} µm</text>',
    ])

    histogram, _ = np.histogram(qualities, bins=np.linspace(0, 1, 21))
    chart_x, chart_y, chart_w, chart_h = stats_panel[0] + 35, stats_panel[1] + 50, 470, 145
    max_count = max(int(histogram.max()), 1)
    out.append(
        f'<line x1="{chart_x}" y1="{chart_y + chart_h}" x2="{chart_x + chart_w}" '
        f'y2="{chart_y + chart_h}" stroke="#78909c"/>'
    )
    for index, count in enumerate(histogram):
        x = chart_x + index * chart_w / len(histogram)
        bar_w = chart_w / len(histogram) - 1
        bar_h = chart_h * count / max_count
        out.append(
            f'<rect x="{x:.2f}" y="{chart_y + chart_h - bar_h:.2f}" '
            f'width="{bar_w:.2f}" height="{bar_h:.2f}" fill="#5470c6"/>'
        )
    threshold = mesh_stats["required_minimum_minSICN_quality"]
    threshold_x = chart_x + threshold * chart_w
    out.extend([
        f'<line x1="{threshold_x:.2f}" y1="{chart_y}" x2="{threshold_x:.2f}" '
        f'y2="{chart_y + chart_h}" stroke="#b33a3a" stroke-width="2"/>',
        f'<text class="small" x="{chart_x}" y="{chart_y + chart_h + 17}">0</text>',
        f'<text class="small" x="{chart_x + chart_w - 7}" y="{chart_y + chart_h + 17}">1</text>',
        f'<text class="small" x="{chart_x + 4}" y="{chart_y + 13}">minSICN</text>',
        f'<text class="small" x="{threshold_x + 4:.2f}" y="{chart_y + 28}">gate {threshold:g}</text>',
    ])

    summary = mesh_stats["quality_summary_minSICN"]
    edge_summary = mesh_stats["realized_edge_length_um"]
    h_summary = mesh_stats["realized_equivalent_h_um"]
    settings = [
        f"Observed min / p05 / median: {summary['minimum']:.3f} / {summary['p05']:.3f} / {summary['median']:.3f}",
        f"Configured characteristic sizes: {mesh_stats['mesh_size_min_um']:g}–{mesh_stats['mesh_size_max_um']:g} µm",
        f"Real edge p05 / median / p95: {edge_summary['p05']:.3f} / {edge_summary['median']:.3f} / {edge_summary['p95']:.3f} µm",
        f"Equivalent h p05 / median / p95: {h_summary['p05']:.3f} / {h_summary['median']:.3f} / {h_summary['p95']:.3f} µm",
        "Element: 4-node linear tetrahedron",
        f"Algorithms: {mesh_stats['surface_algorithm']} / {mesh_stats['volume_algorithm']}",
        "Geometry: OCC extrude + global fragment",
        "Optimization: Gmsh + Netgen",
        f"Seed {mesh_stats['random_seed']}; no local refinement field",
    ]
    settings_x = stats_panel[0] + 545
    for index, setting in enumerate(settings):
        css = "metric" if index < 2 else "small"
        out.append(
            f'<text class="{css}" x="{settings_x}" y="{stats_panel[1] + 55 + index * 20}">{escape(setting)}</text>'
        )

    for index, region in enumerate(regions):
        column, row = index % legend_columns, index // legend_columns
        x, y = 32 + column * 380, legend_y + row * 29
        out.append(
            f'<rect x="{x}" y="{y - 12}" width="14" height="14" fill="{region["color"]}"/>'
        )
        label = (
            f'tag {region["physical_tag"]} · {region["name"]} · '
            f'z {region["z_min_um"]:g}..{region["z_max_um"]:g} µm'
        )
        out.append(f'<text class="small" x="{x + 20}" y="{y}">{escape(label)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def region_table_html(regions, mesh_stats):
    stats_by_tag = {entry["physical_tag"]: entry for entry in mesh_stats["regions"]}
    rows = []
    for region in regions:
        stats = stats_by_tag[region["physical_tag"]]
        area = sum(volume["area_um2"] for volume in region["volumes"])
        rows.append(
            f"<tr><td>{region['physical_tag']}</td><td>{escape(region['name'])}</td>"
            f"<td>L{region['gds_layer']}/D{region['datatype']}</td><td>{escape(region['material'])}</td>"
            f"<td>{region['z_min_um']:g}..{region['z_max_um']:g}</td><td>{region['source_polygon_count']}</td>"
            f"<td>{len(region['volumes'])}</td><td>{area:.3f}</td>"
            f"<td>{stats['tetrahedron_count']}</td><td>{stats['minimum_minSICN_quality']:.3f}</td></tr>"
        )
    return (
        "<table><thead><tr><th>tag</th><th>region</th><th>mask</th><th>material</th><th>z (um)</th>"
        "<th>raw polygons</th><th>union components</th><th>area (um²)</th><th>tets</th><th>minSICN</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )

_HELPER_NAMES = ('signed_area', 'polygon_bbox', 'clean_ring', 'repeated_grid_vertex', 'canonical_ring', 'sha256_file', 'safe_component', 'choose_top_cell', 'extract_layout', 'discover_and_audit', 'audit_html', 'verify_sky130_bundle', 'choose_layer_profile', 'polygon_objects', 'polygon_union_for_specs', 'polygon_area', 'overlap_area', 'scmos_contact_audit', 'sky130_contact_audit', 'contact_audit_html', 'sky130_gallery_svg', 'cell_power_sidecars', 'LayerSpec', 'distinct_color', 'normalized_components', 'component_polygons', 'boolean_components', 'build_process_stack', 'make_region', 'generic_regions', 'dielectric_name', 'build_sky130_regions', 'intervals_overlap', 'assert_no_material_overlaps', 'tetrahedron_volume', 'add_polygon_prism', 'build_and_validate_mesh', 'make_svg', 'read_gmsh_mesh_for_preview', 'tetrahedron_plane_section', 'make_mesh_diagnostics_svg', 'region_table_html')
_HELPER_NAMES += ("region_contract_sha256",)
_CONTEXT_BASE = {name: value for name, value in globals().copy().items()
                 if not name.startswith("_") and name not in _HELPER_NAMES}


def install_helpers(namespace):
    """Bind shared helpers to a caller-owned namespace and fill missing defaults.

    Functions intentionally share the supplied dictionary, so a staged notebook
    can assign ``layout``/``regions`` later. Separate dictionaries are independent.
    No notebook file is opened or compiled by this module at runtime.
    """
    import copy
    from types import FunctionType

    namespace.setdefault("__builtins__", __builtins__)
    namespace.setdefault("__name__", __name__)
    namespace.setdefault("__package__", __package__)
    for name, value in _CONTEXT_BASE.items():
        if name not in namespace:
            namespace[name] = copy.deepcopy(value) if isinstance(value, (dict, list, set)) else value
    for name in _HELPER_NAMES:
        helper = globals()[name]
        if isinstance(helper, FunctionType):
            bound = FunctionType(helper.__code__, namespace, helper.__name__, helper.__defaults__, helper.__closure__)
            bound.__kwdefaults__ = helper.__kwdefaults__
            bound.__annotations__ = dict(helper.__annotations__)
            bound.__doc__ = helper.__doc__
            namespace[name] = bound
        else:
            namespace[name] = helper
    return namespace


def create_context(**overrides):
    """Return one isolated run context; override uppercase configuration keys.

    Typical SRAM adapters set SUBSTRATE_DEPTH_UM, CHANNEL_DEPTH_UM,
    ACTIVE_BACKGROUND_MATERIAL, TOP_PASSIVATION_THICKNESS_UM, and
    TOP_PASSIVATION_MATERIAL explicitly, as well as the desired stack z extents.
    """
    namespace = dict(overrides)
    return install_helpers(namespace)
