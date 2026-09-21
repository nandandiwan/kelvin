"""Notebook-derived, polygon-preserving mesh pipeline for the bundled SRAM.

Geometry is shared with read_gds.ipynb via mesh.gds_notebook, not extracted
from notebook source at runtime. Power is assigned only after matching the
eight source footprints to the verified X0..X7 compact-model instance map.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import gdstk

from gds.bitcell_mapping import (
    REPO, map_bitcell_channels, mapping_revision, validate_device_powers,
)
from gds.read import flatten_by_layer


GDS_PATH = REPO / "data/sram22_64x22m4w22.gds"
CELL_NAME = "sram_sp_cell"
STEM = "sram_sp_cell"


def _contract_hash(manifest):
    """Bind region/tag/material/source semantics as well as the mesh bytes."""
    contract = {key: value for key, value in manifest.items()
                if key != "mesh_manifest_contract_sha256"}
    # JSON object keys are strings on disk; normalize integer physical-tag
    # keys before sorting so a save/load round trip has the same digest.
    contract = json.loads(json.dumps(contract, allow_nan=False))
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def thermal_context_options(refine=1.0):
    """Explicit SRAM settings: preserve the previous frontside thermal model.

    z=0 is now active top, rather than backside. The substrate distance,
    10 nm channel sources, field oxide and nominal SiN cap are unchanged.
    ``refine`` multiplies target lengths (smaller means finer), as in Kelvin.
    These are modeling inputs, not dimensions recovered from GDS.
    """
    if not math.isfinite(refine) or refine <= 0:
        raise ValueError("refine must be finite and positive")
    return {
        "SUBSTRATE_DEPTH_UM": 50.3262,
        "CHANNEL_DEPTH_UM": 0.01,
        "ACTIVE_BACKGROUND_MATERIAL": "SiO2",
        "TOP_PASSIVATION_THICKNESS_UM": 1.0,
        "TOP_PASSIVATION_MATERIAL": "SiN",
        "MESH_SIZE_UM": 0.4 * refine,
        "FINE_MESH_SIZE_UM": 0.04 * refine,
        "REFINEMENT_DISTANCE_UM": 0.35,
        "MIN_TET_QUALITY": 0.025,
        "SKY130_SOLID_Z_UM": {
            "well": (-0.3262, -0.12), "diff": (-0.12, 0.0),
            "poly": (0.0, 0.18), "li1": (0.61, 0.71),
            "mcon": (0.71, 1.05), "met1": (1.05, 1.41),
            "via": (1.41, 1.68), "met2": (1.68, 2.04),
            "via2": (2.0399, 2.4599), "met3": (2.4599, 3.3049),
            "via3": (3.3049, 3.6949), "met4": (3.6949, 4.5399),
            "via4": (4.5399, 5.0449), "met5": (5.0449, 6.3049),
        },
    }


def _mapped_channels():
    library = gdstk.read_gds(GDS_PATH)
    cell = next(cell for cell in library.cells if cell.name == CELL_NAME)
    return map_bitcell_channels(flatten_by_layer(cell))


def match_source_tags(manifest):
    """Match complete polygons, never class names, centroid order, or tag order."""
    if manifest.get("top_cell") != CELL_NAME:
        raise ValueError(f"SRAM pipeline requires the verified {CELL_NAME!r} cell")
    if manifest.get("source_sha256") != hashlib.sha256(GDS_PATH.read_bytes()).hexdigest():
        raise ValueError("Mesh GDS fingerprint does not match the verified SRAM input")
    revision = manifest.get("source_mapping_sha256")
    if revision is not None and revision != mapping_revision():
        raise ValueError("Mesh uses an obsolete SRAM instance map; regenerate/revalidate it")
    channels = _mapped_channels()  # Also verifies the current netlists and pin geometry.
    candidates = [r for r in manifest["regions"] if r.get("role") == "source"]
    if len(candidates) != 8:
        raise ValueError("Expected exactly eight distinct channel-source regions")
    matches = {}
    for channel in channels:
        matching = [r for r in candidates if not gdstk.boolean(
            [gdstk.Polygon(v["footprint_xy_um"]) for v in r["volumes"]],
            [channel.polygon], "xor", precision=1e-6,
        )]
        if len(matching) != 1:
            raise ValueError(f"Missing or ambiguous polygon source for {channel.instance}")
        region = matching[0]
        if region["material"] != "Si_channel":
            raise ValueError(f"{channel.instance} must be assigned to channel silicon")
        if region["physical_tag"] in matches:
            raise ValueError("A physical source tag matched multiple transistor instances")
        matches[int(region["physical_tag"])] = channel.instance
    return matches


def prepare_sram_regions(*, pad_um=0.0, refine=1.0):
    """Return (shared notebook context, manifest), without running Gmsh."""
    from mesh.gds_notebook import build_region_manifest, create_context, region_layout_identity

    if not math.isfinite(pad_um) or pad_um < 0:
        raise ValueError("pad_um must be finite and nonnegative")
    context = create_context(**thermal_context_options(refine))
    layout = context["extract_layout"](GDS_PATH, CELL_NAME)
    # Padding adds background, not neighboring devices; source polygons do not move.
    xmin, ymin, xmax, ymax = layout["bbox_um"]
    layout["bbox_um"] = [xmin-pad_um, ymin-pad_um, xmax+pad_um, ymax+pad_um]
    context.update(layout=layout, layer_profile="sky130", explicit_user_stack=False,
                   OUTPUT_STEM=STEM, SRAM_PIPELINE=True, GDS_PATH=GDS_PATH,
                   TOP_CELL_NAME=CELL_NAME)
    regions, model, used_specs = context["build_sky130_regions"]()
    context["assert_no_material_overlaps"](regions, layout["boolean_grid_um"])
    context.update(regions=regions, process_model=model, used_gds_specs=used_specs,
                   REGION_LAYOUT_IDENTITY=region_layout_identity(layout))
    return context, build_region_manifest(context)


def prepare_sram_manifest(manifest):
    """Validate current SRAM region semantics before the expensive mesh build."""
    from spec.materials import MATERIALS

    manifest = json.loads(json.dumps(manifest, allow_nan=False))
    unknown = {r["material"] for r in manifest["regions"]} - MATERIALS.keys()
    if unknown:
        raise ValueError(f"Unknown Kelvin materials in current SRAM regions: {sorted(unknown)}")
    manifest["process_stack_kind"] = "notebook-polygons-kelvin-sram-thermal-settings"
    manifest["source_mapping_sha256"] = mapping_revision()
    manifest["warnings"] = list(manifest.get("warnings", [])) + [
        "Thermal properties, substrate depth, source depth and passivation are modeling inputs.",
        "Exact mask footprints are preserved; this is not a fabricated-process reconstruction.",
        "Gate oxide and interface thermal resistance remain omitted.",
        "Geometry generation assigns no power; electrical channel/interconnect sources are mapped separately at solve time.",
    ]
    manifest["source_device_by_tag"] = match_source_tags(manifest)
    return manifest


def validated_sram_manifest(manifest, msh_path, *, mesh_stats, mesh_file=None):
    """Bind semantics only with proof of the exact input to the validated mesh.

    A matching GDS/volume alone is insufficient: different materials or swapped
    equal-volume tags can leave the geometry valid but change the thermal model.
    Validation stats carry hashes of both the complete region input and MSH.
    """
    from mesh.gds_notebook import region_contract_sha256

    manifest = json.loads(json.dumps(manifest, allow_nan=False))
    msh_path = Path(msh_path).resolve()
    if not mesh_stats or mesh_stats.get("mesh_regions_sha256") != region_contract_sha256(manifest["regions"]):
        raise ValueError("Mesh validation does not match the current region/material contract; "
                         "rebuild with build_notebook_mesh_bundle")
    mesh_hash = hashlib.sha256(msh_path.read_bytes()).hexdigest()
    if mesh_stats.get("mesh_sha256") != mesh_hash:
        raise ValueError("Mesh changed after validation; rebuild with build_notebook_mesh_bundle")
    manifest["source_device_by_tag"] = match_source_tags(manifest)
    manifest["source_mapping_sha256"] = mapping_revision()
    manifest["mesh_file"] = str(Path(mesh_file).resolve() if mesh_file else msh_path)
    manifest["mesh_sha256"] = mesh_hash
    manifest["mesh_regions_sha256"] = mesh_stats["mesh_regions_sha256"]
    manifest["mesh_manifest_contract_sha256"] = _contract_hash(manifest)
    return manifest


def seal_sram_manifest(manifest_path, msh_path, *, mesh_stats=None):
    """Revalidate a manifest only with the matching mesher-returned proof.

    New runs should call ``build_notebook_mesh_bundle`` instead. The old two-
    argument blind reseal is rejected; it could certify stale notebook settings.
    Previously saved, sealed bundles remain readable by ``load_sram_mesh``.
    """
    manifest_path = Path(manifest_path)
    manifest = validated_sram_manifest(json.loads(manifest_path.read_text()), msh_path,
                                       mesh_stats=mesh_stats)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def generate_sram_mesh(out_dir, *, pad_um=0.0, refine=1.0, renders=True):
    """Generate the notebook's conforming material mesh and return its manifest."""
    from mesh.gds_notebook import build_notebook_mesh_bundle

    context, _ = prepare_sram_regions(pad_um=pad_um, refine=refine)
    print("[notebook mesh] extruding exact GDS polygons and assembling dielectric/substrate", flush=True)
    _, stats, paths = build_notebook_mesh_bundle(context, out_dir, renders=renders)
    manifest_path = paths["manifest"]
    print(f"[notebook mesh] {stats['tetrahedron_count']} tetrahedra; manifest: {manifest_path}", flush=True)
    return manifest_path


def load_sram_mesh(manifest_path, device_power_w, *, bcs=None, msh_path=None):
    """Load a saved notebook mesh with this run's individual device powers."""
    from mesh.gds_import import load_tagged_msh

    powers = validate_device_powers(device_power_w)
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("mesh_manifest_contract_sha256") != _contract_hash(manifest):
        raise ValueError("Notebook manifest contract changed or is unsealed; regenerate or "
                         "rerun the notebook mesh-bundle cell")
    tag_devices = match_source_tags(manifest)
    if not manifest.get("mesh_sha256"):
        raise ValueError("Unsealed notebook mesh: rerun the notebook mesh-bundle cell, then retry")
    msh_path = Path(msh_path or manifest["mesh_file"])
    if not msh_path.is_absolute():
        msh_path = manifest_path.parent / msh_path
    if hashlib.sha256(msh_path.read_bytes()).hexdigest() != manifest["mesh_sha256"]:
        raise ValueError("Saved mesh fingerprint differs from its validated manifest")
    case = load_tagged_msh(
        msh_path, manifest_path, bcs=bcs,
        source_power_by_tag={tag: powers[name] for tag, name in tag_devices.items()},
        source_device_by_tag=tag_devices,
    ).assert_solver_ready()
    return case


def build_sram_mesh(device_power_w, out_dir, *, pad_um=0.0, refine=1.0,
                    renders=True, bcs=None, manifest_path=None):
    """Generate or reuse a notebook mesh; always reassign and audit source power."""
    validate_device_powers(device_power_w)
    if manifest_path is not None and pad_um != 0:
        raise ValueError("Padding is fixed by a saved mesh; omit pad_um when reusing it")
    path = manifest_path or generate_sram_mesh(out_dir, pad_um=pad_um, refine=refine, renders=renders)
    case = load_sram_mesh(path, device_power_w, bcs=bcs)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mesh_import_report.json").write_text(json.dumps(case.report, indent=2) + "\n")
    return case
