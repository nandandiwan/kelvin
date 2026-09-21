"""Import a tagged GDS-derived Gmsh mesh into Kelvin's solver contract.

Kelvin has no monolithic ``Kelvin`` class.  Its steady solver consumes a
dolfinx ``MeshData``, an object whose ``all()`` method returns tagged Regions,
and a chip-like object carrying boundary conditions.  This module builds those
objects from the manifest and MSH written by ``read_gds.ipynb``.

The adapter deliberately does not guess process materials, heat sources, or a
heat-sink boundary.  Those are physical inputs that GDS does not contain.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Optional

import numpy as np
from dolfinx.io import XDMFFile
from dolfinx.io.gmsh import read_from_msh
from dolfinx.mesh import meshtags
from mpi4py import MPI

from mesh.boxes import Region
from mesh.build import FACET_BOTTOM, FACET_TOP
from spec.chip import BoundaryConditions
from spec.materials import MATERIALS


@dataclass(frozen=True)
class ImportedHeatSource:
    """Heat in an arbitrary imported volume, without invented box dimensions.

    The original two-argument density constructor remains valid. Sources
    created by ``load_tagged_msh`` additionally carry a manifest-derived
    volume and intended power so Kelvin's source-budget checks can audit them.
    """

    name: str
    power_density_w_per_m3: float
    volume_m3: Optional[float] = None
    intended_power_w: Optional[float] = None
    device: Optional[str] = None
    kind: str = "imported"

    def __post_init__(self):
        if not math.isfinite(self.power_density_w_per_m3) or self.power_density_w_per_m3 < 0:
            raise ValueError(f"{self.name}: power density must be finite and non-negative")
        if self.device is None:
            object.__setattr__(self, "device", self.name)
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError(f"{self.name}: source device must be a nonempty string")
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ValueError(f"{self.name}: source kind must be a nonempty string")
        if self.volume_m3 is not None and (not math.isfinite(self.volume_m3) or self.volume_m3 <= 0):
            raise ValueError(f"{self.name}: source volume must be finite and positive")
        if self.intended_power_w is not None:
            if not math.isfinite(self.intended_power_w) or self.intended_power_w < 0:
                raise ValueError(f"{self.name}: source power must be finite and non-negative")
            if self.volume_m3 is None:
                raise ValueError(f"{self.name}: intended power requires a source volume")
            if not math.isclose(self.power_density_w_per_m3 * self.volume_m3,
                                self.intended_power_w, rel_tol=1e-12, abs_tol=1e-30):
                raise ValueError(f"{self.name}: density times volume does not match intended power")

    @property
    def power_uw(self):
        if self.intended_power_w is not None:
            return self.intended_power_w * 1e6
        if self.volume_m3 is None:
            raise ValueError(f"{self.name}: source volume is required for a power budget")
        return self.power_density_w_per_m3 * self.volume_m3 * 1e6


class ImportedRegionRegistry:
    """Kelvin registry interface that preserves explicit Gmsh physical tags."""

    def __init__(self, regions):
        self._regions = tuple(regions)
        tags = [region.tag_id for region in self._regions]
        if len(tags) != len(set(tags)):
            raise ValueError("Imported physical volume tags must be unique")

    def all(self):
        return list(self._regions)


@dataclass
class GDSKelvinCase:
    mesh_data: object
    registry: ImportedRegionRegistry
    chip: object
    manifest: dict
    report: dict

    def assert_solver_ready(self, require_sources: bool = True):
        """Fail before PETSc if the imported thermal problem is under-specified."""
        problems = []
        if self.report["unanchored_component_count"]:
            problems.append(
                f"{self.report['unanchored_component_count']} connected component(s) do not "
                "touch the Kelvin backside boundary tag"
            )
        if require_sources and not self.report["source_region_tags"]:
            problems.append("no volumetric heat-source region was assigned")
        if problems:
            raise RuntimeError("GDS mesh is not Kelvin-solve-ready: " + "; ".join(problems))
        return self


def material_mapping_template(manifest_path):
    """Return the masks that need Kelvin material names, without guessing them."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return {
        (int(region["gds_layer"]), int(region["datatype"])): None
        for region in manifest["regions"]
        if region["material"] not in MATERIALS
    }


def _resolve_material(region, material_by_tag, material_by_mask):
    tag = int(region["physical_tag"])
    mask = (int(region["gds_layer"]), int(region["datatype"]))
    candidates = (
        ("physical-tag override", material_by_tag.get(tag)),
        ("mask override", material_by_mask.get(mask)),
        ("manifest", region.get("material")),
    )
    for source, material in candidates:
        if material is None:
            continue
        if material not in MATERIALS:
            raise KeyError(
                f"{source} material {material!r} for physical tag {tag} is not in Kelvin "
                f"MATERIALS; known names are {sorted(MATERIALS)}"
            )
        return material
    raise KeyError(
        f"No Kelvin material supplied for physical tag {tag}, mask L{mask[0]}/D{mask[1]}"
    )


def _manifest_source_volume_m3(region):
    """Exact prism volume from declared polygon areas, not bounding boxes."""
    volumes = region.get("volumes")
    if not volumes:
        raise ValueError(f"Source region {region['name']!r} needs manifest volumes with area_um2")
    areas = []
    for volume in volumes:
        area = float(volume["area_um2"])
        if not math.isfinite(area) or area <= 0:
            raise ValueError(f"Source region {region['name']!r} area must be finite and positive")
        areas.append(area)
    thickness_um = float(region["z_max_um"]) - float(region["z_min_um"])
    volume_m3 = math.fsum(areas) * thickness_um * 1e-18
    if not math.isfinite(volume_m3) or volume_m3 <= 0:
        raise ValueError(f"Source region {region['name']!r} volume must be finite and positive")
    return volume_m3


def _source_volume_report(mesh_data, registry):
    """Compare independently declared source volumes with tagged tetrahedra."""
    import ufl
    from dolfinx.fem import assemble_scalar, form

    mesh = mesh_data.mesh
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=mesh_data.cell_tags)
    report = {}
    for region in registry.all():
        if region.source is None:
            continue
        nominal = region.source.volume_m3
        meshed = float(mesh.comm.allreduce(
            assemble_scalar(form(1.0 * dx(region.tag_id))), op=MPI.SUM))
        if not math.isclose(meshed, nominal, rel_tol=1e-6, abs_tol=1e-30):
            raise ValueError(
                f"Source tag {region.tag_id} ({region.source.device}): meshed volume "
                f"{meshed:.12g} m^3 does not match manifest volume {nominal:.12g} m^3"
            )
        report[region.tag_id] = {
            "device": region.source.device, "kind": region.source.kind,
            "nominal_m3": nominal, "meshed_m3": meshed, "ratio": meshed / nominal,
            "intended_power_w": region.source.power_uw * 1e-6,
            "power_density_w_per_m3": region.source.power_density_w_per_m3,
        }
    return report


def _physical_group_tag(mesh_data, name, fallback):
    group = mesh_data.physical_groups.get(name)
    if group is None:
        return int(fallback)
    if int(group.dim) != 2:
        raise ValueError(f"Physical group {name!r} is dimension {group.dim}, expected 2")
    return int(group.tag)


def _remap_boundary_tags(mesh_data, manifest, robin_region_tags, top_region_tags):
    if mesh_data.facet_tags is None:
        raise ValueError("Imported MSH has no physical facet tags")
    values = np.array(mesh_data.facet_tags.values, copy=True)
    selected_bottom, selected_top = [], []
    by_tag = {int(region["physical_tag"]): region for region in manifest["regions"]}

    for region_tag in sorted(robin_region_tags):
        region = by_tag[region_tag]
        name = f"{region['name']}_external_bottom"
        old_tag = _physical_group_tag(mesh_data, name, 10000 + region_tag)
        if not np.any(values == old_tag):
            raise ValueError(f"Selected Kelvin backside group {name!r} has no mesh facets")
        values[values == old_tag] = FACET_BOTTOM
        selected_bottom.append({"name": name, "original_tag": old_tag})

    for region_tag in sorted(top_region_tags):
        region = by_tag[region_tag]
        name = f"{region['name']}_external_top"
        old_tag = _physical_group_tag(mesh_data, name, 20000 + region_tag)
        if not np.any(values == old_tag):
            continue
        values[values == old_tag] = FACET_TOP
        selected_top.append({"name": name, "original_tag": old_tag})

    if not np.any(values == FACET_BOTTOM):
        raise ValueError("No facets were mapped to Kelvin FACET_BOTTOM=101")
    remapped = meshtags(
        mesh_data.mesh,
        mesh_data.facet_tags.dim,
        np.array(mesh_data.facet_tags.indices, copy=True),
        values,
    )
    remapped.name = "facet_tags"
    mesh_data = mesh_data._replace(facet_tags=remapped)
    return mesh_data, selected_bottom, selected_top


def _connectivity_report(mesh_data):
    mesh = mesh_data.mesh
    if mesh.comm.size != 1:
        raise NotImplementedError(
            "Connected-component validation currently requires a serial import; "
            "load with MPI.COMM_SELF, validate, then partition for production"
        )
    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_connectivity(fdim, tdim)
    facet_to_cell = mesh.topology.connectivity(fdim, tdim)
    num_cells = mesh.topology.index_map(tdim).size_local
    num_facets = mesh.topology.index_map(fdim).size_local

    parent = np.arange(num_cells, dtype=np.int64)

    def find(cell):
        while parent[cell] != cell:
            parent[cell] = parent[parent[cell]]
            cell = parent[cell]
        return int(cell)

    def union(a, b):
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for facet in range(num_facets):
        cells = facet_to_cell.links(facet)
        if len(cells) == 2:
            union(int(cells[0]), int(cells[1]))

    roots = {find(cell) for cell in range(num_cells)}
    anchored = set()
    for facet, value in zip(mesh_data.facet_tags.indices, mesh_data.facet_tags.values):
        if int(value) != FACET_BOTTOM:
            continue
        for cell in facet_to_cell.links(int(facet)):
            anchored.add(find(int(cell)))
    return {
        "connected_component_count": len(roots),
        "anchored_component_count": len(roots & anchored),
        "unanchored_component_count": len(roots - anchored),
    }


def load_tagged_msh(
    msh_path,
    manifest_path,
    *,
    material_by_tag: Optional[Mapping[int, str]] = None,
    material_by_mask: Optional[Mapping[tuple[int, int], str]] = None,
    source_density_by_tag: Optional[Mapping[int, float]] = None,
    source_power_by_tag: Optional[Mapping[int, float]] = None,
    source_device_by_tag: Optional[Mapping[int, str]] = None,
    source_kind_by_tag: Optional[Mapping[int, str]] = None,
    allow_non_candidate_sources: bool = False,
    robin_region_tags=None,
    top_region_tags=None,
    bcs: Optional[BoundaryConditions] = None,
    comm=MPI.COMM_SELF,
):
    """Load the notebook's MSH as a Kelvin-ready object graph.

    By default, only region bottoms at the global minimum z are mapped to
    Kelvin's backside tag 101.  This avoids the physically incorrect shortcut
    of treating every exposed lower metal face as a heat sink.

    ``source_power_by_tag`` supplies independently prescribed WATTS per
    physical tag. The density is power divided by the manifest's polygon
    area times thickness; the tagged mesh volume must match that declaration.
    ``source_device_by_tag`` associates every assigned tag with an instance
    such as X0. Explicit device IDs default to kind="channel"; other sources
    default to kind="imported". ``source_kind_by_tag`` can override the kind.
    Density-only inputs remain supported; their intended powers are derived
    from density times manifest volume, not independent electrical budgets.
    Power and density may be assigned to different tags, never the same tag.
    """
    msh_path, manifest_path = Path(msh_path), Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "gds-material-regions-v2":
        raise ValueError(f"Unsupported manifest schema: {manifest.get('schema')!r}")
    scale_to_m = float(manifest.get("geometry_length_scale_to_m", 1e-6))
    if not math.isfinite(scale_to_m) or scale_to_m <= 0:
        raise ValueError("geometry_length_scale_to_m must be finite and positive")
    if "boolean_grid_um" in manifest:
        boolean_grid_um = float(manifest["boolean_grid_um"])
        if not math.isfinite(boolean_grid_um) or boolean_grid_um <= 0:
            raise ValueError("boolean_grid_um must be finite and positive")
    for region in manifest["regions"]:
        z_min_um = float(region["z_min_um"])
        z_max_um = float(region["z_max_um"])
        if not (math.isfinite(z_min_um) and math.isfinite(z_max_um)):
            raise ValueError(f"Region {region.get('name')!r} has non-finite z extents")
        if z_max_um <= z_min_um:
            raise ValueError(f"Region {region.get('name')!r} must have z_max_um > z_min_um")

    material_by_tag = {int(k): v for k, v in (material_by_tag or {}).items()}
    material_by_mask = {
        (int(layer), int(datatype)): value
        for (layer, datatype), value in (material_by_mask or {}).items()
    }
    source_density_by_tag = {
        int(k): float(v) for k, v in (source_density_by_tag or {}).items()
    }
    source_power_by_tag = {
        int(k): float(v) for k, v in (source_power_by_tag or {}).items()
    }
    duplicate_sources = set(source_density_by_tag) & set(source_power_by_tag)
    if duplicate_sources:
        raise ValueError(f"Source tags have both density and power assignments: {sorted(duplicate_sources)}")
    source_tags = set(source_density_by_tag) | set(source_power_by_tag)
    explicit_devices = source_device_by_tag is not None
    source_device_by_tag = {int(k): v for k, v in (source_device_by_tag or {}).items()}
    if explicit_devices and set(source_device_by_tag) != source_tags:
        raise ValueError("source_device_by_tag must identify every assigned source tag and no others")
    if any(not isinstance(value, str) or not value.strip() for value in source_device_by_tag.values()):
        raise ValueError("Source device IDs must be nonempty strings")
    if len(set(source_device_by_tag.values())) != len(source_device_by_tag):
        raise ValueError("Source device IDs must be unique across physical tags")
    source_kind_by_tag = {int(k): v for k, v in (source_kind_by_tag or {}).items()}
    if set(source_kind_by_tag) - source_tags:
        raise ValueError("source_kind_by_tag contains tags without heat assignments")
    if any(not math.isfinite(value) or value < 0 for value in source_power_by_tag.values()):
        raise ValueError("Source powers must be finite and non-negative")

    manifest_tags = {int(region["physical_tag"]) for region in manifest["regions"]}
    unknown_sources = sorted(source_tags - manifest_tags)
    if unknown_sources:
        raise KeyError(f"Heat-source tags absent from manifest: {unknown_sources}")
    declared_candidates = manifest.get("thermal_source_candidate_tags")
    region_flag_candidates = {
        int(region["physical_tag"])
        for region in manifest["regions"]
        if region.get("thermal_source_candidate") is True
    }
    region_candidate_flags_available = any(
        "thermal_source_candidate" in region for region in manifest["regions"]
    )
    candidate_policy_available = declared_candidates is not None or region_candidate_flags_available
    candidate_source_tags = (
        {int(tag) for tag in declared_candidates}
        if declared_candidates is not None else region_flag_candidates
    )
    unknown_candidates = sorted(candidate_source_tags - manifest_tags)
    if unknown_candidates:
        raise ValueError(f"Manifest source-candidate tags are not volume tags: {unknown_candidates}")
    if (
        declared_candidates is not None
        and region_candidate_flags_available
        and region_flag_candidates != candidate_source_tags
    ):
        raise ValueError(
            "Manifest thermal_source_candidate_tags disagree with per-region candidate flags"
        )
    role_source_tags = {
        int(region["physical_tag"])
        for region in manifest["regions"]
        if region.get("role") == "source"
    }
    if role_source_tags and not candidate_source_tags.issubset(role_source_tags):
        raise ValueError("Manifest source-candidate tags include a region whose role is not 'source'")
    if candidate_policy_available and not allow_non_candidate_sources:
        non_candidate_sources = sorted(source_tags - candidate_source_tags)
        if non_candidate_sources:
            raise ValueError(
                "Heat was assigned to non-candidate tags "
                f"{non_candidate_sources}; pass allow_non_candidate_sources=True only for "
                "intentional sources such as mapped interconnect Joule loss"
            )

    imported_regions = []
    for region in sorted(manifest["regions"], key=lambda item: int(item["physical_tag"])):
        tag = int(region["physical_tag"])
        material = _resolve_material(region, material_by_tag, material_by_mask)
        source = None
        if tag in source_tags:
            volume_m3 = _manifest_source_volume_m3(region)
            intended_power_w = source_power_by_tag.get(tag)
            density = (intended_power_w / volume_m3 if tag in source_power_by_tag
                       else source_density_by_tag[tag])
            source = ImportedHeatSource(
                region["name"], density, volume_m3=volume_m3,
                intended_power_w=intended_power_w,
                device=source_device_by_tag.get(tag, region["name"]),
                kind=source_kind_by_tag.get(tag, "channel" if tag in source_device_by_tag else "imported"),
            )
        imported_regions.append(Region(tag, region["name"], material, source))
    registry = ImportedRegionRegistry(imported_regions)

    mesh_data = read_from_msh(msh_path, comm, rank=0, gdim=3)
    mesh_data.mesh.geometry.x[:] *= scale_to_m
    mesh_data.cell_tags.name = "cell_tags"

    observed_cell_tags = set(int(value) for value in np.unique(mesh_data.cell_tags.values))
    if observed_cell_tags != manifest_tags:
        raise ValueError(
            f"MSH volume tags {sorted(observed_cell_tags)} do not match manifest "
            f"{sorted(manifest_tags)}"
        )
    num_cells = mesh_data.mesh.topology.index_map(mesh_data.mesh.topology.dim).size_local
    if len(mesh_data.cell_tags.indices) != num_cells:
        raise ValueError(
            f"Only {len(mesh_data.cell_tags.indices)} of {num_cells} cells have material tags"
        )
    source_volumes = _source_volume_report(mesh_data, registry)

    global_z_min = min(float(region["z_min_um"]) for region in manifest["regions"])
    global_z_max = max(float(region["z_max_um"]) for region in manifest["regions"])
    tolerance_um = max(float(manifest.get("boolean_grid_um", 0.0)), 1e-9)
    if robin_region_tags is None:
        robin_region_tags = {
            int(region["physical_tag"])
            for region in manifest["regions"]
            if abs(float(region["z_min_um"]) - global_z_min) <= tolerance_um
        }
    else:
        robin_region_tags = {int(tag) for tag in robin_region_tags}
    if top_region_tags is None:
        top_region_tags = {
            int(region["physical_tag"])
            for region in manifest["regions"]
            if abs(float(region["z_max_um"]) - global_z_max) <= tolerance_um
        }
    else:
        top_region_tags = {int(tag) for tag in top_region_tags}
    for label, tags in (("robin_region_tags", robin_region_tags), ("top_region_tags", top_region_tags)):
        unknown = sorted(tags - manifest_tags)
        if unknown:
            raise KeyError(f"{label} contains tags absent from manifest: {unknown}")

    mesh_data, bottom_groups, top_groups = _remap_boundary_tags(
        mesh_data, manifest, robin_region_tags, top_region_tags
    )
    connectivity = _connectivity_report(mesh_data)
    chip = SimpleNamespace(bcs=bcs or BoundaryConditions())
    report = {
        "source_msh": str(msh_path),
        "source_manifest": str(manifest_path),
        "length_scale_to_m": scale_to_m,
        "cell_count": num_cells,
        "material_region_tags": sorted(manifest_tags),
        "source_region_tags": sorted(source_tags),
        "source_power_input_tags": sorted(source_power_by_tag),
        "source_density_input_tags": sorted(source_density_by_tag),
        "source_volumes": source_volumes,
        "kelvin_bottom_groups": bottom_groups,
        "kelvin_top_groups": top_groups,
        **connectivity,
    }
    return GDSKelvinCase(mesh_data, registry, chip, manifest, report)


def write_xdmf(case: GDSKelvinCase, path):
    """Write SI-scaled mesh/cell/facet tags for ParaView or later Kelvin runs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with XDMFFile(case.mesh_data.mesh.comm, path, "w") as xdmf:
        xdmf.write_mesh(case.mesh_data.mesh)
        xdmf.write_meshtags(case.mesh_data.cell_tags, case.mesh_data.mesh.geometry)
        xdmf.write_meshtags(case.mesh_data.facet_tags, case.mesh_data.mesh.geometry)
    return path
