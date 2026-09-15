"""Slide 6a: what the bitcell physically IS -- three views of the same real
geometry, no interpretation layered on top.

  fig6a_layout.png             The GDS as a layout viewer draws it: every mask
                               layer, real polygons, nothing annotated.
  fig6a_stack_3d.svg / .png    Vector isometric of the full layer stack at real
                               SKY130 z-heights, with the 2D mask plan beside
                               it (mesh/gds_svg_viz.py).
  fig6a_cross_section_mesh.png A vertical cut through the actual tetrahedral
                               mesh, coloured by material -- the geometry the
                               solver sees, with its own element edges.

    python cases/fig_bitcell_geometry.py [--refine 1.0] [--skip-mesh]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gdstk
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from gds import techmap
from gds.read import flatten_by_layer
from mesh.gds_svg_viz import make_geometry_svg

OUT = Path("out/presentation")
GDS_PATH = "data/sram22_64x22m4w22.gds"
CELL_NAME = "sram_sp_cell"

# Draw order is bottom-up through the real stack, so upper metals overlay
# lower ones the way a layout viewer shows them. Colours follow
# mesh/gds_svg_viz.py's palette so the 2D layout and the 3D isometric can be
# read against each other.
# Nine mask layers overlap almost everywhere in a 1.2x1.6um cell, so plain
# alpha blending turns the whole plot to mud (tried it). Real layout viewers
# solve this with stipple patterns: solid fills for the FEOL layers at the
# bottom of the stack, hatched/transparent fills for the routing layers above,
# so lower layers stay readable through upper ones.
#   (name, gds key, face, edge, alpha, hatch)
NWELL = (64, 20)
LAYERS = [
    ("nwell",  NWELL,           "#e8f0f8", "#6b8fb5", 0.90, None),
    ("diff",   techmap.DIFF,    "#7fc08a", "#2f6b3d", 0.95, None),
    ("poly",   techmap.POLY,    "#c9a0dc", "#5d3a75", 0.85, None),
    ("licon1", techmap.LICON1,  "#1a1a1a", "#000000", 1.00, None),
    ("li1",    techmap.LI1,     "none",    "#c0392b", 1.00, "///"),
    ("mcon",   techmap.MCON,    "#1a1a1a", "#000000", 1.00, None),
    ("met1",   techmap.MET1,    "none",    "#c8860d", 1.00, "\\\\\\"),
    ("via",    techmap.VIA,     "#1a1a1a", "#000000", 1.00, None),
    ("met2",   techmap.MET2,    "none",    "#2f57a8", 1.00, "..."),
]


def _cell():
    lib = gdstk.read_gds(GDS_PATH)
    cell = next(c for c in lib.cells if c.name == CELL_NAME)
    return cell, flatten_by_layer(cell)


def fig_layout():
    cell, by_layer = _cell()
    (x0, y0), (x1, y1) = cell.bounding_box()

    fig, ax = plt.subplots(figsize=(6.4, 8.0))
    present = []
    for name, key, fc, ec, alpha, hatch in LAYERS:
        polys = by_layer.get(key, [])
        if not polys:
            continue
        present.append((name, fc, ec, alpha, hatch))
        for p in polys:
            ax.add_patch(mpatches.Polygon(p.points, closed=True, facecolor=fc,
                                          edgecolor=ec, lw=1.0, alpha=alpha,
                                          hatch=hatch))

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.legend(handles=[mpatches.Patch(facecolor=fc, edgecolor=ec, alpha=a,
                                      hatch=h, label=n)
                       for n, fc, ec, a, h in present],
              fontsize=8.5, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    fig.savefig(OUT / "fig6a_layout.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/fig6a_layout.png  ({len(present)} layers)")


def fig_stack_svg():
    cell, by_layer = _cell()
    (x0, y0), (x1, y1) = cell.bounding_box()
    svg = make_geometry_svg(by_layer, techmap.FRONTSIDE_STACK,
                            source_name=Path(GDS_PATH).name, top_cell_name=CELL_NAME,
                            window=(x0, x1, y0, y1))
    svg_path = OUT / "fig6a_stack_3d.svg"
    svg_path.write_text(svg)
    print(f"wrote {svg_path}")
    try:
        import cairosvg
        cairosvg.svg2png(bytestring=svg.encode(), write_to=str(OUT / "fig6a_stack_3d.png"),
                         scale=1.8)
        print(f"wrote {OUT}/fig6a_stack_3d.png")
    except ImportError:
        print("  (cairosvg not available -- SVG written, PNG skipped)")


def fig_cross_section(refine):
    """Vertical cut through the real tetrahedral mesh.

    Built from the dolfinx mesh + cell_tags rather than from the GDS
    polygons, deliberately: the point of this figure is that the SOLVER's
    discretisation resolves the stack, so it has to show real elements, not a
    redrawn schematic of them.
    """
    import pyvista as pv
    pv.OFF_SCREEN = True

    from mesh.gds_build import build_gds_3d_mesh
    from mesh.viz import MATERIAL_COLORS

    cell, by_layer = _cell()
    (x0, y0), (x1, y1) = cell.bounding_box()
    mesh_data, registry = build_gds_3d_mesh(
        by_layer, (x0, x1, y0, y1), 1e-6, out_dir="out/bitcell_xsec",
        refine=refine, renders=False, stack=techmap.FRONTSIDE_STACK)

    mesh = mesh_data.mesh
    n_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    conn = mesh.geometry.dofmap.reshape(n_cells, 4)
    cells = np.hstack([np.full((n_cells, 1), 4, dtype=np.int64), conn]).ravel()
    grid = pv.UnstructuredGrid(cells, np.full(n_cells, 10, dtype=np.uint8),
                               mesh.geometry.x * 1e6)

    # tag_id -> material -> colour index. Cells with no tag keep -1 and are
    # dropped, rather than silently inheriting a neighbour's colour.
    mat_by_tag = {r.tag_id: r.material for r in registry.all()}
    palette = sorted(set(mat_by_tag.values()))
    idx_of = {m: i for i, m in enumerate(palette)}
    mat_idx = np.full(n_cells, -1, dtype=np.int32)
    ct = mesh_data.cell_tags
    in_range = ct.indices < n_cells
    mat_idx[ct.indices[in_range]] = [idx_of[mat_by_tag[v]]
                                     for v in ct.values[in_range]]
    grid.cell_data["material_idx"] = mat_idx
    grid = grid.threshold(-0.5, scalars="material_idx")

    # Cut through a DEVICE row, not the geometric middle: the middle of this
    # cell falls between transistors, so the cut shows interconnect only and
    # no gate stack at all. Pick the y of a real channel polygon instead.
    from gds.sources import _dims_um, extract_channels
    chans = [_dims_um(p) for p in extract_channels(by_layer)]
    y_mid = min((c for c in chans if c[3] > 0.05),
                key=lambda c: abs(c[1] - (y0 + y1) / 2))[1]
    sliced = grid.slice(normal="y", origin=(0, y_mid, 0))
    z_bot = techmap.z_bounds(techmap.FRONTSIDE_STACK)[1][0]   # top of Si_substrate
    z_top = max(z1 for _, z1, _ in techmap.z_bounds(techmap.FRONTSIDE_STACK))

    # Draw the slice's own triangles in matplotlib rather than screenshotting
    # the pyvista render: a screenshot letterboxes a tall-narrow slice inside
    # the window, so pasting it under an (x, z) extent puts the geometry at
    # the wrong coordinates (it landed ~5x too narrow, at the wrong x).
    # Here every triangle is placed at its real (x, z).
    from matplotlib.collections import PolyCollection

    tri = sliced.triangulate()
    faces = tri.faces.reshape(-1, 4)[:, 1:]
    pts = tri.points
    verts = pts[faces][:, :, [0, 2]]        # (x, z) only -- the cut plane
    idx = np.asarray(tri.cell_data["material_idx"], dtype=int)
    colors = [MATERIAL_COLORS.get(palette[i], "#999999") for i in idx]

    # Two panels: the whole stack, and a zoom on the FEOL. The gate stack is
    # ~0.2um inside a ~3um stack, so at full scale the channel -- the thing
    # that actually generates the heat -- is a sub-pixel sliver.
    z_feol = techmap.z_bounds(techmap.FRONTSIDE_STACK)[3][1]   # top of poly
    fig, (ax, axz) = plt.subplots(1, 2, figsize=(15.5, 6.6),
                                  gridspec_kw={"width_ratios": [1.45, 1.0]})
    for a, (zlo, zhi), title in (
            (ax, (z_bot - 0.12, z_top + 0.03), "Full stack"),
            (axz, (z_bot - 0.06, z_feol + 0.05), "FEOL zoom — gate, channel, contacts")):
        a.add_collection(PolyCollection(verts, facecolors=colors,
                                        edgecolors="#33333366", linewidths=0.25))
        a.set_xlim(x0, x1)
        a.set_ylim(zlo, zhi)
        a.set_xlabel("x (µm)")
        a.set_ylabel("z (µm)")
        a.set_title(title, fontsize=12)

    used = sorted({mat_by_tag[v] for v in ct.values[in_range]})
    fig.legend(handles=[mpatches.Patch(facecolor=MATERIAL_COLORS.get(m, "#999999"),
                                       edgecolor="k", label=m) for m in used],
               fontsize=9.5, ncol=min(len(used), 9), loc="lower center",
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Cross section through the tetrahedral mesh at y = {y_mid:.3f} µm — "
                 f"{n_cells:,} elements, real SKY130 layer heights", fontsize=13)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "fig6a_cross_section_mesh.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/fig6a_cross_section_mesh.png  ({n_cells:,} cells, "
          f"{len(faces):,} triangles in the cut)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--refine", type=float, default=1.0)
    p.add_argument("--skip-mesh", action="store_true")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    fig_layout()
    fig_stack_svg()
    if not args.skip_mesh:
        fig_cross_section(args.refine)


if __name__ == "__main__":
    main()
