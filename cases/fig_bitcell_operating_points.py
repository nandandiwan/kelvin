"""Slide 6b/6c/6d: steady-state temperature of the real bitcell at four named
operating points, in 2D and in 3D.

Deliberately NOT the same presentation as out/bitcell_gallery/gallery.png.
That figure gives every panel its own auto-scaled colour bar, which makes
HOLD's 2.3e-5 K of essentially-flat field look exactly as structured as
CROWBAR's 31 K -- visually implying the four points are comparable when they
span six orders of magnitude. Three separate figures instead, each honest
about one thing:

  fig6b_operating_points_2d.png  ONE shared colour scale across all four
        points. The real message ("HOLD and WRITE are thermally nothing;
        READ and CROWBAR are everything") is the figure's most obvious
        feature rather than something the caption has to disclaim.
  fig6c_device_detail.png        The two points that actually heat, coloured
        by LOCAL EXCESS above each plane's own floor. Per-device structure is
        a small ripple on a large near-uniform pedestal (the pedestal is set
        by the Robin sink under the whole 1.9um^2 footprint, not by
        conduction), so it is invisible on an absolute scale -- subtracting
        the floor is what makes individual transistors visible at all.
  fig6d_hotspot_3d.png           A cut-away 3D view: the hotspots are in the
        channel layer, and the heat leaves downward through the substrate.

    python cases/fig_bitcell_operating_points.py [--refine 1.0] [--plot-only]
"""

import argparse
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from cases.run_bitcell_gallery import (
    OUT_DIR as GALLERY_DIR, build_bitcell_geometry, channel_sources_for,
)
from gds import techmap
from gds.bitcell_mapping import mapping_revision
from gds.sources import _dims_um, extract_contacts
from gds.spice_power import NAMED_BIAS_POINTS, named_bias_point_power_w
from mesh.gds_build import build_gds_3d_mesh
from physics.coeffs import build_coeffs
from post.budget import power_balance, print_power_balance, verify_device_source_powers
from post.metrics import tmax
from solve.steady import solve_steady_from_fields
from spec.chip import BoundaryConditions, SourceBox

OUT = Path("out/presentation")
CACHE = Path("out/bitcell_oppoints")
POINTS = ["hold_1", "write_1_settled", "read_1", "crowbar"]
POINT_LABEL = {
    "hold_1": "HOLD (idle)",
    "write_1_settled": "WRITE (settled)",
    "read_1": "READ",
    "crowbar": "CROWBAR (worst case)",
}
HOT_POINTS = ["read_1", "crowbar"]


def solve_point(name, refine):
    """One operating point -> (dof coords in um, dT array, Tmax, hot coords).

    A fresh mesh per point, as cases/run_bitcell_gallery.py does: the geometry
    is identical but k/q must stay registry-consistent with the mesh that
    produced them, and a build is cheap at this scale."""
    by_layer, window, channel_info, contacts = build_bitcell_geometry()
    contact_sources = [
        (SourceBox(device=f"co{i}", kind="contact", x_um=_dims_um(p)[0], y_um=_dims_um(p)[1],
                   z0_um=0.0, w_um=_dims_um(p)[2], l_um=_dims_um(p)[3],
                   t_um=techmap.LICON1_THICKNESS_UM, power_uw=0.0), p)
        for i, p in enumerate(contacts)
    ]

    device_power_w = named_bias_point_power_w(name, row_hit_rate=1.0)
    channel_sources = channel_sources_for(channel_info, device_power_w)
    total_power_w = max(sum(b.power_uw for b, _ in channel_sources) * 1e-6, 1e-18)

    mesh_data, registry = build_gds_3d_mesh(
        by_layer, window, total_power_w, out_dir=f"{CACHE}/{name}", refine=refine,
        renders=False, stack=techmap.FRONTSIDE_STACK,
        channel_sources=channel_sources, contact_sources=contact_sources)
    k_f, rho_cp_f, q_f = build_coeffs(mesh_data.mesh, mesh_data.cell_tags, registry,
                                      source_depth_m=None)
    verify_device_source_powers(mesh_data, registry, q_f, device_power_w)
    chip = types.SimpleNamespace(
        bcs=BoundaryConditions(ambient_t_k=300.0, backside_h_eff=20000.0, top_h_eff=None))
    T = solve_steady_from_fields(mesh_data, k_f, q_f, chip)
    print_power_balance(power_balance(mesh_data, registry, chip, T, k_f, q_f,
                                      source_depth_m=None))
    tk, coords = tmax(T)
    print(f"{name}: Tmax={tk:.6f} K  dT={tk-300.0:.6e} K  at z={coords[2]*1e6:.4f} um")

    dof = T.function_space.tabulate_dof_coordinates() * 1e6
    return {"dof_um": dof, "dT": T.x.array - 300.0, "tmax_k": tk,
            "hot_um": np.asarray(coords) * 1e6,
            "total_power_w": total_power_w,
            "source_mapping_sha256": mapping_revision()}


def _plane(res, z_um):
    """dofs lying exactly on one z plane. Uses an exact match against the hot
    dof's own z rather than a tolerance band: the mesh is layered, so a band
    would silently mix two z levels and average away the structure."""
    m = np.isclose(res["dof_um"][:, 2], z_um, atol=1e-7)
    return res["dof_um"][m, 0], res["dof_um"][m, 1], res["dT"][m]


def fig_magnitude(results):
    """How much each operating point heats -- as numbers, not as maps.

    Four temperature MAPS on a shared colour scale was the first attempt and
    is the wrong figure: the four points span six orders of magnitude, so the
    two cold ones render as flat black and the two hot ones as flat yellow --
    four featureless rectangles. Worse, giving each its own scale (what
    out/bitcell_gallery/gallery.png does) makes HOLD's field look structured
    when it is not: measured directly, HOLD's and WRITE's spatial structure
    have the SAME magnitude (1.3480e-6 vs 1.3482e-6 K) and are 99.4%
    correlated, even though their dominant source differs (X0 vs X1) and
    their total power differs 10x. That is the linear solver's absolute
    convergence floor, not silicon. So: magnitudes here, spatial structure in
    fig6c for the only two points that have any.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.8))
    x = np.arange(len(POINTS))
    colors = ["#9e9e9e", "#4c9f70", "#1f77b4", "#d62728"]

    dt = [float(results[n]["dT"].max()) for n in POINTS]
    pw = [float(results[n]["total_power_w"]) for n in POINTS]

    for ax, vals, lab, unit, scale, title in (
            (ax1, dt, "steady-state $\\Delta T_{max}$ above ambient", "K", 1.0,
             "Temperature rise"),
            (ax2, pw, "total channel power into the cell", "µW", 1e6,
             "Electrical power")):
        ax.bar(x, [v * scale for v in vals], 0.62, color=colors)
        for xi, v in zip(x, vals):
            ax.text(xi, v * scale * 1.8, f"{v*scale:.3g}", ha="center", fontsize=10,
                    fontweight="bold")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([POINT_LABEL[n].replace(" (", "\n(") for n in POINTS], fontsize=9.5)
        ax.set_ylabel(f"{lab}  ({unit})")
        ax.set_title(title, fontsize=13)
        # Explicit decade ticks: matplotlib's automatic log locator drops
        # labels unpredictably across a 7-decade span, leaving gaps that read
        # as missing data.
        lo = min(v * scale for v in vals)
        hi = max(v * scale for v in vals)
        e0, e1 = int(np.floor(np.log10(lo))) - 1, int(np.ceil(np.log10(hi))) + 1
        ax.set_yticks([10.0 ** e for e in range(e0, e1 + 1)])
        ax.set_ylim(10.0 ** e0, 10.0 ** (e1 + 1))
        ax.grid(axis="y", alpha=0.3, which="major")

    fig.suptitle("Steady state at four operating points — same geometry and mesh, "
                 "only the per-device power differs", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "fig6b_operating_points_2d.png", dpi=160)
    plt.close(fig)
    print(f"wrote {OUT}/fig6b_operating_points_2d.png")


def fig_device_detail(results, channel_info):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))

    for ax, name in zip(axes, HOT_POINTS):
        res = results[name]
        x, y, t = _plane(res, res["hot_um"][2])
        # "Baseline" = the coolest point in this plane. The whole cell sits on a
        # near-uniform pedestal set by the Robin sink under its 1.9um^2
        # footprint; device-to-device variation is a few tens of mK on top of
        # that. Ambient is 300 K -- the baseline is already tens of K above it.
        excess = t - t.min()
        tpc = ax.tripcolor(mtri.Triangulation(x, y), excess * 1e3, shading="gouraud",
                           cmap="inferno")
        cb = fig.colorbar(tpc, ax=ax, shrink=0.85)
        cb.set_label("$T - T_{\\mathrm{baseline}}$  (mK)")

        # Outline each real channel and label it with its own peak, so the
        # per-device numbers are readable even where the colour ramp is subtle.
        for channel in channel_info:
            cx, cy, x_ext, y_ext = _dims_um(channel.polygon)
            cls = channel.device_class
            ax.add_patch(plt.Rectangle((cx - x_ext / 2, cy - y_ext / 2), x_ext, y_ext,
                                       fill=False, ec="#39d3ff", lw=1.0, zorder=5))
            inside = ((np.abs(x - cx) <= x_ext / 2) & (np.abs(y - cy) <= y_ext / 2))
            if inside.any() and cls != "parasitic":
                ax.text(cx, cy, f"{excess[inside].max()*1e3:.1f}", fontsize=7,
                        color="#39d3ff", ha="center", va="center", zorder=6,
                        fontweight="bold")

        ax.set_title(f"{POINT_LABEL[name]}\n"
                     f"$T_{{\\mathrm{{baseline}}}}$ = 300 K + {t.min():.2f} K   |   "
                     f"device spread {(t.max()-t.min())*1e3:.1f} mK", fontsize=10.5)
        ax.set_xlabel("x (µm)")
        ax.set_aspect("equal")
    axes[0].set_ylabel("y (µm)")

    fig.suptitle("Which transistors are hot", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "fig6c_device_detail.png", dpi=160)
    plt.close(fig)
    print(f"wrote {OUT}/fig6c_device_detail.png")


def fig_layers_svg(results):
    """Slide 6d: the same vector layer view as the geometry figure, but with
    every real GDS polygon filled by its own temperature.

    This replaces a pyvista cut-away of the tet field. The cut-away is correct
    but unreadable as a talk figure: it shows two slicing planes through a
    homogeneous-looking block, and you cannot tell a gate from a via in it.
    Colouring the actual LAYER POLYGONS answers the question a reader has
    ("which parts of the cell get hot?") directly, and keeps the 2D mask plan
    and the 3D isometric side by side.

    Each polygon takes its HOTTEST contained dof, not its centroid: a gate
    stripe's centroid can sit between two devices and miss the channel.
    Colour is local excess above the coolest element, on a scale shared by
    both operating points -- the absolute field is a ~31 K pedestal with a
    ~0.02 K ripple on top, so an absolute scale renders every polygon the
    same colour (see fig6b).
    """
    import gdstk
    from cases.render_transient_layers import _dof_indices_per_volume
    from gds.read import flatten_by_layer
    from mesh.gds_svg_viz import build_layer_regions, render_layer_regions_svg

    lib = gdstk.read_gds("data/sram22_64x22m4w22.gds")
    cell = next(c for c in lib.cells if c.name == "sram_sp_cell")
    by_layer = flatten_by_layer(cell)
    (x0, y0), (x1, y1) = cell.bounding_box()
    regions = build_layer_regions(by_layer, techmap.FRONTSIDE_STACK, (x0, x1, y0, y1))

    excess = {}
    for name in HOT_POINTS:
        res = results[name]
        per_volume = _dof_indices_per_volume(regions, res["dof_um"])
        vals = np.array([res["dT"][idx].max() for _, idx in per_volume])
        # in mK -- the spread is tens of mK, and a colour bar reading
        # "0.0000 .. 0.0253 K" is unreadable at slide distance.
        excess[name] = (per_volume, (vals - vals.min()) * 1e3)
    clim = (0.0, float(max(e.max() for _, e in excess.values())))

    for name in HOT_POINTS:
        per_volume, vals = excess[name]
        for (volume, _), value in zip(per_volume, vals):
            volume["value"] = float(value)
        tmax_k = float(results[name]["tmax_k"])
        title = (f"{POINT_LABEL[name]}   —   Tmax = {tmax_k:.3f} K "
                 f"(ambient 300 K)   —   device spread {vals.max():.1f} mK")
        svg = render_layer_regions_svg(
            regions, clim, title,
            scale_label="temperature above the coolest element (mK)")
        tag = name.replace("_1", "").replace("_settled", "")
        (OUT / f"fig6d_layers_{tag}.svg").write_text(svg)
        try:
            import cairosvg
            cairosvg.svg2png(bytestring=svg.encode(),
                             write_to=str(OUT / f"fig6d_layers_{tag}.png"), scale=1.8)
            print(f"wrote {OUT}/fig6d_layers_{tag}.png (+ .svg)")
        except ImportError:
            print(f"wrote {OUT}/fig6d_layers_{tag}.svg  (cairosvg missing, no PNG)")


def fig_3d(refine):
    """Cut-away 3D view of the CROWBAR field. Re-solves rather than reusing the
    cached arrays: pyvista needs the mesh CONNECTIVITY, which the npz cache
    (coordinates + values only) does not carry."""
    import pyvista as pv
    pv.OFF_SCREEN = True

    by_layer, window, channel_info, contacts = build_bitcell_geometry()
    contact_sources = [
        (SourceBox(device=f"co{i}", kind="contact", x_um=_dims_um(p)[0], y_um=_dims_um(p)[1],
                   z0_um=0.0, w_um=_dims_um(p)[2], l_um=_dims_um(p)[3],
                   t_um=techmap.LICON1_THICKNESS_UM, power_uw=0.0), p)
        for i, p in enumerate(contacts)
    ]
    dp = named_bias_point_power_w("crowbar", row_hit_rate=1.0)
    channel_sources = channel_sources_for(channel_info, dp)
    total_power_w = sum(b.power_uw for b, _ in channel_sources) * 1e-6

    mesh_data, registry = build_gds_3d_mesh(
        by_layer, window, total_power_w, out_dir=f"{CACHE}/crowbar3d", refine=refine,
        renders=False, stack=techmap.FRONTSIDE_STACK,
        channel_sources=channel_sources, contact_sources=contact_sources)
    k_f, _, q_f = build_coeffs(mesh_data.mesh, mesh_data.cell_tags, registry,
                               source_depth_m=None)
    verify_device_source_powers(mesh_data, registry, q_f, dp)
    chip = types.SimpleNamespace(
        bcs=BoundaryConditions(ambient_t_k=300.0, backside_h_eff=20000.0, top_h_eff=None))
    T = solve_steady_from_fields(mesh_data, k_f, q_f, chip)

    from mesh.viz3d import _temperature_grid
    grid = _temperature_grid(T)
    tk, coords = tmax(T)
    z_hot = coords[2] * 1e6
    y_mid = (window[2] + window[3]) / 2

    # Two clips at once: cut away half the cell in y to expose the interior,
    # and keep only the few um of stack around the device layer. The full
    # domain is ~50um of substrate under a 0.2um channel, so an unclipped view
    # is a featureless block. `invert=False` KEEPS the specified box (the
    # default, True, keeps everything outside it -- which renders as a thin
    # sliver of the far stack).
    # A solid clipped block shows only its OUTER faces, so the channel layer
    # ends up hidden inside and the view is a featureless slab (tried first).
    # Two orthogonal cutting planes instead -- the classic thermal-cutaway
    # view: one through the channel layer (where the sources are) and one
    # vertical through the hottest device (where the heat goes).
    b = grid.bounds
    z_lo, z_hi = z_hot - 1.2, z_hot + 0.5
    region = grid.clip_box((b[0], b[1], b[2], b[3], z_lo, z_hi), invert=False)

    y_cut = max(((bx.y_um, bx) for bx, _ in channel_sources),
                key=lambda t: t[1].power_uw)[0]
    s_xy = region.slice(normal="z", origin=(0, 0, z_hot + 0.005))
    s_xz = region.slice(normal="y", origin=(0, y_cut, 0))

    # Colour by local excess on a scale shared by BOTH slices: the pedestal is
    # ~31.2 K (set by the Robin sink under the whole footprint) and the
    # near-source structure is ~0.02 K on top of it, so an absolute scale
    # renders everything one flat colour -- the trap fig6b documents.
    floor = min(s_xy.point_data["dT"].min(), s_xz.point_data["dT"].min())
    for s in (s_xy, s_xz):
        s.point_data["excess_mK"] = (s.point_data["dT"] - floor) * 1e3
    vmax = max(s_xy.point_data["excess_mK"].max(), s_xz.point_data["excess_mK"].max())

    pl = pv.Plotter(off_screen=True, window_size=(1500, 1150))
    pl.set_background("white")
    # The colour bar must be attached to one of the SLICE actors. Calling
    # add_scalar_bar() after add_mesh() binds it to whatever was added last --
    # which, with wireframe boxes in the scene, produced a bar showing the
    # wrong colour map and unusable tick labels.
    pl.add_mesh(s_xy, scalars="excess_mK", cmap="inferno", clim=(0.0, vmax),
                show_scalar_bar=True,
                scalar_bar_args={"title": "local excess above pedestal (mK)",
                                 "vertical": True, "position_x": 0.86,
                                 "position_y": 0.22, "width": 0.05, "height": 0.56,
                                 "title_font_size": 20, "label_font_size": 17,
                                 "color": "#222222", "fmt": "%.1f", "n_labels": 5})
    pl.add_mesh(s_xz, scalars="excess_mK", cmap="inferno", clim=(0.0, vmax),
                show_scalar_bar=False)
    pl.add_mesh(pv.Box((b[0], b[1], b[2], b[3], z_lo, z_hi)),
                color="#888888", style="wireframe", line_width=1.5, opacity=0.6)
    for bx, _poly in channel_sources:
        pl.add_mesh(pv.Box((bx.x_um - bx.w_um / 2, bx.x_um + bx.w_um / 2,
                            bx.y_um - bx.l_um / 2, bx.y_um + bx.l_um / 2,
                            z_hot, z_hot + bx.t_um)),
                    color="#39d3ff", style="wireframe", line_width=4)
    pl.add_text(f"CROWBAR — orthogonal cut-away, real 3D field\n"
                f"horizontal plane: the channel layer, z = {z_hot:.3f} um\n"
                f"vertical plane: through the hottest device, y = {y_cut:.2f} um\n"
                f"Tmax = {tk:.3f} K   (dT = {tk-300.0:.3f} K above ambient)",
                font_size=11, color="#222222")
    pl.camera.parallel_projection = True
    pl.view_isometric()
    pl.camera.azimuth = 20
    pl.reset_camera()
    pl.camera.zoom(1.35)
    pl.screenshot(str(OUT / "fig6d_hotspot_3d.png"))
    pl.close()
    print(f"wrote {OUT}/fig6d_hotspot_3d.png  (excess 0..{vmax:.1f} mK, y_cut={y_cut:.3f})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--refine", type=float, default=1.0)
    p.add_argument("--plot-only", action="store_true",
                   help="reuse the cached fields in out/bitcell_oppoints instead of re-solving")
    # The pyvista cut-away is correct but hard to read as a talk figure --
    # fig_layers_svg superseded it. Kept, but opt-in.
    p.add_argument("--with-3d", action="store_true",
                   help="also render the pyvista tet cut-away (fig6d_hotspot_3d.png)")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    results = {}
    for name in POINTS:
        npz = CACHE / f"{name}.npz"
        if args.plot_only and npz.exists():
            d = np.load(npz)
            if ("source_mapping_sha256" not in d.files
                    or str(d["source_mapping_sha256"]) != mapping_revision()):
                raise ValueError(f"Stale transistor-power mapping in {npz}; "
                                 "rerun without --plot-only")
            results[name] = {k: d[k] for k in d.files}
            print(f"{name}: cached, dT max {results[name]['dT'].max():.6e} K")
            continue
        results[name] = solve_point(name, args.refine)
        np.savez_compressed(npz, **results[name])

    _, _, channel_info, _ = build_bitcell_geometry()
    fig_magnitude(results)
    fig_device_detail(results, channel_info)
    fig_layers_svg(results)
    if args.with_3d:
        fig_3d(args.refine)


if __name__ == "__main__":
    main()
