"""Build + solve the LOD-1 2D cross-section, then check the plan's physical
sanity gates: hot devices measurably hotter than cold ones, via-stack
devices measurably cooler than otherwise-identical no-via ones.

    conda run -n thermals python cases/run_2d_solve.py [row]

Writes out/mesh/{solution.xdmf/.h5, temperature_2d.png} in addition to the
GATE 0 mesh artifacts.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mesh.build import build_2d_mesh
from post import io as post_io
from post import viz as post_viz
from post.metrics import device_temperatures, tmax
from solve.steady import solve_steady
from spec import default_chip
from spec.layout import PITCH_Y_UM

# silicide layer midpoint (z=50.11-50.13um), a proxy for junction temperature
Z_PROBE_UM = 50.12


def main():
    row = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    chip = default_chip()
    mesh_data, registry = build_2d_mesh(chip, row=row)
    # 2D cross-section: homogenize each source over its real row pitch (see
    # physics/coeffs.build_coeffs) so absolute temperatures are physically
    # meaningful, not just qualitatively comparable.
    T, k, q = solve_steady(mesh_data, registry, chip, source_depth_m=PITCH_Y_UM * 1e-6)

    tmax_k, (x_m, z_m) = tmax(T)
    print(f"Tmax = {tmax_k:.3f} K at x={x_m * 1e6:.3f}um z={z_m * 1e6:.3f}um "
          f"(ambient {chip.bcs.ambient_t_k:.1f} K)")

    dev_t = device_temperatures(T, chip, row, Z_PROBE_UM * 1e-6)
    devices = sorted((d for d in chip.layout.devices if d.row == row), key=lambda d: d.x_um)

    print(f"\n{'device':<10}{'power_uW':>10}{'via':>6}{'T (K)':>10}{'dT (K)':>10}")
    for d in devices:
        t = dev_t[d.name]
        print(f"{d.name:<10}{d.power_uw:>10.1f}{str(d.has_via_stack):>6}"
              f"{t:>10.4f}{t - chip.bcs.ambient_t_k:>10.5f}")

    hot = [d for d in devices if d.is_hot]
    cold = [d for d in devices if not d.is_hot]
    hot_via = [d for d in hot if d.has_via_stack]
    hot_novia = [d for d in hot if not d.has_via_stack]
    print("\nsanity checks:")
    if hot and cold:
        dT_hot = max(dev_t[d.name] for d in hot) - chip.bcs.ambient_t_k
        dT_cold = max(dev_t[d.name] for d in cold) - chip.bcs.ambient_t_k
        print(f"  hot vs cold: dT_hot={dT_hot:.5f} K, dT_cold={dT_cold:.5f} K, "
              f"hot > cold: {dT_hot > dT_cold}")
    if hot_via and hot_novia:
        t_via = dev_t[hot_via[0].name]
        t_novia = dev_t[hot_novia[0].name]
        print(f"  via vs no-via (both hot): T_via={t_via:.5f} K, T_novia={t_novia:.5f} K, "
              f"via cooler: {t_via < t_novia}")

    post_io.write_solution(mesh_data, T, k, q, out_dir="out/mesh")
    post_viz.render_temperature(T, chip, row, "out/mesh/temperature_2d.png")
    print("\nwrote out/mesh/{solution.xdmf, temperature_2d.png}")


if __name__ == "__main__":
    main()
