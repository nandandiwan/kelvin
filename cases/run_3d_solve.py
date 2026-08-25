"""Build + solve the LOD-1 3D chip tile (all 30 devices), then check the
same physical sanity gates as the 2D path. Unlike 2D, source_depth_m is
left None here — the 3D mesh resolves each source's real depth directly,
no homogenization needed.

    conda run -n thermals python cases/run_3d_solve.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mesh.build import build_3d_mesh
from post import io as post_io
from post.metrics import device_temperatures, tmax
from solve.steady import solve_steady
from spec import default_chip

Z_PROBE_UM = 50.12


def main():
    chip = default_chip()
    mesh_data, registry = build_3d_mesh(chip)
    T, k, q = solve_steady(mesh_data, registry, chip)

    tmax_k, (x_m, y_m, z_m) = tmax(T)
    print(f"Tmax = {tmax_k:.3f} K at x={x_m * 1e6:.3f}um y={y_m * 1e6:.3f}um "
          f"z={z_m * 1e6:.3f}um (ambient {chip.bcs.ambient_t_k:.1f} K)")

    all_dev_t = {}
    for row in range(5):
        all_dev_t.update(device_temperatures(T, chip, row, Z_PROBE_UM * 1e-6))

    devices = sorted(chip.layout.devices, key=lambda d: (d.row, d.col))
    print(f"\n{'device':<10}{'power_uW':>10}{'via':>6}{'T (K)':>10}{'dT (K)':>10}")
    for d in devices:
        t = all_dev_t[d.name]
        print(f"{d.name:<10}{d.power_uw:>10.1f}{str(d.has_via_stack):>6}"
              f"{t:>10.4f}{t - chip.bcs.ambient_t_k:>10.5f}")

    hot = [d for d in devices if d.is_hot]
    cold = [d for d in devices if not d.is_hot]
    hot_via = [d for d in hot if d.has_via_stack]
    hot_novia = [d for d in hot if not d.has_via_stack]
    print("\nsanity checks:")
    if hot and cold:
        dT_hot = max(all_dev_t[d.name] for d in hot) - chip.bcs.ambient_t_k
        dT_cold = max(all_dev_t[d.name] for d in cold) - chip.bcs.ambient_t_k
        print(f"  hot vs cold: dT_hot={dT_hot:.5f} K, dT_cold={dT_cold:.5f} K, "
              f"hot > cold: {dT_hot > dT_cold}")
    if hot_via and hot_novia:
        t_via = min(all_dev_t[d.name] for d in hot_via)
        t_novia = max(all_dev_t[d.name] for d in hot_novia)
        print(f"  via vs no-via (both hot): min T_via={t_via:.5f} K, "
              f"max T_novia={t_novia:.5f} K, via cooler: {t_via < t_novia}")

    post_io.write_solution(mesh_data, T, k, q, out_dir="out/mesh3d")
    print("\nwrote out/mesh3d/solution.xdmf")


if __name__ == "__main__":
    main()
