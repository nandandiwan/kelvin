"""Tmax/hotspot location and per-device temperature, read directly off the
solved T field. Device lookups are nearest-P1-dof (P1 dofs sit exactly at
mesh vertices, which are dense enough here — sub-nm to few-nm spacing in the
transistor layers — that this is fine for a comparison/sanity check; it is
not a rigorous interpolated point evaluation and shouldn't be used for
anything quantitative beyond that).
"""

import numpy as np


def tmax(T):
    """(T_max_kelvin, coords_m) at the hottest dof — coords_m is (x, z) for a
    2D cross-section (gdim=2, second axis is physical z) or (x, y, z) for 3D.
    """
    idx = int(np.argmax(T.x.array))
    coords = T.function_space.tabulate_dof_coordinates()
    gdim = T.function_space.mesh.geometry.dim
    return float(T.x.array[idx]), tuple(float(c) for c in coords[idx, :gdim])


def device_temperatures(T, chip, row, z_probe_m):
    """{device_name: T_kelvin} at (device position, z_probe_m), one row of
    devices, nearest dof. Works for both the 2D cross-section (gdim=2,
    matches on x only, z_probe is the second axis) and 3D (gdim=3, matches
    on x and y too).
    """
    coords = T.function_space.tabulate_dof_coordinates()
    gdim = T.function_space.mesh.geometry.dim
    devices = [d for d in chip.layout.devices if d.row == row]
    out = {}
    for d in devices:
        x_m = d.x_um * 1e-6
        if gdim == 2:
            dist2 = (coords[:, 0] - x_m) ** 2 + (coords[:, 1] - z_probe_m) ** 2
        else:
            y_m = d.y_um * 1e-6
            dist2 = ((coords[:, 0] - x_m) ** 2 + (coords[:, 1] - y_m) ** 2
                     + (coords[:, 2] - z_probe_m) ** 2)
        idx = int(np.argmin(dist2))
        out[d.name] = float(T.x.array[idx])
    return out
