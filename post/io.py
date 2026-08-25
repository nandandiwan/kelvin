"""XDMF export of the solved fields, for ParaView."""

import os

from mpi4py import MPI

from dolfinx.io import XDMFFile


def write_solution(mesh_data, T, k, q, out_dir="out/mesh"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "solution.xdmf")
    with XDMFFile(MPI.COMM_SELF, path, "w") as xf:
        xf.write_mesh(mesh_data.mesh)
        xf.write_function(T)
        xf.write_function(k)
        xf.write_function(q)
    return path
