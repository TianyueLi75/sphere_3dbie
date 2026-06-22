"""Minimal, dependency-free legacy-VTK (ASCII) writers for ParaView.

Only what the suspension visualization needs:
  - write_structured_points : a regular 3-D grid field (STRUCTURED_POINTS / ImageData)
  - write_polydata          : a surface/point mesh (POLYDATA) given points + polygons
  - grid_quads              : quad connectivity for a structured (nphi, ntheta) sphere grid

Legacy ASCII .vtk is verbose but needs no libraries and ParaView reads it directly.
Point/cell data are dicts {name: array}; structured-grid arrays must be in VTK
x-fastest order (idx = i + nx*j + nx*ny*k).
"""

import numpy as np


def _write_point_data(f, npts, point_vectors=None, point_scalars=None):
    if not point_vectors and not point_scalars:
        return
    f.write(f"POINT_DATA {npts}\n")
    for name, arr in (point_vectors or {}).items():
        arr = np.asarray(arr, dtype=float).reshape(npts, 3)
        f.write(f"VECTORS {name} double\n")
        np.savetxt(f, arr, fmt="%.9g")
    for name, arr in (point_scalars or {}).items():
        arr = np.asarray(arr, dtype=float).reshape(npts)
        f.write(f"SCALARS {name} double 1\n")
        f.write("LOOKUP_TABLE default\n")
        np.savetxt(f, arr.reshape(-1, 1), fmt="%.9g")


def write_structured_points(path, origin, spacing, dims,
                            point_vectors=None, point_scalars=None, title="field"):
    """Regular 3-D grid (STRUCTURED_POINTS). dims=(nx,ny,nz); data in x-fastest order."""
    nx, ny, nz = (int(d) for d in dims)
    npts = nx * ny * nz
    origin = [float(o) for o in origin]
    spacing = [float(s) for s in spacing]
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(title + "\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        f.write("ORIGIN {} {} {}\n".format(*origin))
        f.write("SPACING {} {} {}\n".format(*spacing))
        _write_point_data(f, npts, point_vectors, point_scalars)


def write_polydata(path, points, quads, point_vectors=None, point_scalars=None, title="objects"):
    """POLYDATA surface from points (Np,3) and quad connectivity (Ncell,4)."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    quads = np.asarray(quads, dtype=np.int64).reshape(-1, 4)
    npts, ncell = points.shape[0], quads.shape[0]
    conn = np.hstack([np.full((ncell, 1), 4, dtype=np.int64), quads])
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(title + "\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {npts} double\n")
        np.savetxt(f, points, fmt="%.9g")
        f.write(f"POLYGONS {ncell} {ncell * 5}\n")
        np.savetxt(f, conn, fmt="%d")
        _write_point_data(f, npts, point_vectors, point_scalars)


def grid_quads(nphi, ntheta, offset=0):
    """Quad connectivity for a structured (nphi, ntheta) grid flattened C-order
    (point index = iphi*ntheta + itheta), periodic in phi. Returns (Ncell, 4)."""
    ip = np.arange(nphi)[:, None]
    it = np.arange(ntheta - 1)[None, :]
    ip1 = (ip + 1) % nphi
    a = ip * ntheta + it
    b = ip1 * ntheta + it
    c = ip1 * ntheta + (it + 1)
    d = ip * ntheta + (it + 1)
    return np.stack([a + 0 * it, b + 0 * it, c, d], axis=-1).reshape(-1, 4) + offset
