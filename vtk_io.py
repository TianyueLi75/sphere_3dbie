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


def grid_hexes(nx, ny, nz):
    """Hexahedron connectivity for a regular (nx,ny,nz) grid whose points are stored in
    VTK x-fastest order (idx = i + nx*j + nx*ny*k). Returns (Ncell, 8) with the VTK
    VTK_HEXAHEDRON node ordering (bottom face 0-3 CCW, top face 4-7)."""
    i = np.arange(nx - 1)[:, None, None]
    j = np.arange(ny - 1)[None, :, None]
    k = np.arange(nz - 1)[None, None, :]
    base = i + nx * j + nx * ny * k
    n000 = base
    n100 = base + 1
    n110 = base + 1 + nx
    n010 = base + nx
    off = nx * ny
    return np.stack([n000, n100, n110, n010,
                     n000 + off, n100 + off, n110 + off, n010 + off],
                    axis=-1).reshape(-1, 8)


def _xml_data_array(f, name, arr, ncomp, dtype="Float64", indent="        "):
    arr = np.asarray(arr, dtype=float)
    comp = f' NumberOfComponents="{ncomp}"' if ncomp > 1 else ""
    nm = f' Name="{name}"' if name else ""
    f.write(f'{indent}<DataArray type="{dtype}"{nm}{comp} format="ascii">\n')
    np.savetxt(f, arr.reshape(-1, ncomp), fmt="%.9g")
    f.write(f"{indent}</DataArray>\n")


def write_vtu(path, points, cells, cell_type=12,
              point_vectors=None, point_scalars=None):
    """One XML UnstructuredGrid piece (.vtu, ASCII). points is (Np,3); cells is (Ncell,K)
    connectivity of a single VTK cell type (default 12 == VTK_HEXAHEDRON, K=8)."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    cells = np.asarray(cells, dtype=np.int64).reshape(len(cells), -1) if len(cells) else \
        np.zeros((0, 8), dtype=np.int64)
    npts, ncell, k = points.shape[0], cells.shape[0], cells.shape[1]
    offsets = np.arange(1, ncell + 1, dtype=np.int64) * k
    types = np.full(ncell, cell_type, dtype=np.uint8)
    with open(path, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write("  <UnstructuredGrid>\n")
        f.write(f'    <Piece NumberOfPoints="{npts}" NumberOfCells="{ncell}">\n')
        f.write("      <Points>\n")
        _xml_data_array(f, "", points, 3)
        f.write("      </Points>\n")
        f.write("      <Cells>\n")
        _xml_data_array(f, "connectivity", cells, 1, dtype="Int64")
        _xml_data_array(f, "offsets", offsets, 1, dtype="Int64")
        _xml_data_array(f, "types", types, 1, dtype="UInt8")
        f.write("      </Cells>\n")
        if point_vectors or point_scalars:
            f.write("      <PointData>\n")
            for name, arr in (point_vectors or {}).items():
                _xml_data_array(f, name, arr, 3)
            for name, arr in (point_scalars or {}).items():
                _xml_data_array(f, name, arr, 1)
            f.write("      </PointData>\n")
        f.write("    </Piece>\n")
        f.write("  </UnstructuredGrid>\n")
        f.write("</VTKFile>\n")


def write_pvtu(path, piece_sources, point_vectors=None, point_scalars=None):
    """Master parallel UnstructuredGrid (.pvtu) referencing the given piece .vtu
    <piece_sources> (paths relative to the .pvtu). point_vectors/point_scalars are
    dicts whose keys name the arrays present on every piece (values unused)."""
    with open(path, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="PUnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <PUnstructuredGrid GhostLevel="0">\n')
        f.write("    <PPoints>\n")
        f.write('      <PDataArray type="Float64" NumberOfComponents="3"/>\n')
        f.write("    </PPoints>\n")
        if point_vectors or point_scalars:
            f.write("    <PPointData>\n")
            for name in (point_vectors or {}):
                f.write(f'      <PDataArray type="Float64" Name="{name}" NumberOfComponents="3"/>\n')
            for name in (point_scalars or {}):
                f.write(f'      <PDataArray type="Float64" Name="{name}"/>\n')
            f.write("    </PPointData>\n")
        for src in piece_sources:
            f.write(f'    <Piece Source="{src}"/>\n')
        f.write("  </PUnstructuredGrid>\n")
        f.write("</VTKFile>\n")


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
