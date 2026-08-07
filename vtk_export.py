"""High-level VTK export for the suspension: objects (sphere surfaces) and a regular
3-D grid field. EXPORT ONLY -- this module never evaluates the field or masks; the
caller evaluates on grid["points"] (handling masking by setting non-fluid values to
0 or NaN itself) and passes the result to export_field.

Only a regular 3-D grid target is supported (maps to ParaView STRUCTURED_POINTS, so
Stream Tracer / Slice / Glyph work natively).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vtk_io


def grid_targets(lo, hi, dims):
    """Regular 3-D grid spanning [lo, hi] with dims=(nx,ny,nz). Returns a descriptor
    {points, origin, spacing, dims} with points in VTK x-fastest order so an evaluated
    field array lines up with the structured grid."""
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    nx, ny, nz = (int(d) for d in dims)
    xs = np.linspace(lo[0], hi[0], nx)
    ys = np.linspace(lo[1], hi[1], ny)
    zs = np.linspace(lo[2], hi[2], nz)
    # x-fastest: idx = i + nx*j + nx*ny*k
    px = np.tile(xs, ny * nz)
    py = np.tile(np.repeat(ys, nx), nz)
    pz = np.repeat(zs, nx * ny)
    points = np.stack([px, py, pz], axis=1)
    spacing = [(xs[1] - xs[0]) if nx > 1 else 0.0,
               (ys[1] - ys[0]) if ny > 1 else 0.0,
               (zs[1] - zs[0]) if nz > 1 else 0.0]
    return {"points": points, "origin": lo, "spacing": spacing, "dims": (nx, ny, nz)}


def grid_from_spheres(Sp, Ng=32, pad=0.15):
    """Bounding-box regular grid (Ng^3) enclosing all spheres, padded by <pad> of the span."""
    C = np.asarray(Sp["Xc_lst"], dtype=float)
    r = np.asarray(Sp["r_lst"], dtype=float)
    lo = (C - r[:, None]).min(axis=0)
    hi = (C + r[:, None]).max(axis=0)
    span = hi - lo
    return grid_targets(lo - pad * span, hi + pad * span, (Ng, Ng, Ng))


def export_field(grid, field, path, name=None):
    """Write a field sampled on grid["points"] to a STRUCTURED_POINTS .vtk file.
    field is (N,) scalar (-> <name>/'potential') or (N,3) vector (-> <name>/'velocity'
    plus a 'speed' scalar). Masking must already be baked into field by the caller."""
    field = np.asarray(field)
    N = int(np.prod(grid["dims"]))
    assert field.shape[0] == N, f"field has {field.shape[0]} rows, grid has {N} points"
    if field.ndim == 2 and field.shape[1] == 3:
        vec = np.real(field)
        pv = {name or "velocity": vec}
        ps = {"speed": np.linalg.norm(vec, axis=1)}
    else:
        pv = None
        ps = {name or "potential": np.real(field).reshape(-1)}
    vtk_io.write_structured_points(path, grid["origin"], grid["spacing"], grid["dims"],
                                   point_vectors=pv, point_scalars=ps, title=os.path.basename(path))


def export_field_pvtu(grid, field, path, name=None, npieces=4):
    """Write a field sampled on grid["points"] to a partitioned XML UnstructuredGrid:
    a master <path> (.pvtu) plus <npieces> hexahedral .vtu pieces sliced along z. The
    grid is meshed into VTK hexahedra so ParaView's Stream Tracer / Slice / Glyph work
    on the partitioned dataset too. field is (N,) scalar or (N,3) vector, in the same
    x-fastest order as grid["points"]; masking must already be baked in by the caller."""
    field = np.asarray(field)
    nx, ny, nz = (int(d) for d in grid["dims"])
    N = nx * ny * nz
    assert field.shape[0] == N, f"field has {field.shape[0]} rows, grid has {N} points"
    if not path.endswith(".pvtu"):
        raise ValueError("pvtu path must end in .pvtu")
    pts = np.asarray(grid["points"], dtype=float)

    if field.ndim == 2 and field.shape[1] == 3:
        vec = np.real(field)
        pv_all = {name or "velocity": vec}
        ps_all = {"speed": np.linalg.norm(vec, axis=1)}
    else:
        pv_all = None
        ps_all = {name or "potential": np.real(field).reshape(-1)}

    stem = path[:-len(".pvtu")]
    base = os.path.basename(stem)
    # Slab the grid along z into <npieces> contiguous chunks; adjacent pieces share the
    # boundary z-layer so hexahedra tile the full volume without gaps.
    npieces = max(1, min(int(npieces), nz - 1))
    edges = np.linspace(0, nz, npieces + 1).round().astype(int)
    sources = []
    for p in range(npieces):
        k0, k1 = edges[p], edges[p + 1]        # z-layers [k0, k1); include k1 for shared face
        khi = min(k1 + 1, nz) if k1 < nz else nz
        klayers = np.arange(k0, khi)
        sel = (np.arange(N).reshape(nz, ny, nx)[klayers].reshape(-1))
        pts_p = pts[sel]
        pnz = klayers.shape[0]
        hexes = vtk_io.grid_hexes(nx, ny, pnz)
        pv_p = {k: v[sel] for k, v in pv_all.items()} if pv_all else None
        ps_p = {k: v[sel] for k, v in ps_all.items()} if ps_all else None
        piece_path = f"{stem}_{p}.vtu"
        vtk_io.write_vtu(piece_path, pts_p, hexes,
                         point_vectors=pv_p, point_scalars=ps_p)
        sources.append(f"{base}_{p}.vtu")
    vtk_io.write_pvtu(path, sources, point_vectors=pv_all, point_scalars=ps_all)


def export_objects(path, Sp, bc_vec=None):
    """Write all sphere surfaces to a single POLYDATA .vtk file, with a 'sphere_id' scalar
    and (optional) the per-node boundary velocity 'bc' as point vectors. bc_vec is the flat
    Stokes BC vector (length 3*Nnodes_total) used by the solver."""
    dsp = np.asarray(Sp["Nnodes_dsp"])
    pts_list, quad_list, id_list, bc_list = [], [], [], []
    offset = 0
    bc_flat = np.asarray(bc_vec).reshape(-1) if bc_vec is not None else None
    for s, sph in enumerate(Sp["spheres_lst"]):
        X = np.asarray(sph["Xcart"])           # (nphi, ntheta, 3)
        nphi, ntheta = X.shape[:2]
        P = X.reshape(-1, 3)
        pts_list.append(P)
        quad_list.append(vtk_io.grid_quads(nphi, ntheta, offset=offset))
        id_list.append(np.full(P.shape[0], float(s)))
        if bc_flat is not None:
            blk = bc_flat[3 * int(dsp[s]):3 * int(dsp[s + 1])].reshape(nphi, ntheta, 3).reshape(-1, 3)
            bc_list.append(np.real(blk))
        offset += P.shape[0]
    points = np.concatenate(pts_list, axis=0)
    quads = np.concatenate(quad_list, axis=0)
    ps = {"sphere_id": np.concatenate(id_list)}
    pv = {"bc": np.concatenate(bc_list, axis=0)} if bc_flat is not None else None
    vtk_io.write_polydata(path, points, quads, point_vectors=pv, point_scalars=ps,
                          title=os.path.basename(path))
