"""Dirichlet Stokes BVP for a packed suspension inside a spherical container.

Reads a sphere-packing data file produced by geometry_generators/ (interior spheres
[x y z r] plus an `# outer_radius = ...` header), builds a coupled suspension with the
container as sphere 0 and the packed spheres as interior obstacles, and solves the
coupled boundary-integral Stokes problem:

    - container (sphere 0): interior problem (sgn = -1), tangential squirmer slip BC,
    - interior spheres   : exterior problem (sgn = +1), no-slip (zero velocity) BC.

The domain visualization (sphere surfaces + boundary data, and the volumetric velocity
field interior to the container / exterior to the obstacles) is written to VTK.

This generalizes TEST 3 of suspension.py to N interior spheres read from file, with the
slip moved onto the container.

Usage (production target lmax 500 / 64 are the defaults; verify cheaply with overrides):
    python examples/container_suspension_stokes.py --lmax-container 100 --lmax-interior 20 --Ng 32
"""

import argparse
import os
import sys

# os.environ["XLA_FLAGS"] = (
#     "--xla_cpu_multi_thread_eigen=true "
# )

import numpy as np

# Put the repo root on the path so `import suspension` (and the modules it pulls in) resolve.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

import jax.numpy as jnp

import suspension as susp
from sphere import set_density
from biop import Stk3d
import vtk_export


def read_packing(path):
    """Read a geometry_generators data file. Returns (outer_radius, centers (N,3), radii (N,)).

    The container radius is stored as a `# outer_radius = <val>` comment header; the data
    rows are the interior spheres as [x y z r]."""
    outer_radius = None
    with open(path, "r") as fh:
        for line in fh:
            s = line.strip()
            if not s.startswith("#"):
                break
            if "outer_radius" in s:
                outer_radius = float(s.split("=", 1)[1])
    if outer_radius is None:
        raise ValueError(f"No '# outer_radius = ...' header found in {path}")

    data = np.loadtxt(path).reshape(-1, 4)
    return outer_radius, data[:, :3], data[:, 3]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(
        _REPO_ROOT, "geometry_generators", "sphere_data_100_ball.txt"),
        help="packing data file (with `# outer_radius` header)")
    ap.add_argument("--lmax-container", type=int, default=500,
                    help="spherical-harmonic order on the container (production target 500)")
    ap.add_argument("--lmax-interior", type=int, default=64,
                    help="spherical-harmonic order on each interior sphere (production target 64)")
    ap.add_argument("--Ng", type=int, default=40, help="field grid resolution per axis")
    ap.add_argument("--sep-eta", type=float, default=0.1, help="near/far separation parameter")
    ap.add_argument("--U", type=float, default=1.0, help="container slip speed scale")
    ap.add_argument("--tol", type=float, default=1e-8, help="gmres relative tolerance")
    args = ap.parse_args()

    # --- Geometry: container (sphere 0 at origin) + interior spheres from file ---
    outer_radius, int_centers, int_radii = read_packing(args.data)
    reach = np.max(np.linalg.norm(int_centers, axis=1) + int_radii)
    assert outer_radius >= reach - 1e-9, \
        f"container radius {outer_radius} does not enclose interior spheres (reach {reach})"

    centers = np.vstack([np.zeros((1, 3)), int_centers])           # container first
    radii = np.concatenate([[outer_radius], int_radii])
    print(f"Container radius {outer_radius:.6f}; {int_centers.shape[0]} interior spheres "
          f"(radii {int_radii.min():.4f}..{int_radii.max():.4f}).")

    Sp = susp.build_suspension(jnp.asarray(centers), jnp.asarray(radii), args.sep_eta)
    Ns = Sp["Ns"]
    lmax_lst = jnp.asarray([args.lmax_container] + [args.lmax_interior] * (Ns - 1))
    Sp, sh_lst = susp.quadr_suspension(Sp, lmax_lst)

    # Interior problem on the container, exterior on each obstacle.
    sl_lst = [1.0] * Ns
    dl_lst = [1.0] * Ns
    sgn_lst = [-1.0] + [1.0] * (Ns - 1)

    # --- Dirichlet BC: zero (no-slip) everywhere, tangential squirmer slip on the container ---
    dsp = Sp["Nnodes_dsp"]
    bc = jnp.zeros((3 * int(dsp[-1]),), dtype=jnp.complex128)
    vslip_mag = lambda theta: jnp.sin(theta) * 3. / 2. * args.U   # squirmer tangential speed
    cont = Sp["spheres_lst"][0]
    th0 = cont["Xsph"][:, :, 0]
    ph0 = cont["Xsph"][:, :, 1]
    zeros0 = jnp.zeros_like(th0)
    sx, sy, sz = Stk3d.sph2cart(zeros0, vslip_mag(th0), zeros0, th0, ph0)
    bc_cont = jnp.stack([sx, sy, sz], axis=2)                     # (nphi, ntheta, 3)
    bc = bc.at[3 * int(dsp[0]):3 * int(dsp[1])].set(bc_cont.reshape(-1))

    # --- Coupled solve ---
    Nnodes = dsp[-1].item()
    sigma, t_solve, niter, info, resid = susp.Stk3d_onsurf_solve_spla(bc, Sp, Sp["Ns"], Nnodes, sh_lst, sl_lst, dl_lst, sgn_lst, args.tol)
    print(f"GMRES info (0 == converged): {info}, time for solve is {t_solve}s, num of iters is {niter}, residual = {resid:.3e}")

    # --- Volumetric velocity field: interior to container, exterior to every obstacle ---
    trg_data = vtk_export.grid_from_spheres(Sp, args.Ng, pad=0.001)
    trg_grid = trg_data["points"]
    Cnp = np.asarray(centers)
    Rnp = np.asarray(radii)
    inside = np.linalg.norm(trg_grid - Cnp[0], axis=1) < Rnp[0] * 0.999   # interior to container
    for s in range(1, Ns):
        inside &= np.linalg.norm(trg_grid - Cnp[s], axis=1) > Rnp[s] * 1.001  # outside obstacle s

    Ufield = np.zeros((trg_grid.shape[0], 3), dtype=float)
    if np.any(inside):
        trg_in = jnp.asarray(trg_grid[inside])
        approx = jnp.zeros((trg_in.shape[0], 3), dtype=jnp.complex128)
        for s in range(Ns):
            nphi, ntheta = Sp["spheres_lst"][s]["Xcart"].shape[:2]
            sig_s = sigma[3 * int(dsp[s]):3 * int(dsp[s + 1])].reshape(nphi, ntheta, 3)
            s_sph = set_density(Sp["spheres_lst"][s], sig_s[:, :, 0], sig_s[:, :, 1], sig_s[:, :, 2])
            approx = approx + Stk3d.bio_offsurf_apply(trg_in, s_sph, sh_lst[s], sl_lst[s], dl_lst[s])
        Ufield[inside] = np.real(np.asarray(approx))

    # --- Write VTK ---
    vis_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vis")
    os.makedirs(vis_dir, exist_ok=True)
    vtk_export.export_field(trg_data, Ufield,
                            os.path.join(vis_dir, "container_suspension_field.vtk"),
                            name="velocity")
    bc_vec = np.real(np.asarray(bc)).reshape(-1, 3)
    vtk_export.export_objects(os.path.join(vis_dir, "container_suspension_geometry.vtk"),
                              Sp, bc_vec)
    print("Wrote VTK (field + geometry) to", vis_dir)


if __name__ == "__main__":
    main()
