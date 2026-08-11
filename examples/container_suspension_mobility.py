"""Rigid-body-motion (mobility) Stokes solve for particles in a no-slip container.

Reads a sphere-packing data file produced by geometry_generators/ (interior spheres
[x y z r] plus an `# outer_radius = ...` header), builds a coupled suspension with the
container as sphere 0 and the packed spheres as interior particles, and solves the
MOBILITY problem:

    - container (sphere 0): interior problem (sgn = -1), static no-slip (zero velocity) BC,
    - interior particles  : exterior problem (sgn = +1), rigid-body motion driven by a
                            prescribed external force / torque -- the rigid velocity
                            (U_p, Omega_p) is UNKNOWN and solved for.

Each particle is subject to an external force F_p and torque T_p; by default a uniform
gravity-like body force F_p = (0, 0, -g * volume_p) with zero torque (sedimentation inside
the container). The solver returns each particle's rigid velocity, the recovered
hydrodynamic force/torque, the surface densities, and the interior velocity field (VTK).

NOTE: the force/torque functional carries a sign/scale constant `force_const` that must be
calibrated once against single-sphere Stokes drag (F = -6*pi*mu*a*U); see --force-const and
suspension.net_force_torque. Until calibrated, the reported U/Omega magnitudes are only
correct up to that constant.

Usage (verify cheaply with small orders):
    python examples/container_suspension_mobility.py \
        --data geometry_generators/sphere_data_10_ball.txt \
        --lmax-container 60 --lmax-interior 16 --Ng 32
"""

import argparse
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")   # avoid the eager-shtns MLIR segfault

import numpy as np

# Put the repo root on the path so `import suspension` (and the modules it pulls in) resolve.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

import jax.numpy as jnp

import suspension as susp
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
        _REPO_ROOT, "geometry_generators", "sphere_data_10_ball.txt"),
        help="packing data file (with `# outer_radius` header)")
    ap.add_argument("--lmax-container", type=int, default=200,
                    help="spherical-harmonic order on the container")
    ap.add_argument("--lmax-interior", type=int, default=32,
                    help="spherical-harmonic order on each interior particle")
    ap.add_argument("--Ng", type=int, default=40, help="field grid resolution per axis")
    ap.add_argument("--sep-eta", type=float, default=0.1, help="near/far separation parameter")
    ap.add_argument("--g", type=float, default=1.0,
                    help="gravity-like body-force magnitude (F_p = (0,0,-g*volume_p))")
    ap.add_argument("--mu", type=float, default=1.0, help="fluid viscosity")
    ap.add_argument("--force-const", type=float, default=1.0,
                    help="sign/scale constant of the force/torque functional (calibrate vs Stokes drag)")
    ap.add_argument("--tol", type=float, default=1e-8, help="gmres relative tolerance")
    ap.add_argument("--restart", type=int, default=None,
                    help="gmres Krylov depth per cycle (raise for dense packings that stall; "
                         "None = scipy default 20)")
    ap.add_argument("--maxiter", type=int, default=200, help="gmres max iterations")
    ap.add_argument("--no-field", action="store_true", help="skip the volumetric field evaluation / VTK")
    args = ap.parse_args()

    # --- Geometry: container (sphere 0 at origin) + interior particles from file ---
    outer_radius, int_centers, int_radii = read_packing(args.data)
    reach = np.max(np.linalg.norm(int_centers, axis=1) + int_radii)
    assert outer_radius >= reach - 1e-9, \
        f"container radius {outer_radius} does not enclose interior spheres (reach {reach})"

    centers = np.vstack([np.zeros((1, 3)), int_centers])           # container first
    radii = np.concatenate([[outer_radius], int_radii])
    print(f"Container radius {outer_radius:.6f}; {int_centers.shape[0]} interior particles "
          f"(radii {int_radii.min():.4f}..{int_radii.max():.4f}).")

    Sp = susp.build_suspension(jnp.asarray(centers), jnp.asarray(radii), args.sep_eta)
    Ns = Sp["Ns"]
    lmax_lst = jnp.asarray([args.lmax_container] + [args.lmax_interior] * (Ns - 1))
    Sp, sh_lst = susp.quadr_suspension(Sp, lmax_lst)

    # Interior problem on the container, exterior (rigid particle) on each obstacle.
    sl_lst = [1.0] * Ns
    dl_lst = [1.0] * Ns
    sgn_lst = [-1.0] + [1.0] * (Ns - 1)

    # --- Per-particle external loads: uniform gravity F_p = (0,0,-g*vol_p), no torque ---
    Np = Ns - 1
    vol = (4.0 / 3.0) * np.pi * int_radii ** 3
    forces = np.zeros((Np, 3), dtype=float)
    forces[:, 2] = -args.g * vol
    torques = np.zeros((Np, 3), dtype=float)

    # --- Mobility solve (container no-slip: bc_vec = None -> zero BIE RHS) ---
    dsp = Sp["Nnodes_dsp"]
    Nnodes = dsp[-1].item()
    (sigma, U_lst, Omega_lst, F_lst, T_lst, part_idx,
     t_solve, niter, info, resid) = susp.Stk3d_mobility_solve(
        None, Sp, Ns, Nnodes, sh_lst, sl_lst, dl_lst, sgn_lst,
        forces, torques, tol=args.tol, maxiter=args.maxiter, restart=args.restart,
        mu=args.mu, force_const=args.force_const)

    print(f"GMRES info (0 == converged): {info}, time for solve is {t_solve:.3f}s, "
          f"num of iters is {niter}, residual = {resid:.3e}")
    for i, s in enumerate(part_idx):
        print(f"  particle {s} (r={radii[s]:.4f}): "
              f"U = [{U_lst[i][0]:+.4e}, {U_lst[i][1]:+.4e}, {U_lst[i][2]:+.4e}], "
              f"Omega = [{Omega_lst[i][0]:+.4e}, {Omega_lst[i][1]:+.4e}, {Omega_lst[i][2]:+.4e}]")
        print(f"    applied F = {forces[i]}, recovered F_hydro = {F_lst[i]}, T_hydro = {T_lst[i]}")

    if args.no_field:
        return

    # --- Volumetric velocity field: interior to container, exterior to every particle ---
    trg_data = vtk_export.grid_from_spheres(Sp, args.Ng, pad=0.001)
    trg_grid = trg_data["points"]
    Cnp = np.asarray(centers)
    Rnp = np.asarray(radii)
    inside = np.linalg.norm(trg_grid - Cnp[0], axis=1) < Rnp[0] * 0.999   # interior to container
    for s in range(1, Ns):
        inside &= np.linalg.norm(trg_grid - Cnp[s], axis=1) > Rnp[s] * 1.001  # outside particle s

    Ufield = np.zeros((trg_grid.shape[0], 3), dtype=float)
    if np.any(inside):
        trg_in = jnp.asarray(trg_grid[inside])
        approx = jnp.zeros((trg_in.shape[0], 3), dtype=jnp.complex128)
        for s in range(Ns):
            s_sph = Sp["spheres_lst"][s]
            nphi, ntheta = s_sph["Xcart"].shape[:2]
            sig_s = sigma[3 * int(dsp[s]):3 * int(dsp[s + 1])].reshape(nphi, ntheta, 3)
            vwx_s = jnp.stack(Stk3d.sig_xyz2vwx(sig_s[:, :, 0], sig_s[:, :, 1], sig_s[:, :, 2],
                                                s_sph["Xsph"][:, :, 0], s_sph["Xsph"][:, :, 1], sh_lst[s]))
            approx = approx + Stk3d.bio_offsurf_apply(trg_in, vwx_s, s_sph, sh_lst[s], sl_lst[s], dl_lst[s])
        Ufield[inside] = np.real(np.asarray(approx))

    # --- Write VTK (field + geometry) ---
    vis_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vis")
    os.makedirs(vis_dir, exist_ok=True)
    vtk_export.export_field(trg_data, Ufield,
                            os.path.join(vis_dir, "container_suspension_mobility_field.vtk"),
                            name="velocity")
    vtk_export.export_objects(os.path.join(vis_dir, "container_suspension_mobility_geometry.vtk"), Sp)
    print("Wrote VTK (field + geometry) to", vis_dir)


if __name__ == "__main__":
    main()
