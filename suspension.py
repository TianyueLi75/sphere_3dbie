"""
Suspension Class:
    A dictionary object containing information about the suspension system.
    Each suspension item contains spheres (SphereDict object from sphere.py),
    also includes Ns (number of spheres), BC (enumeration of BC states on each sphere, no-slip, passive, RBM, vesicle)
    handles inter-spherical tasks like separation and collision checks, as well as
    provide parent functions that calls the same function on each individual sphere.

Data model notes:
    - Object containers (spheres, sht objects) are stored as plain Python lists, NOT jax
      arrays: jax arrays cannot hold arbitrary Python objects.
    - A suspension density / BC is a flat 1-D complex vector of length Nnodes_dsp[-1].
      Sphere s owns entries Nnodes_dsp[s] : Nnodes_dsp[s+1]. Each per-sphere block is the
      C-order flatten of the (nphi, ntheta) surface grid (matching sphere["Xcart"].shape[:2]).
    - Targets are seperated into on-surface, near, and far off-surface. 
      This can be surface collocation nodes during solve or in-domain points during evaluation.

Main examples:
    - Stokes suspension solver: manufactured solutions test
    - Container geometry (TODO)
"""

from typing import Dict, Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import os
import sys

import shtns
import shtns_jax
jax.config.update("jax_enable_x64", True)  # support float64

# Put the repo root on the path so that `from sphere import *` (used inside biop) and
# `from biop import Lap3d` both resolve, regardless of the working directory.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), "vis"))
from sphere import *
from biop import Lap3d
from biop import Stk3d
import vtk_export

import scipy.sparse.linalg as spla

SuspensionDict = Dict[str, Any]

def build_suspension(center_lst: jax.Array, radius_lst: jax.Array, sep_eta: float = -1.0) -> SuspensionDict:
    """
    create a suspension dictionary with given list of sphere centers and radii,
    along with the corresponding (un-quadratured) sphere objects.

    center_lst : (Ns, 3) array of sphere centers (a (3, Ns) array is accepted and transposed).
    radius_lst : (Ns,) array of sphere radii (any 1-D / (Ns,1) / (1,Ns) shape is flattened).
    """

    center_lst = jnp.asarray(center_lst, dtype=jnp.float64)
    radius_lst = jnp.asarray(radius_lst, dtype=jnp.float64).reshape(-1)
    Ns_ = radius_lst.shape[0]

    # Accept a (3, Ns) center list by transposing to (Ns, 3).
    if center_lst.ndim == 2 and center_lst.shape[1] != 3 and center_lst.shape[0] == 3:
        center_lst = center_lst.T
    assert center_lst.shape == (Ns_, 3), \
        f"center_lst must be (Ns, 3) matching {Ns_} radii; got {center_lst.shape}"

    spheres_lst = [build_sphere(center_lst[i], float(radius_lst[i])) for i in range(Ns_)]

    return {
        "Xc_lst": center_lst,                              # (Ns, 3)
        "r_lst": radius_lst,                               # (Ns,)
        "spheres_lst": spheres_lst,                        # Python list of SphereDict
        "sh_lst": [None] * Ns_,                            # filled by quadr_suspension
        "Nnodes_lst": jnp.zeros((Ns_,), dtype=int),        # nodes per sphere
        "Nnodes_dsp": jnp.zeros((Ns_ + 1,), dtype=int),    # cumulative node offsets
        "Ns": Ns_,
        "sep_eta": sep_eta,
    }


def quadr_suspension(Sp: SuspensionDict, lmax_lst: jax.Array) -> Tuple[SuspensionDict, list]:
    """
    Set up the quadrature grid on each sphere from lmax_lst, returning the updated suspension
    and the list of corresponding shtns objects (also stored in Sp["sh_lst"]).
    """
    lmax_lst = jnp.asarray(lmax_lst).reshape(-1)
    assert lmax_lst.shape[0] == Sp["Ns"]  # TODO: allow a scalar lmax broadcast to Ns

    sh_lst = []
    nnodes = []
    for sind in range(Sp["Ns"]):
        sphere_updated, sh_sind = quadr_sphere(Sp["spheres_lst"][sind], int(lmax_lst[sind]))
        Sp["spheres_lst"][sind] = sphere_updated
        sh_lst.append(sh_sind)
        nphi, ntheta = sphere_updated["Xcart"].shape[:2]
        nnodes.append(int(nphi * ntheta))

    Sp["sh_lst"] = sh_lst
    Sp["Nnodes_lst"] = jnp.asarray(nnodes, dtype=int)
    Sp["Nnodes_dsp"] = jnp.concatenate([jnp.zeros((1,), dtype=int),
                                        jnp.cumsum(jnp.asarray(nnodes, dtype=int))])

    return Sp, sh_lst


def separate_spheres(Sp: SuspensionDict) -> jax.Array:
    """
    return a Ns x Ns array of [0,1]s that is the separation matrix among spheres in the suspension.
    Heuristically, if two spheres' closest surface points fall within sep_eta, consider them
    close (0); otherwise far (1). The diagonal is 0 (self-interaction handled separately).
    """

    if Sp["sep_eta"] < 0.:
        print("Warning: no separation eta set by user; setting and using default value 1e-3.")
        Sp["sep_eta"] = 0.001

    Ns = Sp["Ns"]
    spheres = Sp["spheres_lst"]
    sep_mat = jnp.zeros((Ns, Ns), dtype=int)

    for sind in range(Ns):
        for tind in range(sind + 1, Ns):  # only the upper triangle; mirror below
            s_sph = spheres[sind]
            t_sph = spheres[tind]
            gap = jnp.linalg.norm(s_sph["Xc"] - t_sph["Xc"]) - s_sph["r"] - t_sph["r"]
            max_r = max(s_sph["r"], t_sph["r"])   # Python floats; jnp.max rejects a list
            is_far = int(gap > Sp["sep_eta"] * max_r)
            sep_mat = sep_mat.at[tind, sind].set(is_far)
            sep_mat = sep_mat.at[sind, tind].set(is_far)

    return sep_mat


def separate_targets(trg: jax.Array, Sp: SuspensionDict) -> jax.Array:
    """
    Given a list of targets, return a (Ntrg x Ns) array of [0,1]s giving whether each target
    is far (1) or near (0) for each source sphere, using sep_eta.
    """

    assert Sp["sep_eta"] >= 0.

    # reshape target to Ntrg x 3
    if trg.shape[1] != 3:
        if trg.shape[0] != 3:
            raise Exception("target should be a Ntrg x 3 array.")
        else:
            trg = jnp.transpose(trg)

    cols = [separate_target(trg, Sp["spheres_lst"][sind], Sp["sep_eta"]).astype(int)
            for sind in range(Sp["Ns"])]
    return jnp.stack(cols, axis=1)  # (Ntrg, Ns)


def _block_slice(Sp: SuspensionDict, sind: int) -> slice:
    """Row range owned by sphere `sind` in a flat suspension density/BC vector."""
    dsp = Sp["Nnodes_dsp"]
    return slice(int(dsp[sind]), int(dsp[sind + 1]))


def Lap3d_onsurf_apply(sigma: jax.Array, Sp: SuspensionDict, sh_lst: list,
                       sl_scal_lst, dl_scal_lst, sgn_lst,
                       sep_mat: jax.Array = None) -> jax.Array:
    """
    Apply the suspension on-surface operator K[sigma] for Laplace:
        for each target sphere t,  (K sigma)_t = sum_s K_{t,s}[sigma_s]
        - s == t : self term <dl_scal>*(1/2*sgn I + D) + <sl_scal>*S   (Lap3d.bio_onsurf_apply)
        - s != t : sphere-to-sphere layer potential evaluated at t's surface nodes
                   (Lap3d.bio_offsurf_apply, the off-surface point evaluator).

    sigma : flat 1-D complex array, length Sp["Nnodes_dsp"][-1].
    sl_scal_lst, dl_scal_lst, sgn_lst : per-sphere scalars (length Ns lists/arrays).
    sep_mat : optional (Ns,Ns) far/near matrix; computed via separate_spheres if None.

    Returns a flat 1-D complex array of the same length.
    """
    Ns = Sp["Ns"]
    spheres = Sp["spheres_lst"]
    sigma = jnp.asarray(sigma, dtype=jnp.complex128).reshape(-1)
    assert int(Sp["Nnodes_dsp"][-1]) == sigma.shape[0]

    if sep_mat is None:
        sep_mat = separate_spheres(Sp)

    y = jnp.zeros_like(sigma)

    for tind in range(Ns):
        t_sph = spheres[tind]
        t_slice = _block_slice(Sp, tind)
        acc = jnp.zeros((int(Sp["Nnodes_lst"][tind]),), dtype=jnp.complex128)

        for sind in range(Ns):
            s_sph = spheres[sind]
            s_grid = s_sph["Xcart"].shape[:2]
            sigma_s = sigma[_block_slice(Sp, sind)]

            if sind == tind:
                # Self interaction on the source/target grid (includes the DL jump).
                self_out = Lap3d.bio_onsurf_apply(
                    sigma_s.reshape(s_grid), sh_lst[sind],
                    sl_scal_lst[sind], dl_scal_lst[sind], sgn_lst[sind])
                acc = acc + self_out.reshape(-1)
            else:
                # Evaluate sphere `sind`'s layer potential at sphere `tind`'s surface nodes.
                s_sph = set_density(s_sph, sigma_s.reshape(s_grid))
                trg = t_sph["Xcart"].reshape(-1, 3)
                # if sep_mat[tind, sind] == 0: 
                #     print("\n in near regime")
                # else:
                #     print("\n in far regime")
                cross = Lap3d.bio_offsurf_apply(
                    trg, s_sph, sh_lst[sind], sl_scal_lst[sind], dl_scal_lst[sind], sep_mat[tind, sind] == 0)

                acc = acc + cross.reshape(-1)   

        y = y.at[t_slice].set(acc)

    return y


def Lap3d_onsurf_solve(bc_pot: jax.Array, Sp: SuspensionDict, sh_lst: list,
                       sl_scal_lst, dl_scal_lst, sgn_lst,
                       tol: float = 1e-13, atol: float = 1e-15, maxiter: int = 100,
                       precond: bool = True):
    """
    Solve K[sigma] = bc_pot for the suspension surface density sigma.

    The coupled operator mixes the JAX-spectral self blocks with the (non-traceable, numpy)
    off-surface point evaluator for the cross blocks, so the GMRES is driven eagerly with
    scipy.sparse.linalg.gmres rather than lineax.

    When precond=True, a block-Jacobi preconditioner applies each sphere's spectral direct
    self-solve (Lap3d.bio_onsurf_direct_solve) -- the exact inverse of the diagonal blocks.

    Returns (sigma_flat, info, resid) where info is scipy's GMRES status (0 == converged).
    """
    dsp = Sp["Nnodes_dsp"]
    N = int(dsp[-1])
    sep_mat = separate_spheres(Sp)
    bc_pot = jnp.asarray(bc_pot, dtype=jnp.complex128).reshape(-1)

    def matvec(x):
        y = Lap3d_onsurf_apply(jnp.asarray(x), Sp, sh_lst,
                               sl_scal_lst, dl_scal_lst, sgn_lst, sep_mat)
        # np.array (not asarray): scipy's GMRES mutates the matvec output in place, so it
        # must be a writable copy rather than a read-only view of the immutable jax array.
        return np.array(y, dtype=np.complex128)

    A = spla.LinearOperator((N, N), matvec=matvec, dtype=np.complex128)

    M = None
    if precond:
        def psolve(r):
            r = jnp.asarray(r, dtype=jnp.complex128).reshape(-1)
            z = jnp.zeros_like(r)
            for sind in range(Sp["Ns"]):
                sl = _block_slice(Sp, sind)
                grid = Sp["spheres_lst"][sind]["Xcart"].shape[:2]
                zs = Lap3d.bio_onsurf_direct_solve(
                    r[sl].reshape(grid), sh_lst[sind],
                    sl_scal_lst[sind], dl_scal_lst[sind], sgn_lst[sind])
                z = z.at[sl].set(zs.reshape(-1))
            return np.array(z, dtype=np.complex128)
        M = spla.LinearOperator((N, N), matvec=psolve, dtype=np.complex128)

    b = np.asarray(bc_pot, dtype=np.complex128)
    try:
        sol, info = spla.gmres(A, b, M=M, rtol=tol, atol=atol, maxiter=maxiter)
    except TypeError:
        # Older SciPy: `tol` instead of `rtol`.
        sol, info = spla.gmres(A, b, M=M, tol=tol, atol=atol, maxiter=maxiter)

    sigma = jnp.asarray(sol, dtype=jnp.complex128)
    resid = float(jnp.linalg.norm(matvec(sigma) - b))
    return sigma, info, resid


def _block_slice3(Sp: SuspensionDict, sind: int) -> slice:
    """Row range owned by sphere `sind` in a flat Stokes suspension density/BC vector
    (3 velocity components per node, so 3x the scalar offsets)."""
    dsp = Sp["Nnodes_dsp"]
    return slice(3 * int(dsp[sind]), 3 * int(dsp[sind + 1]))


def Stk3d_onsurf_apply(sigma: jax.Array, Sp: SuspensionDict, sh_lst: list,
                       sl_scal_lst, dl_scal_lst, sgn_lst,
                       sep_mat: jax.Array = None) -> jax.Array:
    """
    Apply the suspension on-surface operator K[sigma] for Stokes (vector density).
        for each target sphere t,  (K sigma)_t = sum_s K_{t,s}[sigma_s]
        - s == t : self term (Stk3d.bio_onsurf_apply, includes the DL jump), with the
                   source-sphere radius threaded so the SL block scales correctly.
        - s != t : sphere-to-sphere layer potential at t's surface nodes
                   (Stk3d.bio_offsurf_apply).

    sigma : flat 1-D complex array, length 3 * Sp["Nnodes_dsp"][-1] (3 components/node).
            Each per-sphere block is the C-order flatten of the (nphi, ntheta, 3) field.
    Returns a flat 1-D complex array of the same length.
    """
    Ns = Sp["Ns"]
    spheres = Sp["spheres_lst"]
    sigma = jnp.asarray(sigma, dtype=jnp.complex128).reshape(-1)
    assert 3 * int(Sp["Nnodes_dsp"][-1]) == sigma.shape[0]

    if sep_mat is None:
        sep_mat = separate_spheres(Sp)

    y = jnp.zeros_like(sigma)

    for tind in range(Ns):
        t_sph = spheres[tind]
        acc = jnp.zeros((3 * int(Sp["Nnodes_lst"][tind]),), dtype=jnp.complex128)

        for sind in range(Ns):
            s_sph = spheres[sind]
            nphi, ntheta = s_sph["Xcart"].shape[:2]
            sigma_s = sigma[_block_slice3(Sp, sind)].reshape(nphi, ntheta, 3)

            if sind == tind:
                self_out = Stk3d.bio_onsurf_apply(
                    sigma_s, s_sph["Xsph"][:, :, 0], s_sph["Xsph"][:, :, 1], sh_lst[sind],
                    sl_scal_lst[sind], dl_scal_lst[sind], sgn_lst[sind], radius=s_sph["r"])
                acc = acc + self_out.reshape(-1)
            else:
                s_sph = set_density(s_sph, sigma_s[:, :, 0], sigma_s[:, :, 1], sigma_s[:, :, 2])
                trg = t_sph["Xcart"].reshape(-1, 3)
                # cross = Stk3d.bio_offsurf_apply(
                #     trg, s_sph, sh_lst[sind], sl_scal_lst[sind], dl_scal_lst[sind],
                #     sep_mat[tind, sind] == 0)
                cross = Stk3d.point_n_shoot(
                    t_sph, sh_lst[tind], s_sph, sh_lst[sind], sl_scal_lst[sind], dl_scal_lst[sind],
                    sep_mat[tind, sind] == 0)
                acc = acc + cross.reshape(-1)

        y = y.at[_block_slice3(Sp, tind)].set(acc)

    return y


def Stk3d_onsurf_solve(bc_vec: jax.Array, Sp: SuspensionDict, sh_lst: list,
                       sl_scal_lst, dl_scal_lst, sgn_lst,
                       tol: float = 1e-13, atol: float = 1e-15, maxiter: int = 200,
                       precond: bool = True, callback=None):
    """
    Solve K[sigma] = bc_vec for the suspension Stokes surface density sigma (vector).

    Mirrors Lap3d_onsurf_solve: scipy GMRES on the matrix-free coupled operator, with an
    optional block-Jacobi preconditioner that applies each sphere's spectral direct self-solve
    (Stk3d.stokes_onsurf_direct_solve, radius-aware).

    <callback> (optional) is forwarded to scipy.sparse.linalg.gmres (e.g. to count iterations).

    Returns (sigma_flat, info, resid).
    """
    dsp = Sp["Nnodes_dsp"]
    N = 3 * int(dsp[-1])
    sep_mat = separate_spheres(Sp)
    bc_vec = jnp.asarray(bc_vec, dtype=jnp.complex128).reshape(-1)

    def matvec(x):
        y = Stk3d_onsurf_apply(jnp.asarray(x), Sp, sh_lst,
                               sl_scal_lst, dl_scal_lst, sgn_lst, sep_mat)
        return np.array(y, dtype=np.complex128)

    A = spla.LinearOperator((N, N), matvec=matvec, dtype=np.complex128)

    M = None
    if precond:
        def psolve(r):
            r = jnp.asarray(r, dtype=jnp.complex128).reshape(-1)
            z = jnp.zeros_like(r)
            for sind in range(Sp["Ns"]):
                s_sph = Sp["spheres_lst"][sind]
                nphi, ntheta = s_sph["Xcart"].shape[:2]
                sl = _block_slice3(Sp, sind)
                zs = Stk3d.stokes_onsurf_direct_solve(
                    r[sl].reshape(nphi, ntheta, 3), s_sph["Xsph"][:, :, 0], s_sph["Xsph"][:, :, 1],
                    sh_lst[sind], sl_scal_lst[sind], dl_scal_lst[sind], sgn_lst[sind],
                    radius=s_sph["r"])
                z = z.at[sl].set(zs.reshape(-1))
            return np.array(z, dtype=np.complex128)
        M = spla.LinearOperator((N, N), matvec=psolve, dtype=np.complex128)

    b = np.asarray(bc_vec, dtype=np.complex128)
    try:
        sol, info = spla.gmres(A, b, M=M, rtol=tol, atol=atol, maxiter=maxiter, callback=callback)
    except TypeError:
        sol, info = spla.gmres(A, b, M=M, tol=tol, atol=atol, maxiter=maxiter, callback=callback)

    sigma = jnp.asarray(sol, dtype=jnp.complex128)
    resid = float(jnp.linalg.norm(matvec(sigma) - b))
    return sigma, info, resid


if __name__ == "__main__":
    """
    Two-sphere exterior Laplace manufactured-solution test.

    Two well-separated unit spheres; one interior point charge per sphere generates a field
    that is harmonic in the exterior of both. We solve the coupled BIE for the surface
    densities, then evaluate the combined layer potential at exterior check points (far from
    both spheres) and compare against the exact potential.
    """

    lmax = 36
    sl_scal = 1.0
    dl_scal = 1.0
    sep_eta = 0.1

    # ===== TEST 1 & TEST 2 commented out (kept verbatim); only TEST 3 runs below =====
    '''
    print("=========== TEST 1: Manufactured solution exterior to two spheres ==============")
    centers = jnp.array([[0., 0., 0.], [3., 0., 0.]])
    radii = jnp.array([1.0, 1.0])
    Sp = build_suspension(centers, radii, sep_eta)
    Sp, sh_lst = quadr_suspension(Sp, jnp.array([lmax, lmax]))
    Ns = Sp["Ns"]

    # Exterior problem on every sphere.
    sl_lst = [sl_scal] * Ns
    dl_lst = [dl_scal] * Ns
    sgn_lst = [1.0] * Ns

    # One interior point charge per sphere (singularities live inside the spheres).
    ptsrc = jnp.array([[0.1, 0.3, 0.15], [2.8, 0.1, -0.15]])
    force = jnp.array([[1.0], [-0.7]])

    # Boundary condition: exact potential sampled on each sphere's surface nodes.
    dsp = Sp["Nnodes_dsp"]
    bc = jnp.zeros((int(dsp[-1]),), dtype=jnp.complex128)
    for s in range(Ns):
        nodes = Sp["spheres_lst"][s]["Xcart"].reshape(-1, 3)
        pot = Lap3d.compute_potential(nodes, ptsrc, force)  # (Nn, 1) complex
        bc = bc.at[int(dsp[s]):int(dsp[s + 1])].set(pot.reshape(-1))

    # Coupled solve.
    sigma, info, resid = Lap3d_onsurf_solve(bc, Sp, sh_lst, sl_lst, dl_lst, sgn_lst)
    print(f"GMRES info (0 == converged): {info}, residual = {resid:.3e}")

    # Accuracy at exterior check points, well separated from both spheres.
    chk = jnp.array([[6., 1., 0.5], [1.5, 5., -2.], [-4., -3., 2.]])
    true_pot = jnp.real(Lap3d.compute_potential(chk, ptsrc, force))
    approx = jnp.zeros((chk.shape[0], 1), dtype=jnp.complex128)
    for s in range(Ns):
        grid = Sp["spheres_lst"][s]["Xcart"].shape[:2]
        s_sph = set_density(Sp["spheres_lst"][s],
                            sigma[int(dsp[s]):int(dsp[s + 1])].reshape(grid))
        approx = approx + Lap3d.bio_offsurf_apply(chk, s_sph, sh_lst[s], sl_scal, dl_scal)
    approx = jnp.real(approx)

    rel_err = jnp.max(jnp.abs(true_pot - approx)) / jnp.max(jnp.abs(true_pot))
    print(f"Two-sphere exterior Laplace: max relative error at exterior targets "
          f"(lmax={lmax}) = {float(rel_err):.3e}")


    print("=========== TEST 2: Manufactured solution exterior to two Stokes spheres (radii != 1) ==========")
    centers = jnp.array([[0., 0., 0.], [3., 0., 0.]])
    radii = jnp.array([1.0, 0.5])  # second sphere is non-unit: exercises solid-harmonic radius scaling
    Sp = build_suspension(centers, radii, sep_eta)
    Sp, sh_lst = quadr_suspension(Sp, jnp.array([lmax, lmax]))
    Ns = Sp["Ns"]

    # Exterior Stokes problem on every sphere.
    sl_lst = [sl_scal] * Ns
    dl_lst = [dl_scal] * Ns
    sgn_lst = [1.0] * Ns

    # One interior point Stokeslet per sphere (singularities live inside the spheres).
    ptsrc = jnp.array([[0.1, 0.3, 0.15], [2.95, 0.1, 0.05]])
    force = jnp.array([[1.0, 0.5, -0.3], [-0.7, 0.2, 0.4]])

    # Boundary condition: exact Stokeslet velocity sampled on each sphere's surface nodes
    # (flat length 3*dsp[-1]; each block is the C-order flatten of (nphi, ntheta, 3)).
    dsp = Sp["Nnodes_dsp"]
    bc = jnp.zeros((3 * int(dsp[-1]),), dtype=jnp.complex128)
    for s in range(Ns):
        nodes = Sp["spheres_lst"][s]["Xcart"].reshape(-1, 3)
        vel = Stk3d.compute_field(nodes, ptsrc, force)  # (Nn, 3) complex
        bc = bc.at[3 * int(dsp[s]):3 * int(dsp[s + 1])].set(vel.reshape(-1))

    # Coupled solve.
    sigma, info, resid = Stk3d_onsurf_solve(bc, Sp, sh_lst, sl_lst, dl_lst, sgn_lst)
    print(f"GMRES info (0 == converged): {info}, residual = {resid:.3e}")

    # Accuracy at exterior check points, well separated from both spheres.
    chk = jnp.array([[6., 1., 0.5], [1.5, 5., -2.], [-4., -3., 2.]])
    true_vel = jnp.real(Stk3d.compute_field(chk, ptsrc, force))
    approx = jnp.zeros((chk.shape[0], 3), dtype=jnp.complex128)
    for s in range(Ns):
        nphi, ntheta = Sp["spheres_lst"][s]["Xcart"].shape[:2]
        sig_s = sigma[3 * int(dsp[s]):3 * int(dsp[s + 1])].reshape(nphi, ntheta, 3)
        s_sph = set_density(Sp["spheres_lst"][s], sig_s[:, :, 0], sig_s[:, :, 1], sig_s[:, :, 2])
        approx = approx + Stk3d.bio_offsurf_apply(chk, s_sph, sh_lst[s], sl_scal, dl_scal)
    approx = jnp.real(approx)

    rel_err = jnp.max(jnp.abs(true_vel - approx)) / jnp.max(jnp.abs(true_vel))
    print(f"Two-sphere exterior Stokes: max relative error at exterior targets "
          f"(lmax={lmax}, radii={[float(r) for r in radii]}) = {float(rel_err):.3e}")


    '''


    print("======= TEST 3: An obstacle with slip inside a no-slip container ===============")
    centers = jnp.array([[0., 0., 0.], [0.3, 0.1, -0.05]])
    radii = jnp.array([1.0, 0.2])
    Sp = build_suspension(centers, radii, sep_eta)
    Sp, sh_lst = quadr_suspension(Sp, jnp.array([lmax, lmax]))
    Ns = Sp["Ns"]

    sl_lst = [sl_scal] * Ns
    dl_lst = [dl_scal] * Ns
    sgn_lst = [-1.0, 1.0]  # interior problem on outer container, exterior problem on obstacle

    # Boundary condition (Stokes: 3 velocity components per node, flat length 3*dsp[-1]).
    dsp = Sp["Nnodes_dsp"]
    bc = jnp.zeros((3 * int(dsp[-1]),), dtype=jnp.complex128)  # no-slip container (sphere 0)
    U = 1.0
    vslip_mag = lambda theta: jnp.sin(theta) * 3. / 2. * U   # squirmer tangential surface speed
    # Tangential slip u = vslip_mag(theta) e_theta on the obstacle (sphere 1).
    obst = Sp["spheres_lst"][1]
    th1 = obst["Xsph"][:, :, 0]
    ph1 = obst["Xsph"][:, :, 1]
    zeros1 = jnp.zeros_like(th1)
    sx, sy, sz = Stk3d.sph2cart(zeros1, vslip_mag(th1), zeros1, th1, ph1)
    bc_obs = jnp.stack([sx, sy, sz], axis=2)                 # (nphi, ntheta, 3)
    bc = bc.at[3 * int(dsp[1]):3 * int(dsp[2])].set(bc_obs.reshape(-1))

    # Coupled solve (may not fully converge; the field is judged visually).
    sigma, info, resid = Stk3d_onsurf_solve(bc, Sp, sh_lst, sl_lst, dl_lst, sgn_lst)
    print(f"GMRES info (0 == converged): {info}, residual = {resid:.3e}")

    # Evaluate the flow on a grid interior to the container and exterior to the obstacle.
    Ng = 26
    trg_data = vtk_export.grid_from_spheres(Sp, Ng, pad=0.001)
    trg_grid = trg_data["points"]
    rr_0 = jnp.linalg.norm(trg_grid - centers[0, :], axis=1)
    rr_1 = jnp.linalg.norm(trg_grid - centers[1, :], axis=1)
    in_0 = rr_0 < radii[0] * 0.999   # strictly interior to container
    in_1 = rr_1 > radii[1] * 1.001   # strictly exterior to obstacle

    Ufield = np.zeros((trg_grid.shape[0], 3), dtype=float) # for plotting
    if np.any(in_0 & in_1):
        trg_in = jnp.asarray(trg_grid[in_0 & in_1])
        approx = jnp.zeros((trg_in.shape[0], 3), dtype=jnp.complex128)
        for s in range(Ns):
            nphi, ntheta = Sp["spheres_lst"][s]["Xcart"].shape[:2]
            sig_s = sigma[3 * int(dsp[s]):3 * int(dsp[s + 1])].reshape(nphi, ntheta, 3)
            s_sph = set_density(Sp["spheres_lst"][s], sig_s[:, :, 0], sig_s[:, :, 1], sig_s[:, :, 2])
            approx = approx + Stk3d.bio_offsurf_apply(trg_in, s_sph, sh_lst[s], sl_scal, dl_scal)
        Ufield[in_0 & in_1] = np.real(np.asarray(approx))
    
    vis_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vis")
    os.makedirs(vis_dir, exist_ok=True)
    vtk_export.export_field(trg_data, Ufield, os.path.join(vis_dir, "container_obstacle_field.vtk"),
                                name="velocity")
    print("Wrote VTK (field) to", vis_dir)
    
    # Surface boundary condition per node (zero on container, slip on obstacle), for plotting.
    bc_vec = np.real(np.asarray(bc)).reshape(-1, 3)
    vtk_export.export_objects(os.path.join(vis_dir, "container_obstacle_geometry.vtk"), Sp, bc_vec)
    print("Wrote VTK (geometry) to", vis_dir)

