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

import os
# os.environ.setdefault("OMP_NUM_THREADS", "1")

from typing import Dict, Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np

import sys
import time

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
from biop import Stk3d_np
import vtk_export

import scipy.sparse.linalg as spla
import lineax as lx
from lineax._norm import two_norm
from functools import partial

SuspensionDict = Dict[str, Any]

class IterationCounter:
    """Counts GMRES iterations (passed as the scipy gmres callback)."""
    def __init__(self):
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1

class LineaxMatchingCallback:
    def __init__(self, A, b):
        self.A = A
        self.b = b
        self.count = 0
        self.residuals = []

    def __call__(self, x_k):
        self.count += 1
        
        # Lineax evaluates the actual residual norm ||b - Ax||_2 
        # at the end of its JAX while_loop steps.
        current_residual = np.linalg.norm(self.b - self.A @ x_k)
        self.residuals.append(current_residual)
        
        print(f"Step {self.count} | Residual Norm: {current_residual:.6e}")

def build_suspension(center_lst: jax.Array, radius_lst: jax.Array, sep_eta: float = 0.01) -> SuspensionDict:
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


def quadr_suspension(Sp: SuspensionDict, lmax_lst: jax.Array) -> Tuple[SuspensionDict, tuple]:
    """
    Set up the quadrature grid on each sphere from lmax_lst, returning the updated suspension
    and the list of corresponding shtns objects.
    Note that Sp no longer stores sh_lst due to JAX tracers, and sh_lst is stored as a tuple which is hashable.
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

    # Sp["sh_lst"] = sh_lst
    Sp["Nnodes_lst"] = jnp.asarray(nnodes, dtype=int)
    Sp["Nnodes_dsp"] = jnp.concatenate([jnp.zeros((1,), dtype=int),
                                        jnp.cumsum(jnp.asarray(nnodes, dtype=int))])

    return Sp, tuple(sh_lst)


def separate_spheres(Sp: SuspensionDict) -> jax.Array:
    """
    return a Ns x Ns array of [0,1]s that is the separation matrix among spheres in the suspension.
    Heuristically, if two spheres' closest surface points fall within sep_eta, consider them
    close (0); otherwise far (1). The diagonal is 0 (self-interaction handled separately).
    """

    # NOTE: for jax tracing, defaulting without notice.
    # if Sp["sep_eta"] < 0.:
    #     print("Warning: no separation eta set by user; setting and using default value 1e-3.")
    #     Sp["sep_eta"] = 0.001

    Ns = int(Sp["Ns"])
    spheres = Sp["spheres_lst"]

    centers = jnp.stack([sph["Xc"] for sph in spheres])          # (Ns, 3)
    radii = jnp.stack([jnp.asarray(sph["r"]) for sph in spheres])  # (Ns,)

    # Pairwise closest-surface gap. Two non-overlapping configurations give a positive
    # clearance: external separation (gap_ext = dist - r_i - r_j) and containment, where one
    # sphere sits inside the other (gap_int = R_outer - R_inner - dist, the clearance between
    # the inner sphere's farthest point and the outer wall). The true surface gap is whichever
    # is positive, so take the max. Note gap_ext - gap_int = 2*(dist - max_r), so gap_int only
    # wins when dist < max_r (a nested pair) -- for every non-nested pair this reduces exactly
    # to the old gap_ext, leaving obstacle<->obstacle routing byte-identical. Without gap_int a
    # containment pair (interior sphere inside the container) always clamps to 0 -> near, so the
    # container<->interior pairs never reached the far path even when well clear of the wall.
    dist = jnp.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)  # (Ns, Ns)
    max_r = jnp.maximum(radii[:, None], radii[None, :])                         # (Ns, Ns)
    min_r = jnp.minimum(radii[:, None], radii[None, :])                         # (Ns, Ns)
    gap_ext = dist - radii[:, None] - radii[None, :]                            # external separation
    gap_int = max_r - min_r - dist                                             # containment clearance
    gap = jnp.maximum(gap_ext, gap_int)                                        # (Ns, Ns)
    gap = jnp.maximum(gap, jnp.zeros_like(gap))                                 # (Ns, Ns), take gap or 0, only diag should be <0.

    sep_mat = (gap > Sp["sep_eta"] * max_r).astype(int)

    # Diagonal is 0 (self-interaction handled separately); off-diagonal is symmetric.
    # sep_mat = is_far * (1 - jnp.eye(is_far.shape[0], dtype=int))

    return sep_mat


def separate_targets(trg: jax.Array, Sp: SuspensionDict) -> jax.Array:
    """
    Given a list of targets, return a (Ntrg x Ns) array of [0,1]s giving whether each target
    is far (1) or near (0) for each source sphere, using sep_eta.
    """

    # assert Sp["sep_eta"] >= 0.

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
                # sep_mat[t, s] == 1 means sphere s is FAR from t (separate_spheres), so that
                # is the smooth-quadrature branch; near pairs take the spectral point eval.
                # (This flag used to be passed inverted, sending far pairs through the eager
                # per-point synthesis and near pairs through the far quadrature.)
                cross = Lap3d.bio_offsurf_apply(
                    trg, s_sph, sh_lst[sind], sl_scal_lst[sind], dl_scal_lst[sind], sep_mat[tind, sind] == 1)

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


def _block_bounds3(Sp: SuspensionDict, Ns: int) -> list:
    """Per-sphere (start, stop) row ranges in a flat Stokes density vector, computed from the
    grid SHAPES (3 * nphi * ntheta each). Shapes are static even when the underlying arrays are
    JAX tracers, so these bounds stay CONCRETE inside lineax's traced operator -- unlike
    int(Sp["Nnodes_dsp"][...]), which concretizes a (closure-lifted) tracer and fails."""
    spheres = Sp["spheres_lst"]
    bounds, start = [], 0
    for s in range(Ns):
        nphi, ntheta = spheres[s]["Xcart"].shape[:2]
        stop = start + 3 * nphi * ntheta
        bounds.append((start, stop))
        start = stop
    return bounds


def build_ps_evaluators(Sp: SuspensionDict, Ns: int, sh_lst, sep_mat: jax.Array) -> dict:
    """Eagerly build the jitted point-and-shoot evaluator for every ordered NEAR off-diagonal
    pair (tind != sind and sep_mat[tind, sind] == 0), keyed (tind, sind). Each evaluator
    (Stk3d.point_n_shoot_evaluator) closes over its pair's CONCRETE geometry (rotation
    C-objects, ring constants) and maps a source density (nphi_s, ntheta_s, 3) -> velocity at
    sphere tind's grid. Built once here, outside the lineax-traced matvec, so the eager
    rotation/keying work never runs under a trace.

    FAR pairs are intentionally excluded: they are handled by the rotation-free smooth-quadrature
    far evaluators (build_far_evaluators), which removes their expensive per-pair rotation FFI
    primitives from the fused matvec entirely."""
    spheres = Sp["spheres_lst"]
    evals = {}
    for tind in range(Ns):
        for sind in range(Ns):
            if sind != tind and bool(sep_mat[tind, sind] == 0):
                evals[(tind, sind)] = Stk3d.point_n_shoot_evaluator(
                    spheres[tind], sh_lst[tind], spheres[sind], sh_lst[sind], near=True)
    return evals


def build_far_evaluators(Sp: SuspensionDict, Ns: int, sh_lst, sep_mat: jax.Array,
                         far_chunk: int = None) -> dict:
    """Eagerly build one jitted direct far evaluator per SOURCE sphere that has >= 1 far target
    (sep_mat[tind, sind] == 1). Each source's far targets (the surface nodes of every far target
    sphere) are concatenated into a single static Xtrg list, and the evaluator maps the source
    density (nphi_s, ntheta_s, 3) -> velocity at all those targets via the rotation-free
    smooth-quadrature Stk3d.bio_offsurf_apply(..., far=True) (a pure-JAX dense Nystrom sum).
    The far kernel is forced (far=True) rather than bio_offsurf_apply's per-target far/near
    split because this runs inside a jitted matvec and the spectral near-eval is eager-only;
    the sphere-level sep_mat already guarantees every node of a far source's target is far.

    Peak memory is bounded by the far kernel itself, which tiles its target loop (see
    sphere.far_tile_map): the tile is sized from Nsrc by sphere.far_tile_size, or forced with
    <far_chunk> if that is given. Tiling inside the kernel (rather than around it, as this
    function used to do) means the density is synthesized to the source grid ONCE per matvec
    instead of once per operator per chunk.

    Returns {sind: (far_eval, dest)} where:
      - far_eval(vwx_s, sl, dl) -> (Ntrg_s, 3), vwx_s is the source density VWX coefficients
        (3, nlm_s); the far kernel synthesizes them back to the source grid internally.
      - dest is a static int index array of shape (3*Ntrg_s,): the flat rows in the suspension
        output vector that far_eval's flattened (Ntrg_s, 3) output writes to, in Xtrg order. This
        encodes the entire scatter (one .at[dest].add), so no per-chunk loop is needed.
    Only vwx_s/sl/dl are dynamic; Xtrg and source geometry (Xcart, Xncart, r, lmax) and sh are
    closed over concretely, so this traces cleanly inside the lineax matvec."""
    spheres = Sp["spheres_lst"]
    bounds = _block_bounds3(Sp, Ns)
    evals = {}
    for sind in range(Ns):
        far_t = [tind for tind in range(Ns)
                 if tind != sind and bool(sep_mat[tind, sind] == 1)]
        if not far_t:
            continue
        Xtrg_s = jnp.concatenate([spheres[t]["Xcart"].reshape(-1, 3) for t in far_t])  # (Ntrg_s, 3)
        Ntrg_s = int(Xtrg_s.shape[0])
        # Static scatter destination: flat output rows for each far target block, in Xtrg order.
        dest = jnp.concatenate([jnp.arange(bounds[t][0], bounds[t][1]) for t in far_t])  # (3*Ntrg_s,)

        s_sph = spheres[sind]
        sh_s = sh_lst[sind]

        def _make_far_eval(s_sph, sh_s, Xtrg_s):
            @jax.jit
            def far_eval(vwx_s, sl, dl):
                # far=True forces the pure-JAX smooth quadrature for every target: this runs
                # inside a jitted matvec, and bio_offsurf_apply's per-target far/near split
                # (far=None) would evaluate the spectral near-eval, which is eager-only (C
                # per-point synthesis, not traceable). The sphere-level sep_mat already
                # guarantees every node of a far target sphere is far, so no split is needed.
                # The density is passed as VWX coefficients (vwx_s); bio_offsurf_apply's far
                # kernel synthesizes them back to the source grid to run the direct quadrature.
                return Stk3d._stk_far(Xtrg_s, vwx_s, s_sph, sh_s, ("sl", "dl"),
                                      sl_scal=sl, dl_scal=dl, tile=far_chunk)  # (Ntrg_s, 3)
            return far_eval

        evals[sind] = (_make_far_eval(s_sph, sh_s, Xtrg_s), dest)
    return evals


def Stk3d_onsurf_apply(sigma: jax.Array, Sp: SuspensionDict, Ns: int, sh_lst: list,
                       sl_scal_lst, dl_scal_lst, sgn_lst, ps_evals: dict,
                       far_evals: dict) -> jax.Array:
    """
    Apply the suspension on-surface operator K[sigma] for Stokes (vector density).
        for each target sphere t,  (K sigma)_t = sum_s K_{t,s}[sigma_s]
        - s == t   : self term (Stk3d.bio_onsurf_apply, includes the DL jump), with the
                     source-sphere radius threaded so the SL block scales correctly.
        - s != t near : sphere-to-sphere point-and-shoot layer potential at t's surface nodes,
                     via the pre-built evaluator ps_evals[(t, s)] (only near pairs are keyed).
        - s != t far  : rotation-free smooth-quadrature layer potential of source s evaluated
                     at ALL its far targets in one call, via far_evals[s], then scattered to
                     each far target block.

    This is PURE JAX: block ranges come from static grid shapes (_block_bounds3) and the
    cross terms call already-compiled evaluators, so no eager int()/float()/point_n_shoot
    build runs here. Safe to hand to lineax (closure-converted, geometry stays constant).

    sigma : flat 1-D complex array, length 3 * Sp["Nnodes_dsp"][-1] (3 components/node).
            Each per-sphere block is the C-order flatten of the (nphi, ntheta, 3) field.
    Returns a flat 1-D complex array of the same length.
    """
    spheres = Sp["spheres_lst"]
    sigma = jnp.asarray(sigma, dtype=jnp.complex128).reshape(-1)
    bounds = _block_bounds3(Sp, Ns)

    # self blocks + near cross (point-and-shoot): each target block is filled exactly once and
    # the blocks tile the output contiguously, so assemble by concatenation (no scatter).
    ylist = []
    for tind in range(Ns):
        t_sph = spheres[tind]
        nphi_t, ntheta_t = t_sph["Xcart"].shape[:2]
        st, sp = bounds[tind]

        # self interaction on sphere tind (includes the DL jump)
        sigma_t = sigma[st:sp].reshape(nphi_t, ntheta_t, 3)
        acc = Stk3d.bio_onsurf_apply(
            sigma_t, t_sph["Xsph"][:, :, 0], t_sph["Xsph"][:, :, 1], sh_lst[tind],
            sl_scal_lst[tind], dl_scal_lst[tind], sgn_lst[tind], radius=t_sph["r"]).reshape(-1)

        # near cross: point-and-shoot layer potential of source sind at sphere tind's grid
        for sind in range(Ns):
            if (tind, sind) in ps_evals:
                ss, se = bounds[sind]
                nphi_s, ntheta_s = spheres[sind]["Xcart"].shape[:2]
                sigma_s = sigma[ss:se].reshape(nphi_s, ntheta_s, 3)
                cross = ps_evals[(tind, sind)](sigma_s, sl_scal_lst[sind], dl_scal_lst[sind])
                acc = acc + cross.reshape(-1)
        ylist.append(acc)

    y = jnp.concatenate(ylist)

    # far cross: one chunked smooth-quadrature call per source over all its far targets, then a
    # SINGLE scatter-add over the precomputed destination indices (repeated indices across
    # sources hitting the same target block accumulate correctly under scatter-add).
    if far_evals:
        u_all = jnp.concatenate([
            far_evals[sind][0](
                sigma[bounds[sind][0]:bounds[sind][1]].reshape(*spheres[sind]["Xcart"].shape[:2], 3),
                sl_scal_lst[sind], dl_scal_lst[sind]).reshape(-1)
            for sind in far_evals])
        dest_all = jnp.concatenate([dest for (_, dest) in far_evals.values()])
        y = y.at[dest_all].add(u_all)

    return y


# ---------------------------------------------------------------------------
# NUMPY-ONLY TWINS (temporary diagnostic; selected by Stk3d_onsurf_solve_spla(numpy=True)).
# Mirror build_ps_evaluators / build_far_evaluators / Stk3d_onsurf_apply with all JAX removed
# (Stk3d_np.* kernels, np arrays, np.add.at scatter, Python chunk loop instead of jax.lax.map).
# ---------------------------------------------------------------------------
def build_ps_evaluators_np(Sp: SuspensionDict, Ns: int, sh_lst, sep_mat: jax.Array) -> dict:
    """Numpy twin of build_ps_evaluators: eagerly build the plain-Python point-and-shoot
    evaluator (Stk3d_np.point_n_shoot_evaluator_np) for every NEAR off-diagonal pair."""
    spheres = Sp["spheres_lst"]
    evals = {}
    for tind in range(Ns):
        for sind in range(Ns):
            if sind != tind and bool(sep_mat[tind, sind] == 0):
                evals[(tind, sind)] = Stk3d_np.point_n_shoot_evaluator_np(
                    spheres[tind], sh_lst[tind], spheres[sind], sh_lst[sind], near=True)
    return evals


def build_far_evaluators_np(Sp: SuspensionDict, Ns: int, sh_lst, sep_mat: jax.Array,
                            far_chunk: int = 2048) -> dict:
    """Numpy twin of build_far_evaluators. One plain far evaluator per source sphere with >=1
    far target; the (Ntrg_s x Nsrc x 3) broadcast in Stk3d_np.bio_offsurf_apply_np is bounded by
    a Python loop over target CHUNKS of <far_chunk> (numpy has no trace-memory constraint, so
    this is a simple for-loop rather than jax.lax.map). Returns {sind: (far_eval, dest)} with
    dest the flat output rows (np.arange) the (Ntrg_s,3) output scatters to, in Xtrg order."""
    spheres = Sp["spheres_lst"]
    bounds = _block_bounds3(Sp, Ns)
    evals = {}
    for sind in range(Ns):
        far_t = [tind for tind in range(Ns)
                 if tind != sind and bool(sep_mat[tind, sind] == 1)]
        if not far_t:
            continue
        Xtrg_s = np.concatenate([np.asarray(spheres[t]["Xcart"], dtype=np.float64).reshape(-1, 3)
                                 for t in far_t])                       # (Ntrg_s, 3)
        Ntrg_s = int(Xtrg_s.shape[0])
        dest = np.concatenate([np.arange(bounds[t][0], bounds[t][1]) for t in far_t])  # (3*Ntrg_s,)

        s_sph = spheres[sind]
        sh_s = sh_lst[sind]
        nphi_s, ntheta_s = s_sph["Xcart"].shape[:2]

        def _make_far_eval(s_sph, sh_s, Xtrg_s, Ntrg_s, nphi_s, ntheta_s):
            def far_eval(sigma_s, sl, dl):
                S = {**s_sph, "Sigma": np.asarray(sigma_s, dtype=np.complex128).reshape(nphi_s, ntheta_s, 3)}
                out = np.empty((Ntrg_s, 3), dtype=np.complex128)
                for st in range(0, Ntrg_s, far_chunk):
                    en = min(st + far_chunk, Ntrg_s)
                    out[st:en] = Stk3d_np.bio_offsurf_apply_np(Xtrg_s[st:en], S, sh_s, sl, dl, far=True)
                return out                                              # (Ntrg_s, 3)
            return far_eval

        evals[sind] = (_make_far_eval(s_sph, sh_s, Xtrg_s, Ntrg_s, nphi_s, ntheta_s), dest)
    return evals


def Stk3d_onsurf_apply_np(sigma, Sp: SuspensionDict, Ns: int, sh_lst: list,
                          sl_scal_lst, dl_scal_lst, sgn_lst, ps_evals: dict,
                          far_evals: dict):
    """Numpy twin of Stk3d_onsurf_apply. Same block structure (self + near via concatenation,
    far via a single scatter-add) but pure numpy: np.concatenate and np.add.at replace
    jnp.concatenate and .at[].add. Returns a flat complex128 numpy array."""
    spheres = Sp["spheres_lst"]
    sigma = np.asarray(sigma, dtype=np.complex128).reshape(-1)
    bounds = _block_bounds3(Sp, Ns)

    ylist = []
    for tind in range(Ns):
        t_sph = spheres[tind]
        nphi_t, ntheta_t = t_sph["Xcart"].shape[:2]
        st, sp = bounds[tind]

        sigma_t = sigma[st:sp].reshape(nphi_t, ntheta_t, 3)
        acc = Stk3d_np.bio_onsurf_apply_np(
            sigma_t, np.asarray(t_sph["Xsph"][:, :, 0], dtype=np.float64),
            np.asarray(t_sph["Xsph"][:, :, 1], dtype=np.float64), sh_lst[tind],
            sl_scal_lst[tind], dl_scal_lst[tind], sgn_lst[tind], radius=float(t_sph["r"])).reshape(-1)

        for sind in range(Ns):
            if (tind, sind) in ps_evals:
                ss, se = bounds[sind]
                nphi_s, ntheta_s = spheres[sind]["Xcart"].shape[:2]
                sigma_s = sigma[ss:se].reshape(nphi_s, ntheta_s, 3)
                cross = ps_evals[(tind, sind)](sigma_s, sl_scal_lst[sind], dl_scal_lst[sind])
                acc = acc + np.asarray(cross, dtype=np.complex128).reshape(-1)
        ylist.append(acc)

    y = np.concatenate(ylist)

    if far_evals:
        u_all = np.concatenate([
            far_evals[sind][0](
                sigma[bounds[sind][0]:bounds[sind][1]].reshape(*spheres[sind]["Xcart"].shape[:2], 3),
                sl_scal_lst[sind], dl_scal_lst[sind]).reshape(-1)
            for sind in far_evals])
        dest_all = np.concatenate([dest for (_, dest) in far_evals.values()])
        np.add.at(y, dest_all, u_all)

    return y


def Stk3d_onsurf_solve(bc_vec: jax.Array, Sp: SuspensionDict, Ns, Nnodes: int, sh_lst: list,
                       sl_scal_lst, dl_scal_lst, sgn_lst,
                       tol: float = 1e-11, atol: float = 1e-11, maxiter: int = 200,
                       precond: bool = True, far_chunk: int = None):
    """
    Solve K[sigma] = bc_vec for the suspension Stokes surface density sigma (vector).

    lineax GMRES on the matrix-free coupled operator. The operator (matvec) closes over the
    CONCRETE geometry in Sp / sh_lst / sep_mat, so only the flat density vector is dynamic --
    lineax then treats all geometry as constant operator data. The function is intentionally
    NOT jitted: the cross-sphere point_n_shoot build (shtns rotation C objects, whose angle is
    frozen in a C config) is eager, and this routine also times the solve on the host.

    An optional block-Jacobi preconditioner applies each sphere's spectral direct self-solve
    (Stk3d.stokes_onsurf_direct_solve, radius-aware) -- but ONLY for exterior spheres (sgn == +1).
    An interior-formulation sphere (sgn == -1) carries a double-layer nullspace, so its self-block
    is near-singular and inverting it (cond ~1e20) amplifies the near-null modes and stalls GMRES;
    that block is left as identity in the preconditioner instead.

    Convergence is judged from the recomputed true relative residual ||K sigma - bc|| / ||bc||,
    NOT from lineax's own result flag: lineax GMRES uses a componentwise, Cauchy-step (`diff`)
    criterion that with a tiny `atol` never registers a clean convergence exit at a
    machine-precision plateau (it falls out via `stagnation` instead), so its flag reads as a
    failure even when the density solves the system. We use `two_norm` and a reachable `atol`
    so the loop still terminates early, then report success from the true residual.

    Returns (sigma_flat, time_solve, niters, info, resid); info == 0 means the true relative
    residual is within `tol`. `resid` is the un-normalized ||K sigma - bc||.
    """
    N = 3 * int(Nnodes)                        # system size: 3 velocity components per node
    sep_mat = separate_spheres(Sp)             # eager -> concrete near/far flags
    bc_vec = jnp.asarray(bc_vec, dtype=jnp.complex128).reshape(-1)
    struct = jax.ShapeDtypeStruct((N,), jnp.complex128)
    bounds = _block_bounds3(Sp, Ns)            # static per-sphere flat row ranges

    # Pre-build the cross-evaluators EAGERLY (concrete geometry), outside the matvec -- lineax
    # closure-converts / eval_shapes the operator, which abstracts every closed-over array, so
    # eager rotation builds / _ps_geom_key (float(r), np.asarray(Xc)) cannot run under trace.
    # NEAR pairs -> point-and-shoot; FAR targets -> rotation-free smooth-quadrature per source.
    ps_evals = build_ps_evaluators(Sp, Ns, sh_lst, sep_mat)
    far_evals = build_far_evaluators(Sp, Ns, sh_lst, sep_mat, far_chunk=far_chunk)

    # Operator matvec: pure JAX, only `sigma` dynamic; concrete geometry closed over.
    def matvec(x):
        return Stk3d_onsurf_apply(x, Sp, Ns, sh_lst,
                                  sl_scal_lst, dl_scal_lst, sgn_lst, ps_evals, far_evals)
    # Eager warmup (concrete input): compiles the per-pair _core kernels + self blocks once,
    # before any tracing.
    jax.block_until_ready(matvec(jnp.zeros((N,), dtype=jnp.complex128)))

    gmres_func = lx.FunctionLinearOperator(matvec, struct)
    # norm=two_norm: lineax defaults to max_norm with a componentwise scale, which blows up on
    # small-|b| rows and never clears at tight tol; two_norm gives a standard relative criterion.
    solver = lx.GMRES(rtol=tol, atol=atol, max_steps=maxiter, stagnation_iters=50, norm=two_norm)

    options = {"y0": jnp.zeros((N,), dtype=jnp.complex128)}
    if precond:
        def psolve(r):
            r = jnp.asarray(r, dtype=jnp.complex128).reshape(-1)
            zlist = []
            for sind in range(Ns):
                s_sph = Sp["spheres_lst"][sind]
                nphi, ntheta = s_sph["Xcart"].shape[:2]
                ss, se = bounds[sind]
                # Interior formulation (sgn == -1): its self-block carries a double-layer
                # nullspace, so the direct self-solve is a near-singular inverse (cond ~1e20)
                # that amplifies the near-null modes and stalls GMRES. Use identity for that
                # block; only exterior spheres (sgn == +1) get the block-Jacobi self-solve.
                if float(sgn_lst[sind]) < 0.0:
                    zlist.append(r[ss:se])
                    continue
                zs = Stk3d.stokes_onsurf_direct_solve(
                    r[ss:se].reshape(nphi, ntheta, 3), s_sph["Xsph"][:, :, 0], s_sph["Xsph"][:, :, 1],
                    sh_lst[sind], sl_scal_lst[sind], dl_scal_lst[sind], sgn_lst[sind],
                    radius=s_sph["r"])
                zlist.append(zs.reshape(-1))
            return jnp.concatenate(zlist)   # jax array: keeps the solve on-device (no host sync)

        options["preconditioner"] = lx.FunctionLinearOperator(psolve, struct)

    # throw=False: on stagnation / non-convergence (e.g. the interior container problem, which
    # has a double-layer nullspace) lineax otherwise raises; instead return the best iterate and
    # report it via `info` below. Warmup solve compiles the fused GMRES loop; second solve timed.
    solution = lx.linear_solve(gmres_func, bc_vec, solver=solver, options=options, throw=False)
    jax.block_until_ready(solution.value)
    tstart = time.time()
    solution = lx.linear_solve(gmres_func, bc_vec, solver=solver, options=options, throw=False)
    jax.block_until_ready(solution.value)
    time_solve = time.time() - tstart

    sigma = solution.value
    niters = solution.stats["num_steps"]
    # Judge convergence from the true residual, not lineax's flag (see docstring): its Cauchy-step
    # criterion reports `stagnation` at a machine-precision plateau even when the solve is exact.
    resid = jnp.linalg.norm(matvec(sigma) - bc_vec)
    bc_norm = jnp.linalg.norm(bc_vec)
    resid_rel = resid / bc_norm if bc_norm > 0 else resid
    info = 0 if float(resid_rel) <= tol else 1

    return sigma, time_solve, niters, info, resid_rel

def Stk3d_onsurf_solve_spla(bc_vec: jax.Array, Sp: SuspensionDict, Ns, Nnodes: int, sh_lst: list,
                       sl_scal_lst, dl_scal_lst, sgn_lst,
                       tol: float = 1e-10, atol: float = 1e-14, maxiter: int = 200,
                       precond: bool = True, far_chunk: int = None, numpy: bool = False):
    """
    Solve K[sigma] = bc_vec for the suspension Stokes surface density sigma (vector).

    An optional block-Jacobi preconditioner applies each sphere's spectral direct self-solve
    (Stk3d.stokes_onsurf_direct_solve, radius-aware) -- but ONLY for exterior spheres (sgn == +1).
    An interior-formulation sphere (sgn == -1) carries a double-layer nullspace, so its self-block
    is near-singular and inverting it (cond ~1e20) amplifies the near-null modes and stalls GMRES;
    that block is left as identity in the preconditioner instead.

    numpy=True (TEMPORARY DIAGNOSTIC): run the operator/preconditioner through the pure-numpy
    Stk3d_np twins (plain shtns C calls, no JAX trace) instead of the JAX kernels, so the cost is
    attributable in an ordinary CPU profiler. The geometry, sep_mat, GMRES call and convergence
    reporting are otherwise identical; only the matvec/psolve internals differ.

    Returns (sigma_flat, time_solve, niters, info, resid); info == 0 means GMRES converged.
    """
    N = 3 * int(Nnodes)                        # system size: 3 velocity components per node
    sep_mat = separate_spheres(Sp)             # eager -> concrete near/far flags
    bc_vec = jnp.asarray(bc_vec, dtype=jnp.complex128).reshape(-1)
    struct = jax.ShapeDtypeStruct((N,), jnp.complex128)
    bounds = _block_bounds3(Sp, Ns)            # static per-sphere flat row ranges

    if numpy:
        ps_evals = build_ps_evaluators_np(Sp, Ns, sh_lst, sep_mat)
        far_evals = build_far_evaluators_np(Sp, Ns, sh_lst, sep_mat, far_chunk=far_chunk or 2048)
    else:
        ps_evals = build_ps_evaluators(Sp, Ns, sh_lst, sep_mat)
        far_evals = build_far_evaluators(Sp, Ns, sh_lst, sep_mat, far_chunk=far_chunk)

    # Operator matvec: numpy twin (pure numpy, no trace) or JAX (only `sigma` dynamic).
    def matvec(x):
        if numpy:
            return Stk3d_onsurf_apply_np(x, Sp, Ns, sh_lst,
                                         sl_scal_lst, dl_scal_lst, sgn_lst, ps_evals, far_evals)
        y = Stk3d_onsurf_apply(jnp.asarray(x), Sp, Ns, sh_lst,
                                  sl_scal_lst, dl_scal_lst, sgn_lst, ps_evals, far_evals)
        return np.array(y, dtype=np.complex128)

    A = spla.LinearOperator((N, N), matvec=matvec, dtype=np.complex128)

    counter = IterationCounter()

    M = None
    if precond:
        def psolve(r):
            # Interior formulation (sgn == -1): its self-block carries a double-layer nullspace,
            # so the direct self-solve is a near-singular inverse (cond ~1e20) that amplifies the
            # near-null modes and stalls GMRES. Use identity for that block; only exterior spheres
            # (sgn == +1) get the block-Jacobi self-solve.
            if numpy:
                r = np.asarray(r, dtype=np.complex128).reshape(-1)
                zlist = []
                for sind in range(Sp["Ns"]):
                    s_sph = Sp["spheres_lst"][sind]
                    nphi, ntheta = s_sph["Xcart"].shape[:2]
                    sl = _block_slice3(Sp, sind)
                    if float(sgn_lst[sind]) < 0.0:
                        zlist.append(r[sl])
                        continue
                    zs = Stk3d_np.stokes_onsurf_direct_solve_np(
                        r[sl].reshape(nphi, ntheta, 3),
                        np.asarray(s_sph["Xsph"][:, :, 0], dtype=np.float64),
                        np.asarray(s_sph["Xsph"][:, :, 1], dtype=np.float64),
                        sh_lst[sind], sl_scal_lst[sind], dl_scal_lst[sind], sgn_lst[sind],
                        radius=float(s_sph["r"]))
                    zlist.append(zs.reshape(-1))
                return np.concatenate(zlist)

            r = jnp.asarray(r, dtype=jnp.complex128).reshape(-1)
            z = jnp.zeros_like(r)
            for sind in range(Sp["Ns"]):
                s_sph = Sp["spheres_lst"][sind]
                nphi, ntheta = s_sph["Xcart"].shape[:2]
                sl = _block_slice3(Sp, sind)
                if float(sgn_lst[sind]) < 0.0:
                    z = z.at[sl].set(r[sl])
                    continue
                zs = Stk3d.stokes_onsurf_direct_solve(
                    r[sl].reshape(nphi, ntheta, 3), s_sph["Xsph"][:, :, 0], s_sph["Xsph"][:, :, 1],
                    sh_lst[sind], sl_scal_lst[sind], dl_scal_lst[sind], sgn_lst[sind],
                    radius=s_sph["r"])
                z = z.at[sl].set(zs.reshape(-1))
            return np.array(z, dtype=np.complex128)
        M = spla.LinearOperator((N, N), matvec=psolve, dtype=np.complex128)

    # Eager warmup (concrete zeros input): compile the jitted per-pair point-and-shoot / far
    # kernels + self blocks in matvec (and the preconditioner's direct self-solves) once here,
    # so the timed GMRES loop below pays no JIT-compile cost. (numpy=True: harmless no-compile
    # warm call.)
    jax.block_until_ready(matvec(np.zeros((N,), dtype=np.complex128)))
    if M is not None:
        jax.block_until_ready(psolve(np.zeros((N,), dtype=np.complex128)))

    b = np.asarray(bc_vec, dtype=np.complex128)

    tstart = time.time()
    sol, info = spla.gmres(A, b, M=M, rtol=tol, atol=atol, maxiter=maxiter, callback=counter)
    tend = time.time()
    t_solve = tend - tstart

    sigma = jnp.asarray(sol, dtype=jnp.complex128)
    bnorm = float(jnp.linalg.norm(b))
    resid = float(jnp.linalg.norm(matvec(sigma) - b)) if bnorm <= 0 else float(jnp.linalg.norm(matvec(sigma) - b)) / bnorm
    niters = counter.count
    return sigma, t_solve, niters, info, resid



if __name__ == "__main__":
    """
    Two-sphere exterior Laplace manufactured-solution test.

    Two well-separated unit spheres; one interior point charge per sphere generates a field
    that is harmonic in the exterior of both. We solve the coupled BIE for the surface
    densities, then evaluate the combined layer potential at exterior check points (far from
    both spheres) and compare against the exact potential.
    """

    # lmax = 36
    lmax = 100
    sl_scal = 1.0
    dl_scal = 1.0
    sep_eta = 0.1

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

    '''

    '''

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
    Nnodes = dsp[-1].item()
    # Max_nodes = int(max(Sp["Nnodes_lst"]))
    sigma, t_solve, niter, info, resid = Stk3d_onsurf_solve_spla(bc, Sp, Sp["Ns"], Nnodes, sh_lst, sl_lst, dl_lst, sgn_lst)
    print(f"GMRES info (0 == converged): {info}, time of solve is {t_solve}s, niter = {niter}, residual = {resid:.3e}")

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

    # '''
    print("======= TEST 3: An obstacle with slip inside a no-slip container ===============")
    centers = jnp.array([[0., 0., 0.], [0.3, 0.1, -0.05]])
    radii = jnp.array([1.0, 0.2])
    Sp = build_suspension(centers, radii, sep_eta)
    Sp, sh_lst = quadr_suspension(Sp, jnp.array([lmax, lmax]))
    Ns = Sp["Ns"]

    # sl_lst = [sl_scal] * Ns
    sl_lst = [sl_scal, sl_scal]
    dl_lst = [dl_scal, dl_scal]
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

    # # CHECK sum u \dot n in case solver instability caused by numerical error
    # xx = jnp.sin(th1) * jnp.cos(ph1)
    # yy = jnp.sin(th1) * jnp.sin(ph1)
    # zz = jnp.cos(th1)
    # udotn = bc_obs * jnp.stack([xx,yy,zz], axis=2) # (nphi, ntheta, 3)
    # udotn = jnp.sum(jnp.abs(jnp.sum(udotn, axis=2)))
    # print(f"u dot n on obstacle: {udotn:.8e}", flush=True)

    # Coupled solve (may not fully converge; the field is judged visually).
    Nnodes = dsp[-1].item()
    sigma, t_solve, niter, info, resid = Stk3d_onsurf_solve_spla(bc, Sp, Sp["Ns"], Nnodes, sh_lst, sl_lst, dl_lst, sgn_lst)
    print(f"GMRES info (0 == converged): {info}, time for solve is {t_solve}s, num of iters is {niter}, residual = {resid:.3e}")

    # # Evaluate the flow on a grid interior to the container and exterior to the obstacle.
    # Ng = 26
    # trg_data = vtk_export.grid_from_spheres(Sp, Ng, pad=0.001)
    # trg_grid = trg_data["points"]
    # rr_0 = jnp.linalg.norm(trg_grid - centers[0, :], axis=1)
    # rr_1 = jnp.linalg.norm(trg_grid - centers[1, :], axis=1)
    # in_0 = rr_0 < radii[0] * 0.999   # strictly interior to container
    # in_1 = rr_1 > radii[1] * 1.001   # strictly exterior to obstacle

    # Ufield = np.zeros((trg_grid.shape[0], 3), dtype=float) # for plotting
    # if np.any(in_0 & in_1):
    #     trg_in = jnp.asarray(trg_grid[in_0 & in_1])
    #     approx = jnp.zeros((trg_in.shape[0], 3), dtype=jnp.complex128)
    #     for s in range(Ns):
    #         nphi, ntheta = Sp["spheres_lst"][s]["Xcart"].shape[:2]
    #         sig_s = sigma[3 * int(dsp[s]):3 * int(dsp[s + 1])].reshape(nphi, ntheta, 3)
    #         s_sph = set_density(Sp["spheres_lst"][s], sig_s[:, :, 0], sig_s[:, :, 1], sig_s[:, :, 2])
    #         approx = approx + Stk3d.bio_offsurf_apply(trg_in, s_sph, sh_lst[s], sl_lst[s], dl_lst[s])
    #     Ufield[in_0 & in_1] = np.real(np.asarray(approx))
    
    # vis_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vis")
    # os.makedirs(vis_dir, exist_ok=True)
    # vtk_export.export_field(trg_data, Ufield, os.path.join(vis_dir, "container_obstacle_field.vtk"),
    #                             name="velocity")
    # print("Wrote VTK (field) to", vis_dir)
    
    # # Surface boundary condition per node (zero on container, slip on obstacle), for plotting.
    # bc_vec = np.real(np.asarray(bc)).reshape(-1, 3)
    # vtk_export.export_objects(os.path.join(vis_dir, "container_obstacle_geometry.vtk"), Sp, bc_vec)
    # print("Wrote VTK (geometry) to", vis_dir)
    # '''
