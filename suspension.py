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
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

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


# --- grid <-> spectral-coefficient change of basis (per-sphere blocks) ---------------------
# Used to run GMRES in COEFFICIENT space: the flat unknown is the concatenation of each sphere's
# diagonalizing-basis coefficients (Laplace: SH qlm, length nlm_cplx; Stokes: stacked VWX, length
# 3*nlm_cplx). The change of basis is applied at the solve boundaries (bc -> coeff before, solution
# coeff -> grid after) and inside the coeff-space matvec, which wraps the existing grid operator.

def _coeff_len_lap(Ns: int, sh_lst) -> int:
    # return int(sum(sh_lst[s].nlm_cplx for s in range(Ns)))   # complex/full SH layout
    return int(sum(sh_lst[s].nlm for s in range(Ns)))          # real/truncated SH layout

def _coeff_len_stk(Ns: int, sh_lst) -> int:
    # return int(sum(3 * sh_lst[s].nlm_cplx for s in range(Ns)))   # complex/full VSHT layout
    return int(sum(3 * sh_lst[s].nlm for s in range(Ns)))          # real/truncated VSHT layout

def grid2coeff_lap(vg: jax.Array, Sp: SuspensionDict, Ns: int, sh_lst) -> jax.Array:
    """Flat scalar grid vector -> concatenated per-sphere SH coefficients (analys per block)."""
    # vg = jnp.asarray(vg, dtype=jnp.complex128).reshape(-1)   # complex grid for the cplx analysis
    vg = jnp.real(jnp.asarray(vg)).reshape(-1)                 # real/truncated analys_jax takes float64
    out = []
    for s in range(Ns):
        grid = Sp["spheres_lst"][s]["Xcart"].shape[:2]
        # out.append(sh_lst[s].analys_cplx_jax(vg[_block_slice(Sp, s)].reshape(grid)))   # cplx
        out.append(sh_lst[s].analys_jax(vg[_block_slice(Sp, s)].reshape(grid)))          # real
    return jnp.concatenate(out)

def coeff2grid_lap(vc: jax.Array, Sp: SuspensionDict, Ns: int, sh_lst) -> jax.Array:
    """Concatenated per-sphere SH coefficients -> flat scalar grid vector (synth per block)."""
    vc = jnp.asarray(vc, dtype=jnp.complex128).reshape(-1)
    out, c0 = [], 0
    for s in range(Ns):
        # n = sh_lst[s].nlm_cplx; out.append(sh_lst[s].synth_cplx_jax(vc[c0:c0 + n]).reshape(-1))   # cplx
        n = sh_lst[s].nlm                                                                            # real
        out.append(sh_lst[s].synth_jax(vc[c0:c0 + n]).reshape(-1)); c0 += n
    return jnp.concatenate(out)

def grid2coeff_stk(vg: jax.Array, Sp: SuspensionDict, Ns: int, sh_lst, bounds: list) -> jax.Array:
    """Flat Cartesian grid vector -> concatenated per-sphere stacked-VWX coefficients (sig_xyz2vwx)."""
    # vg = jnp.asarray(vg, dtype=jnp.complex128).reshape(-1)   # complex grid for the cplx analysis
    vg = jnp.real(jnp.asarray(vg)).reshape(-1)                 # real/truncated analysis takes float64 grid
    out = []
    for s in range(Ns):
        sph = Sp["spheres_lst"][s]; nphi, ntheta = sph["Xcart"].shape[:2]; gst, gsp = bounds[s]
        th, ph = sph["Xsph"][:, :, 0], sph["Xsph"][:, :, 1]
        v = vg[gst:gsp].reshape(nphi, ntheta, 3)
        out.append(jnp.stack(Stk3d.sig_xyz2vwx(v[:, :, 0], v[:, :, 1], v[:, :, 2], th, ph, sh_lst[s])).reshape(-1))
    return jnp.concatenate(out)

def coeff2grid_stk(vc: jax.Array, Sp: SuspensionDict, Ns: int, sh_lst) -> jax.Array:
    """Concatenated per-sphere stacked-VWX coefficients -> flat Cartesian grid vector (sig_vwx2xyz)."""
    vc = jnp.asarray(vc, dtype=jnp.complex128).reshape(-1)
    out, c0 = [], 0
    for s in range(Ns):
        sph = Sp["spheres_lst"][s]; nphi, ntheta = sph["Xcart"].shape[:2]
        th, ph = sph["Xsph"][:, :, 0], sph["Xsph"][:, :, 1]
        # nlm = sh_lst[s].nlm_cplx; vwx = vc[c0:c0 + 3 * nlm].reshape(3, nlm); c0 += 3 * nlm   # cplx layout
        nlm = sh_lst[s].nlm; vwx = vc[c0:c0 + 3 * nlm].reshape(3, nlm); c0 += 3 * nlm          # real/truncated layout
        vx, vy, vz = Stk3d.sig_vwx2xyz(vwx[0], vwx[1], vwx[2], th, ph, sh_lst[s])
        out.append(jnp.stack([vx, vy, vz], axis=2).reshape(-1))
    return jnp.concatenate(out)


def Lap3d_onsurf_apply(sig_coeff: jax.Array, Sp: SuspensionDict, sh_lst: list,
                       sl_scal_lst, dl_scal_lst, sgn_lst,
                       sep_mat: jax.Array = None) -> jax.Array:
    """
    Apply the suspension on-surface operator K in COEFFICIENT space (Laplace):
        for each target sphere t,  (K c)_t = sum_s K_{t,s}[c_s]
        - s == t : self term <dl_scal>*(1/2*sgn I + D) + <sl_scal>*S   (Lap3d.bio_onsurf_apply,
                   diagonal in the SH basis, coeff -> coeff)
        - s != t : sphere-to-sphere layer potential evaluated at t's surface nodes
                   (Lap3d.bio_offsurf_apply) -- returns point VALUES, which are analysed back to
                   t's SH coefficients (the only change of basis needed; the far output is grid
                   point-values by nature).

    <sig_coeff> : flat 1-D complex array = concatenation of each sphere's SH coefficients
    (nlm_cplx per sphere). Returns a coefficient vector of the same layout. Taking/returning
    coefficients lets the GMRES matvec avoid the coeff->grid->coeff round trip (see the solvers).
    sl_scal_lst, dl_scal_lst, sgn_lst : per-sphere scalars. sep_mat : optional (Ns,Ns) far/near
    matrix; computed via separate_spheres if None.
    """
    Ns = Sp["Ns"]
    spheres = Sp["spheres_lst"]
    sig_coeff = jnp.asarray(sig_coeff, dtype=jnp.complex128).reshape(-1)

    if sep_mat is None:
        sep_mat = separate_spheres(Sp)

    # unpack per-sphere SH coefficient blocks
    cb, c0 = [], 0
    for s in range(Ns):
        # n = sh_lst[s].nlm_cplx; cb.append((c0, c0 + n)); c0 += n   # cplx layout
        n = sh_lst[s].nlm; cb.append((c0, c0 + n)); c0 += n          # real/truncated layout
    qlm_blocks = [sig_coeff[cb[s][0]:cb[s][1]] for s in range(Ns)]

    out = []
    for tind in range(Ns):
        t_sph = spheres[tind]
        # self interaction (diagonal in the SH basis, includes the DL jump), coeff -> coeff
        acc = Lap3d.bio_onsurf_apply(
            qlm_blocks[tind], sh_lst[tind], sl_scal_lst[tind], dl_scal_lst[tind], sgn_lst[tind])
        # cross: accumulate sphere-to-sphere layer potentials as point VALUES on t's grid,
        # then analyse once back to t's SH coefficients.
        if Ns > 1:
            trg = t_sph["Xcart"].reshape(-1, 3)
            # cross = jnp.zeros((trg.shape[0], 1), dtype=jnp.complex128)   # cplx bio_offsurf
            cross = jnp.zeros((trg.shape[0], 1), dtype=jnp.float64)        # real bio_offsurf -> float64
            for sind in range(Ns):
                if sind == tind:
                    continue
                # sep_mat[t, s] == 1 means sphere s is FAR from t (separate_spheres), so that
                # is the smooth-quadrature branch; near pairs take the spectral point eval.
                # (This flag used to be passed inverted, sending far pairs through the eager
                # per-point synthesis and near pairs through the far quadrature.)
                cross = cross + Lap3d.bio_offsurf_apply(
                    trg, qlm_blocks[sind], spheres[sind], sh_lst[sind],
                    sl_scal_lst[sind], dl_scal_lst[sind], sep_mat[tind, sind] == 1)
            # acc = acc + sh_lst[tind].analys_cplx_jax(cross.reshape(t_sph["Xcart"].shape[:2]))   # cplx
            acc = acc + sh_lst[tind].analys_jax(cross.reshape(t_sph["Xcart"].shape[:2]))          # real
        out.append(acc)

    return jnp.concatenate(out)


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

    GMRES runs in COEFFICIENT space: the unknown is the concatenation of each sphere's SH
    coefficients, and Lap3d_onsurf_apply is itself coeff -> coeff, so the matvec calls it directly
    (no coeff<->grid round trip). The change of basis only happens at the boundaries: bc -> coeffs
    before the solve, solution coeffs -> grid density after. Working in the coefficient (physical-DOF)
    space -- rather than the oversampled grid -- makes the system square/full-rank, so there is no
    un-clearable oversampling-complement residual.

    Returns (sigma_flat, info, resid) with sigma_flat the GRID-space density; info is scipy's GMRES
    status (0 == converged); resid is the coefficient-space residual ||K c - bc_c||.
    """
    Ns = Sp["Ns"]
    sep_mat = separate_spheres(Sp)
    Nc = _coeff_len_lap(Ns, sh_lst)                                   # coefficient system size
    bc_coeff = np.asarray(grid2coeff_lap(bc_pot, Sp, Ns, sh_lst), dtype=np.complex128)  # COB before

    def matvec(xc):
        # Lap3d_onsurf_apply is coeff -> coeff; no change of basis inside the matvec.
        y = Lap3d_onsurf_apply(jnp.asarray(xc), Sp, sh_lst, sl_scal_lst, dl_scal_lst, sgn_lst, sep_mat)
        # np.array (not asarray): scipy's GMRES mutates the matvec output in place.
        return np.array(y, dtype=np.complex128)

    A = spla.LinearOperator((Nc, Nc), matvec=matvec, dtype=np.complex128)

    M = None
    if precond:
        def psolve(rc):
            # block-Jacobi direct self-solve, natively in coefficient space (no COB needed)
            rc = jnp.asarray(rc, dtype=jnp.complex128).reshape(-1)
            z, c0 = [], 0
            for sind in range(Ns):
                n = sh_lst[sind].nlm   # real/truncated layout (was nlm_cplx)
                z.append(Lap3d.bio_onsurf_direct_solve(
                    rc[c0:c0 + n], sh_lst[sind],
                    sl_scal_lst[sind], dl_scal_lst[sind], sgn_lst[sind])); c0 += n
            return np.array(jnp.concatenate(z), dtype=np.complex128)
        M = spla.LinearOperator((Nc, Nc), matvec=psolve, dtype=np.complex128)

    try:
        sol, info = spla.gmres(A, bc_coeff, M=M, rtol=tol, atol=atol, maxiter=maxiter)
    except TypeError:
        # Older SciPy: `tol` instead of `rtol`.
        sol, info = spla.gmres(A, bc_coeff, M=M, tol=tol, atol=atol, maxiter=maxiter)

    resid = float(jnp.linalg.norm(matvec(sol) - bc_coeff))
    sigma = coeff2grid_lap(jnp.asarray(sol), Sp, Ns, sh_lst)          # COB after -> grid density
    return sigma, info, resid




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


def Stk3d_onsurf_apply(sig_coeff: jax.Array, Sp: SuspensionDict, Ns: int, sh_lst: list,
                       sl_scal_lst, dl_scal_lst, sgn_lst, ps_evals: dict,
                       far_evals: dict) -> jax.Array:
    """
    Apply the suspension on-surface operator K in COEFFICIENT space (Stokes, vector density).
        for each target sphere t,  (K c)_t = sum_s K_{t,s}[c_s]
        - s == t     : self term (Stk3d.bio_onsurf_apply, diagonal in VWX, includes the DL jump),
                       radius-threaded so the SL block scales correctly. coeff -> coeff.
        - s != t near: point-and-shoot layer potential of source s at t's grid, returning
                       target-basis VWX coefficients (ps_evals[(t, s)]). coeff -> coeff.
        - s != t far : rotation-free smooth-quadrature layer potential of source s at ALL its far
                       targets in one call (far_evals[s]) -- returns grid point VALUES, scattered
                       to the far target blocks and then analysed back to each target's VWX coeffs.

    <sig_coeff> is the flat concatenation of each sphere's stacked-VWX coefficients (3*nlm_cplx per
    sphere); the return uses the same layout. Taking/returning coefficients lets the GMRES matvec
    call this directly with no coeff<->grid round trip -- only the far contribution needs a change
    of basis (analys of its grid point-values), since the smooth quadrature is inherently on the
    grid. Pure JAX (safe for lineax).
    """
    spheres = Sp["spheres_lst"]
    sig_coeff = jnp.asarray(sig_coeff, dtype=jnp.complex128).reshape(-1)
    bounds = _block_bounds3(Sp, Ns)          # grid row ranges (for the far scatter/analys)

    # unpack per-sphere stacked-VWX coefficient blocks
    cb, c0 = [], 0
    for s in range(Ns):
        # n = 3 * sh_lst[s].nlm_cplx; cb.append((c0, c0 + n)); c0 += n           # cplx layout
        n = 3 * sh_lst[s].nlm; cb.append((c0, c0 + n)); c0 += n                  # real/truncated layout
    # vwx_blocks = [sig_coeff[cb[s][0]:cb[s][1]].reshape(3, sh_lst[s].nlm_cplx) for s in range(Ns)]  # cplx
    vwx_blocks = [sig_coeff[cb[s][0]:cb[s][1]].reshape(3, sh_lst[s].nlm) for s in range(Ns)]          # real

    # self + near cross: both produce target-basis VWX coefficients, summed in-basis (no COB).
    out_blocks = []
    for tind in range(Ns):
        t_sph = spheres[tind]
        acc = Stk3d.bio_onsurf_apply(
            vwx_blocks[tind], sh_lst[tind],
            sl_scal_lst[tind], dl_scal_lst[tind], sgn_lst[tind], radius=t_sph["r"])
        for sind in range(Ns):
            if (tind, sind) in ps_evals:
                acc = acc + ps_evals[(tind, sind)](
                    vwx_blocks[sind], sl_scal_lst[sind], dl_scal_lst[sind])
        out_blocks.append(acc)

    # far cross: grid point-values at all far targets (one call per source; the kernel tiles its
    # own target loop), scattered into a grid buffer, then each target block is analysed back to
    # VWX coefficients and added to out_blocks.
    if far_evals:
        Ngrid = bounds[-1][1]
        u_all = jnp.concatenate([
            far_evals[sind][0](vwx_blocks[sind], sl_scal_lst[sind], dl_scal_lst[sind]).reshape(-1)
            for sind in far_evals])
        dest_all = jnp.concatenate([dest for (_, dest) in far_evals.values()])
        # far_grid = jnp.zeros(Ngrid, dtype=jnp.complex128).at[dest_all].add(u_all)   # cplx far kernel
        far_grid = jnp.zeros(Ngrid, dtype=jnp.float64).at[dest_all].add(u_all)        # real _stk_far -> float64
        for tind in range(Ns):
            t_sph = spheres[tind]; nphi_t, ntheta_t = t_sph["Xcart"].shape[:2]
            th_t, ph_t = t_sph["Xsph"][:, :, 0], t_sph["Xsph"][:, :, 1]
            gt = far_grid[bounds[tind][0]:bounds[tind][1]].reshape(nphi_t, ntheta_t, 3)
            out_blocks[tind] = out_blocks[tind] + jnp.stack(
                Stk3d.sig_xyz2vwx(gt[:, :, 0], gt[:, :, 1], gt[:, :, 2], th_t, ph_t, sh_lst[tind]))

    return jnp.concatenate([b.reshape(-1) for b in out_blocks])


def Stk3d_onsurf_solve(bc_vec: jax.Array, Sp: SuspensionDict, Ns, Nnodes: int, sh_lst: list,
                       sl_scal_lst, dl_scal_lst, sgn_lst,
                       tol: float = 1e-10, atol: float = 1e-12, maxiter: int = 4,
                       restart: int = 40, precond: bool = True, far_chunk: int = None):
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

    Preconditioning (precond=True) is baked in as RIGHT preconditioning -- the operator is A@M and
    the unknown is u with x = M@u -- with M the block-Jacobi self-solve on the EXTERIOR spheres only
    (interior/container sgn<0 blocks stay identity; see the preconditioner comment below). This makes
    lineax's monitored residual the TRUE residual b - A@x against the raw-b scale (the criterion scipy
    uses), and keeps the preconditioned system nonsingular.

    Convergence is judged from the recomputed true relative residual ||K sigma - bc|| / ||bc||, NOT
    from lineax's flag: lineax 0.1.0's stopping test does not fire cleanly on this problem (the
    no-slip container BC is exactly zero, so its elementwise atol+rtol|b| scale collapses to the
    atol floor on those rows, and the container nullspace keeps the Cauchy-step criterion from
    settling), so it would grind to machine precision over many cycles.

    The solve is therefore a SINGLE `lx.linear_solve` from a zero initial guess -- lineax does its
    own restarting internally, `maxiter` caps the number of restart cycles via max_steps, and the
    true residual is recomputed once at the end to set `info`. There is deliberately no warm start
    from a previous iterate.

    The consequence, which the caller must size for: lineax keeps cycling past the point where the
    true residual has cleared `tol` -- sometimes to the full `maxiter` (TEST 3 lmax=16 runs all 51
    of a 51-cycle budget), sometimes exiting on its own criterion a few cycles late (5-sphere
    container lmax=20, restart=40: 3 cycles for a residual that is already ~1e-15 after 1). So
    `maxiter` is kept SMALL and convergence rests on `restart` being >= the problem's Krylov depth.
    One cycle of GMRES(40) already reaches ~1e-15 on
    the 2-sphere container (depth ~23) and the 5-sphere container (depth ~39). When `restart` is
    below the depth, restarted GMRES discards its Krylov space each cycle and convergence degrades
    sharply -- measured on the 5-sphere container at lmax=20, restart=10 needs ~12 cycles and is
    still at 5.9e-6 after 4 -- so a `restart` bump, not a `maxiter` bump, is the fix; raising
    `maxiter` costs its full price whether or not it is needed (TEST 3 lmax=16: maxiter=50 spends
    2040 matvecs / 10.3s to reach the same 2.3e-16 that one cycle reaches in 0.34s).
    See [[lineax-gmres-restart-stokes-solve]].

    Returns (sigma_flat, time_solve, niters, info, resid); `niters` is the number of real restart
    cycles run (each is up to `restart` matvecs); info == 0 means the true relative residual is
    within `tol`. `resid` is the relative ||K sigma - bc|| / ||bc||.
    """
    sep_mat = separate_spheres(Sp)             # eager -> concrete near/far flags
    bc_vec = jnp.asarray(bc_vec, dtype=jnp.complex128).reshape(-1)
    bounds = _block_bounds3(Sp, Ns)            # static per-sphere flat row ranges
    Nc = _coeff_len_stk(Ns, sh_lst)            # coefficient system size (concatenated VWX)
    struct = jax.ShapeDtypeStruct((Nc,), jnp.complex128)
    bc_coeff = grid2coeff_stk(bc_vec, Sp, Ns, sh_lst, bounds)   # change of basis before the solve

    # Pre-build the cross-evaluators EAGERLY (concrete geometry), outside the matvec -- lineax
    # closure-converts / eval_shapes the operator, which abstracts every closed-over array, so
    # eager rotation builds / _ps_geom_key (float(r), np.asarray(Xc)) cannot run under trace.
    # NEAR pairs -> point-and-shoot; FAR targets -> rotation-free smooth-quadrature per source.
    ps_evals = build_ps_evaluators(Sp, Ns, sh_lst, sep_mat)
    far_evals = build_far_evaluators(Sp, Ns, sh_lst, sep_mat, far_chunk=far_chunk)

    # Coefficient-space matvec: Stk3d_onsurf_apply is itself coeff -> coeff, so call it directly
    # (no coeff<->grid round trip). GMRES runs entirely in coefficient space, so the system is
    # square in the physical DOFs (no oversampled-grid complement / residual floor).
    def matvec(xc):
        return Stk3d_onsurf_apply(xc, Sp, Ns, sh_lst, sl_scal_lst, dl_scal_lst, sgn_lst, ps_evals, far_evals)
    # Eager warmup (concrete input): compiles the per-pair _core kernels + self blocks once.
    jax.block_until_ready(matvec(jnp.zeros((Nc,), dtype=jnp.complex128)))

    # Block-Jacobi preconditioner M: per-sphere spectral direct self-solve, natively in coefficient
    # (VWX) space -- no COB needed. Its safe_div handles the l=0 toroidal phantom / zero-diagonal
    # modes. EXCEPTION: interior-formulation spheres (sgn < 0, e.g. the no-slip container) carry a
    # genuine double-layer nullspace, so their self-block is near-singular; a direct-solve inverse of
    # it does not remove that nullspace, it just feeds an amplifying near-singular operator into the
    # Krylov system, and lineax's stopping logic (residual + Cauchy-step) can then never early-exit
    # -- it grinds to ~1e-16 over 30-150 restart cycles regardless of tol / left-vs-right fold /
    # restart. Leaving that block as IDENTITY keeps the preconditioned system nonsingular so lineax
    # converges cleanly (~4 iters). Exterior spheres (sgn > 0) still get the block-Jacobi self-solve.
    # See [[lineax-gmres-restart-stokes-solve]].
    def psolve(rc):
        rc = jnp.asarray(rc, dtype=jnp.complex128).reshape(-1)
        zlist, c0 = [], 0
        for sind in range(Ns):
            n = 3 * sh_lst[sind].nlm   # real/truncated layout (was nlm_cplx)
            blk = rc[c0:c0 + n]; c0 += n
            if float(sgn_lst[sind]) < 0.0:      # interior/container nullspace block -> identity
                zlist.append(blk)
                continue
            vwx_z = Stk3d.stokes_onsurf_direct_solve(
                blk.reshape(3, sh_lst[sind].nlm), sh_lst[sind],
                sl_scal_lst[sind], dl_scal_lst[sind], sgn_lst[sind], radius=Sp["spheres_lst"][sind]["r"])
            zlist.append(vwx_z.reshape(-1))
        return jnp.concatenate(zlist)   # jax array: keeps the solve on-device (no host sync)

    # Preconditioning is baked into the operator/RHS as RIGHT preconditioning: solve (A@M) u = b,
    # then recover x = M@u. We do NOT pass M to lineax as options["preconditioner"] (that is LEFT
    # preconditioning) for two reasons, both rooted in lineax 0.1.0's convergence test
    # `||preconditioner @ (b - A y)|| / (atol + rtol*|b|)` (gmres.py):
    #   1. LEFT: the numerator is the PRECONDITIONED residual ||M(b-Ay)|| but the scale is built from
    #      the RAW b, so the ratio is inconsistent -- and because M is a near-singular inverse
    #      (container/interior double-layer nullspace, cond ~1e20) that AMPLIFIES the residual on the
    #      near-null modes, GMRES must grind those modes to ~1e-16 true residual just to clear the
    #      test (measured: 77+ restart cycles, overshooting to ~1e-16, vs scipy's 13 iters at 1e-12).
    #   2. RIGHT: with op = A@M and rhs = b, lineax's residual b - A@M@u == b - A@x is the TRUE
    #      residual against the raw-b scale -- exactly the criterion scipy uses -- so it converges in
    #      ~the same iteration count as scipy and stops without over-resolving the amplified modes.
    # See [[lineax-gmres-restart-stokes-solve]].
    if precond:
        op_apply = lambda u: matvec(psolve(u))   # A @ M
        rhs = bc_coeff                            # unchanged (right preconditioning)
    else:
        op_apply = matvec
        rhs = bc_coeff

    gmres_func = lx.FunctionLinearOperator(op_apply, struct)
    # norm=two_norm: lineax defaults to max_norm with a componentwise scale, which blows up on
    # small-|b| rows and never clears at tight tol; two_norm gives a standard relative criterion.
    #
    # restart / max_steps, and why the restart cycles are driven from PYTHON:
    #   * restart = Krylov depth per cycle. lineax's inner Arnoldi loop always runs the FULL `restart`
    #     matvecs (no per-iteration tolerance exit, gmres.py _gmres_compute) and discards the subspace
    #     at each restart, so once `restart` drops below the depth needed to converge, convergence
    #     degrades from one cycle to many (measured, 5-sphere container lmax=20, scipy depth 39:
    #     restart>=40 -> 1 cycle to 2.6e-13; restart=20 -> 2 cycles; restart=10 -> 12 cycles, and
    #     only 5.9e-6 if cut off at 4). A fixed small cycle cap therefore silently returns a
    #     half-solved density on any problem deeper than `restart`.
    #   * lineax's stopping test does NOT fire on this problem, so it cannot be trusted to end the
    #     cycles. Two structural reasons it never self-terminates here: (1) the no-slip container BC is
    #     exactly 0, so half of b is zero and lineax's ELEMENTWISE scale atol+rtol|b| collapses to the
    #     absolute floor atol on those rows, forcing the residual there to ~atol before the ratio
    #     clears; scipy avoids this with a GLOBAL norm ||b-Ax||/max(rtol||b||,atol). (2) the container's
    #     double-layer nullspace makes the operator singular, so the Cauchy-step (`diff`) criterion
    #     never settles. Neither is reachable through lineax 0.1.0's public API.
    #   * So the restart cycles are left to lineax (one call, y0 = 0, NO warm start from a previous
    #     iterate) and `maxiter` bounds them through max_steps, with convergence judged afterwards
    #     from the recomputed global true residual (`info` below). The internal test is late or
    #     absent, so the call runs well past the point where the true residual has cleared -- keep
    #     `maxiter` small (measured, TEST 3 lmax=16: max_steps=51 spends all 51 cycles / 2040
    #     matvecs / 10.3s to reach the same 2.3e-16 that one cycle reaches in 0.34s). Raising it
    #     costs proportionally, so raise `restart` instead when a problem needs more depth.
    #   * max_steps = maxiter + 1: lineax's step 0 is a NO-OP that only computes the residual
    #     (gmres.py initialises r0 to zeros and `first_gmres` returns y unchanged, to save compiling
    #     an extra matvec), so N max_steps perform N-1 real Arnoldi cycles -- the +1 makes `maxiter`
    #     mean the number of REAL restart cycles. (The old max_steps=maxiter was 3 cycles, not 4.)
    restart = int(min(restart, Nc))
    solver = lx.GMRES(rtol=tol, atol=atol, max_steps=int(maxiter) + 1, restart=restart,
                      stagnation_iters=50, norm=two_norm)
    options = {"y0": jnp.zeros((Nc,), dtype=jnp.complex128)}

    # throw=False: on stagnation / non-convergence (e.g. the interior container problem, which
    # has a double-layer nullspace) lineax otherwise raises; instead return the best iterate and
    # report it via `info` below. Warmup solve compiles the fused GMRES loop; second solve timed.
    solution = lx.linear_solve(gmres_func, rhs, solver=solver, options=options, throw=False)
    jax.block_until_ready(solution.value)
    tstart = time.time()
    solution = lx.linear_solve(gmres_func, rhs, solver=solver, options=options, throw=False)
    jax.block_until_ready(solution.value)
    time_solve = time.time() - tstart

    # Right preconditioning solves for u; the density coeffs are x = M@u (identity when precond=off).
    sol_coeff = psolve(solution.value) if precond else solution.value
    niters = int(solution.stats["num_steps"]) - 1        # drop the no-op step 0
    # Judge convergence from the true (coefficient-space) residual, not lineax's flag (see docstring):
    # its Cauchy-step criterion reports `stagnation` at a machine-precision plateau even when exact.
    resid = jnp.linalg.norm(matvec(sol_coeff) - bc_coeff)
    bc_norm = jnp.linalg.norm(bc_coeff)
    resid_rel = resid / bc_norm if bc_norm > 0 else resid
    info = 0 if float(resid_rel) <= tol else 1

    sigma = coeff2grid_stk(sol_coeff, Sp, Ns, sh_lst)          # change of basis after the solve
    return sigma, time_solve, niters, info, resid_rel

def Stk3d_onsurf_solve_spla(bc_vec: jax.Array, Sp: SuspensionDict, Ns, Nnodes: int, sh_lst: list,
                       sl_scal_lst, dl_scal_lst, sgn_lst,
                       tol: float = 1e-10, atol: float = 1e-14, maxiter: int = 200,
                       precond: bool = True, precond_interior: bool = True, far_chunk: int = None):
    """
    Solve K[sigma] = bc_vec for the suspension Stokes surface density sigma (vector).

    An optional block-Jacobi preconditioner applies each sphere's spectral direct self-solve
    (Stk3d.stokes_onsurf_direct_solve, radius-aware). By default (precond_interior=True) EVERY
    sphere is preconditioned, including interior/container spheres (sgn == -1); set
    precond_interior=False to leave those blocks as identity (exterior-only preconditioning).

    Interior-formulation spheres carry a double-layer nullspace (2 null modes per sphere: the
    l=0 V/W modes), so their diagonal self-block has 2 zero eigenvalues -- but it is otherwise
    perfectly conditioned (nonzero eigenvalues in [-1, -1/3], cond 3, INDEPENDENT of lmax). The
    direct self-solve's safe_div leaves those 2 null modes as identity, so inverting the block is
    stable under scipy GMRES's global-norm stopping test (unlike lineax's elementwise test in
    Stk3d_onsurf_solve, which is why THAT path still skips it -- see its docstring). Preconditioning
    the interior block roughly HALVES the Krylov depth, which matters at high lmax: with it left as
    identity the un-preconditioned container is the far source whose VWX->QST synthesis amplifies
    coefficient roundoff by ~0.58*l (~300x at lmax=512), raising the far-coupling residual floor
    toward `tol`; restarted GMRES(20) then needs ~7 restart cycles (146 iters) to grind under it,
    whereas one cycle suffices when the block is preconditioned (measured container BVP lmax=512:
    146 iters/2593s -> ~20 iters(1 cycle)/307s, same 7e-11 residual). See
    [[suspension-real-transform-migration]].

    Returns (sigma_flat, time_solve, niters, info, resid); info == 0 means GMRES converged.
    """
    sep_mat = separate_spheres(Sp)             # eager -> concrete near/far flags
    bc_vec = jnp.asarray(bc_vec, dtype=jnp.complex128).reshape(-1)
    bounds = _block_bounds3(Sp, Ns)            # static per-sphere flat row ranges

    ps_evals = build_ps_evaluators(Sp, Ns, sh_lst, sep_mat)
    far_evals = build_far_evaluators(Sp, Ns, sh_lst, sep_mat, far_chunk=far_chunk)

    # The JAX path solves in COEFFICIENT space -- change of basis at the boundaries (bc -> coeff
    # here, solution coeff -> grid at the end) and inside the matvec, so the system is square in
    # the physical DOFs (no oversampled-grid complement / residual floor).
    Nsys = _coeff_len_stk(Ns, sh_lst)
    b = np.asarray(grid2coeff_stk(bc_vec, Sp, Ns, sh_lst, bounds), dtype=np.complex128)

    # Operator matvec: JAX coeff-space operator, called directly.
    def matvec(x):
        y = Stk3d_onsurf_apply(jnp.asarray(x), Sp, Ns, sh_lst,
                               sl_scal_lst, dl_scal_lst, sgn_lst, ps_evals, far_evals)
        return np.array(y, dtype=np.complex128)

    A = spla.LinearOperator((Nsys, Nsys), matvec=matvec, dtype=np.complex128)

    counter = IterationCounter()

    M = None
    if precond:
        def psolve(r):
            # Block-Jacobi direct self-solve, natively in coefficient (VWX) space (no COB).
            # Interior/container spheres (sgn == -1) carry a double-layer nullspace; their direct
            # self-solve's safe_div leaves the 2 null modes as identity, so it is stable under
            # scipy's global-norm GMRES and preconditioning them roughly halves the Krylov depth
            # (see the docstring). precond_interior=False reverts to exterior-only (identity on the
            # sgn<0 blocks) if a caller wants the old behaviour.
            rc = jnp.asarray(r, dtype=jnp.complex128).reshape(-1)
            zlist, c0 = [], 0
            for sind in range(Sp["Ns"]):
                n = 3 * sh_lst[sind].nlm
                blk = rc[c0:c0 + n]; c0 += n
                if float(sgn_lst[sind]) < 0.0 and not precond_interior:   # interior block -> identity
                    zlist.append(blk)
                    continue
                vwx_z = Stk3d.stokes_onsurf_direct_solve(
                    blk.reshape(3, sh_lst[sind].nlm), sh_lst[sind],
                    sl_scal_lst[sind], dl_scal_lst[sind], sgn_lst[sind], radius=Sp["spheres_lst"][sind]["r"])
                zlist.append(vwx_z.reshape(-1))
            return np.array(jnp.concatenate(zlist), dtype=np.complex128)
        M = spla.LinearOperator((Nsys, Nsys), matvec=psolve, dtype=np.complex128)

    # Eager warmup (concrete zeros input): compile the jitted per-pair point-and-shoot / far
    # kernels + self blocks in matvec (and the preconditioner's direct self-solves) once here,
    # so the timed GMRES loop below pays no JIT-compile cost.
    jax.block_until_ready(matvec(np.zeros((Nsys,), dtype=np.complex128)))
    if M is not None:
        jax.block_until_ready(psolve(np.zeros((Nsys,), dtype=np.complex128)))

    tstart = time.time()
    sol, info = spla.gmres(A, b, M=M, rtol=tol, atol=atol, maxiter=maxiter, callback=counter)
    tend = time.time()
    t_solve = tend - tstart

    bnorm = float(np.linalg.norm(b))
    r_abs = float(np.linalg.norm(matvec(sol) - b))
    resid = r_abs if bnorm <= 0 else r_abs / bnorm
    niters = counter.count
    # change of basis after the solve: coeff -> grid density.
    sigma = coeff2grid_stk(jnp.asarray(sol), Sp, Ns, sh_lst)
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
    # Small sep_eta so every off-diagonal pair is classified FAR: the cross blocks are then
    # handled by the direct smooth-quadrature far evaluator (bio_offsurf_apply far=True), not the
    # near point-and-shoot. (Keep 0.0001 for now.)
    sep_eta = 0.0001

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
        qlm_s = sh_lst[s].analys_cplx_jax(sigma[int(dsp[s]):int(dsp[s + 1])].reshape(grid))
        approx = approx + Lap3d.bio_offsurf_apply(chk, qlm_s, Sp["spheres_lst"][s], sh_lst[s], sl_scal, dl_scal)
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
        s_sph = Sp["spheres_lst"][s]
        nphi, ntheta = s_sph["Xcart"].shape[:2]
        sig_s = sigma[3 * int(dsp[s]):3 * int(dsp[s + 1])].reshape(nphi, ntheta, 3)
        vwx_s = jnp.stack(Stk3d.sig_xyz2vwx(sig_s[:, :, 0], sig_s[:, :, 1], sig_s[:, :, 2],
                                            s_sph["Xsph"][:, :, 0], s_sph["Xsph"][:, :, 1], sh_lst[s]))
        approx = approx + Stk3d.bio_offsurf_apply(chk, vwx_s, s_sph, sh_lst[s], sl_scal, dl_scal, far=True)
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
    sigma, t_solve, niter, info, resid = Stk3d_onsurf_solve(bc, Sp, Sp["Ns"], Nnodes, sh_lst, sl_lst, dl_lst, sgn_lst)
    print(f"LINEAX GMRES info (0 == converged): {info}, time for solve is {t_solve}s, num of iters is {niter}, residual = {resid:.3e}")

    # sigma, t_solve, niter, info, resid = Stk3d_onsurf_solve_spla(bc, Sp, Sp["Ns"], Nnodes, sh_lst, sl_lst, dl_lst, sgn_lst)
    # print(f"SCIPY GMRES info (0 == converged): {info}, time for solve is {t_solve}s, num of iters is {niter}, residual = {resid:.3e}")


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
    #         s_sph = Sp["spheres_lst"][s]
    #         nphi, ntheta = s_sph["Xcart"].shape[:2]
    #         sig_s = sigma[3 * int(dsp[s]):3 * int(dsp[s + 1])].reshape(nphi, ntheta, 3)
    #         vwx_s = jnp.stack(Stk3d.sig_xyz2vwx(sig_s[:, :, 0], sig_s[:, :, 1], sig_s[:, :, 2],
    #                                             s_sph["Xsph"][:, :, 0], s_sph["Xsph"][:, :, 1], sh_lst[s]))
    #         approx = approx + Stk3d.bio_offsurf_apply(trg_in, vwx_s, s_sph, sh_lst[s], sl_lst[s], dl_lst[s], far=True)
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
