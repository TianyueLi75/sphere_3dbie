"""
Stokes Operator Class:
    SL and DL operators on a sphere, using spectra and solid harmonics
    TODO:
        Made SHqst_to_point_cplx using np locally in this script -- need to incorporate into SHTns and have jax wrap
        Allow on-surface evaluation in bio_offsurf_apply()
        onsurf_diag_solve() l=0 currently set to BC values. Throw exception instead?
        SL traction near and far off-surface eval.
"""

from typing import Dict, Any, Tuple
from functools import partial

import scipy
import numpy as np
import jax
import jax.numpy as jnp
# import scipy.sparse.linalg
import lineax as lx
import shtns
import shtns_jax

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sphere import *

SphereDict = Dict[str, Any]

def cart2sph(Vx: jax.Array, Vy: jax.Array, Vz: jax.Array, theta: jax.Array, phi: jax.Array) -> tuple([jax.Array, jax.Array, jax.Array]):
    """
    Transform vector fields from Cartesian to spherical coordinates
    """

    Vr = Vx * jnp.sin(theta) * jnp.cos(phi) + Vy * jnp.sin(theta) * jnp.sin(phi) + Vz * jnp.cos(theta)
    Vtheta = Vx * jnp.cos(theta) * jnp.cos(phi) + Vy * jnp.cos(theta) * jnp.sin(phi) - Vz * jnp.sin(theta)
    Vphi = -Vx * jnp.sin(phi) + Vy * jnp.cos(phi)
    return Vr, Vtheta, Vphi

def sph2cart(Vr: jax.Array, Vtheta: jax.Array, Vphi: jax.Array, theta: jax.Array, phi: jax.Array) -> tuple([jax.Array, jax.Array, jax.Array]):
    """
    Transform vector fields from spherical to Cartesian coordinates
    """

    Vx = Vr * jnp.sin(theta) * jnp.cos(phi) + Vtheta * jnp.cos(theta) * jnp.cos(phi) - Vphi * jnp.sin(phi)
    Vy = Vr * jnp.sin(theta) * jnp.sin(phi) + Vtheta * jnp.cos(theta) * jnp.sin(phi) + Vphi * jnp.cos(phi)
    Vz = Vr * jnp.cos(theta) - Vtheta * jnp.sin(theta)
    return Vx, Vy, Vz

def qst2vwx(qlm: jax.Array, slm: jax.Array, tlm: jax.Array, sh: shtns_jax.sht) -> tuple([jax.Array, jax.Array, jax.Array]):
    """
    Transform coefficients in Q,S,T basis (used in SHTns)
        to the V,W,X basis (diagonalizing basis for Stokes LP operators)
    """
    
    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64)
    vlm = (l_vals * slm - qlm) / (2.0*l_vals + 1.0)
    wlm = ((l_vals + 1.0) * slm + qlm) / (2.0*l_vals + 1.0)
    xlm = -tlm
    return vlm, wlm, xlm

def vwx2qst(vlm: jax.Array, wlm: jax.Array, xlm: jax.Array, sh: shtns_jax.sht) -> tuple([jax.Array, jax.Array, jax.Array]):
    """
    Transform coefficients in V,W,X basis (diagonalizing basis for Stokes LP operators)
        to the Q,S,T basis (used in SHTns)
    """
    
    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64)
    slm = vlm + wlm
    qlm = l_vals * (wlm - vlm) - vlm
    tlm = -xlm
    return qlm, slm, tlm

# TODO: to clean up, can work with only the (3, _, _) arrays
def vec_stack(v1, v2, v3):
    """
    Takes the x, y, z or q, s, t vectors 
        and stack them in (3, _, _) shape
    """
    return jnp.stack([v1, v2, v3], axis=0)

def vec_distr(v_all):
    """
    Takes the output size (3, _, _) from synth or analys
        and distribute into arrays v1, v2, v3
    """
    v1 = v_all[0,...]
    v2 = v_all[1,...]
    v3 = v_all[2,...]
    return v1, v2, v3

def sig_xyz2vwx(sigma_x: jax.Array, sigma_y: jax.Array, sigma_z: jax.Array, theta: jax.Array, phi: jax.Array, sh: shtns_jax.sht) -> tuple([jax.Array,jax.Array,jax.Array]):
    """
    Transforms vector fields in Cartesian coordinates  
        into its coefficients in V,W,X basis
    """

    sigma_r, sigma_t, sigma_p = cart2sph(sigma_x,sigma_y,sigma_z,theta,phi)
    sigma_rtp = vec_stack(sigma_r, sigma_t, sigma_p)
    qstlm = sh.analys_vec_cplx_jax(sigma_rtp)
    qlm, slm, tlm = vec_distr(qstlm) 
    vlm, wlm, xlm = qst2vwx(qlm, slm, tlm, sh)
    return vlm, wlm, xlm

def sig_vwx2xyz(vlm: jax.Array, wlm: jax.Array, xlm: jax.Array, theta: jax.Array, phi: jax.Array, sh: shtns_jax.sht) -> tuple([jax.Array, jax.Array, jax.Array]):
    """
    Transforms coefficients in V,W,X basis
        into its Cartesian coordinates
    """

    qlm, slm, tlm = vwx2qst(vlm,wlm,xlm,sh)
    qstlm = vec_stack(qlm, slm, tlm)
    vrtp = sh.synth_vec_cplx_jax(qstlm)
    vr, vt, vp = vec_distr(vrtp)
    vx, vy, vz = sph2cart(vr, vt, vp, theta, phi)
    return vx, vy, vz

def Stk3d_sl_VWX_diag(sh: shtns_jax.sht) -> tuple([jax.Array, jax.Array, jax.Array]):
    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64)
    diag_V = l_vals / (2.0*l_vals + 1.0) / (2.0*l_vals + 3.0)
    diag_W = (l_vals + 1.0) / (2.0*l_vals + 1.0) / (2.0*l_vals - 1.0)
    diag_X = 1.0 / (2.0*l_vals + 1.0)
    return diag_V, diag_W, diag_X

def Stk3d_dl_VWX_diag(sh: shtns_jax.sht) -> tuple([jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]):
    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64)

    diag_V_ext = (2.0*l_vals*l_vals + 4*l_vals + 3) / (2.0*l_vals + 1.0) / (2.0*l_vals + 3.0)
    diag_W_ext = 2.0*(l_vals + 1.0)*(l_vals - 1.0) / (2.0*l_vals + 1.0) / (2.0*l_vals - 1.0)
    diag_X_ext = (l_vals - 1.0) / (2.0*l_vals + 1.0)

    diag_V_int = -2.0*l_vals*(l_vals + 2) / (2.0*l_vals + 1.0) / (2.0*l_vals + 3.0)
    diag_W_int = -(2.0*l_vals*l_vals + 1.0) / (2.0*l_vals + 1.0) / (2.0*l_vals - 1.0)
    diag_X_int = -(l_vals + 2.0) / (2.0*l_vals + 1.0)

    return diag_V_ext, diag_W_ext, diag_X_ext, diag_V_int, diag_W_int, diag_X_int

def Stk3d_dsl_VWX_diag(sh: shtns_jax.sht) -> tuple([jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]):
    # dSL is the DL diagonal with the exterior/interior triples swapped.
    diag_V_ext, diag_W_ext, diag_X_ext, diag_V_int, diag_W_int, diag_X_int = Stk3d_dl_VWX_diag(sh)
    return diag_V_int, diag_W_int, diag_X_int, diag_V_ext, diag_W_ext, diag_X_ext

def diag_W2V(sh: shtns_jax.sht, which: str) -> jax.Array:
    """Exterior W->V off-surface coupling coefficient of the SL ('sl') or DL ('dl')
    solid-harmonic scaling, in the V/W/X diagonalizing basis."""
    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64)
    if which == "sl":
        return l_vals / (4.0*l_vals + 2.0)
    return 2.0*l_vals*(l_vals - 1.0) / (4.0*l_vals + 2.0)          # dl

def diag_V2W(sh: shtns_jax.sht, which: str) -> jax.Array:
    """Interior V->W off-surface coupling coefficient of the SL ('sl') or DL ('dl')
    solid-harmonic scaling, in the V/W/X diagonalizing basis."""
    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64)
    if which == "sl":
        return (l_vals + 1.0) / (4.0*l_vals + 2.0)
    return (l_vals + 1.0)*(l_vals + 2.0) / (2.0*l_vals + 1.0)      # dl

def Stk3d_sl_far(trg: jax.Array, S: SphereDict, sh: shtns_jax.sht) -> jax.Array:
    """
    Off-surface evaluation of the Stokes SL (Stokeslet) velocity with the vector
    density S["Sigma"]
        with <trg>: Ntrg x 3
        using the smooth surface quadrature (accurate only for targets well
        away from the surface).
    Returns the velocity at the targets: Ntrg x 3.

    Stokeslet kernel (matches compute_field): with r = trg - src, d = |r|,
        u_j(x) = (1/8pi) sum_src [ sigma_j / d + r_j (r.sigma) / d^3 ] * w.
    """
    assert trg.shape[1] == 3

    # Flatten the (nphi, ntheta, ...) grid arrays to per-source-point lists.
    grid_shape = S["Xcart"].shape[:2]
    ysrc = S["Xcart"].reshape(-1, 3)        # Nsrc x 3
    fsrc = S["Sigma"].reshape(-1, 3)        # Nsrc x 3, vector density
    # Gauss weights (1 x ntheta) broadcast over the (nphi, ntheta) grid, plus
    # the r^2 surface Jacobian for a sphere of radius S["r"].
    wts = jnp.broadcast_to(sh._grid_weights(), grid_shape).reshape(-1) * S["r"] ** 2

    r = trg[:, None, :] - ysrc[None, :, :]  # Ntrg x Nsrc x 3, convention r = trg - src
    d = jnp.linalg.norm(r, axis=2)          # Ntrg x Nsrc
    rdotf = jnp.sum(r * fsrc[None, :, :], axis=2)  # Ntrg x Nsrc
    term = fsrc[None, :, :] / d[..., None] + r * (rdotf / d ** 3)[..., None]
    prefac = 1.0 / 8.0 / jnp.pi
    SL_sigma = prefac * jnp.sum(term * wts[None, :, None], axis=1)  # Ntrg x 3
    return SL_sigma

def Stk3d_dl_far(trg: jax.Array, S: SphereDict, sh: shtns_jax.sht) -> jax.Array:
    """
    Off-surface evaluation of the Stokes DL (stresslet) velocity with the vector
    density S["Sigma"]
        with <trg>: Ntrg x 3
        using the smooth surface quadrature (accurate only for targets well
        away from the surface).
    Returns the velocity at the targets: Ntrg x 3.

    Stresslet kernel T_ijk = 6 r_i r_j r_k / d^5. With r = trg - src, d = |r|,
    unit outward normal n = S["Xncart"] and density sigma = S["Sigma"],
        u_j(x) = (6/8pi) sum_src [ (r.sigma)(r.n) / d^5 ] r_j * w.
    The sign matches the spectral Stk3d_dl_r_1sph convention (verified in __main__).
    """
    assert trg.shape[1] == 3

    grid_shape = S["Xcart"].shape[:2]
    ysrc = S["Xcart"].reshape(-1, 3)        # Nsrc x 3
    fsrc = S["Sigma"].reshape(-1, 3)        # Nsrc x 3, vector density
    nsrc = S["Xncart"].reshape(-1, 3)       # Nsrc x 3, unit outward normals
    wts = jnp.broadcast_to(sh._grid_weights(), grid_shape).reshape(-1) * S["r"] ** 2

    r = trg[:, None, :] - ysrc[None, :, :]  # Ntrg x Nsrc x 3, convention r = trg - src
    d = jnp.linalg.norm(r, axis=2)          # Ntrg x Nsrc
    rdotn = jnp.sum(r * nsrc[None, :, :], axis=2)   # Ntrg x Nsrc
    rdots = jnp.sum(r * fsrc[None, :, :], axis=2)   # Ntrg x Nsrc
    prefac = 6.0 / 8.0 / jnp.pi
    DL_sigma = prefac * jnp.sum(r * (rdotn * rdots / d ** 5 * wts[None, :])[..., None], axis=1)
    return DL_sigma


def _ps_rotation(t_vec: jax.Array, lmax_src: int, lmax_trg: int) -> tuple:
    """Forward/inverse shtns rotations that bring the target-center direction onto
    the +z pole (and back). Built with set_angle_axis (axis = t_hat x z_hat, angle
    = acos(t_z/d)); verified so the rotated expansion sampled at the pole equals the
    original sampled at t_hat. Degenerate t along +/-z falls back to an x-axis flip.
    The forward rotation acts on the SOURCE coefficients (stage 1) so it is built at
    <lmax_src>; the inverse acts on the TARGET coefficients (stage 3) so it is built
    at <lmax_trg>. Returns (rot_fwd, rot_inv, d)."""
    t = jnp.asarray(t_vec).reshape(3)
    d = jnp.linalg.norm(t)
    that = t / d
    beta = jnp.arccos(jnp.clip(that[2], -1.0, 1.0))
    axis = jnp.array([that[1], -that[0], 0.0])   # t_hat x z_hat
    nrm = jnp.linalg.norm(axis)
    axis = jnp.array([1.0, 0.0, 0.0]) if nrm < 1e-14 else axis / nrm
    # shtns set_angle_axis is a C (SWIG) binding taking Python doubles, so the
    # angle/axis must be concrete host scalars (this is the eager rotation-setup boundary).
    beta = float(beta); ax = (float(axis[0]), float(axis[1]), float(axis[2]))
    rot_fwd = shtns_jax.rotation(lmax_src, lmax_src, 0)   # mmax=lmax => exact rotation
    rot_fwd.set_angle_axis(beta, *ax)
    rot_inv = shtns_jax.rotation(lmax_trg, lmax_trg, 0)
    rot_inv.set_angle_axis(-beta, *ax)
    return rot_fwd, rot_inv, float(d)

def _ps_target_rings(d: float, R_t: float, theta_std: jax.Array) -> tuple:
    """Source-centered (r_j, cos_theta_src_j) for each target-local theta ring,
    after the target center is placed on the +z axis at distance d (radius R_t).
    Within a ring all nphi points share these (translation along +z preserves phi)."""
    theta_std = jnp.asarray(theta_std, dtype=jnp.float64)
    ct = jnp.cos(theta_std); st = jnp.sin(theta_std)
    z = d + R_t * ct
    r = jnp.sqrt((R_t * st) ** 2 + z * z)
    return r, z / r

def _latlm_maps(sh: shtns_jax.sht) -> tuple:
    """Cached cplx->real-layout gather indices + parity for the G/H split used by
    _stk_latitude_cplx. For each real-layout index (degree l, order m>=0): gather
    from the cplx layout at k_pos = l(l+1)+m and k_neg = l(l+1)-m, with parity (-1)^m."""
    cached = getattr(sh, "_ps_latlm_maps", None)
    if cached is not None:
        return cached
    lr = np.asarray(sh.l, dtype=np.int64)
    mr = np.asarray(sh.m, dtype=np.int64)
    kpos = jnp.asarray(lr * (lr + 1) + mr, dtype=jnp.int64)
    kneg = jnp.asarray(lr * (lr + 1) - mr, dtype=jnp.int64)
    parity = jnp.asarray((-1.0) ** mr, dtype=jnp.float64)
    maps = (kpos, kneg, parity)
    sh._ps_latlm_maps = maps
    return maps

def _stk_latitude_ring(qlm: jax.Array, slm: jax.Array, tlm: jax.Array,
                       cos_src, sh: shtns_jax.sht) -> tuple:
    """Single-ring FFT synthesis: complex Q/S/T coefficient vectors (each (nlm_cplx,),
    already radius-scaled) evaluated at ONE ring latitude cos_src (scalar) over all nphi
    longitudes. Reuses the vetted real FFT path sht.SHqst_to_lat_jax by splitting each
    complex coefficient vector into the real-layout coefficients of its real (G) and
    imaginary (H) parts. Returns (vr, vt, vp), each (nphi,) complex, in the source-centered
    spherical basis. _stk_latitude_cplx vmaps this over rings."""
    kpos, kneg, parity = _latlm_maps(sh)

    def split(z):   # z: (nlm_cplx,) -> real-layout G, H each (nlm,)
        zp = z[kpos]; zn = z[kneg]
        aG = 0.5 * (zp + parity * jnp.conj(zn))
        aH = (zp - parity * jnp.conj(zn)) / 2j
        return aG, aH

    Qg, Qh = split(qlm); Sg, Sh = split(slm); Tg, Th = split(tlm)
    c = jnp.asarray(cos_src, dtype=jnp.float64)
    VrG, VtG, VpG = sh.SHqst_to_lat_jax(Qg, Sg, Tg, c, sh.nphi)   # each (nphi,) float64
    VrH, VtH, VpH = sh.SHqst_to_lat_jax(Qh, Sh, Th, c, sh.nphi)
    return VrG + 1j * VrH, VtG + 1j * VtH, VpG + 1j * VpH

def _stk_latitude_cplx(qlm: jax.Array, slm: jax.Array, tlm: jax.Array,
                       cos_src: jax.Array, sh: shtns_jax.sht) -> tuple:
    """FFT-accelerated evaluation of complex Q/S/T expansions -- one per target ring
    (coeff arrays shaped (ntheta, nlm_cplx), already radius-scaled) -- at each ring's
    latitude cos_src[j] over all nphi longitudes. Vmaps _stk_latitude_ring over the ring
    axis. Returns (vr, vt, vp), each (nphi, ntheta) complex, in the source-centered
    spherical basis."""
    vr, vt, vp = jax.vmap(_stk_latitude_ring, in_axes=(0, 0, 0, 0, None))(
        qlm, slm, tlm, cos_src, sh)
    # ring-major (ntheta, nphi) -> (nphi, ntheta)
    return vr.T, vt.T, vp.T

def _ps_scale_coeffs(sh: shtns_jax.sht, exterior: bool, which: str) -> tuple:
    """Ring-INDEPENDENT VWX diagonal + coupling symbols for operator 'which' ('sl'/'dl') on the
    exterior/interior branch. Depend only on sh.zl and the static branch, so precomputed ONCE per
    evaluator (in point_n_shoot_evaluator's eager setup) and closed over by the jitted _core --
    hoisted out of the per-ring vmap. (Under jit these fold to constants anyway; hoisting keeps
    the jaxpr small and mirrors the Stk3d_np twin.) Returns (diag_V, diag_W, diag_X, coup)."""
    if which == "sl":
        diag_V, diag_W, diag_X = Stk3d_sl_VWX_diag(sh)
    else:
        (dVe, dWe, dXe, dVi, dWi, dXi) = Stk3d_dl_VWX_diag(sh)
        diag_V, diag_W, diag_X = (dVe, dWe, dXe) if exterior else (dVi, dWi, dXi)
    coup = diag_W2V(sh, which) if exterior else diag_V2W(sh, which)
    return diag_V, diag_W, diag_X, coup


def _ps_scale_vwx(vlm: jax.Array, wlm: jax.Array, xlm: jax.Array,
                  pw_V: jax.Array, pw_W: jax.Array, pw_X: jax.Array, exterior: bool,
                  which: str, coeffs: tuple) -> tuple:
    """Apply the per-ring solid-harmonic radial scaling of the SL ('sl') or DL ('dl')
    operator (in the V/W/X diagonalizing basis) to the rotated source coefficients, using the
    precomputed ring-independent <coeffs> (diag_V, diag_W, diag_X, coup) from _ps_scale_coeffs
    and this ring's precomputed radial-power slices pw_V/pw_W/pw_X (each (nlm,)). These are the
    solid-harmonic powers of the source-centered radius -- exterior: (rho^{-l-2}, rho^{-l},
    rho^{-l-1}); interior: (rho^{l+1}, rho^{l-1}, rho^{l}) -- geometry-fixed, so they are
    precomputed once per evaluator (_ps_ring_core, hoisted out of the per-matvec path) rather
    than re-evaluated as transcendental powers on every GMRES matvec. """
    diag_V, diag_W, diag_X, coup_coef = coeffs
    if exterior:
        coup = (pw_V - pw_W) * coup_coef * wlm
        vlm_o = pw_V * diag_V * vlm + coup
        wlm_o = pw_W * diag_W * wlm
        xlm_o = pw_X * diag_X * xlm
    else:
        # SL: +(rho^{l+1} - rho^{l-1}); DL: -(rho^{l+1} - rho^{l-1}) (sign per operator)
        sgn = 1.0 if which == "sl" else -1.0
        coup = sgn * (pw_V - pw_W) * coup_coef * vlm
        vlm_o = pw_V * diag_V * vlm
        wlm_o = pw_W * diag_W * wlm + coup
        xlm_o = pw_X * diag_X * xlm
    return vlm_o, wlm_o, xlm_o

def _ps_ring_core(vlm: jax.Array, wlm: jax.Array, xlm: jax.Array,
                  pw_V_j: jax.Array, rho_j, cos_src_j, theta_src_j, th_e_j,
                  phi_ring: jax.Array, exterior: bool,
                  a: float, sl_scal, dl_scal, sh_eval: shtns_jax.sht,
                  sl_coeffs: tuple, dl_coeffs: tuple) -> tuple:
    """Traceable point-and-shoot core for ONE target ring. Consumes the already-rotated,
    pole-aligned source VWX coefficients (each (nlm_e,)) plus this ring's explicit geometry
    -- source-centered radius rho_j = r_j/a and latitude cos_src_j (scalars), source-centered
    polar angle theta_src_j (scalar), target-local polar angle th_e_j (scalar), this ring's
    precomputed leading radial-power slice pw_V_j (nlm_e,), and the shared ring longitudes
    phi_ring (nphi_e,) -- and returns the ring's velocity as target-local spherical components
    (vr_e, vt_e, vp_e), each (nphi_e,), still in the ROTATED frame (the caller's Stage 3
    rotates back). Batched over rings via jax.vmap (in_axes maps the per-ring slices). Stage 2
    of Corona & Veerapaneni 2018. sl_coeffs/dl_coeffs are the ring-independent VWX scaling
    symbols (from _ps_scale_coeffs), hoisted out of the vmap.

    The two remaining radial powers are derived from the precomputed pw_V_j by cheap scalar
    multiplies (exterior: rho^{-l} = pw_V*rho^2, rho^{-l-1} = pw_V*rho; interior: rho^{l-1} =
    pw_V/rho^2, rho^{l} = pw_V/rho), so only one (ntheta, nlm) power table is stored per
    evaluator and NO transcendental powers run per matvec."""
    # derive the W/X radial powers from the precomputed V power (scalar-mult, no transcendental)
    if exterior:
        pw_W_j = pw_V_j * rho_j * rho_j        # rho^{-l}   = rho^{-l-2} * rho^2
        pw_X_j = pw_V_j * rho_j                # rho^{-l-1} = rho^{-l-2} * rho
    else:
        pw_W_j = pw_V_j / (rho_j * rho_j)      # rho^{l-1}  = rho^{l+1} * rho^{-2}
        pw_X_j = pw_V_j / rho_j                # rho^{l}    = rho^{l+1} * rho^{-1}
    # per-ring solid-harmonic radial scaling of the SL and DL operators (V/W/X basis)
    vSL, wSL, xSL = _ps_scale_vwx(vlm, wlm, xlm, pw_V_j, pw_W_j, pw_X_j, exterior, "sl", sl_coeffs)
    vDL, wDL, xDL = _ps_scale_vwx(vlm, wlm, xlm, pw_V_j, pw_W_j, pw_X_j, exterior, "dl", dl_coeffs)
    vK = sl_scal * a * vSL + dl_scal * vDL          # SL scales by source radius a
    wK = sl_scal * a * wSL + dl_scal * wDL
    xK = sl_scal * a * xSL + dl_scal * xDL
    qlm_K, slm_K, tlm_K = vwx2qst(vK, wK, xK, sh_eval)                  # VWX -> QST (nlm_e,)
    # latitude FFT synthesis at this ring: source-centered spherical, (nphi_e,)
    vr, vt, vp = _stk_latitude_ring(qlm_K, slm_K, tlm_K, cos_src_j, sh_eval)
    # source-centered spherical -> Cartesian (rotated frame) -> target-local spherical
    uxR, uyR, uzR = sph2cart(vr, vt, vp, theta_src_j, phi_ring)
    return cart2sph(uxR, uyR, uzR, th_e_j, phi_ring)

# Cache of jitted point-and-shoot evaluators, keyed on geometry so the compiled
# sigma->velocity kernel is reused across calls with the same spheres (e.g. every
# GMRES iteration of the suspension matvec, where only the density changes).
_PS_EVALUATOR_CACHE: Dict[tuple, Any] = {}

def clear_point_n_shoot_cache() -> None:
    """Drop all cached point-and-shoot evaluators (and the shtns rotation objects they
    hold). Call to bound memory (up to ~N^2 entries for N spheres) or between problems."""
    _PS_EVALUATOR_CACHE.clear()

def _ps_geom_key(Strg: SphereDict, shtrg: shtns_jax.sht, S: SphereDict, sh: shtns_jax.sht) -> tuple:
    """Stable, value-based signature that fully determines the evaluator (rotations,
    exterior branch, all geometry constants): source/target lmax, centers, and radii.
    Identical across GMRES iterations even though the source sphere dict is rebuilt each
    matvec, since centers / radii / grids / sh config are unchanged."""
    return (int(S["lmax"]), int(Strg["lmax"]),
            tuple(np.asarray(S["Xc"], dtype=np.float64).tolist()), float(S["r"]),
            tuple(np.asarray(Strg["Xc"], dtype=np.float64).tolist()), float(Strg["r"]))

def point_n_shoot_evaluator(Strg: SphereDict, shtrg: shtns_jax.sht, S: SphereDict, sh: shtns_jax.sht,
                            near: bool = False):
    """
    Build a jitted point-and-shoot evaluator for a FIXED source/target geometry.
    Returns a callable apply(sigma_cart, sl_scal, dl_scal) -> velocity (nphi_t, ntheta_t, 3)
    for the combined Stokes layer potential K = sl_scal*SL + dl_scal*DL of source sphere
    <S> at the surface grid of target sphere <Strg> (non-concentric). Implements the
    3-stage FFT-accelerated near-evaluation of Corona & Veerapaneni 2018 (JCP 362),
    Sec. 4.2 / Fig. 2:
      (1) rotate the source Q/S/T density coefficients so the target center lies on
          the +z pole (Wigner-D, shtns_jax rotation.apply_cplx_jax, each scalar
          potential rotated independently);
      (2) on the now pole-aligned target rings (constant source-centered r, theta per
          ring), apply the solid-harmonic radial scaling and evaluate with one FFT in
          longitude per ring (O(p^3 log p));
      (3) rotate the sampled field back to the global frame.

    The eager setup here (reforming shts, building the shtns C rotation objects, and
    precomputing all geometry-only constants) runs once. The returned jax.jit kernel
    closes over them, so it compiles once and is reused for any density / scaling on this
    geometry. <near> is accepted for API symmetry with bio_offsurf_apply but does not
    change the radial branch (fixed by geometry: target fully exterior or interior to S).
    """
    if S["lmax"] != sh.lmax:
        print("S lmax does not match sht's lmax, reform sht.")
        sh = shtns_jax.sht(S["lmax"], S["lmax"])
    if Strg["lmax"] != shtrg.lmax:
        print("Strg lmax does not match sht_trg's lmax, reform sht_trg.")
        shtrg = shtns_jax.sht(Strg["lmax"], Strg["lmax"])

    a = float(S["r"]); R_t = float(Strg["r"])
    t_vec = jnp.asarray(Strg["Xc"], dtype=jnp.float64) - jnp.asarray(S["Xc"], dtype=jnp.float64)
    # Evaluate stages 2-3 at the finer of the two resolutions so no source content is
    # dropped before the radial scaling; band-limit to the target only at the final
    # synthesis. sh_eval owns that working grid (sh or shtrg, whichever has higher lmax).
    sh_eval = sh if sh.lmax >= shtrg.lmax else shtrg
    rot_fwd, rot_inv, d = _ps_rotation(t_vec, sh.lmax, sh_eval.lmax)

    # Distances from the source center to the target sphere surface span
    # [|d - R_t|, d + R_t]; pick the (uniform) exterior or interior radial branch.
    if abs(d - R_t) > a:
        exterior = True
    elif d + R_t < a:
        exterior = False
    else:
        raise ValueError(
            f"point_n_shoot requires the target sphere to be wholly exterior or "
            f"interior to the source (non-overlapping); got d={d}, a={a}, R_t={R_t}.")

    # ---- geometry-only constants (computed once; the jitted core closes over them) ----
    th_s = S["Xsph"][:, :, 0]; ph_s = S["Xsph"][:, :, 1]        # source-grid angles
    th_t = Strg["Xsph"][:, :, 0]; ph_t = Strg["Xsph"][:, :, 1]  # target-grid angles
    nlm_e = sh_eval.nlm_cplx; nlm_t = shtrg.nlm_cplx
    l_vals = jnp.asarray(sh_eval.zl, dtype=jnp.float64)         # (nlm_e,)
    theta_e = jnp.arccos(jnp.asarray(sh_eval.cos_theta))        # (ntheta_e,)
    r_j, cos_src = _ps_target_rings(d, R_t, theta_e)           # each (ntheta_e,)
    # Per-ring geometry passed into the vmapped single-ring core (all (ntheta_e,), except
    # the shared ring longitudes ph_ring_1d which are identical across rings).
    rho_ring = r_j / a                                         # (ntheta_e,) source-centered r/a
    theta_src_ring = jnp.arccos(cos_src)                       # (ntheta_e,) source-centered theta
    th_e_ring = theta_e                                        # (ntheta_e,) target-local theta
    ph_ring_1d = jnp.arange(sh_eval.nphi) * (2.0*jnp.pi/sh_eval.nphi)  # (nphi_e,) longitudes
    # Geometry-fixed leading radial-power table: rho^{-l-2} (exterior) or rho^{l+1} (interior),
    # shape (ntheta_e, nlm_e). Profiling showed the per-ring transcendental rho powers in
    # _ps_scale_vwx (rho_j is a vmapped tracer, so XLA cannot fold them) dominate the matvec at
    # O(lmax^3); precompute the one leading power here (once, eager) and derive the other two by
    # scalar multiplies in _ps_ring_core, so the per-matvec scaling is pure elementwise mults.
    if exterior:
        pw_V_ring = rho_ring[:, None] ** (-l_vals - 2.0)       # (ntheta_e, nlm_e)
    else:
        pw_V_ring = rho_ring[:, None] ** (l_vals + 1.0)
    # Ring-independent VWX scaling symbols: computed ONCE here (depend only on sh_eval.zl and the
    # static exterior branch), closed over by _core and reused across every ring in the vmap.
    sl_coeffs = _ps_scale_coeffs(sh_eval, exterior, "sl")
    dl_coeffs = _ps_scale_coeffs(sh_eval, exterior, "dl")
    _latlm_maps(sh_eval)   # populate the (sh-static) latlm cache eagerly, so _core reads
                           # concrete arrays rather than caching trace-scoped ones (leak)

    def _pad(z, n):   # lmax_eval >= lmax_src, so on the source this only zero-pads
        if n == z.shape[0]:
            return z                                              # #4: no-op for uniform lmax
        return jnp.concatenate([z, jnp.zeros(n - z.shape[0], dtype=jnp.complex128)]) if n > z.shape[0] else z[:n]

    def _pad3(zzz, n):    # pad a stacked (3, m) array along axis 1 to (3, n); no-op if m == n
        if n == zzz.shape[1]:
            return zzz
        if n > zzz.shape[1]:
            return jnp.concatenate(
                [zzz, jnp.zeros((3, n - zzz.shape[1]), dtype=jnp.complex128)], axis=1)
        return zzz[:, :n]

    def _core(sigma_cart: jax.Array, sl_scal, dl_scal) -> jax.Array:
        # ---- source density -> Q/S/T coefficients (cplx layout) ----
        sr, st_s, sp_s = cart2sph(sigma_cart[:, :, 0], sigma_cart[:, :, 1], sigma_cart[:, :, 2], th_s, ph_s)
        qst_src = sh.analys_vec_cplx_jax(vec_stack(sr, st_s, sp_s))   # (3, nlm_src)

        # ---- STAGE 1: rotate the target center onto +z. #2: rotate all three scalar
        #      potentials in ONE vmapped FFI call over the stacked (3, nlm_src) array
        #      (was three separate apply_cplx_jax calls + per-component re-stacking). ----
        qstR = _pad3(jax.vmap(rot_fwd.apply_cplx_jax)(qst_src), nlm_e)   # (3, nlm_e)

        # ---- STAGE 2: per-ring radial scaling + latitude FFT + reframe to target-local
        #      spherical, one ring at a time, batched over rings with vmap (sh_eval grid) ----
        vlm, wlm, xlm = qst2vwx(qstR[0], qstR[1], qstR[2], sh_eval)  # rotated source VWX (nlm_e,)
        # #5: out_axes=1 places the mapped (ring) axis at position 1, so each output is built
        # directly as (nphi_e, ntheta_e) -- no post-hoc .T transpose of the ring-major result.
        vr_e, vt_e, vp_e = jax.vmap(
            _ps_ring_core,
            in_axes=(None, None, None, 0, 0, 0, 0, 0,
                     None, None, None, None, None, None, None, None),
            out_axes=1)(
            vlm, wlm, xlm, pw_V_ring, rho_ring, cos_src, theta_src_ring, th_e_ring,
            ph_ring_1d, exterior, a, sl_scal, dl_scal, sh_eval,
            sl_coeffs, dl_coeffs)

        # ---- STAGE 3: rotate the sampled field back (again a single vmapped FFI call over the
        #      stacked (3, nlm_e) analysis output), band-limit to the target grid. ----
        qst_R = sh_eval.analys_vec_cplx_jax(vec_stack(vr_e, vt_e, vp_e))   # (3, nlm_e)
        qst_g = _pad3(jax.vmap(rot_inv.apply_cplx_jax)(qst_R), nlm_t)      # (3, nlm_t)
        vr_g, vt_g, vp_g = vec_distr(shtrg.synth_vec_cplx_jax(qst_g))
        ux, uy, uz = sph2cart(vr_g, vt_g, vp_g, th_t, ph_t)
        return jnp.stack([ux, uy, uz], axis=2)

    return jax.jit(_core)

def point_n_shoot(Strg: SphereDict, shtrg: shtns_jax.sht, S: SphereDict, sh: shtns_jax.sht,
                  sl_scal: float, dl_scal: float, near: bool = False) -> jax.Array:
    """
    Point-and-shoot (move-pole) evaluation of the combined Stokes layer potential
        K = sl_scal*SL + dl_scal*DL
    of source sphere <S> (density S["Sigma"]) at the surface grid of target sphere <Strg>.
    Thin wrapper over point_n_shoot_evaluator: fetches (or builds) the jitted evaluator for
    this geometry from a module cache, then applies it to S["Sigma"]. Repeated calls with
    the same geometry (e.g. every GMRES iteration of the suspension matvec) reuse the same
    compiled kernel. See point_n_shoot_evaluator for the algorithm and clear_point_n_shoot_cache
    to release cached kernels. Returns velocity at Strg's grid: (nphi_t, ntheta_t, 3) complex.
    """
    key = _ps_geom_key(Strg, shtrg, S, sh)
    evaluator = _PS_EVALUATOR_CACHE.get(key)
    if evaluator is None:
        evaluator = point_n_shoot_evaluator(Strg, shtrg, S, sh, near=near)
        _PS_EVALUATOR_CACHE[key] = evaluator
    return evaluator(S["Sigma"], sl_scal, dl_scal)

def Stk3d_dl_point_and_shoot(Strg: SphereDict, shtrg: shtns_jax.sht, S: SphereDict, sh: shtns_jax.sht) -> jax.Array:
    """DL-only point-and-shoot (thin wrapper over point_n_shoot)."""
    return point_n_shoot(Strg, shtrg, S, sh, 0.0, 1.0, near=False)





def _stk_trg_sph(trg: jax.Array, S: SphereDict) -> tuple([jax.Array, jax.Array, jax.Array]):
    """Spherical coordinates (dr, theta, phi) of targets <trg> relative to S["Xc"]."""
    trg_dx = trg[:,0] - S["Xc"][0]
    trg_dy = trg[:,1] - S["Xc"][1]
    trg_dz = trg[:,2] - S["Xc"][2]
    trg_dr = jnp.sqrt(trg_dx*trg_dx + trg_dy*trg_dy + trg_dz*trg_dz)
    trg_phi = jnp.atan2(trg_dy, trg_dx)
    trg_theta = jnp.acos(trg_dz / trg_dr)
    return trg_dr, trg_theta, trg_phi

def _stk_qst_to_points(qlm: jax.Array, slm: jax.Array, tlm: jax.Array, trg_theta: jax.Array, trg_phi: jax.Array, sh: shtns_jax.sht) -> jax.Array:
    """Evaluate the complex Q/S/T vector expansion (per-target coefficients,
    shape Ntrg x nlm_cplx) at the target points (trg_theta, trg_phi) and return
    the Cartesian velocity Ntrg x 3.  Mirrors the per-target loop of Lap3d_dl: the
    point synthesis uses the scalar shtns SHqst_to_point_cplx, only the C call is
    looped (coefficients differ per target)."""
    # Materialize every array to host once, up front. The per-target loop below calls into
    # synchronous C (sh.SHqst_to_point_cplx), so nothing it touches may be a live JAX array:
    # per-element host transfers inside the loop race the C call and can segfault.
    qlm = np.asarray(qlm, dtype=np.complex128)
    slm = np.asarray(slm, dtype=np.complex128)
    tlm = np.asarray(tlm, dtype=np.complex128)
    trg_theta = np.asarray(trg_theta, dtype=np.float64)
    trg_phi = np.asarray(trg_phi, dtype=np.float64)
    cost = np.cos(trg_theta)
    def _helper(i):
        vr, vt, vp = sh.SHqst_to_point_cplx(qlm[i], slm[i], tlm[i], float(cost[i]), float(trg_phi[i]))
        vx, vy, vz = sph2cart(vr, vt, vp, trg_theta[i], trg_phi[i])
        return vx, vy, vz
    vals = [_helper(trg_i) for trg_i in range(qlm.shape[0])]
    return jnp.array(vals, dtype=jnp.complex128)

def Stk3d_sl(trg: jax.Array, S: SphereDict, sh: shtns_jax.sht) -> jax.Array:
    """
    Off-surface evaluation of the Stokes SL velocity at arbitrary points <trg>: Ntrg x 3
        From source <S> with density <S["Sigma"]>, source uses SHT object <sh>.
    Spectral solid-harmonic scaling identical to Stk3d_sl_r_1sph, but per-target
    radius and point synthesis.  Returns Ntrg x 3.
    """
    assert trg.shape[1] == 3
    if S["lmax"] != sh.lmax:
        print("S lmax does not match sht's lmax, reform sht.")
        sh = shtns_jax.sht(S["lmax"], S["lmax"])

    vlm_sigma, wlm_sigma, xlm_sigma = sig_xyz2vwx(
        S["Sigma"][:,:,0], S["Sigma"][:,:,1], S["Sigma"][:,:,2],
        S["Xsph"][:,:,0], S["Xsph"][:,:,1], sh)

    trg_dr, trg_theta, trg_phi = _stk_trg_sph(trg, S)
    trg_dr = trg_dr[:,None]   # Ntrg x 1, broadcasts against (nlm,)
    a = S['r']
    rho = trg_dr / a          # solid harmonics evaluated as if src sphere were unit
    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64)
    diag_V, diag_W, diag_X = Stk3d_sl_VWX_diag(sh)
    rpowers_V_ext = rho ** (-l_vals-2.0)
    rpowers_V_int = rho ** (l_vals+1.0)
    vlm_SL_sigma_ext = rpowers_V_ext * diag_V * vlm_sigma
    vlm_SL_sigma_int = rpowers_V_int * diag_V * vlm_sigma

    rpowers_W_ext = rho ** (-l_vals)
    rpowers_W_int = rho ** (l_vals - 1.0)
    wlm_SL_sigma_ext = rpowers_W_ext * diag_W * wlm_sigma
    wlm_SL_sigma_int = rpowers_W_int * diag_W * wlm_sigma

    rpowers_X_ext = rho ** (-l_vals - 1.0)
    rpowers_X_int = rho ** (l_vals)
    xlm_SL_sigma_ext = rpowers_X_ext * diag_X * xlm_sigma
    xlm_SL_sigma_int = rpowers_X_int * diag_X * xlm_sigma

    diag_V2W_int = diag_V2W(sh, "sl")
    diag_W2V_ext = diag_W2V(sh, "sl")
    rpowers_V2W_int = rho ** (l_vals+1.0) - rho ** (l_vals - 1.0) # Note: TYPO IN PAPER
    rpowers_W2V_ext = rho ** (-l_vals - 2.0) - rho ** (-l_vals)
    V2Wlm_SL_sigma_int = rpowers_V2W_int * diag_V2W_int * vlm_sigma
    W2Vlm_SL_sigma_ext = rpowers_W2V_ext * diag_W2V_ext * wlm_sigma

    is_ext = trg_dr > a
    vlm_SL_sigma = jnp.where(is_ext, vlm_SL_sigma_ext + W2Vlm_SL_sigma_ext, vlm_SL_sigma_int)
    wlm_SL_sigma = jnp.where(is_ext, wlm_SL_sigma_ext, wlm_SL_sigma_int + V2Wlm_SL_sigma_int)
    xlm_SL_sigma = jnp.where(is_ext, xlm_SL_sigma_ext, xlm_SL_sigma_int)

    qlm, slm, tlm = vwx2qst(vlm_SL_sigma, wlm_SL_sigma, xlm_SL_sigma, sh)
    return a * _stk_qst_to_points(qlm, slm, tlm, trg_theta, trg_phi, sh)   # SL scales by a

def Stk3d_dl(trg: jax.Array, S: SphereDict, sh: shtns_jax.sht) -> jax.Array:
    """
    Off-surface evaluation of the Stokes DL velocity at arbitrary points <trg>: Ntrg x 3
        From source <S> with density <S["Sigma"]>, source uses SHT object <sh>.
    Spectral solid-harmonic scaling identical to Stk3d_dl_r_1sph, but per-target
    radius and point synthesis.  Returns Ntrg x 3.
    """
    assert trg.shape[1] == 3
    if S["lmax"] != sh.lmax:
        print("S lmax does not match sht's lmax, reform sht.")
        sh = shtns_jax.sht(S["lmax"], S["lmax"])

    vlm_sigma, wlm_sigma, xlm_sigma = sig_xyz2vwx(
        S["Sigma"][:,:,0], S["Sigma"][:,:,1], S["Sigma"][:,:,2],
        S["Xsph"][:,:,0], S["Xsph"][:,:,1], sh)

    trg_dr, trg_theta, trg_phi = _stk_trg_sph(trg, S)
    trg_dr = trg_dr[:,None]
    a = S['r']
    rho = trg_dr / a          # solid harmonics evaluated as if src sphere were unit
    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64)
    diag_V_ext, diag_W_ext, diag_X_ext, diag_V_int, diag_W_int, diag_X_int = Stk3d_dl_VWX_diag(sh)

    rpowers_V_ext = rho ** (- l_vals - 2.0)
    rpowers_V_int = rho ** (l_vals + 1.0)
    vlm_DL_sigma_ext = rpowers_V_ext * diag_V_ext * vlm_sigma
    vlm_DL_sigma_int = rpowers_V_int * diag_V_int * vlm_sigma

    rpowers_W_ext = rho ** (-l_vals)
    rpowers_W_int = rho ** (l_vals - 1.0)
    wlm_DL_sigma_ext = rpowers_W_ext * diag_W_ext * wlm_sigma
    wlm_DL_sigma_int = rpowers_W_int * diag_W_int * wlm_sigma

    rpowers_X_ext = rho ** (-l_vals - 1.0)
    rpowers_X_int = rho ** (l_vals)
    xlm_DL_sigma_ext = rpowers_X_ext * diag_X_ext * xlm_sigma
    xlm_DL_sigma_int = rpowers_X_int * diag_X_int * xlm_sigma

    diag_V2W_int = diag_V2W(sh, "dl")
    diag_W2V_ext = diag_W2V(sh, "dl")
    rpowers_V2W_int = - rho ** (l_vals + 1.0) + rho ** (l_vals - 1.0)
    rpowers_W2V_ext = rho ** (-l_vals - 2.0) - rho ** (-l_vals)
    V2Wlm_DL_sigma_int = rpowers_V2W_int * diag_V2W_int * vlm_sigma
    W2Vlm_DL_sigma_ext = rpowers_W2V_ext * diag_W2V_ext * wlm_sigma

    is_ext = trg_dr > a
    vlm_DL_sigma = jnp.where(is_ext, vlm_DL_sigma_ext + W2Vlm_DL_sigma_ext, vlm_DL_sigma_int)
    wlm_DL_sigma = jnp.where(is_ext, wlm_DL_sigma_ext, wlm_DL_sigma_int + V2Wlm_DL_sigma_int)
    xlm_DL_sigma = jnp.where(is_ext, xlm_DL_sigma_ext, xlm_DL_sigma_int)

    qlm, slm, tlm = vwx2qst(vlm_DL_sigma, wlm_DL_sigma, xlm_DL_sigma, sh)
    return _stk_qst_to_points(qlm, slm, tlm, trg_theta, trg_phi, sh)

def bio_offsurf_apply(trg: jax.Array, S: SphereDict, sh: shtns_jax.sht, sl_scal: float, dl_scal: float, far: bool = None, sep_eta: float = 1e-1) -> jax.Array:
    """
    Evaluate the KL formulation of <S> with density <S["Sigma"]> at arbitrary target <trg>: Ntrg x 3.

    The two SL/DL kernels have complementary accuracy: the spectral solid-harmonic
    synthesis (Stk3d_sl/Stk3d_dl) is accurate for targets close to the surface, while
    the rotation-free smooth quadrature (Stk3d_sl_far/Stk3d_dl_far) is cheap and
    accurate only for targets well away from the surface. Rather than the caller picking
    one kernel for the whole target list, this function separates targets per-target and
    routes each to the appropriate kernel:
        far == None (default): split trg into far/near via separate_target (surface gap
                               > sep_eta * r) and combine with jnp.where -- far targets
                               use the smooth quadrature, near targets the spectral eval.
        far == True          : force the smooth quadrature for every target.
        far == False         : force the spectral synthesis for every target.
    (jnp.where keeps shapes static so the routed default stays jit-traceable; both kernels
    are evaluated for all targets and selected pointwise.)
    """
    if far is True:
        SLsigma = Stk3d_sl_far(trg, S, sh)
        DLsigma = Stk3d_dl_far(trg, S, sh)
    elif far is False:
        SLsigma = Stk3d_sl(trg, S, sh)
        DLsigma = Stk3d_dl(trg, S, sh)
    else:
        # far is None: separate targets into far/near per-target using sep_eta.
        sep_far = separate_target(trg, S, sep_eta)[:, None]   # (Ntrg, 1) bool, True == far
        SLsigma = jnp.where(sep_far, Stk3d_sl_far(trg, S, sh), Stk3d_sl(trg, S, sh))
        DLsigma = jnp.where(sep_far, Stk3d_dl_far(trg, S, sh), Stk3d_dl(trg, S, sh))
    Ksigma = sl_scal * SLsigma + dl_scal * DLsigma
    return Ksigma

@partial(jax.jit, static_argnames=["sh"])
def bio_onsurf_apply(sigma_tens: jax.Array, theta: jax.Array, phi: jax.Array, sh: shtns_jax.sht, sl_scal: float, dl_scal: float, sgn: float, dsl_scal: float = 0., radius: float = 1.0) -> jax.Array:
    """
    Apply the combined DL potential operator K = sl_scal * SL + dl_scal * DL
        to the density <sigma_tens> defined on the <sh> grid
        taking into account the DL jump condition with sign <sgn>.
    Returns the resulting function, also defined on the <sh> grid.
    The SL block scales by the source-sphere <radius> (DL/dSL/jump are scale-invariant).
    """

    sigma_x = sigma_tens[:,:,0]
    sigma_y = sigma_tens[:,:,1]
    sigma_z = sigma_tens[:,:,2]
    vlm, wlm, xlm = sig_xyz2vwx(sigma_x, sigma_y, sigma_z, theta, phi, sh)

    diag_V, diag_W, diag_X = Stk3d_sl_VWX_diag(sh)
    vlm_SL_sigma = radius * diag_V * vlm
    wlm_SL_sigma = radius * diag_W * wlm
    xlm_SL_sigma = radius * diag_X * xlm
    
    diag_V_ext, diag_W_ext, diag_X_ext, diag_V_int, diag_W_int, diag_X_int = Stk3d_dl_VWX_diag(sh)
    vlm_DL_sigma = 0.5*(diag_V_int + diag_V_ext) * vlm
    wlm_DL_sigma = 0.5*(diag_W_int + diag_W_ext) * wlm
    xlm_DL_sigma = 0.5*(diag_X_int + diag_X_ext) * xlm

    diag_V_ext, diag_W_ext, diag_X_ext, diag_V_int, diag_W_int, diag_X_int = Stk3d_dsl_VWX_diag(sh)
    vlm_dSL_sigma = 0.5*(diag_V_int + diag_V_ext) * vlm
    wlm_dSL_sigma = 0.5*(diag_W_int + diag_W_ext) * wlm
    xlm_dSL_sigma = 0.5*(diag_X_int + diag_X_ext) * xlm

    vlm_op = sl_scal * vlm_SL_sigma + dl_scal * vlm_DL_sigma + dsl_scal * vlm_dSL_sigma
    wlm_op = sl_scal * wlm_SL_sigma + dl_scal * wlm_DL_sigma + dsl_scal * wlm_dSL_sigma
    xlm_op = sl_scal * xlm_SL_sigma + dl_scal * xlm_DL_sigma + dsl_scal * xlm_dSL_sigma

    vx,vy,vz = sig_vwx2xyz(vlm_op,wlm_op,xlm_op,theta,phi,sh)
    vx = vx + 0.5 * sgn * dl_scal * sigma_x + 0.5 * (-1*sgn) * dsl_scal * sigma_x
    vy = vy + 0.5 * sgn * dl_scal * sigma_y + 0.5 * (-1*sgn) * dsl_scal * sigma_y
    vz = vz + 0.5 * sgn * dl_scal * sigma_z + 0.5 * (-1*sgn) * dsl_scal * sigma_z
    V = jnp.stack([vx, vy, vz], axis=2)

    return V

@partial(jax.jit, static_argnames=["sh"])
def stokes_onsurf_direct_solve(bc_vec: jax.Array, theta: jax.Array, phi: jax.Array, sh: shtns_jax.sht, sl_scal: float, dl_scal: float, sgn: float, dsl_scal: float = 0., radius: float = 1.0) -> jax.Array:
    """
    Directly solves the Stokes BIO equation using the VWX diagonal property.
    The SL block scales by the source-sphere <radius> (DL/dSL/jump are scale-invariant).
    """

    vlm_bc, wlm_bc, xlm_bc = sig_xyz2vwx(bc_vec[:,:,0], bc_vec[:,:,1], bc_vec[:,:,2], theta, phi, sh)
    diag_V_sl, diag_W_sl, diag_X_sl = Stk3d_sl_VWX_diag(sh)
    diag_V_ext, diag_W_ext, diag_X_ext, diag_V_int, diag_W_int, diag_X_int = Stk3d_dl_VWX_diag(sh)
    diag_V_dl = 0.5 * (diag_V_int + diag_V_ext)
    diag_W_dl = 0.5 * (diag_W_int + diag_W_ext)
    diag_X_dl = 0.5 * (diag_X_int + diag_X_ext)
    diag_V_ext, diag_W_ext, diag_X_ext, diag_V_int, diag_W_int, diag_X_int = Stk3d_dsl_VWX_diag(sh)
    diag_V_dsl = 0.5 * (diag_V_int + diag_V_ext)
    diag_W_dsl = 0.5 * (diag_W_int + diag_W_ext)
    diag_X_dsl = 0.5 * (diag_X_int + diag_X_ext)
    
    op_diag_V = (0.5 * dl_scal * sgn) + (dl_scal * diag_V_dl) + (radius * sl_scal * diag_V_sl) + (0.5 * dsl_scal * (-1*sgn)) + (dsl_scal * diag_V_dsl)
    op_diag_W = (0.5 * dl_scal * sgn) + (dl_scal * diag_W_dl) + (radius * sl_scal * diag_W_sl) + (0.5 * dsl_scal * (-1*sgn)) + (dsl_scal * diag_W_dsl)
    op_diag_X = (0.5 * dl_scal * sgn) + (dl_scal * diag_X_dl) + (radius * sl_scal * diag_X_sl) + (0.5 * dsl_scal * (-1*sgn)) + (dsl_scal * diag_X_dsl)

    eps = 1e-14
    def safe_div(bc_lm, op_diag):
        safe = jnp.where(jnp.abs(op_diag) > eps, op_diag, 1.0+0j)
        res = bc_lm / safe
        # Return BC value where diag is zero (null space)
        return jnp.where(jnp.abs(op_diag) <= eps, bc_lm, res)

    vlm_sigma = safe_div(vlm_bc, op_diag_V)
    wlm_sigma = safe_div(wlm_bc, op_diag_W)
    xlm_sigma = safe_div(xlm_bc, op_diag_X)

    sig_x, sig_y, sig_z = sig_vwx2xyz(vlm_sigma, wlm_sigma, xlm_sigma, theta, phi, sh)
    
    return jnp.stack([sig_x, sig_y, sig_z], axis=-1)

def compute_field(trg: jax.Array, src: jax.Array, force: jax.Array) -> jax.Array:
    """
    Compute the field generated by Stokeslets
        positioned at <src>: Nsrc x 3 
        with strenght <force>: Nsrc x 3
        at target positioned at <trg>: Ntrg x 3
    """

    assert trg.shape[1] == 3 and src.shape[1] == 3 and force.shape[1] == 3
    assert force.shape[0] == src.shape[0]

    srcx = src[:,0][None,:] 
    srcy = src[:,1][None,:]
    srcz = src[:,2][None,:]
    dx = trg[:,0][:,None] - srcx 
    dy = trg[:,1][:,None] - srcy
    dz = trg[:,2][:,None] - srcz
    dr = jnp.sqrt(dx*dx + dy*dy + dz*dz)
    
    r_vec = jnp.stack([dx, dy, dz], axis=-1) 
    r_norm = dr[..., None]  
    r_hat = r_vec / r_norm
    force_expanded = force[None, :, :]  
    dot_prod = jnp.sum(force_expanded * r_vec, axis=-1, keepdims=True)  
    u_contrib = (1/(8*jnp.pi)) * (force_expanded / r_norm + dot_prod * r_hat / (r_norm**2))
    u = jnp.sum(u_contrib, axis=1)  
    
    return u.astype(jnp.complex128)

def compute_traction(trg: jax.Array, trgN: jax.Array, src: jax.Array, force: jax.Array) -> jax.Array:
    """
    Compute the field generated by Stokeslets
        positioned at <src>: Nsrc x 3 
        with strenght <force>: Nsrc x 3
        at target positioned at <trg>: Ntrg x 3 with target normal <trgN>: Ntrg x 3
    """

    assert trg.shape[1] == 3 and trgN.shape[1] == 3 and src.shape[1] == 3 and force.shape[1] == 3
    assert force.shape[0] == src.shape[0]

    srcx = src[:,0][None,:] 
    srcy = src[:,1][None,:]
    srcz = src[:,2][None,:]
    dx = trg[:,0][:,None] - srcx 
    dy = trg[:,1][:,None] - srcy
    dz = trg[:,2][:,None] - srcz
    dr = jnp.sqrt(dx*dx + dy*dy + dz*dz)
    
    r_vec = jnp.stack([dx, dy, dz], axis=-1) 
    r_norm = dr[..., None]  
    invr = 1./r_norm
    invr3 = invr * invr * invr

    force_expanded = force[None, :, :]  
    rdotf = jnp.sum(force_expanded * r_vec, axis=-1, keepdims=True)  
    trgN_expanded = trgN[:, None, :]
    rdotn = jnp.sum(trgN_expanded * r_vec, axis=-1, keepdims=True)  
    u_contrib = -(3/(4*jnp.pi)) * (rdotf * r_vec * rdotn * invr3 * invr * invr)
    u = jnp.sum(u_contrib, axis=1)  
    
    return u.astype(jnp.complex128)
    
if __name__ == "__main__":
    """
    Test Stokes operators on spheres using manufactured solutions.
    """

    # Geometry setup
    lmax = 36
    center = jnp.array([0.,0.,0.])
    radius = 1.
    S = build_sphere(center, radius)
    S, sh = quadr_sphere(S, lmax)
    # BIOp parameter
    sl_scal = 1.0
    dl_scal = 1.0

    # Targets -- exterior
    print("\n Manufactured solutions test Stokes 3D solver on the unit sphere ---- Exterior Dirichlet problem")
    # Non-concentric exterior target sphere so point_n_shoot (to_lat_jax) applies:
    # wholly exterior to S (d - Rtrg = 2 > a = 1).
    trg_center = jnp.array([3.,0.,0.])
    Rtrg = radius
    sgn = 1.0
    Strg = build_sphere(trg_center, Rtrg)
    Strg, shtrg = quadr_sphere(Strg, lmax)

    # Manufactured solutions test
    ptsrc = jnp.array([[0.1,0.3,0.15],[-0.35,0.2,0.]])
    force = jnp.array([[1,1,1],[-1,0,0]])

    x = S["Xcart"][:,:,0]
    y = S["Xcart"][:,:,1]
    z = S["Xcart"][:,:,2]
    theta = S["Xsph"][:,:,0]
    phi = S["Xsph"][:,:,1]
    trg_sphere = jnp.column_stack([jnp.reshape(x,-1), jnp.reshape(y,-1), jnp.reshape(z,-1)])
    BC_pot = compute_field(trg_sphere, ptsrc, force)
    BC_pot = jnp.reshape(BC_pot, S["Xcart"].shape)

    # DIRECT solve
    sig_direct = stokes_onsurf_direct_solve(BC_pot, theta, phi, sh, sl_scal, dl_scal, sgn)
    bc_check_direct = bio_onsurf_apply(sig_direct, theta, phi, sh, sl_scal, dl_scal, sgn)
    resid_direct = jnp.linalg.norm(bc_check_direct - BC_pot)
    print("Residual of DIRECT solve = {a}".format(a=resid_direct))

    # Accuracy
    xtrg = Strg["Xcart"][:,:,0] 
    ytrg = Strg["Xcart"][:,:,1]
    ztrg = Strg["Xcart"][:,:,2]
    trg_sphere2 = jnp.column_stack([jnp.reshape(xtrg,-1), jnp.reshape(ytrg,-1), jnp.reshape(ztrg,-1)])
    true_field = compute_field(trg_sphere2, ptsrc, force) 
    true_field = jnp.reshape(true_field, Strg["Xcart"].shape)
    true_field = jnp.real(true_field)

    S = set_density(S, sig_direct[:,:,0], sig_direct[:,:,1], sig_direct[:,:,2])
    Ksig_direct = point_n_shoot(Strg, shtrg, S, sh, sl_scal, dl_scal)
    Ksig_direct = jnp.real(Ksig_direct)

    diff_direct = jnp.max(true_field - Ksig_direct) / jnp.max(true_field)
    print("Max relative error of order {lmax} solver at target radius {Rtrg} for direct solver is {d}".format(lmax=lmax, Rtrg=Rtrg, d=diff_direct))


    print("\n Manufactured solutions test Stokes 3D solver on the unit sphere ---- Exterior Neumann problem")
    # Formulation: u(x) = S[sigma](x), match du/dn(gamma) = dSn[sigma](gamma)
    xn = S["Xncart"][:,:,0]
    yn = S["Xncart"][:,:,1]
    zn = S["Xncart"][:,:,2]
    trgN_sphere = jnp.column_stack([jnp.reshape(xn,-1), jnp.reshape(yn,-1), jnp.reshape(zn,-1)])
    BC_flux = compute_traction(trg_sphere, trgN_sphere, ptsrc, force) 
    BC_flux = jnp.reshape(BC_flux, S["Xcart"].shape)
    # DIRECT solve
    sig_direct = stokes_onsurf_direct_solve(
        BC_flux, theta, phi,
        sh=sh,
        sl_scal=0.,
        dl_scal=0.,
        sgn=sgn,
        dsl_scal=1.0
    )
    bc_check_direct = bio_onsurf_apply(sig_direct, theta, phi, sh, 0., 0., sgn, 1.0)
    resid_direct = jnp.linalg.norm(bc_check_direct - BC_flux)
    jax.debug.print("Residual of DIRECT solve: {a}", a=resid_direct)
    # Accuracy
    xtrg = Strg["Xcart"][:,:,0] 
    ytrg = Strg["Xcart"][:,:,1]
    ztrg = Strg["Xcart"][:,:,2]
    trg_sphere2 = jnp.column_stack([jnp.reshape(xtrg,-1), jnp.reshape(ytrg,-1), jnp.reshape(ztrg,-1)])
    true_field = compute_field(trg_sphere2, ptsrc, force)
    true_field = jnp.reshape(true_field, Strg["Xcart"].shape)
    true_field = jnp.real(true_field)

    S = set_density(S, sig_direct[:,:,0], sig_direct[:,:,1], sig_direct[:,:,2])
    Ksig_direct = point_n_shoot(Strg, shtrg, S, sh, 1.0, 0.)
    Ksig_direct = jnp.real(Ksig_direct)

    diff_direct = jnp.max(true_field - Ksig_direct) / jnp.max(true_field)
    jax.debug.print("Max relative error of order {lmax} solver at target radius {Rtrg} for direct solver is {d}", lmax=lmax, Rtrg=Rtrg, d=diff_direct)



    # Targets -- interior
    print("\n Manufactured solutions test Stokes 3D solver on the unit sphere ---- Interior Dirichlet problem")
    # Non-concentric interior target sphere so point_n_shoot (to_lat_jax) applies:
    # wholly interior to S (d + Rtrg = 0.7 < a = 1).
    trg_center = jnp.array([0.2,0.,0.])
    Rtrg = radius * 0.5
    sgn = -1.0
    Strg = build_sphere(trg_center, Rtrg)
    Strg, shtrg = quadr_sphere(Strg, lmax)

    ptsrc = jnp.array([[1.3,1.75,-2],[-1.3,-1.,2.32]])
    force = jnp.array([[1,-0.93,1.25],[-0.2,1.37,0]])

    x = S["Xcart"][:,:,0]
    y = S["Xcart"][:,:,1]
    z = S["Xcart"][:,:,2]
    theta = S["Xsph"][:,:,0]
    phi = S["Xsph"][:,:,1]
    trg_sphere = jnp.column_stack([jnp.reshape(x,-1), jnp.reshape(y,-1), jnp.reshape(z,-1)])
    BC_pot = compute_field(trg_sphere, ptsrc, force)
    BC_pot = jnp.reshape(BC_pot, S["Xcart"].shape)

    # DIRECT solve
    sig_direct = stokes_onsurf_direct_solve(BC_pot, theta, phi, sh, sl_scal, dl_scal, sgn)
    bc_check_direct = bio_onsurf_apply(sig_direct, theta, phi, sh, sl_scal, dl_scal, sgn)
    resid_diag = jnp.linalg.norm(bc_check_direct - BC_pot)
    print("Residual of DIRECT solve = {a}".format(a=resid_diag))

    # Accuracy
    xtrg = Strg["Xcart"][:,:,0] 
    ytrg = Strg["Xcart"][:,:,1]
    ztrg = Strg["Xcart"][:,:,2]
    trg_sphere2 = jnp.column_stack([jnp.reshape(xtrg,-1), jnp.reshape(ytrg,-1), jnp.reshape(ztrg,-1)])
    true_field = compute_field(trg_sphere2, ptsrc, force) 
    true_field = jnp.reshape(true_field, S["Xcart"].shape)
    true_field = jnp.real(true_field)

    S = set_density(S, sig_direct[:,:,0], sig_direct[:,:,1], sig_direct[:,:,2])
    Ksig_direct = point_n_shoot(Strg, shtrg, S, sh, sl_scal, dl_scal)
    Ksig_direct = jnp.real(Ksig_direct)

    diff_direct = jnp.max(true_field - Ksig_direct) / jnp.max(true_field)
    print("Max relative error of order {lmax} solver at target radius {Rtrg} for direct solver is {d}".format(lmax=lmax, Rtrg=Rtrg, d=diff_direct))

    # Arbitrary-point spectral eval (bio_offsurf_apply) vs point_n_shoot (to_lat_jax) to check far eval formulas
    K_pns = jnp.real(jnp.reshape(point_n_shoot(Strg, shtrg, S, sh, sl_scal, dl_scal), (-1, 3)))
    K_far = jnp.real(bio_offsurf_apply(trg_sphere2, S, sh, sl_scal, dl_scal, far=True))
    err_far = jnp.max(jnp.abs(K_far - K_pns)) / jnp.max(jnp.abs(K_pns))
    jax.debug.print("Max relative error of bio_offsurf_apply (far eval) vs point_n_shoot at radius {Rtrg} = {e}", Rtrg=Rtrg, e=err_far)

    #  ...  to check SHqst_to_point_cplx
    K_pt = jnp.real(bio_offsurf_apply(trg_sphere2, S, sh, sl_scal, dl_scal, far=False))
    err_pt = jnp.max(jnp.abs(K_pt - K_pns)) / jnp.max(jnp.abs(K_pns))
    jax.debug.print("Max relative error of bio_offsurf_apply (point eval) vs point_n_shoot at radius {Rtrg} = {e}", Rtrg=Rtrg, e=err_pt)

