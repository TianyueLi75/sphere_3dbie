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
    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64)

    diag_V_int = (2.0*l_vals*l_vals + 4*l_vals + 3) / (2.0*l_vals + 1.0) / (2.0*l_vals + 3.0)
    diag_W_int = 2.0*(l_vals + 1.0)*(l_vals - 1.0) / (2.0*l_vals + 1.0) / (2.0*l_vals - 1.0)
    diag_X_int = (l_vals - 1.0) / (2.0*l_vals + 1.0)

    diag_V_ext = -2.0*l_vals*(l_vals + 2) / (2.0*l_vals + 1.0) / (2.0*l_vals + 3.0)
    diag_W_ext = -(2.0*l_vals*l_vals + 1.0) / (2.0*l_vals + 1.0) / (2.0*l_vals - 1.0)
    diag_X_ext = -(l_vals + 2.0) / (2.0*l_vals + 1.0)

    return diag_V_ext, diag_W_ext, diag_X_ext, diag_V_int, diag_W_int, diag_X_int

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

@partial(jax.jit, static_argnames=["sh", "shtrg", "exterior"])
def _stk_sl_1sph_kernel(Sigma: jax.Array, theta: jax.Array, phi: jax.Array,
                        theta_trg: jax.Array, phi_trg: jax.Array, a: float, trg_dr: float,
                        sh: shtns_jax.sht, shtrg: shtns_jax.sht, exterior: bool) -> jax.Array:
    """Jitted core of Stk3d_sl_r_1sph. <exterior> (the trg_dr > a branch) and the
    nlm pad/truncate are compile-time constants (static args)."""
    vlm_sigma, wlm_sigma, xlm_sigma = sig_xyz2vwx(
        Sigma[:, :, 0], Sigma[:, :, 1], Sigma[:, :, 2], theta, phi, sh)
    rho = trg_dr / a          # solid harmonics evaluated as if src sphere were unit
    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64)
    diag_V, diag_W, diag_X = Stk3d_sl_VWX_diag(sh)

    if exterior:
        diag_W2V_ext = l_vals / (4.0*l_vals+2.0)
        W2Vlm_SL_sigma_ext = (rho ** (-l_vals - 2.0) - rho ** (-l_vals)) * diag_W2V_ext * wlm_sigma
        vlm_SL_sigma = rho ** (-l_vals-2.0) * diag_V * vlm_sigma + W2Vlm_SL_sigma_ext
        wlm_SL_sigma = rho ** (-l_vals) * diag_W * wlm_sigma
        xlm_SL_sigma = rho ** (-l_vals - 1.0) * diag_X * xlm_sigma
    else:
        diag_V2W_int = (l_vals+1.0) / (4.0*l_vals+2.0)
        # rpowers_V2W_int has a typo in the paper; kept as previously verified.
        V2Wlm_SL_sigma_int = (rho ** (l_vals+1.0) - rho ** (l_vals - 1.0)) * diag_V2W_int * vlm_sigma
        vlm_SL_sigma = rho ** (l_vals+1.0) * diag_V * vlm_sigma
        wlm_SL_sigma = rho ** (l_vals - 1.0) * diag_W * wlm_sigma + V2Wlm_SL_sigma_int
        xlm_SL_sigma = rho ** (l_vals) * diag_X * xlm_sigma

    # Interpolate to new grid by padding or truncating coefficients to Strg['lmax']
    nlm_src = sh.nlm_cplx
    nlm_trg = shtrg.nlm_cplx
    if nlm_trg > nlm_src:
        vlm_SL_sigma = jnp.pad(vlm_SL_sigma, (0, nlm_trg - nlm_src), constant_values=0)
        wlm_SL_sigma = jnp.pad(wlm_SL_sigma, (0, nlm_trg - nlm_src), constant_values=0)
        xlm_SL_sigma = jnp.pad(xlm_SL_sigma, (0, nlm_trg - nlm_src), constant_values=0)
    elif nlm_trg < nlm_src:
        vlm_SL_sigma = vlm_SL_sigma[:nlm_trg]
        wlm_SL_sigma = wlm_SL_sigma[:nlm_trg]
        xlm_SL_sigma = xlm_SL_sigma[:nlm_trg]
    val_x, val_y, val_z = sig_vwx2xyz(vlm_SL_sigma, wlm_SL_sigma, xlm_SL_sigma, theta_trg, phi_trg, shtrg)
    return a * jnp.stack([val_x, val_y, val_z], axis=2)   # SL scales by a

def Stk3d_sl_r_1sph(Strg: SphereDict, shtrg: shtns_jax.sht, S: SphereDict, sh: shtns_jax.sht) -> jax.Array:
    """
    Off-surface evaluation at a spherical grid of targets
        From source <S> with density <S["Sigma"]>, source uses SHT object <sh>
        To target <Strg>, target uses SHT object <shtrg>
    """

    if S["lmax"] != sh.lmax:
        print("S lmax does not match sht's lmax, reform sht.")
        sh = shtns_jax.sht(S["lmax"], S["lmax"])
    if Strg["lmax"] != shtrg.lmax:
        print("Strg lmax does not match sht_trg's lmax, reform sht_trg.")
        shtrg = shtns_jax.sht(Strg["lmax"], Strg["lmax"])

    exterior = bool(Strg["r"] > S["r"])
    return _stk_sl_1sph_kernel(S["Sigma"], S["Xsph"][:, :, 0], S["Xsph"][:, :, 1],
                               Strg["Xsph"][:, :, 0], Strg["Xsph"][:, :, 1],
                               S["r"], Strg["r"], sh, shtrg, exterior)

@partial(jax.jit, static_argnames=["sh", "shtrg", "exterior"])
def _stk_dl_1sph_kernel(Sigma: jax.Array, theta: jax.Array, phi: jax.Array,
                        theta_trg: jax.Array, phi_trg: jax.Array, a: float, trg_dr: float,
                        sh: shtns_jax.sht, shtrg: shtns_jax.sht, exterior: bool) -> jax.Array:
    """Jitted core of Stk3d_dl_r_1sph. <exterior> (the trg_dr > a branch) and the
    nlm pad/truncate are compile-time constants (static args). DL is scale-invariant."""
    vlm_sigma, wlm_sigma, xlm_sigma = sig_xyz2vwx(
        Sigma[:, :, 0], Sigma[:, :, 1], Sigma[:, :, 2], theta, phi, sh)
    rho = trg_dr / a          # solid harmonics evaluated as if src sphere were unit
    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64)
    diag_V_ext, diag_W_ext, diag_X_ext, diag_V_int, diag_W_int, diag_X_int = Stk3d_dl_VWX_diag(sh)

    if exterior:
        diag_W2V_ext = 2. * l_vals * (l_vals - 1.0) / (4.0*l_vals+2.0)
        W2Vlm_DL_sigma_ext = (rho ** (-l_vals - 2.0) - rho ** (-l_vals)) * diag_W2V_ext * wlm_sigma
        vlm_DL_sigma = rho ** (- l_vals - 2.0) * diag_V_ext * vlm_sigma + W2Vlm_DL_sigma_ext
        wlm_DL_sigma = rho ** (-l_vals) * diag_W_ext * wlm_sigma
        xlm_DL_sigma = rho ** (-l_vals - 1.0) * diag_X_ext * xlm_sigma
    else:
        diag_V2W_int = (l_vals+1.0) * (l_vals + 2.0) / (2.0*l_vals+1.0)
        V2Wlm_DL_sigma_int = (- rho ** (l_vals + 1.0) + rho ** (l_vals - 1.0)) * diag_V2W_int * vlm_sigma
        vlm_DL_sigma = rho ** (l_vals + 1.0) * diag_V_int * vlm_sigma
        wlm_DL_sigma = rho ** (l_vals - 1.0) * diag_W_int * wlm_sigma + V2Wlm_DL_sigma_int
        xlm_DL_sigma = rho ** (l_vals) * diag_X_int * xlm_sigma

    # Interpolate to new grid by padding or truncating coefficients to Strg['lmax']
    nlm_src = sh.nlm_cplx
    nlm_trg = shtrg.nlm_cplx
    if nlm_trg > nlm_src:
        vlm_DL_sigma = jnp.pad(vlm_DL_sigma, (0, nlm_trg - nlm_src), constant_values=0)
        wlm_DL_sigma = jnp.pad(wlm_DL_sigma, (0, nlm_trg - nlm_src), constant_values=0)
        xlm_DL_sigma = jnp.pad(xlm_DL_sigma, (0, nlm_trg - nlm_src), constant_values=0)
    elif nlm_trg < nlm_src:
        vlm_DL_sigma = vlm_DL_sigma[:nlm_trg]
        wlm_DL_sigma = wlm_DL_sigma[:nlm_trg]
        xlm_DL_sigma = xlm_DL_sigma[:nlm_trg]
    val_x, val_y, val_z = sig_vwx2xyz(vlm_DL_sigma, wlm_DL_sigma, xlm_DL_sigma, theta_trg, phi_trg, shtrg)
    return jnp.stack([val_x, val_y, val_z], axis=2)

def Stk3d_dl_r_1sph(Strg: SphereDict, shtrg: shtns_jax.sht, S: SphereDict, sh: shtns_jax.sht) -> jax.Array:
    """
    Off-surface evaluation at a spherical grid of targets
        From source <S> with density <S["Sigma"]>, source uses SHT object <sh>
        To target <Strg>, target uses SHT object <shtrg>
    """

    if S["lmax"] != sh.lmax:
        print("S lmax does not match sht's lmax, reform sht.")
        sh = shtns_jax.sht(S["lmax"], S["lmax"])
    if Strg["lmax"] != shtrg.lmax:
        print("Strg lmax does not match sht_trg's lmax, reform sht_trg.")
        shtrg = shtns_jax.sht(Strg["lmax"], Strg["lmax"])

    exterior = bool(Strg["r"] > S["r"])
    return _stk_dl_1sph_kernel(S["Sigma"], S["Xsph"][:, :, 0], S["Xsph"][:, :, 1],
                               Strg["Xsph"][:, :, 0], Strg["Xsph"][:, :, 1],
                               S["r"], Strg["r"], sh, shtrg, exterior)

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
    qlm = np.asarray(qlm, dtype=np.complex128)
    slm = np.asarray(slm, dtype=np.complex128)
    tlm = np.asarray(tlm, dtype=np.complex128)
    cost = np.cos(np.asarray(trg_theta))
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

    diag_V2W_int = (l_vals+1.0) / (4.0*l_vals+2.0)
    diag_W2V_ext = l_vals / (4.0*l_vals+2.0)
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

    diag_V2W_int = (l_vals+1.0) * (l_vals + 2.0) / (2.0*l_vals+1.0)
    diag_W2V_ext = 2. * l_vals * (l_vals - 1.0) / (4.0*l_vals+2.0)
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

def bio_offsurf_apply(trg: jax.Array, S: SphereDict, sh: shtns_jax.sht, sl_scal: float, dl_scal: float, far: bool = False) -> jax.Array:
    """
    Evaluate the KL formulation of <S> with density <S["Sigma"]> at arbitrary target <trg>: Ntrg x 3.
    """
    if not far:
        SLsigma = Stk3d_sl(trg, S, sh)
        DLsigma = Stk3d_dl(trg, S, sh)
    else:
        SLsigma = Stk3d_sl_far(trg, S, sh)
        DLsigma = Stk3d_dl_far(trg, S, sh)
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

def bio_offsurf_apply_1sph(Strg: SphereDict, shtrg: shtns_jax.sht, S: SphereDict, sh: shtns_jax.sht, sl_scal: float, dl_scal: float) -> jax.Array:
    """
    Evaluate the KL formulation of <S> with density <S["Sigma"]> at a spherical grid of targets on <Strg>
    """
    
    SLsigma = Stk3d_sl_r_1sph(Strg, shtrg, S, sh) 
    DLsigma = Stk3d_dl_r_1sph(Strg, shtrg, S, sh) 
    Ksigma = sl_scal * SLsigma + dl_scal * DLsigma
    return Ksigma

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
    Rtrg = radius * 1.025
    sgn = 1.0 
    Strg = build_sphere(center, Rtrg)
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
    Ksig_direct = bio_offsurf_apply_1sph(Strg, shtrg, S, sh, sl_scal, dl_scal)
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
    Ksig_direct = bio_offsurf_apply_1sph(Strg, shtrg, S, sh, 1.0, 0.)
    Ksig_direct = jnp.real(Ksig_direct)

    diff_direct = jnp.max(true_field - Ksig_direct) / jnp.max(true_field)
    jax.debug.print("Max relative error of order {lmax} solver at target radius {Rtrg} for direct solver is {d}", lmax=lmax, Rtrg=Rtrg, d=diff_direct)



    # Targets -- interior
    print("\n Manufactured solutions test Stokes 3D solver on the unit sphere ---- Interior Dirichlet problem")
    Rtrg = radius * 0.62
    sgn = -1.0
    Strg = build_sphere(center, Rtrg)
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
    Ksig_direct = bio_offsurf_apply_1sph(Strg, shtrg, S, sh, sl_scal, dl_scal)
    Ksig_direct = jnp.real(Ksig_direct)

    diff_direct = jnp.max(true_field - Ksig_direct) / jnp.max(true_field)
    print("Max relative error of order {lmax} solver at target radius {Rtrg} for direct solver is {d}".format(lmax=lmax, Rtrg=Rtrg, d=diff_direct))

    # Arbitrary-point spectral eval (bio_offsurf_apply) vs the grid _1sph to check far eval formulas
    K_1sph = jnp.real(jnp.reshape(bio_offsurf_apply_1sph(Strg, shtrg, S, sh, sl_scal, dl_scal), (-1, 3)))
    K_far = jnp.real(bio_offsurf_apply(trg_sphere2, S, sh, sl_scal, dl_scal, far=True))
    err_far = jnp.max(jnp.abs(K_far - K_1sph)) / jnp.max(jnp.abs(K_1sph))
    jax.debug.print("Max relative error of bio_offsurf_apply (far eval) vs bio_offsurf_apply_1sph at radius {Rtrg} = {e}", Rtrg=Rtrg, e=err_far)

    #  ...  to check SHqst_to_point_cplx
    K_pt = jnp.real(bio_offsurf_apply(trg_sphere2, S, sh, sl_scal, dl_scal, far=False))
    err_pt = jnp.max(jnp.abs(K_pt - K_1sph)) / jnp.max(jnp.abs(K_1sph))
    jax.debug.print("Max relative error of bio_offsurf_apply (point eval) vs bio_offsurf_apply_1sph at radius {Rtrg} = {e}", Rtrg=Rtrg, e=err_pt)

