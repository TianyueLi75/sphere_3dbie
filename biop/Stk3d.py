"""
Stokes Operator Class:
    SL and DL operators on a sphere, using spectra and solid harmonics
    TODO:
        everything in numpy at the moment.
        Allow on-surface evaluation in bio_offsurf_apply()
        onsurf_diag_solve() l=0 currently set to BC values. Throw exception instead?
        SL traction
        solid harmonics r should be scaled s.t. src sphere has r = 1
"""

from typing import Dict, Any, Tuple
from functools import partial

import scipy
import numpy as np
import jax
import jax.numpy as jnp
import scipy.sparse.linalg
import shtns
import shtns_jax

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from sphere_np import *
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

    Sigma_x = S["Sigma"][:,:,0] 
    Sigma_y = S["Sigma"][:,:,1] 
    Sigma_z = S["Sigma"][:,:,2] 
    theta = S["Xsph"][:,:,0]
    phi = S["Xsph"][:,:,1]
    vlm_sigma, wlm_sigma, xlm_sigma = sig_xyz2vwx(Sigma_x, Sigma_y, Sigma_z, theta, phi, sh)
    
    trg_dr = Strg["r"]
    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64) 
    diag_V, diag_W, diag_X = Stk3d_sl_VWX_diag(sh) 
    rpowers_V_ext = trg_dr ** (-l_vals-2.0) 
    rpowers_V_int = trg_dr ** (l_vals+1.0) 
    vlm_SL_sigma_ext = rpowers_V_ext * diag_V * vlm_sigma 
    vlm_SL_sigma_int = rpowers_V_int * diag_V * vlm_sigma

    rpowers_W_ext = trg_dr ** (-l_vals)
    rpowers_W_int = trg_dr ** (l_vals - 1.0)
    wlm_SL_sigma_ext = rpowers_W_ext * diag_W * wlm_sigma 
    wlm_SL_sigma_int = rpowers_W_int * diag_W * wlm_sigma 

    rpowers_X_ext = trg_dr ** (-l_vals - 1.0)
    rpowers_X_int = trg_dr ** (l_vals)
    xlm_SL_sigma_ext = rpowers_X_ext * diag_X * xlm_sigma
    xlm_SL_sigma_int = rpowers_X_int * diag_X * xlm_sigma

    diag_V2W_int = (l_vals+1.0) / (4.0*l_vals+2.0) 
    diag_W2V_ext = l_vals / (4.0*l_vals+2.0) 
    rpowers_V2W_int = trg_dr ** (l_vals+1.0) - trg_dr ** (l_vals - 1.0) # Note: TYPO IN PAPER
    rpowers_W2V_ext = trg_dr ** (-l_vals - 2.0) - trg_dr ** (-l_vals)
    V2Wlm_SL_sigma_int = rpowers_V2W_int * diag_V2W_int * vlm_sigma
    W2Vlm_SL_sigma_ext = rpowers_W2V_ext * diag_W2V_ext * wlm_sigma

    vlm_SL_sigma = vlm_SL_sigma_ext + W2Vlm_SL_sigma_ext if trg_dr > S['r'] else vlm_SL_sigma_int 
    wlm_SL_sigma = wlm_SL_sigma_ext if trg_dr > S['r'] else wlm_SL_sigma_int + V2Wlm_SL_sigma_int
    xlm_SL_sigma = xlm_SL_sigma_ext if trg_dr > S['r'] else xlm_SL_sigma_int

    theta_trg = Strg["Xsph"][:,:,0]
    phi_trg = Strg["Xsph"][:,:,1]
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
    SL_sigma = jnp.stack([val_x, val_y, val_z], axis=2)

    return SL_sigma

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

    Sigma_x = S["Sigma"][:,:,0] 
    Sigma_y = S["Sigma"][:,:,1] 
    Sigma_z = S["Sigma"][:,:,2] 
    theta = S["Xsph"][:,:,0]
    phi = S["Xsph"][:,:,1]
    vlm_sigma, wlm_sigma, xlm_sigma = sig_xyz2vwx(Sigma_x, Sigma_y, Sigma_z, theta, phi, sh)

    trg_dr = Strg["r"]
    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64) 
    diag_V_ext, diag_W_ext, diag_X_ext, diag_V_int, diag_W_int, diag_X_int = Stk3d_dl_VWX_diag(sh)
    
    rpowers_V_ext = trg_dr ** (- l_vals - 2.0) 
    rpowers_V_int = trg_dr ** (l_vals + 1.0) 
    vlm_DL_sigma_ext = rpowers_V_ext * diag_V_ext * vlm_sigma 
    vlm_DL_sigma_int = rpowers_V_int * diag_V_int * vlm_sigma

    rpowers_W_ext = trg_dr ** (-l_vals)
    rpowers_W_int = trg_dr ** (l_vals - 1.0)
    wlm_DL_sigma_ext = rpowers_W_ext * diag_W_ext * wlm_sigma
    wlm_DL_sigma_int = rpowers_W_int * diag_W_int * wlm_sigma 

    rpowers_X_ext = trg_dr ** (-l_vals - 1.0)
    rpowers_X_int = trg_dr ** (l_vals)
    xlm_DL_sigma_ext = rpowers_X_ext * diag_X_ext * xlm_sigma
    xlm_DL_sigma_int = rpowers_X_int * diag_X_int * xlm_sigma

    diag_V2W_int = (l_vals+1.0) * (l_vals + 2.0) / (2.0*l_vals+1.0) 
    diag_W2V_ext = 2. * l_vals * (l_vals - 1.0) / (4.0*l_vals+2.0)
    rpowers_V2W_int = - trg_dr ** (l_vals + 1.0) + trg_dr ** (l_vals - 1.0)
    rpowers_W2V_ext = trg_dr ** (-l_vals - 2.0) - trg_dr ** (-l_vals) 
    V2Wlm_DL_sigma_int = rpowers_V2W_int * diag_V2W_int * vlm_sigma
    W2Vlm_DL_sigma_ext = rpowers_W2V_ext * diag_W2V_ext * wlm_sigma

    vlm_DL_sigma = vlm_DL_sigma_ext + W2Vlm_DL_sigma_ext if trg_dr > S['r'] else vlm_DL_sigma_int
    wlm_DL_sigma = wlm_DL_sigma_ext if trg_dr > S['r'] else wlm_DL_sigma_int + V2Wlm_DL_sigma_int
    xlm_DL_sigma = xlm_DL_sigma_ext if trg_dr > S['r'] else xlm_DL_sigma_int

    theta_trg = Strg["Xsph"][:,:,0]
    phi_trg = Strg["Xsph"][:,:,1]
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
    DL_sigma = jnp.stack([val_x, val_y, val_z], axis=2)

    return DL_sigma

def bio_onsurf_apply(sigma_tens: jax.Array, theta: jax.Array, phi: jax.Array, sh: shtns_jax.sht, sl_scal: float, dl_scal: float, sgn: float) -> jax.Array:
    """
    Apply the combined DL potential operator K = sl_scal * SL + dl_scal * DL
        to the density <sigma_tens> (flattened) defined on the <sh> grid
        taking into account the DL jump condition with sign <sgn>.
    Returns the resulting function, also defined on the <sh> grid, flattened.
    """

    sigma_tens = sigma_tens.reshape(theta.shape[0], theta.shape[1], 3)
    sigma_x = sigma_tens[:,:,0]
    sigma_y = sigma_tens[:,:,1]
    sigma_z = sigma_tens[:,:,2]
    vlm, wlm, xlm = sig_xyz2vwx(sigma_x, sigma_y, sigma_z, theta, phi, sh)

    diag_V, diag_W, diag_X = Stk3d_sl_VWX_diag(sh)
    vlm_SL_sigma = diag_V * vlm
    wlm_SL_sigma = diag_W * wlm
    xlm_SL_sigma = diag_X * xlm
    
    diag_V_ext, diag_W_ext, diag_X_ext, diag_V_int, diag_W_int, diag_X_int = Stk3d_dl_VWX_diag(sh)
    vlm_DL_sigma = 0.5*(diag_V_int + diag_V_ext) * vlm
    wlm_DL_sigma = 0.5*(diag_W_int + diag_W_ext) * wlm
    xlm_DL_sigma = 0.5*(diag_X_int + diag_X_ext) * xlm

    vlm_op = sl_scal * vlm_SL_sigma + dl_scal * vlm_DL_sigma
    wlm_op = sl_scal * wlm_SL_sigma + dl_scal * wlm_DL_sigma
    xlm_op = sl_scal * xlm_SL_sigma + dl_scal * xlm_DL_sigma

    vx,vy,vz = sig_vwx2xyz(vlm_op,wlm_op,xlm_op,theta,phi,sh)
    vx = vx + 0.5 * sgn * dl_scal * sigma_x
    vy = vy + 0.5 * sgn * dl_scal * sigma_y
    vz = vz + 0.5 * sgn * dl_scal * sigma_z
    V = jnp.stack([vx, vy, vz], axis=2)

    return V.flatten()

def stokes_onsurf_direct_solve(bc_vec: jax.Array, theta: jax.Array, phi: jax.Array, sh: shtns_jax.sht, sl_scal: float, dl_scal: float, sgn: float) -> jax.Array:
    """
    Directly solves the Stokes BIO equation using the VWX diagonal property.
    """

    vlm_bc, wlm_bc, xlm_bc = sig_xyz2vwx(bc_vec[:,:,0], bc_vec[:,:,1], bc_vec[:,:,2], theta, phi, sh)
    diag_V_sl, diag_W_sl, diag_X_sl = Stk3d_sl_VWX_diag(sh)
    diag_V_ext, diag_W_ext, diag_X_ext, diag_V_int, diag_W_int, diag_X_int = Stk3d_dl_VWX_diag(sh)
    diag_V_dl = 0.5 * (diag_V_int + diag_V_ext)
    diag_W_dl = 0.5 * (diag_W_int + diag_W_ext)
    diag_X_dl = 0.5 * (diag_X_int + diag_X_ext)
    
    op_diag_V = (0.5 * dl_scal * sgn) + (dl_scal * diag_V_dl) + (sl_scal * diag_V_sl)
    op_diag_W = (0.5 * dl_scal * sgn) + (dl_scal * diag_W_dl) + (sl_scal * diag_W_sl)
    op_diag_X = (0.5 * dl_scal * sgn) + (dl_scal * diag_X_dl) + (sl_scal * diag_X_sl)

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

    # GMRES solve
    StkK_apply = partial(
        bio_onsurf_apply,
        theta = theta,
        phi = phi,
        sh=sh,
        sl_scal=sl_scal, 
        dl_scal=dl_scal, 
        sgn=sgn
    )
    # TODO: fix the following gmres to be lx gmres, not np
    # total_dofs = S["Xcart"].size
    # gmres_func = scipy.sparse.linalg.LinearOperator((total_dofs, total_dofs), \
    #                                                 matvec=StkK_apply, \
    #                                                 dtype=jnp.complex128)
    # x, info = scipy.sparse.linalg.gmres(gmres_func, BC_pot.flatten(), x0=jnp.zeros(total_dofs, dtype=jnp.complex128), \
    #                                     atol = 1e-14, rtol = 1e-13, maxiter=200)
    # sig_gmres = x.reshape(theta.shape[0], theta.shape[1], 3)
    # bc_check = bio_onsurf_apply(x, theta, phi, sh, sl_scal, dl_scal, sgn)
    # resid_gmres = jnp.linalg.norm(bc_check - BC_pot.flatten())
    # print("Residual of GMRES solve = {a}, exitcode (0:successful): {b}".format(a=resid_gmres, b=info))

    # DIRECT solve
    sig_direct = stokes_onsurf_direct_solve(BC_pot, theta, phi, sh, sl_scal, dl_scal, sgn)
    bc_check_direct = bio_onsurf_apply(sig_direct.flatten(), theta, phi, sh, sl_scal, dl_scal, sgn)
    resid_direct = jnp.linalg.norm(bc_check_direct - BC_pot.flatten())
    print("Residual of DIRECT solve = {a}".format(a=resid_direct))

    # Accuracy
    xtrg = Strg["Xcart"][:,:,0] 
    ytrg = Strg["Xcart"][:,:,1]
    ztrg = Strg["Xcart"][:,:,2]
    trg_sphere2 = jnp.column_stack([jnp.reshape(xtrg,-1), jnp.reshape(ytrg,-1), jnp.reshape(ztrg,-1)])
    true_field = compute_field(trg_sphere2, ptsrc, force) 
    true_field = jnp.reshape(true_field, Strg["Xcart"].shape)
    true_field = jnp.real(true_field)

    # S = set_density(S, sig_gmres[:,:,0], sig_gmres[:,:,1], sig_gmres[:,:,2])
    # Ksig_gmres = bio_offsurf_apply_1sph(Strg, shtrg, S, sh, sl_scal, dl_scal)
    # Ksig_gmres = jnp.real(Ksig_gmres)

    S = set_density(S, sig_direct[:,:,0], sig_direct[:,:,1], sig_direct[:,:,2])
    Ksig_direct = bio_offsurf_apply_1sph(Strg, shtrg, S, sh, sl_scal, dl_scal)
    Ksig_direct = jnp.real(Ksig_direct)

    # diff_gmres = jnp.max(true_field - Ksig_gmres) / jnp.max(true_field)
    diff_direct = jnp.max(true_field - Ksig_direct) / jnp.max(true_field)
    # print("Max relative error of order {lmax} solver at target radius {Rtrg} for GMRES solver is {d1}, for direct solver is {d2}".format(lmax=lmax, Rtrg=Rtrg, d1=diff_gmres, d2=diff_direct))
    print("Max relative error of order {lmax} solver at target radius {Rtrg} for direct solver is {d2}".format(lmax=lmax, Rtrg=Rtrg, d2=diff_direct))


    # Targets -- interior
    Rtrg = radius * 0.5
    sgn = -1.0
    Strg = build_sphere(center, Rtrg)
    Strg, shtrg = quadr_sphere(Strg, lmax)

    ptsrc = jnp.array([[1.3,1.75,-2],[-1.3,-1.75,2]])
    force = jnp.array([[1,1,1],[-1,-1,-1]]) # net force zero for interior flow

    x = S["Xcart"][:,:,0]
    y = S["Xcart"][:,:,1]
    z = S["Xcart"][:,:,2]
    theta = S["Xsph"][:,:,0]
    phi = S["Xsph"][:,:,1]
    trg_sphere = jnp.column_stack([jnp.reshape(x,-1), jnp.reshape(y,-1), jnp.reshape(z,-1)])
    BC_pot = compute_field(trg_sphere, ptsrc, force)
    BC_pot = jnp.reshape(BC_pot, S["Xcart"].shape)

    # GMRES solve
    StkK_apply = partial(
        bio_onsurf_apply,
        theta = theta,
        phi = phi,
        sh=sh,
        sl_scal=sl_scal, 
        dl_scal=dl_scal, 
        sgn=sgn
    )
    # total_dofs = S["Xcart"].size
    # gmres_func = scipy.sparse.linalg.LinearOperator((total_dofs, total_dofs), \
    #                                                 matvec=StkK_apply, \
    #                                                 dtype=jnp.complex128)
    # x, info = scipy.sparse.linalg.gmres(gmres_func, BC_pot.flatten(), x0=jnp.zeros(total_dofs, dtype=jnp.complex128), \
    #                                     atol = 1e-14, rtol = 1e-13, maxiter=200)
    # sig_gmres = x.reshape(theta.shape[0], theta.shape[1], 3)
    # bc_check = bio_onsurf_apply(x, theta, phi, sh, sl_scal, dl_scal, sgn)
    # resid_gmres = jnp.linalg.norm(bc_check - BC_pot.flatten())
    # print("Residual of GMRES solve = {a}, exitcode (0:successful): {b}".format(a=resid_gmres, b=info))

    # DIRECT solve
    sig_direct = stokes_onsurf_direct_solve(BC_pot, theta, phi, sh, sl_scal, dl_scal, sgn)
    bc_check_direct = bio_onsurf_apply(sig_direct.flatten(), theta, phi, sh, sl_scal, dl_scal, sgn)
    resid_diag = jnp.linalg.norm(bc_check_direct - BC_pot.flatten())
    print("Residual of DIRECT solve = {a}".format(a=resid_diag))

    # Accuracy
    xtrg = Strg["Xcart"][:,:,0] 
    ytrg = Strg["Xcart"][:,:,1]
    ztrg = Strg["Xcart"][:,:,2]
    trg_sphere2 = jnp.column_stack([jnp.reshape(xtrg,-1), jnp.reshape(ytrg,-1), jnp.reshape(ztrg,-1)])
    true_field = compute_field(trg_sphere2, ptsrc, force) 
    true_field = jnp.reshape(true_field, S["Xcart"].shape)
    true_field = jnp.real(true_field)

    # S = set_density(S, sig_gmres[:,:,0], sig_gmres[:,:,1], sig_gmres[:,:,2])
    # Ksig_gmres = bio_offsurf_apply_1sph(Strg, shtrg, S, sh, sl_scal, dl_scal)
    # Ksig_gmres = jnp.real(Ksig_gmres)

    S = set_density(S, sig_direct[:,:,0], sig_direct[:,:,1], sig_direct[:,:,2])
    Ksig_direct = bio_offsurf_apply_1sph(Strg, shtrg, S, sh, sl_scal, dl_scal)
    Ksig_direct = jnp.real(Ksig_direct)

    # diff_gmres = jnp.max(true_field - Ksig_gmres) / jnp.max(true_field)
    diff_direct = jnp.max(true_field - Ksig_direct) / jnp.max(true_field)
    # print("Max relative error of order {lmax} solver at target radius {Rtrg} for GMRES solver is {d1}, for direct solver is {d2}".format(lmax=lmax, Rtrg=Rtrg, d1=diff_gmres, d2=diff_direct))
    print("Max relative error of order {lmax} solver at target radius {Rtrg} for direct solver is {d2}".format(lmax=lmax, Rtrg=Rtrg,  d2=diff_direct))
