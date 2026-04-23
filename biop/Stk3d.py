from typing import Dict, Any, Tuple
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import lineax as lx
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sphere import *
import shtns

jax.config.update("jax_enable_x64", True)  # support float64

SphereDict = Dict[str, Any]

def Stk3d_sl_VWX_diag(sh: shtns.sht) -> tuple([jax.Array, jax.Array, jax.Array]):
    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64)
    diag_V = l_vals / (2.0*l_vals + 1.0) / (2.0*l_vals + 3.0)
    diag_W = (l_vals + 1.0) / (2.0*l_vals + 1.0) / (2.0*l_vals - 1.0)
    diag_X = 1.0 / (2.0*l_vals + 1.0)

    return diag_V, diag_W, diag_X

def Stk3d_dl_VWX_diag(sh: shtns.sht) -> tuple([jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]):
    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64)

    diag_V_ext = (2.0*l_vals*l_vals + 4*l_vals + 3) / (2.0*l_vals + 1.0) / (2.0*l_vals + 3.0)
    diag_W_ext = 2.0*(l_vals + 1.0)*(l_vals - 1.0) / (2.0*l_vals + 1.0) / (2.0*l_vals - 1.0)
    diag_X_ext = (l_vals - 1.0) / (2.0*l_vals + 1.0)

    diag_V_int = -2.0*l_vals*(l_vals + 2) / (2.0*l_vals + 1.0) / (2.0*l_vals + 3.0)
    diag_W_int = -(2.0*l_vals*l_vals + 1.0) / (2.0*l_vals + 1.0) / (2.0*l_vals - 1.0)
    diag_X_int = -(l_vals + 2.0) / (2.0*l_vals + 1.0)

    return diag_V_ext, diag_W_ext, diag_X_ext, diag_V_int, diag_W_int, diag_X_int
    
# TODO: S traction

def Stk3d_sl(trg: jax.Array, S: SphereDict, sh: shtns.sht) -> jax.Array:
    if S["lmax"] != sh.lmax:
        print("S lmax does not match sht's lmax, reform sht.")
        sh = shtns.sht(S["lmax"], S["lmax"])

    Sigma_x = S["Sigma"][:,:,0] 
    Sigma_y = S["Sigma"][:,:,1] 
    Sigma_z = S["Sigma"][:,:,2] 
    Sigma_r,Sigma_theta,Sigma_phi = cart2sph(Sigma_x,Sigma_y,Sigma_z) # TODO: convert x-y-z field to r-t-p
    qlm_sigma, slm_sigma, tlm_sigma = sh.analys_cplx(Sigma_r, Sigma_theta, Sigma_phi)
    vlm_sigma, wlm_sigma, xlm_sigma = qst2vwx(qlm_sigma, slm_sigma, tlm_sigma) # TODO: qst basis is indeed the Y-G-X basis not V-W-M. paper has conversions.

    assert trg.shape[1] == 3

    Ntrg = trg.shape[0]
    trg_dx = trg[:,0] - S["Xc"][0]
    trg_dy = trg[:,1] - S["Xc"][1]
    trg_dz = trg[:,2] - S["Xc"][2]
    trg_dr = jnp.sqrt(trg_dx*trg_dx + trg_dy*trg_dy + trg_dz*trg_dz)
    trg_phi = jnp.atan2(trg_dy, trg_dx)
    trg_theta = jnp.acos(trg_dz / trg_dr)
    trg_dr = trg_dr[:,None] # Ntrg x 1

    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64) # shape ((p+1)^2, )
    diag_V, diag_W, diag_X = Stk3d_sl_VWX_diag(sh) # Coeff for V-V, W-W, X-X matches for SL
    diag_W2V_int = (l_vals+1.0) / (4.0*l_vals+2.0) # additional coeff for V<-W, W<-V
    diag_V2W_ext = l_vals / (4.0*l_vals+2.0) # additional coeff for V<-W, W<-V

    rpowers_V_ext = trg_dr ** (-l_vals-2.0) # Ntrg x Nlm
    rpowers_V_int = trg_dr ** (l_vals+1.0) 
    rpowers_W2V_int = trg_dr ** (l_vals-1.0) - trg_dr ** (l_vals + 1.0)
    vlm_SL_sigma_ext = rpowers_V_ext * diag_V * vlm_sigma # resulting Vnm coeff from sigma_Vnm terms (SL[V] = aV + bW)
    vlm_SL_sigma_int = rpowers_V_int * diag_V * vlm_sigma + rpowers_W2V_int * diag_W2V_int * wlm_sigma

    rpowers_W_ext = trg_dr ** (-l_vals)
    rpowers_W_int = trg_dr ** (l_vals - 1.0)
    rpowers_V2W_ext = trg_dr ** (-l_vals - 2.0) - trg_dr ** (-l_vals)
    wlm_SL_sigma_ext = rpowers_W_ext * diag_W * wlm_sigma + rpowers_V2W_ext * diag_V2W_ext * vlm_sigma 
    wlm_SL_sigma_int = rpowers_W_int * diag_W * wlm_sigma 

    rpowers_X_ext = trg_dr ** (-l_vals - 1.0)
    rpowers_X_int = trg_dr ** (l_vals)
    xlm_SL_sigma_ext = rpower_X_ext * diag_X * xlm_sigma
    xlm_SL_sigma_int = rpower_X_int * diag_X * xlm_sigma

    vlm_SL_sigma = jnp.where(trg_dr > S['r'], qlm_SL_sigma_ext, qlm_SL_sigma_int)
    wlm_SL_sigma = jnp.where(trg_dr > S['r'], slm_SL_sigma_ext, slm_SL_sigma_int)
    xlm_SL_sigma = jnp.where(trg_dr > S['r'], tlm_SL_sigma_ext, tlm_SL_sigma_int)

    qlm_SL_sigma, slm_SL_sigma, tlm_SL_sigma = vwm2qst(vlm_SL_sigma, wlm_SL_sigma, xlm_SL_sigma) # TODO

    qlm_SL_sigma = np.array(qlm_SL_sigma, dtype=np.complex128)
    slm_SL_sigma = np.array(slm_SL_sigma, dtype=np.complex128)
    tlm_SL_sigma = np.array(tlm_SL_sigma, dtype=np.complex128)
    trg_phi = np.array(trg_phi)
    trg_costheta = np.cos(trg_theta)

    # TODO: use jax.vmap over all targets
    # TODO: complex version of SHqst?
    SL_sigma = jnp.zeros((Ntrg,3), dtype = jnp.complex128)
    for trg_i in range(Ntrg):
        val_r, val_t, val_p = sh.SHqst_to_point(qlm_SL_sigma[trg_i,:], slm_SL_sigma[trg_i,:], tlm_SL_sigma[trg_i,:], trg_costheta[trg_i], trg_phi[trg_i])
        val_x, val_y, val_z = sph2cart(val_r, val_t, val_p) # TODO: convert spherical back to x-y-z
        SL_sigma = SL_sigma.at[trg_i,:].set(jnp.array([val_x, val_y, val_z]))

    return SL_sigma

    # TODO: StkDL