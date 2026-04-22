from typing import Dict, Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Only for testing
from sphere import *
import matplotlib.pyplot as plt
import shtns

SphereDict = Dict[str, Any]

jax.clear_caches()

def Lap3d_sl_self(S: SphereDict, sh: shtns.sht) -> jax.Array:
    if S["lmax"] != sh.lmax:
        print("S lmax does not match sht's lmax, reform sht.")
        sh = shtns.sht(S["lmax"], S["lmax"])

    Sigma = S["Sigma"][:,:,0] # scalar operator, so only take Sigma_x
    qlm_sigma = sh.analys_cplx_jax(Sigma)

    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64)
    diag = 1.0 / (2.0 * l_vals + 1.0)

    SL_sigma = sh.synth_cplx_jax(qlm_sigma * diag)

    return SL_sigma

def Lap3d_dl_self(S: SphereDict, sh: shtns.sht) -> jax.Array:
    if S["lmax"] != sh.lmax:
        print("S lmax does not match sht's lmax, reform sht.")
        sh = shtns.sht(S["lmax"])

    Sigma = S["Sigma"][:,:,0] # scalar operator, so only take Sigma_x
    qlm_sigma = sh.analys_cplx_jax(Sigma)

    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64)
    diag = -0.5 / (2.0 * l_vals + 1.0) # double layer principal value: avg( l / (2l+1), -(l+1) / (2l+1) )

    DL_sigma = sh.synth_cplx_jax(qlm_sigma * diag)

    return DL_sigma

def Lap3d_sl(trg: jax.Array, S: SphereDict, sh: shtns.sht) -> jax.Array:
    if S["lmax"] != sh.lmax:
        print("S lmax does not match sht's lmax, reform sht.")
        sh = shtns.sht(S["lmax"], S["lmax"])

    Sigma = S["Sigma"][:,:,0] # scalar operator, so only take Sigma_x
    qlm_sigma = sh.analys_cplx_jax(Sigma)

    trg_dx = trg[:,0] - S["Xc"][0]
    trg_dy = trg[:,1] - S["Xc"][1]
    trg_dz = trg[:,2] - S["Xc"][2]
    trg_dr = jnp.sqrt(trg_dx*trg_dx + trg_dy*trg_dy + trg_dz*trg_dz)
    trg_dr = trg_dr[:,None] # Ntrg x 1
    trg_phi = jnp.atan(trg_dy / trg_dz)
    trg_theta = jnp.acos(trg_dz / trg_dr)

    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64) # shape ((p+1)^2, )
    diag_ext = 1.0 / (2.0 * l_vals + 1.0)
    diag_int = 1.0 / (2.0 * l_vals + 1.0)
    rpowers_ext = trg_dr ** (-l_vals-1) # Ntrg x Nlm
    rpowers_int = trg_dr ** (l_vals) # Ntrg x Nlm
    qlm_SL_sigma_ext = rpowers_ext * qlm_sigma * diag_ext
    qlm_SL_sigma_int = rpowers_int * qlm_sigma * diag_int

    qlm_SL_sigma = jnp.where(trg_dr > S['r'], qlm_SL_sigma_ext, qlm_SL_sigma_int)

    jax.debug.print("size of trg_theta: {}", trg_theta)

    # SL_sigma = sh.SH_to_point_cplx(qlm_SL_sigma, trg_theta, trg_phi)

    # jax.debug.print("shape of qlm_sigma * diag is {a}, shape of rpowers * that is {b}, shape of SLsigma after synth is {c}", a=(qlm_sigma*diag).shape, b=qlm_SL_sigma.shape, c=SL_sigma.shape)

    # return SL_sigma
    return 0

def Lap3d_dl(trg: jax.Array, S: SphereDict, sh: shtns.sht) -> jax.Array:
    if S["lmax"] != sh.lmax:
        print("S lmax does not match sht's lmax, reform sht.")
        sh = shtns.sht(S["lmax"])

    Sigma = S["Sigma"][:,:,0] # scalar operator, so only take Sigma_x
    qlm_sigma = sh.analys_cplx_jax(Sigma)

    trg_dx = trg[:,0] - S["Xc"][0]
    trg_dy = trg[:,1] - S["Xc"][1]
    trg_dz = trg[:,2] - S["Xc"][2]
    trg_dr = jnp.sqrt(trg_dx*trg_dx + trg_dy*trg_dy + trg_dz*trg_dz)
    trg_dr = trg_dr[:,None] # Ntrg x 1
    trg_phi = jnp.atan(trg_dy / trg_dz)
    trg_theta = jnp.acos(trg_dz / trg_dr)

    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64)
    diag_ext = l_vals / (2.0 * l_vals + 1.0) 
    diag_int = - (l_vals + 1) / (2.0 * l_vals + 1.0)
    rpowers_ext = trg_dr ** (-l_vals-1) # Ntrg x Nlm
    rpowers_int = trg_dr ** (l_vals) # Ntrg x Nlm
    qlm_DL_sigma_ext = rpowers_ext * qlm_sigma * diag_ext
    qlm_DL_sigma_int = rpowers_int * qlm_sigma * diag_int

    qlm_DL_sigma = jnp.where(trg_dr > S['r'], qlm_DL_sigma_ext, qlm_DL_sigma_int)
    
    DL_sigma = sh.SH_to_point_cplx(qlm_DL_sigma, trg_theta, trg_phi)

    return DL_sigma

# Given source S with density <dens>, compute the Laplace potential at <trg>
def compute_potential(trg: jax.Array, src: jax.Array, force: jax.Array) -> jax.Array:
    # trg and src are Ntrg x 3, Nsrc x 3 arrays of coordinates; force is Nsrc x 1 or Nsrc x , array of strengths
    assert trg.shape[1] == 3 and src.shape[1] == 3
    assert force.shape[0] == src.shape[0]
    if len(force.shape) > 1:
        assert force.shape[1] == 1
    else:
        force = force[:,None] # make Nsrc x 1

    dx = trg[:,0] - src[:,0]
    dy = trg[:,1] - src[:,1]
    dz = trg[:,2] - src[:,2]
    dr = jnp.sqrt(dx*dx + dy*dy + dz*dz)
    G = 1/4./jnp.pi / dr * force

    return G

def solve_bio(S):
    return


if __name__ == "__main__":

    lmax = 4
    center = jnp.array([0.,0.,0.])
    radius = 1.
    S = build_sphere(center, radius)
    S, sh = quadr_sphere(S, lmax)
    x = S["Xcart"][:,:,0]
    y = S["Xcart"][:,:,1]
    z = S["Xcart"][:,:,2]
    xn = S["Xncart"][:,:,0]
    yn = S["Xncart"][:,:,1]
    zn = S["Xncart"][:,:,2]

    # Simple density in x
    seed = 1701
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    Ycoeff_r = jax.random.uniform(subkey, shape=(sh.nlm_cplx,))
    key, subkey = jax.random.split(key)
    Ycoeff_i = jax.random.uniform(subkey, shape=(sh.nlm_cplx,)) 
    Ycoeff = jax.lax.complex(Ycoeff_r, Ycoeff_i)
    
    Ynm = sh.synth_cplx_jax(Ycoeff) 
    sig_x = Ynm
    S = set_density(S, sig_x)
    sigmax = S["Sigma"][:,:,0]
    # sigmay = S["Sigma"][:,1]
    # sigmaz = S["Sigma"][:,2]
    sigmay = jnp.zeros(sigmax.shape)
    sigmaz = jnp.zeros(sigmax.shape)

    # Lap op close
    SLsigma = Lap3d_sl_self(S, sh)
    DLsigma = Lap3d_dl_self(S, sh)
    sl_scal = 1.0
    dl_scal = 1.0
    sgn_dl = 1.0 # +1 for exterior problem, -1 for interior problem
    BIOsigma = sl_scal * SLsigma + dl_scal * (sgn_dl * 0.5 * sigmax + DLsigma) # shape mismatch when using S["Sigma"] r.n. since DLsigma doesn't support vector-valued SH yet. 

    # Lap op far
    Strg = build_sphere(center, radius*1.5)
    Strg, shtrg = quadr_sphere(Strg, lmax)
    xtrg = Strg["Xcart"][:2,10,0] # First take arbitrary point on target sphere
    ytrg = Strg["Xcart"][:2,10,1]
    ztrg = Strg["Xcart"][:2,10,2]
    trg_sphere2 = jnp.column_stack([jnp.reshape(xtrg,-1), jnp.reshape(ytrg,-1), jnp.reshape(ztrg,-1)])
    SLsigma = Lap3d_sl(trg_sphere2, S, sh)
    DLsigma = Lap3d_dl(trg_sphere2, S, sh)
    BIOsigma = sl_scal * SLsigma + dl_scal * DLsigma
    jax.debug.print("at target {a} with R = 1.5*R0: {b}", a=trg_sphere2, b=BIOsigma)

    # # 2) Manufactured solutions
    # ptsrc = jnp.zeros((1,3))
    # force = jnp.ones((1,1))
    # # trg_temp = jnp.ones((2,3))
    # # trg_temp = trg_temp.at[1,1].set(1.25)
    # # pot = compute_potential(trg_temp, ptsrc, force)
    # # jax.debug.print("potential at [1,1,1] and [1,1.25,1] from [0,0,0] with strength 1 are {}", pot)

    # trg_sphere = jnp.column_stack([jnp.reshape(x,-1), jnp.reshape(y,-1), jnp.reshape(z,-1)])
    # BC_pot = compute_potential(trg_sphere, ptsrc, force)
    # # TODO: BIO with Lap

    # # TODO: solve using gmres or lstsq

    # # Error at trg_sphere2
    # xtrg = Strg["Xcart"][:,:,0] 
    # ytrg = Strg["Xcart"][:,:,1]
    # ztrg = Strg["Xcart"][:,:,2]
    # trg_sphere2 = jnp.column_stack([jnp.reshape(xtrg,-1), jnp.reshape(ytrg,-1), jnp.reshape(ztrg,-1)])
    # true_pot = compute_potential(trg_sphere2, ptsrc, force)
    # SLsigma = Lap3d_sl(trg_sphere2, S, sh)
    # DLsigma = Lap3d_dl(trg_sphere2, S, sh)
    # BIOsigma = sl_scal * SLsigma + dl_scal * DLsigma
    # jax.debug.print("at targets, first test far eval")



