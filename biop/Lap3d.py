from typing import Dict, Any, Tuple
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import lineax as lx
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Only for testing
from sphere import *
import matplotlib.pyplot as plt
import shtns

SphereDict = Dict[str, Any]

# jax.clear_caches()

def Lap3d_sl_diag(sh: shtns.sht) -> jax.Array:
    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64)
    diag = 1.0 / (2.0 * l_vals + 1.0)
    return diag

def Lap3d_dl_diag(sh: shtns.sht) -> tuple([jax.Array, jax.Array]):
    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64)
    diag_ext = l_vals / (2.0 * l_vals + 1.0) 
    diag_int = - (l_vals + 1) / (2.0 * l_vals + 1.0)
    return diag_ext, diag_int

# @partial(jax.jit, static_argnames=["sh"])
def Lap3d_sl_self(S: SphereDict, sh: shtns.sht) -> jax.Array:
    if S["lmax"] != sh.lmax:
        print("S lmax does not match sht's lmax, reform sht.")
        sh = shtns.sht(S["lmax"], S["lmax"])

    Sigma = S["Sigma"][:,:,0] # scalar operator, so only take Sigma_x
    qlm_sigma = sh.analys_cplx_jax(Sigma)

    diag = Lap3d_sl_diag(sh)
    qlm_SL_sigma = qlm_sigma * diag
    SL_sigma = sh.synth_cplx_jax(qlm_SL_sigma)

    return SL_sigma

# @partial(jax.jit, static_argnames=["sh"])
def Lap3d_dl_self(S: SphereDict, sh: shtns.sht) -> jax.Array:
    if S["lmax"] != sh.lmax:
        print("S lmax does not match sht's lmax, reform sht.")
        sh = shtns.sht(S["lmax"])

    Sigma = S["Sigma"][:,:,0] # scalar operator, so only take Sigma_x
    qlm_sigma = sh.analys_cplx_jax(Sigma)

    [diag_ext, diag_int] = Lap3d_dl_diag(sh)
    diag = 0.5*(diag_ext + diag_int) # on-surface DL is average of the two spectra
    qlm_DL_sigma = qlm_sigma * diag
    DL_sigma = sh.synth_cplx_jax(qlm_DL_sigma)

    return DL_sigma

# @partial(jax.jit, static_argnames=["sh"])
# TODO: consider how much of solid harmonics to take out as separate function in spectral space.
def Lap3d_sl(trg: jax.Array, S: SphereDict, sh: shtns.sht) -> jax.Array:
    if S["lmax"] != sh.lmax:
        print("S lmax does not match sht's lmax, reform sht.")
        sh = shtns.sht(S["lmax"], S["lmax"])

    Sigma = S["Sigma"][:,:,0] # scalar operator, so only take Sigma_x
    qlm_sigma = sh.analys_cplx_jax(Sigma)

    Ntrg = trg.shape[0]
    trg_dx = trg[:,0] - S["Xc"][0]
    trg_dy = trg[:,1] - S["Xc"][1]
    trg_dz = trg[:,2] - S["Xc"][2]
    trg_dr = jnp.sqrt(trg_dx*trg_dx + trg_dy*trg_dy + trg_dz*trg_dz)
    trg_phi = jnp.atan(trg_dy / trg_dx)
    trg_theta = jnp.acos(trg_dz / trg_dr)
    trg_dr = trg_dr[:,None] # Ntrg x 1

    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64) # shape ((p+1)^2, )
    diag = Lap3d_sl_diag(sh)
    rpowers_ext = trg_dr ** (-l_vals-1) # Ntrg x Nlm
    rpowers_int = trg_dr ** (l_vals) # Ntrg x Nlm
    qlm_SL_sigma_ext = rpowers_ext * qlm_sigma * diag
    qlm_SL_sigma_int = rpowers_int * qlm_sigma * diag

    qlm_SL_sigma = jnp.where(trg_dr > S['r'], qlm_SL_sigma_ext, qlm_SL_sigma_int)

    qlm_SL_sigma = np.array(qlm_SL_sigma, dtype=np.complex128)
    trg_phi = np.array(trg_phi)
    trg_costheta = np.cos(trg_theta)

    # TODO: use jax.vmap over all targets
    SL_sigma = jnp.zeros((Ntrg,), dtype = jnp.complex128)
    for trg_i in range(trg_theta.shape[0]):
        val = sh.SH_to_point_cplx(qlm_SL_sigma[trg_i,:], trg_costheta[trg_i], trg_phi[trg_i])
        jax.debug.print("val at cost = {a}, phi={b} is {c}", a=trg_costheta[trg_i], b=trg_phi[trg_i], c=val)
        SL_sigma = SL_sigma.at[trg_i].set(val)

    return SL_sigma

def Lap3d_dl(trg: jax.Array, S: SphereDict, sh: shtns.sht) -> jax.Array:
    if S["lmax"] != sh.lmax:
        print("S lmax does not match sht's lmax, reform sht.")
        sh = shtns.sht(S["lmax"])

    Sigma = S["Sigma"][:,:,0] # scalar operator, so only take Sigma_x
    qlm_sigma = sh.analys_cplx_jax(Sigma)

    Ntrg = trg.shape[0]
    trg_dx = trg[:,0] - S["Xc"][0]
    trg_dy = trg[:,1] - S["Xc"][1]
    trg_dz = trg[:,2] - S["Xc"][2]
    trg_dr = jnp.sqrt(trg_dx*trg_dx + trg_dy*trg_dy + trg_dz*trg_dz)
    trg_phi = jnp.atan(trg_dy / trg_dz)
    trg_theta = jnp.acos(trg_dz / trg_dr)
    trg_dr = trg_dr[:,None] # Ntrg x 1

    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64)
    [diag_ext, diag_int] = Lap3d_dl_diag(sh)
    rpowers_ext = trg_dr ** (-l_vals-1) # Ntrg x Nlm
    rpowers_int = trg_dr ** (l_vals) # Ntrg x Nlm
    qlm_DL_sigma_ext = rpowers_ext * qlm_sigma * diag_ext
    qlm_DL_sigma_int = rpowers_int * qlm_sigma * diag_int

    qlm_DL_sigma = jnp.where(trg_dr > S['r'], qlm_DL_sigma_ext, qlm_DL_sigma_int)
    
    qlm_DL_sigma = np.array(qlm_DL_sigma, dtype=np.complex128)
    trg_phi = np.array(trg_phi)
    trg_costheta = np.cos(trg_theta)

    # TODO: use jax.vmap over all targets
    DL_sigma = jnp.zeros((Ntrg,), dtype = jnp.complex128)
    for trg_i in range(trg_theta.shape[0]):
        DL_sigma = DL_sigma.at[trg_i].set(sh.SH_to_point_cplx(qlm_DL_sigma[trg_i,:], trg_costheta[trg_i], trg_phi[trg_i]))

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

# TODO: decompose the LapSL and focus on the diag-multiply part for BIO solve.
# @partial(jax.jit, static_argnames=["sh"])
# def bio_self_apply(S: SphereDict, sh: shtns.sht, density: jax.Array, sl_scal: float, dl_scal: float, sgn: float): # +1 for exterior problem, -1 for interior problem
#     set_density(S, density)
    
#     SLsigma = Lap3d_sl_self(S, sh)
#     DLsigma = Lap3d_dl_self(S, sh)

#     BIOsigma = sl_scal * SLsigma + dl_scal * (sgn * 0.5 * sigmax + DLsigma) # shape mismatch when using S["Sigma"] r.n. since DLsigma doesn't support vector-valued SH yet. 
    
#     return BIOsigma

# Form K = dl_scal*(sgn*0.5*I + DL) + sl_scal*SL
@partial(jax.jit, static_argnames=["sh"])
def bio_diag_mat(sh: shtns.sht, sl_scal: float, dl_scal: float, sgn: float) -> jax.Array:
    sl_diag = Lap3d_sl_diag(sh)
    [dl_diag_ext, dl_diag_int] = Lap3d_dl_diag(sh)
    dl_diag = 0.5*(dl_diag_ext + dl_diag_int)
    SL = jnp.diag(sl_diag)
    DL = jnp.diag(dl_diag)
    K = dl_scal * (sgn * 0.5 * jnp.eye(dl_diag.shape[0]) + DL) + sl_scal * SL
    return K


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

    # # Simple density in x
    # seed = 1701
    # key = jax.random.PRNGKey(seed)
    # key, subkey = jax.random.split(key)
    # Ycoeff_r = jax.random.uniform(subkey, shape=(sh.nlm_cplx,))
    # key, subkey = jax.random.split(key)
    # Ycoeff_i = jax.random.uniform(subkey, shape=(sh.nlm_cplx,)) 
    # Ycoeff = jax.lax.complex(Ycoeff_r, Ycoeff_i)
    Ycoeff = jnp.zeros((sh.nlm_cplx,),dtype = jnp.complex128)
    Ycoeff = Ycoeff.at[2].set(1.0) # coeff(Y10) = 1, all others 0
    
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
    # DEBUG: Analytical SL[Y_1^0] spectra compared to sl and dl evals -- checked for m=0, l=1
    # jax.debug.print("SL of Y10: {a}", a=SLsigma)
    # jax.debug.print("DL of Y10: {b}", b=DLsigma)

    # Lap op far
    ext = True
    if ext:
        Rtrg = radius * 1.5
        sgn = 1.0 # exterior problem, sgn = +1
    else:
        Rtrg = radius * 0.5
        sgn = -1.0

    Strg = build_sphere(center, Rtrg)
    Strg, shtrg = quadr_sphere(Strg, lmax)
    # # Check TWO TARGETS, same r, no vmap yet -- checked for m=0, l=1
    # xtrg = Strg["Xcart"][2,10:12,0] 
    # ytrg = Strg["Xcart"][2,10:12,1]
    # ztrg = Strg["Xcart"][2,10:12,2]
    # trg_sphere2 = jnp.column_stack([jnp.reshape(xtrg,-1), jnp.reshape(ytrg,-1), jnp.reshape(ztrg,-1)])
    # SLsigma = Lap3d_sl(trg_sphere2, S, sh)
    # DLsigma = Lap3d_dl(trg_sphere2, S, sh)
    # jax.debug.print("at target {a}, R = {b}, SL = {c}, DL = {d}", a=trg_sphere2, b=radius*0.5, c=SLsigma, d = DLsigma) 

    # 2) Manufactured solutions
    ptsrc = jnp.zeros((1,3))
    force = jnp.ones((1,1))

    trg_sphere = jnp.column_stack([jnp.reshape(x,-1), jnp.reshape(y,-1), jnp.reshape(z,-1)])
    BC_pot = compute_potential(trg_sphere, ptsrc, force)
    # Attempt 2: direct solve
    K = bio_diag_mat(sh, 1.0, 1.0, sgn) #-- checked for individual SL and DL diagonals..
    jax.debug.print("K mat: {}", K)


    # # BIO and gmres operator; solve
    # # TODO: consider SLop in basis only
    # # TODO: solve by just multiply inverse diagonal?
    # LapK_apply = partial(
    #     bio_self_apply,
    #     S=S,
    #     sh=sh,
    #     sl_scal=1.0, 
    #     dl_scal=1.0, 
    #     sgn=sgn
    # )
    # gmres_func = lx.FunctionLinearOperator(
    #     LapK_apply, jax.eval_shape(lambda: sigmax)
    # )
    # solver = lx.GMRES(rtol=1e-10, atol=1e-12, max_steps=200)
    # solution = lx.linear_solve(
    #     operator,
    #     BC_pot,
    #     solver=solver,
    #     options={"y0": jnp.zeros_like(sigmax)},
    # )
    # sigmax_sol = solution.value

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



