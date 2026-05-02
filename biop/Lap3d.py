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
# TODO: consider how much of solid harmonics to take out as separate function in spectral space.
def Lap3d_sl(trg: jax.Array, S: SphereDict, sh: shtns.sht) -> jax.Array:
    if S["lmax"] != sh.lmax:
        print("S lmax does not match sht's lmax, reform sht.")
        sh = shtns.sht(S["lmax"], S["lmax"])

    Sigma = S["Sigma"][:,:,0] # scalar operator, so only take Sigma_x
    qlm_sigma = sh.analys_cplx_jax(Sigma)

    assert trg.shape[1] == 3

    Ntrg = trg.shape[0]
    trg_dx = trg[:,0] - S["Xc"][0]
    trg_dy = trg[:,1] - S["Xc"][1]
    trg_dz = trg[:,2] - S["Xc"][2]
    trg_dr = jnp.sqrt(trg_dx*trg_dx + trg_dy*trg_dy + trg_dz*trg_dz)
    trg_phi = jnp.atan2(trg_dy, trg_dx)
    trg_theta = jnp.acos(trg_dz / trg_dr)
    trg_dr = trg_dr[:,None] # Ntrg x 1    
    # trg_costheta = jnp.cos(trg_theta)
    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64) # shape ((p+1)^2, )
    diag = Lap3d_sl_diag(sh)
    rpowers_ext = trg_dr ** (-l_vals-1) # Ntrg x Nlm
    rpowers_int = trg_dr ** (l_vals) # Ntrg x Nlm
    qlm_SL_sigma_ext = rpowers_ext * qlm_sigma * diag
    qlm_SL_sigma_int = rpowers_int * qlm_sigma * diag # Ntrg x Nlm

    qlm_SL_sigma = jnp.where(trg_dr > S['r'], qlm_SL_sigma_ext, qlm_SL_sigma_int)

    # OLD CODE (commented out):
    qlm_SL_sigma = np.array(qlm_SL_sigma, dtype=np.complex128)
    trg_phi = np.array(trg_phi)
    trg_costheta = np.cos(trg_theta)
    SL_sigma = jnp.zeros((Ntrg,1), dtype = jnp.complex128)
    for trg_i in range(Ntrg):
        val = sh.SH_to_point_cplx(qlm_SL_sigma[trg_i,:], trg_costheta[trg_i], trg_phi[trg_i])
        SL_sigma = SL_sigma.at[trg_i,0].set(val)

    # TODO: cannot use vmap on naive python function since vmap passes in tracers not values.
    # SL_sigma = jax.vmap(lambda qlm, ct, p: sh.SH_to_point_cplx(qlm, ct.astype(float), p.astype(float)))(qlm_SL_sigma, trg_costheta, trg_phi)[:,None]

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
    trg_phi = jnp.atan2(trg_dy, trg_dx)
    trg_theta = jnp.acos(trg_dz / trg_dr)
    trg_dr = trg_dr[:,None] # Ntrg x 1    
    # trg_costheta = jnp.cos(trg_theta)
    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64)
    [diag_ext, diag_int] = Lap3d_dl_diag(sh)
    rpowers_ext = trg_dr ** (-l_vals-1) # Ntrg x Nlm
    rpowers_int = trg_dr ** (l_vals) # Ntrg x Nlm
    qlm_DL_sigma_ext = rpowers_ext * qlm_sigma * diag_ext
    qlm_DL_sigma_int = rpowers_int * qlm_sigma * diag_int

    qlm_DL_sigma = jnp.where(trg_dr > S['r'], qlm_DL_sigma_ext, qlm_DL_sigma_int)
    
    # OLD CODE (commented out):
    qlm_DL_sigma = np.array(qlm_DL_sigma, dtype=np.complex128)
    trg_phi = np.array(trg_phi)
    trg_costheta = np.cos(trg_theta)
    DL_sigma = jnp.zeros((Ntrg,1), dtype = jnp.complex128)
    for trg_i in range(trg_theta.shape[0]):
        DL_sigma = DL_sigma.at[trg_i,0].set(sh.SH_to_point_cplx(qlm_DL_sigma[trg_i,:], trg_costheta[trg_i], trg_phi[trg_i]))

    # TODO: see sl comment
    # DL_sigma = jax.vmap(lambda qlm, ct, p: sh.SH_to_point_cplx(qlm, ct, p))(qlm_DL_sigma, trg_costheta, trg_phi)[:, None]

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

    srcx = src[:,0][None,:] # reshape source x,y,z into 1 x Nsrc 
    srcy = src[:,1][None,:]
    srcz = src[:,2][None,:]
    dx = trg[:,0][:,None] - srcx # size = Ntrg x Nsrc
    dy = trg[:,1][:,None] - srcy
    dz = trg[:,2][:,None] - srcz
    dr = jnp.sqrt(dx*dx + dy*dy + dz*dz)
    G = jnp.matmul(1/4./jnp.pi / dr, force) # 1/dr has size Ntrg x Nsrc, force has size Nsrc x 1, G has size Ntrg x 1
    G = G + 0j

    return G

# Apply SL and DL self-eval spectra to sigma
@partial(jax.jit, static_argnames=["sh"])
def bio_diag_apply(qlm_sigma: jax.Array, sh: shtns.sht, sl_scal: float, dl_scal: float) -> jax.Array:
    sl_diag = Lap3d_sl_diag(sh)
    [dl_diag_ext, dl_diag_int] = Lap3d_dl_diag(sh)
    dl_diag = 0.5*(dl_diag_ext + dl_diag_int)
    SL = sl_diag * qlm_sigma
    DL = dl_diag * qlm_sigma
    KL = dl_scal * DL + sl_scal * SL
    return KL

# returns K[sigma] = [dl_scal * (pm I/2 + DL) + sl_scal * SL] [sigma]
@partial(jax.jit, static_argnames=["sh"])
def bio_onsurf_apply(sigma: jax.Array, sh: shtns.sht, sl_scal: float, dl_scal: float, sgn: float) -> jax.Array:
    qlm_sigma = sh.analys_cplx_jax(sigma)
    qlm_KL_sigma = bio_diag_apply(qlm_sigma, sh, sl_scal, dl_scal)
    KL_sigma = sh.synth_cplx_jax(qlm_KL_sigma)
    return 0.5 * dl_scal * sgn * sigma + KL_sigma

def bio_offsurf_apply(trg: jax.Array, S: SphereDict, sh: shtns.sht, sl_scal: float, dl_scal: float) -> jax.Array:
    SLsigma = Lap3d_sl(trg, S, sh)
    DLsigma = Lap3d_dl(trg, S, sh)
    Ksigma = sl_scal * SLsigma + dl_scal * DLsigma 
    return Ksigma

if __name__ == "__main__":

    import time

    # Geometry setup
    lmax = 36
    center = jnp.array([0.,0.,0.])
    radius = 1.
    S = build_sphere(center, radius)
    S, sh = quadr_sphere(S, lmax)
    # Lap op far
    ext = False
    if ext:
        Rtrg = radius * 1.00025
        sgn = 1.0 # exterior problem, sgn = +1
    else:
        Rtrg = radius * 0.5
        sgn = -1.0
    Strg = build_sphere(center, Rtrg)
    Strg, shtrg = quadr_sphere(Strg, lmax)
    # BIOp parameter
    sl_scal = 1.0
    dl_scal = 1.0

    # # TEST 1: Evaluation using simple scalar density
    # # # random density
    # # seed = 1701
    # # key = jax.random.PRNGKey(seed)
    # # key, subkey = jax.random.split(key)
    # # Ycoeff_r = jax.random.uniform(subkey, shape=(sh.nlm_cplx,))
    # # key, subkey = jax.random.split(key)
    # # Ycoeff_i = jax.random.uniform(subkey, shape=(sh.nlm_cplx,)) 
    # # Ycoeff = jax.lax.complex(Ycoeff_r, Ycoeff_i)
    # # Y10
    # Ycoeff = jnp.zeros((sh.nlm_cplx,),dtype = jnp.complex128)
    # Ycoeff = Ycoeff.at[2].set(1.0) 
    # Ynm = sh.synth_cplx_jax(Ycoeff) 
    # sig_x = Ynm
    # S = set_density(S, sig_x)
    # sigmax = S["Sigma"][:,:,0]
    # sigmay = jnp.zeros(sigmax.shape)
    # sigmaz = jnp.zeros(sigmax.shape)


    # TEST 2: Manufactured solutions
    if ext:
        ptsrc = jnp.array([[0.1,0.3,0.15]]) # shifted source to avoid constant potential on all of S
        force = jnp.ones((1,1)) 
    else:
        ptsrc = jnp.array([[1.5,2,1.5],[-1.5,-2,-1.5]])
        force = jnp.array([[1],[-1]])

    x = S["Xcart"][:,:,0]
    y = S["Xcart"][:,:,1]
    z = S["Xcart"][:,:,2]
    trg_sphere = jnp.column_stack([jnp.reshape(x,-1), jnp.reshape(y,-1), jnp.reshape(z,-1)])
    BC_pot = compute_potential(trg_sphere, ptsrc, force)
    BC_pot = jnp.reshape(BC_pot, x.shape)

    # BIO and gmres operator; solve
    LapK_apply = partial(
        bio_onsurf_apply,
        sh=sh,
        sl_scal=sl_scal, 
        dl_scal=dl_scal, 
        sgn=sgn
    )
    gmres_func = lx.FunctionLinearOperator(
        LapK_apply, jax.eval_shape(lambda: jnp.zeros(x.shape, dtype=jnp.complex128))
    )
    solver = lx.GMRES(rtol=1e-10, atol=1e-12, max_steps=200)
    time_solver_start = time.time()
    solution = lx.linear_solve(
        gmres_func,
        BC_pot,
        solver=solver,
        options={"y0": jnp.zeros(x.shape, dtype=jnp.complex128)},
    )
    time_solver_end = time.time()

    time_solver_start2 = time.time()
    solution = lx.linear_solve(
        gmres_func,
        BC_pot,
        solver=solver,
        options={"y0": jnp.zeros(x.shape, dtype=jnp.complex128)},
    )
    time_solver_end2 = time.time()
    print(f"Timing results, first solve: {time_solver_end - time_solver_start}, second solve: {time_solver_end2 - time_solver_start2}")

    sig_fromBC = solution.value # This will have the ntheta x nphi grid size
    stats = solution.stats
    # Manually check residual
    bc_check = bio_onsurf_apply(sig_fromBC, sh, sl_scal, dl_scal, sgn)
    resid_gmres = jnp.linalg.norm(bc_check - BC_pot)
    jax.debug.print("Checking residual of solve: {a}, number of iterations needed: {b}", a=resid_gmres, b=stats["num_steps"])


    # Compare with true solution at target sphere
    xtrg = Strg["Xcart"][:,:,0] 
    ytrg = Strg["Xcart"][:,:,1]
    ztrg = Strg["Xcart"][:,:,2]
    trg_sphere2 = jnp.column_stack([jnp.reshape(xtrg,-1), jnp.reshape(ytrg,-1), jnp.reshape(ztrg,-1)])
    S = set_density(S, sig_fromBC)
    time_eval_start = time.time()
    Ksigma = bio_offsurf_apply(trg_sphere2, S, sh, sl_scal, dl_scal) # trg_sphere2 reshaped to be Ntrg x 3, so output is Ntrg x 1.
    Ksigma.block_until_ready()
    time_eval_end = time.time()
    time_eval_start2 = time.time()
    Ksigma = bio_offsurf_apply(trg_sphere2, S, sh, sl_scal, dl_scal) # trg_sphere2 reshaped to be Ntrg x 3, so output is Ntrg x 1.
    Ksigma.block_until_ready()
    time_eval_end2 = time.time()
    print(f"Timing results, first off-surface eval: {time_eval_end - time_eval_start}, second eval: {time_eval_end2 - time_eval_start2}")

    true_pot = compute_potential(trg_sphere2, ptsrc, force) # true_pot also computed as Ntrg x 1
    # For scalar electric potential calculation, only real values
    Ksigma = jnp.real(Ksigma)
    true_pot = jnp.real(true_pot)
    diff = jnp.max(true_pot - Ksigma) / jnp.max(true_pot )
    jax.debug.print("At target sphere Rtrg = {a}, relative error from true potential using lmax = {b} is {c}", a=Rtrg, b=lmax, c=diff)



