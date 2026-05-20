"""
Laplace Operator Class:
    SL and DL operators on a sphere, using spectra and solid harmonics
    TODO:
        SH to point evaluation function needs to be jax-enabled, or vectorized
        Allow on-surface evaluation in bio_offsurf_apply()
        onsurf_diag_solve() l=0 currently set to BC values. Throw exception instead?
        solid harmonics r should be scaled s.t. src sphere has r = 1
"""

from typing import Dict, Any, Tuple
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import lineax as lx
import shtns
import shtns_jax

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sphere import *

jax.config.update("jax_enable_x64", True)  # support float64

SphereDict = Dict[str, Any]

def Lap3d_sl_diag(sh: shtns_jax.sht) -> jax.Array:
    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64)
    diag = 1.0 / (2.0 * l_vals + 1.0)
    return diag

def Lap3d_dl_diag(sh: shtns_jax.sht) -> tuple([jax.Array, jax.Array]):
    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64)
    diag_ext = l_vals / (2.0 * l_vals + 1.0) 
    diag_int = - (l_vals + 1) / (2.0 * l_vals + 1.0)
    return diag_ext, diag_int

def Lap3d_sl(trg: jax.Array, S: SphereDict, sh: shtns_jax.sht) -> jax.Array:
    """
    Off-surface evaluation at points <trg>: Ntrg x 3
        From source <S> with density <S["Sigma"]>
        source uses SHT object <sh>
    """

    assert trg.shape[1] == 3

    if S["lmax"] != sh.lmax:
        print("S lmax does not match sht's lmax, reform sht.")
        sh = shtns_jax.sht(S["lmax"], S["lmax"])

    Sigma = S["Sigma"][:,:,0] 
    qlm_sigma = sh.analys_cplx_jax(Sigma)

    Ntrg = trg.shape[0]
    trg_dx = trg[:,0] - S["Xc"][0]
    trg_dy = trg[:,1] - S["Xc"][1]
    trg_dz = trg[:,2] - S["Xc"][2]
    trg_dr = jnp.sqrt(trg_dx*trg_dx + trg_dy*trg_dy + trg_dz*trg_dz)
    trg_phi = jnp.atan2(trg_dy, trg_dx)
    trg_theta = jnp.acos(trg_dz / trg_dr)
    trg_dr = trg_dr[:,None] 

    # Solid harmonics
    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64) 
    diag = Lap3d_sl_diag(sh)
    rpowers_ext = trg_dr ** (-l_vals-1) 
    rpowers_int = trg_dr ** (l_vals) 
    qlm_SL_sigma_ext = rpowers_ext * qlm_sigma * diag
    qlm_SL_sigma_int = rpowers_int * qlm_sigma * diag 
    qlm_SL_sigma = jnp.where(trg_dr > S['r'], qlm_SL_sigma_ext, qlm_SL_sigma_int)

    # Evaluation at target
    qlm_SL_sigma = np.array(qlm_SL_sigma, dtype=np.complex128)
    trg_phi = np.array(trg_phi)
    trg_costheta = np.cos(trg_theta)
    SL_sigma = jnp.zeros((Ntrg,1), dtype = jnp.complex128)
    for trg_i in range(Ntrg):
        val = sh.SH_to_point_cplx(qlm_SL_sigma[trg_i,:], trg_costheta[trg_i], trg_phi[trg_i])
        SL_sigma = SL_sigma.at[trg_i,0].set(val)

    return SL_sigma

def Lap3d_dl(trg: jax.Array, S: SphereDict, sh: shtns_jax.sht) -> jax.Array:
    """
    Off-surface evaluation at points <trg>: Ntrg x 3
        From source <S> with density <S["Sigma"]>
        source uses SHT object <sh>
    """

    assert trg.shape[1] == 3

    if S["lmax"] != sh.lmax:
        print("S lmax does not match sht's lmax, reform sht.")
        sh = shtns_jax.sht(S["lmax"])

    Sigma = S["Sigma"][:,:,0] # scalar operator, so only take Sigma_x
    qlm_sigma = sh.analys_cplx_jax(Sigma)

    Ntrg = trg.shape[0]
    trg_dx = trg[:,0] - S["Xc"][0]
    trg_dy = trg[:,1] - S["Xc"][1]
    trg_dz = trg[:,2] - S["Xc"][2]
    trg_dr = jnp.sqrt(trg_dx*trg_dx + trg_dy*trg_dy + trg_dz*trg_dz)
    trg_phi = jnp.atan2(trg_dy, trg_dx)
    trg_theta = jnp.acos(trg_dz / trg_dr)
    trg_dr = trg_dr[:,None] 

    # Solid harmonics
    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64)
    [diag_ext, diag_int] = Lap3d_dl_diag(sh)
    rpowers_ext = trg_dr ** (-l_vals-1) 
    rpowers_int = trg_dr ** (l_vals) 
    qlm_DL_sigma_ext = rpowers_ext * qlm_sigma * diag_ext
    qlm_DL_sigma_int = rpowers_int * qlm_sigma * diag_int
    qlm_DL_sigma = jnp.where(trg_dr > S['r'], qlm_DL_sigma_ext, qlm_DL_sigma_int)
    
    # Evaluation at target
    qlm_DL_sigma = np.array(qlm_DL_sigma, dtype=np.complex128)
    trg_phi = np.array(trg_phi)
    trg_costheta = np.cos(trg_theta)
    DL_sigma = jnp.zeros((Ntrg,1), dtype = jnp.complex128)
    for trg_i in range(trg_theta.shape[0]):
        DL_sigma = DL_sigma.at[trg_i,0].set(sh.SH_to_point_cplx(qlm_DL_sigma[trg_i,:], trg_costheta[trg_i], trg_phi[trg_i]))

    return DL_sigma

def Lap3d_sl_r_1sph(Strg: SphereDict, shtrg: shtns_jax.sht, S: SphereDict, sh: shtns_jax.sht) -> jax.Array:
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

    Sigma = S["Sigma"][:,:,0] 
    qlm_sigma = sh.analys_cplx_jax(Sigma)

    trg_dr = Strg["r"]
    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64) 
    diag = Lap3d_sl_diag(sh)
    rpowers_ext = trg_dr ** (-l_vals-1) 
    rpowers_int = trg_dr ** (l_vals) 
    qlm_SL_sigma_ext = rpowers_ext * qlm_sigma * diag
    qlm_SL_sigma_int = rpowers_int * qlm_sigma * diag
    qlm_SL_sigma = qlm_SL_sigma_ext if trg_dr > S['r'] else qlm_SL_sigma_int

    # Interpolate to new grid by padding or truncating coefficients to Strg['lmax']
    nlm_src = sh.nlm_cplx
    nlm_trg = shtrg.nlm_cplx
    if nlm_trg > nlm_src:
        qlm_SL_sigma = jnp.pad(qlm_SL_sigma, (0, nlm_trg - nlm_src), constant_values=0)
    elif nlm_trg < nlm_src:
        qlm_SL_sigma = qlm_SL_sigma[:nlm_trg]

    theta_trg = Strg["Xsph"][:,:,0]
    phi_trg = Strg["Xsph"][:,:,1]
    SL_sigma = shtrg.synth_cplx_jax(qlm_SL_sigma)

    return SL_sigma

def Lap3d_dl_r_1sph(Strg: SphereDict, shtrg: shtns_jax.sht, S: SphereDict, sh: shtns_jax.sht) -> jax.Array:
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

    Sigma = S["Sigma"][:,:,0] 
    qlm_sigma = sh.analys_cplx_jax(Sigma)

    trg_dr = Strg["r"]
    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64)
    [diag_ext, diag_int] = Lap3d_dl_diag(sh)
    rpowers_ext = trg_dr ** (-l_vals-1) 
    rpowers_int = trg_dr ** (l_vals) 
    qlm_DL_sigma_ext = rpowers_ext * qlm_sigma * diag_ext
    qlm_DL_sigma_int = rpowers_int * qlm_sigma * diag_int
    qlm_DL_sigma = qlm_DL_sigma_ext if trg_dr > S['r'] else qlm_DL_sigma_int

    # Interpolate to new grid by padding or truncating coefficients to Strg['lmax']
    nlm_src = sh.nlm_cplx
    nlm_trg = shtrg.nlm_cplx
    if nlm_trg > nlm_src:
        qlm_DL_sigma = jnp.pad(qlm_DL_sigma, (0, nlm_trg - nlm_src), constant_values=0)
    elif nlm_trg < nlm_src:
        qlm_DL_sigma = qlm_DL_sigma[:nlm_trg]

    theta_trg = Strg["Xsph"][:,:,0]
    phi_trg = Strg["Xsph"][:,:,1]
    DL_sigma = shtrg.synth_cplx_jax(qlm_DL_sigma)

    return DL_sigma

def compute_potential(trg: jax.Array, src: jax.Array, force: jax.Array) -> jax.Array:
    """
    Compute the Laplace potential 
        from source positioned at <src>: Nsrc x 3 
            with charge <force>: Nsrc x 1
        at target positioned at <trg>: Ntrg x 3
    """

    assert trg.shape[1] == 3 and src.shape[1] == 3
    assert force.shape[0] == src.shape[0]
    if len(force.shape) > 1:
        assert force.shape[1] == 1
    else:
        force = force[:,None] 

    srcx = src[:,0][None,:] 
    srcy = src[:,1][None,:]
    srcz = src[:,2][None,:]
    dx = trg[:,0][:,None] - srcx 
    dy = trg[:,1][:,None] - srcy
    dz = trg[:,2][:,None] - srcz
    dr = jnp.sqrt(dx*dx + dy*dy + dz*dz)
    G = jnp.matmul(1/4./jnp.pi / dr, force) 
    G = G + 0j

    return G

@partial(jax.jit, static_argnames=["sh"])
def bio_diag_apply(qlm_sigma: jax.Array, sh: shtns_jax.sht, sl_scal: float, dl_scal: float) -> jax.Array:
    """
    Apply the combined DL potential operator K = sl_scal * SL + dl_scal * DL
        to the density with coefficients <qlm_sigma> in SH basis 
    Returns the resulting function coefficients <qlm_KL_sigma> in SH basis,
        where DL is evaluated in the P.V. sense.
    """

    sl_diag = Lap3d_sl_diag(sh)
    [dl_diag_ext, dl_diag_int] = Lap3d_dl_diag(sh)
    dl_diag = 0.5*(dl_diag_ext + dl_diag_int)
    qlm_SL_sigma = sl_diag * qlm_sigma
    qlm_DL_sigma = dl_diag * qlm_sigma
    qlm_KL_sigma = dl_scal * qlm_DL_sigma + sl_scal * qlm_SL_sigma
    return qlm_KL_sigma

@partial(jax.jit, static_argnames=["sh"])
def bio_onsurf_apply(sigma: jax.Array, sh: shtns_jax.sht, sl_scal: float, dl_scal: float, sgn: float) -> jax.Array:
    """
    Apply the combined DL potential operator K = sl_scal * SL + dl_scal * DL
        to the density <sigma> defined on the <sh> grid
        taking into account the DL jump condition with sign <sgn>.
    """

    qlm_sigma = sh.analys_cplx_jax(sigma)
    qlm_KL_sigma = bio_diag_apply(qlm_sigma, sh, sl_scal, dl_scal)
    KL_sigma = sh.synth_cplx_jax(qlm_KL_sigma)
    return 0.5 * dl_scal * sgn * sigma + KL_sigma

@partial(jax.jit, static_argnames=["sh"])
def bio_onsurf_direct_solve(bc_pot: jax.Array, sh: shtns_jax.sht, sl_scal: float, dl_scal: float, sgn: float) -> jax.Array:
    """
    Directly solves the BIO equation in the spectral domain.
    Equation: [0.5 * dl_scal * sgn * I + KL] sigma = bc_pot
    
    Note: The l=0 mode is in the null space and is set to the BC value.
    """

    qlm_bc = sh.analys_cplx_jax(bc_pot)
    sl_diag = Lap3d_sl_diag(sh)
    [dl_diag_ext, dl_diag_int] = Lap3d_dl_diag(sh)
    dl_diag = 0.5 * (dl_diag_ext + dl_diag_int)
    operator_diag = (0.5 * dl_scal * sgn) + (dl_scal * dl_diag) + (sl_scal * sl_diag)
    
    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64)
    # Create a safe diagonal by replacing near-zero entries with 1.0 (dummy) where we won't divide
    safe_diag = jnp.where(jnp.abs(operator_diag) > 1e-14, operator_diag, 1.0)
    qlm_sigma = qlm_bc / safe_diag
    # For l=0 modes (index 0), set to BC value 
    qlm_sigma = jnp.where(l_vals == 0.0, qlm_bc, qlm_sigma)
    
    sigma = sh.synth_cplx_jax(qlm_sigma)
    
    return sigma

def bio_offsurf_apply(trg: jax.Array, S: SphereDict, sh: shtns_jax.sht, sl_scal: float, dl_scal: float) -> jax.Array:
    """
    Evaluate the KL formulation of <S> with density <S["Sigma"]> at target <trg>
    """

    SLsigma = Lap3d_sl(trg, S, sh)
    DLsigma = Lap3d_dl(trg, S, sh)
    Ksigma = sl_scal * SLsigma + dl_scal * DLsigma 
    return Ksigma

def bio_offsurf_apply_1sph(Strg: SphereDict, shtrg: shtns_jax.sht, S: SphereDict, sh: shtns_jax.sht, sl_scal: float, dl_scal: float) -> jax.Array:
    """
    Evaluate the KL formulation of <S> with density <S["Sigma"]> at a spherical grid of targets on <Strg>
    """

    SLsigma = Lap3d_sl_r_1sph(Strg, shtrg, S, sh)
    DLsigma = Lap3d_dl_r_1sph(Strg, shtrg, S, sh)
    Ksigma = sl_scal * SLsigma + dl_scal * DLsigma 
    return Ksigma

if __name__ == "__main__":
    """
    Test Laplace operators on spheres using manufactured solutions.
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
    Rtrg = radius * 1.00025
    sgn = 1.0 # exterior problem, sgn = +1
    Strg = build_sphere(center, Rtrg)
    Strg, shtrg = quadr_sphere(Strg, lmax)

    # Manufactured solutions test
    ptsrc = jnp.array([[0.1,0.3,0.15]]) # shifted source to avoid constant potential on all of S
    force = jnp.ones((1,1)) 

    x = S["Xcart"][:,:,0]
    y = S["Xcart"][:,:,1]
    z = S["Xcart"][:,:,2]
    trg_sphere = jnp.column_stack([jnp.reshape(x,-1), jnp.reshape(y,-1), jnp.reshape(z,-1)])
    BC_pot = compute_potential(trg_sphere, ptsrc, force)
    BC_pot = jnp.reshape(BC_pot, x.shape)

    # GMRES solve
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
    solution = lx.linear_solve(
        gmres_func,
        BC_pot,
        solver=solver,
        options={"y0": jnp.zeros(x.shape, dtype=jnp.complex128)},
    )
    sig_gmres = solution.value 
    stats = solution.stats
    # Manually check residual
    bc_check = bio_onsurf_apply(sig_gmres, sh, sl_scal, dl_scal, sgn)
    resid_gmres = jnp.linalg.norm(bc_check - BC_pot)
    jax.debug.print("Residual of GMRES solve = {a}, number of iterations = {b}", a=resid_gmres, b=stats["num_steps"])

    # DIRECT solve
    sig_direct = bio_onsurf_direct_solve(
        bc_pot=BC_pot,
        sh=sh,
        sl_scal=sl_scal,
        dl_scal=dl_scal,
        sgn=sgn
    )
    bc_check_direct = bio_onsurf_apply(sig_direct, sh, sl_scal, dl_scal, sgn)
    resid_direct = jnp.linalg.norm(bc_check_direct - BC_pot)
    jax.debug.print("Residual of DIRECT solve: {a}", a=resid_direct)

    # Accuracy
    xtrg = Strg["Xcart"][:,:,0] 
    ytrg = Strg["Xcart"][:,:,1]
    ztrg = Strg["Xcart"][:,:,2]
    trg_sphere2 = jnp.column_stack([jnp.reshape(xtrg,-1), jnp.reshape(ytrg,-1), jnp.reshape(ztrg,-1)])
    true_pot = compute_potential(trg_sphere2, ptsrc, force)
    true_pot = jnp.real(true_pot)

    S = set_density(S, sig_gmres)
    Ksig_gmres = bio_offsurf_apply_1sph(Strg, shtrg, S, sh, sl_scal, dl_scal)
    Ksig_gmres = jnp.real(jnp.reshape(Ksig_gmres,(-1,1)))

    S = set_density(S, sig_direct)
    Ksig_direct = bio_offsurf_apply_1sph(Strg, shtrg, S, sh, sl_scal, dl_scal)
    Ksig_direct = jnp.real(jnp.reshape(Ksig_direct,(-1,1)))

    diff_gmres = jnp.max(true_pot - Ksig_gmres) / jnp.max(true_pot)
    diff_direct = jnp.max(true_pot - Ksig_direct) / jnp.max(true_pot)
    jax.debug.print("Max relative error of order {lmax} solver at target radius {Rtrg} for GMRES solver is {d1}, for direct solver is {d2}", lmax=lmax, Rtrg=Rtrg, d1=diff_gmres, d2=diff_direct)


    # Target -- interior
    Rtrg = radius * 0.5
    sgn = -1.0
    Strg = build_sphere(center, Rtrg)
    Strg, shtrg = quadr_sphere(Strg, lmax)

    ptsrc = jnp.array([[1.5,2,1.5],[-1.5,-2,-1.5]])
    force = jnp.array([[1],[-1]])

    x = S["Xcart"][:,:,0]
    y = S["Xcart"][:,:,1]
    z = S["Xcart"][:,:,2]
    trg_sphere = jnp.column_stack([jnp.reshape(x,-1), jnp.reshape(y,-1), jnp.reshape(z,-1)])
    BC_pot = compute_potential(trg_sphere, ptsrc, force)
    BC_pot = jnp.reshape(BC_pot, x.shape)

    # GMRES solve
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
    solution = lx.linear_solve(
        gmres_func,
        BC_pot,
        solver=solver,
        options={"y0": jnp.zeros(x.shape, dtype=jnp.complex128)},
    )
    sig_gmres = solution.value 
    stats = solution.stats
    # Manually check residual
    bc_check = bio_onsurf_apply(sig_gmres, sh, sl_scal, dl_scal, sgn)
    resid_gmres = jnp.linalg.norm(bc_check - BC_pot)
    jax.debug.print("Residual of GMRES solve = {a}, number of iterations = {b}", a=resid_gmres, b=stats["num_steps"])

    # DIRECT solve
    sig_direct = bio_onsurf_direct_solve(
        bc_pot=BC_pot,
        sh=sh,
        sl_scal=sl_scal,
        dl_scal=dl_scal,
        sgn=sgn
    )
    bc_check_direct = bio_onsurf_apply(sig_direct, sh, sl_scal, dl_scal, sgn)
    resid_direct = jnp.linalg.norm(bc_check_direct - BC_pot)
    jax.debug.print("Residual of DIRECT solve: {a}", a=resid_direct)

    # Accuracy
    xtrg = Strg["Xcart"][:,:,0] 
    ytrg = Strg["Xcart"][:,:,1]
    ztrg = Strg["Xcart"][:,:,2]
    trg_sphere2 = jnp.column_stack([jnp.reshape(xtrg,-1), jnp.reshape(ytrg,-1), jnp.reshape(ztrg,-1)])
    true_pot = compute_potential(trg_sphere2, ptsrc, force)
    true_pot = jnp.real(true_pot)

    S = set_density(S, sig_gmres)
    Ksig_gmres = bio_offsurf_apply_1sph(Strg, shtrg, S, sh, sl_scal, dl_scal)
    Ksig_gmres = jnp.real(jnp.reshape(Ksig_gmres,(-1,1)))

    S = set_density(S, sig_direct)
    Ksig_direct = bio_offsurf_apply_1sph(Strg, shtrg, S, sh, sl_scal, dl_scal)
    Ksig_direct = jnp.real(jnp.reshape(Ksig_direct,(-1,1)))

    diff_gmres = jnp.max(true_pot - Ksig_gmres) / jnp.max(true_pot)
    diff_direct = jnp.max(true_pot - Ksig_direct) / jnp.max(true_pot)
    jax.debug.print("Max relative error of order {lmax} solver at target radius {Rtrg} for GMRES solver is {d1}, for direct solver is {d2}", lmax=lmax, Rtrg=Rtrg, d1=diff_gmres, d2=diff_direct)



