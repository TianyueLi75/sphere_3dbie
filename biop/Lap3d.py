"""
Laplace Operator Class:
    SL and DL operators on a sphere, using spectra and solid harmonics
    TODO:
        SH to point evaluation function needs to be jax-enabled, or vectorized
        Allow on-surface evaluation in bio_offsurf_apply()
        onsurf_diag_solve() l=0 currently set to BC values. Throw exception instead?
        solid harmonics r should be scaled s.t. src sphere has r = 1

    NOTE: (Jun 17, 2026)
        added SL traction far eval and on-surface eval, for far and for Neumann BC formulation. No near formula for SL traction yet.
"""

from typing import Dict, Any, Tuple
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import lineax as lx
import shtns
import shtns_jax
import matplotlib.pyplot as plt
import mpld3

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

def Lap3d_dsl_diag(sh: shtns_jax.sht) -> tuple([jax.Array, jax.Array]):
    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64)
    diag_int = l_vals / (2.0 * l_vals + 1.0) 
    diag_ext = - (l_vals + 1) / (2.0 * l_vals + 1.0)
    return diag_ext, diag_int

def Lap3d_sl_far(trg: jax.Array, S:SphereDict, sh: shtns_jax.sht) -> jax.Array:
    """
    Off-surface evaluation of the Lap3d SL potential with S["Sigma"]
        with <trg>: Ntrg x 3
        using the smooth surface quadrature (accurate only for targets well
        away from the surface).
    Returns the potential at the targets: Ntrg x 1.
    """
    assert trg.shape[1] == 3

    # Flatten the (nphi, ntheta, ...) grid arrays to per-source-point lists.
    grid_shape = S["Xcart"].shape[:2]
    ysrc = S["Xcart"].reshape(-1, 3)        # Nsrc x 3
    fsrc = S["Sigma"][:, :, 0].reshape(-1)  # Nsrc
    # Gauss weights (1 x ntheta) broadcast over the (nphi, ntheta) grid, plus
    # the r^2 surface Jacobian for a sphere of radius S["r"].
    wts = jnp.broadcast_to(sh._grid_weights(), grid_shape).reshape(-1) * S["r"] ** 2

    r = trg[:, None, :] - ysrc[None, :, :]  # Ntrg x Nsrc x 3, convention r = trg - src
    invd = 1.0 / jnp.linalg.norm(r, axis=2) # Ntrg x Nsrc
    prefac = 1.0 / 4.0 / jnp.pi
    SL_sigma = prefac * jnp.sum(invd * (wts * fsrc)[None, :], axis=1)
    return SL_sigma[:, None]

def Lap3d_dl_far(trg: jax.Array, S:SphereDict, sh: shtns_jax.sht) -> jax.Array:
    """
    Off-surface evaluation of the Lap3d DL potential with S["Sigma"]
        with <trg>: Ntrg x 3
        using the smooth surface quadrature (accurate only for targets well
        away from the surface).
    Returns the potential at the targets: Ntrg x 1.

    The DL kernel is d/dn_y (1/(4 pi |x-y|)) = (1/4pi) (x-y).n_y / |x-y|^3.
    With r = trg - src = x - y, this is prefac * (r.n) / |r|^3.
    """
    assert trg.shape[1] == 3

    grid_shape = S["Xcart"].shape[:2]
    ysrc = S["Xcart"].reshape(-1, 3)        # Nsrc x 3
    fsrc = S["Sigma"][:, :, 0].reshape(-1)  # Nsrc
    ynsrc = S["Xncart"].reshape(-1, 3)      # Nsrc x 3, unit outward normals
    wts = jnp.broadcast_to(sh._grid_weights(), grid_shape).reshape(-1) * S["r"] ** 2

    r = trg[:, None, :] - ysrc[None, :, :]  # Ntrg x Nsrc x 3, convention r = trg - src
    d = jnp.linalg.norm(r, axis=2)          # Ntrg x Nsrc
    invd3 = 1.0 / (d * d * d)
    rdotn = jnp.sum(r * ynsrc[None, :, :], axis=2)  # Ntrg x Nsrc
    prefac = 1.0 / 4.0 / jnp.pi
    DL_sigma = prefac * jnp.sum(invd3 * rdotn * (wts * fsrc)[None, :], axis=1)
    return DL_sigma[:, None]

def Lap3d_dsl_far(trg: jax.Array, trgN: jax.Array, S:SphereDict, sh: shtns_jax.sht) -> jax.Array:
    """
    Off-surface evaluation of the surface-normal derviatve of the Lap3d SL potential ("T")
        with <trg>: Ntrg x 3
        against target normal <trgN>: Ntrg x 3
        using the smooth surface quadrature (accurate only for targets well
        away from the surface).
    Returns the traction at the targets: Ntrg x 1.

    The SL traction kernel is -(1/4pi) (x-y).n_x / |x-y|^3.
    With r = trg - src = x - y
    """
    assert trg.shape[1] == 3 and trgN.shape[1] == 3

    grid_shape = S["Xcart"].shape[:2]
    ysrc = S["Xcart"].reshape(-1, 3)        # Nsrc x 3
    fsrc = S["Sigma"][:, :, 0].reshape(-1)  # Nsrc
    wts = jnp.broadcast_to(sh._grid_weights(), grid_shape).reshape(-1) * S["r"] ** 2

    r = trg[:, None, :] - ysrc[None, :, :]  # Ntrg x Nsrc x 3, convention r = trg - src
    d = jnp.linalg.norm(r, axis=2)          # Ntrg x Nsrc
    invd3 = 1.0 / (d * d * d)
    rdotn = jnp.sum(r * trgN[None, :, :], axis=2)  # Ntrg x Nsrc
    prefac = - 1.0 / 4.0 / jnp.pi
    T_sigma = prefac * jnp.sum(invd3 * rdotn * (wts * fsrc)[None, :], axis=1)
    return T_sigma[:, None]

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

    # Solid harmonics (evaluated as if src sphere were unit; SL scales by a = S['r'])
    a = S['r']
    rho = trg_dr / a
    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64)
    diag = Lap3d_sl_diag(sh)
    rpowers_ext = rho ** (-l_vals-1)
    rpowers_int = rho ** (l_vals)
    qlm_SL_sigma_ext = rpowers_ext * qlm_sigma * diag
    qlm_SL_sigma_int = rpowers_int * qlm_sigma * diag
    qlm_SL_sigma = a * jnp.where(trg_dr > a, qlm_SL_sigma_ext, qlm_SL_sigma_int)

    # Evaluation at target
    qlm_SL_sigma = np.array(qlm_SL_sigma, dtype=np.complex128)
    trg_phi = np.array(trg_phi)
    trg_costheta = np.cos(trg_theta)
    SL_sigma = jnp.array(
        [sh.SH_to_point_cplx(qlm_SL_sigma[trg_i, :], trg_costheta[trg_i], trg_phi[trg_i])
         for trg_i in range(trg_theta.shape[0])],
        dtype=jnp.complex128,
    )[:, None]

    # TODO: use SH_to_lat (cplx) for near target that lie on a sphere, with rotations pre- and post-. 

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

    # Solid harmonics (evaluated as if src sphere were unit; DL is scale-invariant)
    a = S['r']
    rho = trg_dr / a
    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64)
    [diag_ext, diag_int] = Lap3d_dl_diag(sh)
    rpowers_ext = rho ** (-l_vals-1)
    rpowers_int = rho ** (l_vals)
    qlm_DL_sigma_ext = rpowers_ext * qlm_sigma * diag_ext
    qlm_DL_sigma_int = rpowers_int * qlm_sigma * diag_int
    qlm_DL_sigma = jnp.where(trg_dr > a, qlm_DL_sigma_ext, qlm_DL_sigma_int)
    
    # Evaluation at target
    qlm_DL_sigma = np.array(qlm_DL_sigma, dtype=np.complex128)
    trg_phi = np.array(trg_phi)
    trg_costheta = np.cos(trg_theta)
    DL_sigma = jnp.array(
        [sh.SH_to_point_cplx(qlm_DL_sigma[trg_i, :], trg_costheta[trg_i], trg_phi[trg_i])
         for trg_i in range(trg_theta.shape[0])],
        dtype=jnp.complex128,
    )[:, None]

    return DL_sigma

def Lap3d_dsl(trg: jax.Array, trgN: jax.Array, S: SphereDict, sh: shtns_jax.sht) -> jax.Array:
    """
    Off-surface evaluation at points <trg>: Ntrg x 3 with target normal <trgN>: Ntrg x 3
        From source <S> with density <S["Sigma"]>
        source uses SHT object <sh>
    """

    assert trg.shape[1] == 3 and trgN.shape[1] == 3

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

    # Solid harmonics (evaluated as if src sphere were unit; SL traction is scale-invariant)
    a = S['r']
    rho = trg_dr / a
    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64)
    [diag_ext, diag_int] = Lap3d_dsl_diag(sh)
    rpowers_ext = rho ** (-l_vals-1)
    rpowers_int = rho ** (l_vals)
    qlm_T_sigma_ext = rpowers_ext * qlm_sigma * diag_ext
    qlm_T_sigma_int = rpowers_int * qlm_sigma * diag_int
    qlm_T_sigma = jnp.where(trg_dr > a, qlm_T_sigma_ext, qlm_T_sigma_int)
    
    # Evaluation at target
    qlm_T_sigma = np.array(qlm_T_sigma, dtype=np.complex128)
    trg_phi = np.array(trg_phi)
    trg_costheta = np.cos(trg_theta)
    T_sigma = jnp.array(
        [sh.SH_to_point_cplx(qlm_T_sigma[trg_i, :], trg_costheta[trg_i], trg_phi[trg_i])
         for trg_i in range(trg_theta.shape[0])],
        dtype=jnp.complex128,
    )[:, None]

    return T_sigma

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
    a = S['r']
    rho = trg_dr / a          # solid harmonics as if src sphere were unit; SL scales by a
    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64)
    diag = Lap3d_sl_diag(sh)
    rpowers_ext = rho ** (-l_vals-1)
    rpowers_int = rho ** (l_vals)
    qlm_SL_sigma_ext = a * rpowers_ext * qlm_sigma * diag
    qlm_SL_sigma_int = a * rpowers_int * qlm_sigma * diag
    qlm_SL_sigma = qlm_SL_sigma_ext if trg_dr > a else qlm_SL_sigma_int

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
    a = S['r']
    rho = trg_dr / a          # solid harmonics as if src sphere were unit; DL is scale-invariant
    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64)
    [diag_ext, diag_int] = Lap3d_dl_diag(sh)
    rpowers_ext = rho ** (-l_vals-1)
    rpowers_int = rho ** (l_vals)
    qlm_DL_sigma_ext = rpowers_ext * qlm_sigma * diag_ext
    qlm_DL_sigma_int = rpowers_int * qlm_sigma * diag_int
    qlm_DL_sigma = qlm_DL_sigma_ext if trg_dr > a else qlm_DL_sigma_int

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

def compute_flux(trg: jax.Array, trgN: jax.Array, src: jax.Array, force: jax.Array) -> jax.Array:
    """
    Compute the outward normal derivative n . grad u (the Neumann data du/dn)
        of the point-source potential u (see compute_potential)
        from source positioned at <src>: Nsrc x 3
            with charge <force>: Nsrc x 1
        at target positioned at <trg>: Ntrg x 3 with normal <trgN>

    With r = trg - src, grad_x (1/|r|) = -r/|r|^3, so
        du/dn = -(1/4pi) (r.n) / |r|^3 * q.
    This matches the SL traction operator (Lap3d_dsl), which returns the true
    normal derivative d(S sigma)/dn.
    """

    assert trg.shape[1] == 3 and src.shape[1] == 3 and trgN.shape[1] == 3
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
    invdr = 1./dr
    invdr3 = invdr * invdr * invdr
    rdotn = dx * trgN[:,0][:,None] + dy * trgN[:,1][:,None] + dz * trgN[:,2][:,None]
    K = -jnp.matmul(1/4./jnp.pi * invdr3 * rdotn, force)  # du/dn = -(1/4pi)(r.n)/|r|^3 q
    K = K + 0j

    return K

@partial(jax.jit, static_argnames=["sh"])
def bio_diag_apply(qlm_sigma: jax.Array, sh: shtns_jax.sht, sl_scal: float, dl_scal: float, dsl_scal: float = 0., radius: float = 1.0) -> jax.Array:
    """
    Apply the combined DL potential operator K = sl_scal * SL + dl_scal * DL
        to the density with coefficients <qlm_sigma> in SH basis
    Returns the resulting function coefficients <qlm_KL_sigma> in SH basis,
        where DL is evaluated in the P.V. sense.
    Added dSL, the traction kernel for singular layer potential. Formulation should be either a combination of SL+DL, or T, not both.
    The SL block scales by the source-sphere <radius> (DL/dSL are scale-invariant).
    """

    sl_diag = radius * Lap3d_sl_diag(sh)
    [dl_diag_ext, dl_diag_int] = Lap3d_dl_diag(sh)
    dl_diag = 0.5*(dl_diag_ext + dl_diag_int)
    [dsl_diag_ext, dsl_diag_int] = Lap3d_dsl_diag(sh)
    dsl_diag = 0.5*(dsl_diag_ext + dsl_diag_int)
    qlm_SL_sigma = sl_diag * qlm_sigma
    qlm_DL_sigma = dl_diag * qlm_sigma
    qlm_T_sigma = dsl_diag * qlm_sigma
    qlm_KL_sigma = dl_scal * qlm_DL_sigma + sl_scal * qlm_SL_sigma + dsl_scal * qlm_T_sigma
    return qlm_KL_sigma

@partial(jax.jit, static_argnames=["sh"])
def bio_onsurf_apply(sigma: jax.Array, sh: shtns_jax.sht, sl_scal: float, dl_scal: float, sgn: float, dsl_scal: float = 0., radius: float = 1.0) -> jax.Array:
    """
    Apply the combined DL potential operator K = sl_scal * SL + dl_scal * DL
        to the density <sigma> defined on the <sh> grid
        taking into account the DL jump condition with sign <sgn>.
    Traction kernel should have opposite sign to DL, but only one should appear per formulation.
    The SL block scales by the source-sphere <radius> (DL/dSL/jump are scale-invariant).
    """
    # assert dl_scal * dsl_scal == 0 # check that only one is nonzero. -- ran into jit compile problem

    qlm_sigma = sh.analys_cplx_jax(sigma)
    qlm_KL_sigma = bio_diag_apply(qlm_sigma, sh, sl_scal, dl_scal, dsl_scal, radius)
    KL_sigma = sh.synth_cplx_jax(qlm_KL_sigma)
    return 0.5 * dl_scal * sgn * sigma + 0.5 * dsl_scal * (-1*sgn) * sigma + KL_sigma

@partial(jax.jit, static_argnames=["sh"])
def bio_onsurf_direct_solve(bc_pot: jax.Array, sh: shtns_jax.sht, sl_scal: float, dl_scal: float, sgn: float, dsl_scal: float = 0., radius: float = 1.0) -> jax.Array:
    """
    Directly solves the BIO equation in the spectral domain.
    Equation: [0.5 * dl_scal * sgn * I + KL] sigma = bc_pot

    Note: The l=0 mode is in the null space and is set to the BC value.
    The SL block scales by the source-sphere <radius> (DL/dSL/jump are scale-invariant).
    """
    # assert dl_scal * dsl_scal == 0

    qlm_bc = sh.analys_cplx_jax(bc_pot)
    sl_diag = radius * Lap3d_sl_diag(sh)
    [dl_diag_ext, dl_diag_int] = Lap3d_dl_diag(sh)
    dl_diag = 0.5 * (dl_diag_ext + dl_diag_int)
    [dsl_diag_ext, dsl_diag_int] = Lap3d_dsl_diag(sh)
    dsl_diag = 0.5 * (dsl_diag_ext + dsl_diag_int)
    operator_diag = (0.5 * dl_scal * sgn) + (dl_scal * dl_diag) + (sl_scal * sl_diag) + (0.5 * dsl_scal * (-1*sgn)) + (dsl_scal * dsl_diag)
    
    # Where the operator is singular (e.g. the l=0 nullspace of the pure-DL
    # Dirichlet formulation) division is undefined; set those modes to the BC
    # value as a heuristic. Modes with a nonzero eigenvalue (e.g. the
    # Neumann/traction formulation, where l=0 has eigenvalue -1) are solved
    # normally -- forcing them to the BC value would corrupt the solution.
    near_zero = jnp.abs(operator_diag) <= 1e-14
    safe_diag = jnp.where(near_zero, 1.0, operator_diag)
    qlm_sigma = qlm_bc / safe_diag
    qlm_sigma = jnp.where(near_zero, qlm_bc, qlm_sigma)
    
    sigma = sh.synth_cplx_jax(qlm_sigma)
    
    return sigma

def bio_offsurf_apply(trg: jax.Array, S: SphereDict, sh: shtns_jax.sht, sl_scal: float, dl_scal: float, far: bool = False) -> jax.Array:
    """
    Evaluate the KL formulation of <S> with density <S["Sigma"]> at target <trg>
    """
    if not far:
        SLsigma = Lap3d_sl(trg, S, sh)
        DLsigma = Lap3d_dl(trg, S, sh)
    else:
        SLsigma = Lap3d_sl_far(trg, S, sh)
        DLsigma = Lap3d_dl_far(trg, S, sh)
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
    print("\n Manufactured solutions test Laplace 3D solver on the unit sphere ---- Exterior Dirichlet problem")
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

    S = set_density(S, sig_direct)
    Ksig_direct = bio_offsurf_apply_1sph(Strg, shtrg, S, sh, sl_scal, dl_scal)
    Ksig_direct = jnp.real(jnp.reshape(Ksig_direct,(-1,1)))

    diff_direct = jnp.max(true_pot - Ksig_direct) / jnp.max(true_pot)
    jax.debug.print("Max relative error of order {lmax} solver at target radius {Rtrg} for direct solver is {d}", lmax=lmax, Rtrg=Rtrg, d=diff_direct)

    # Exterior Neumann problem
    print("\n Manufactured solutions test Laplace 3D solver on the unit sphere ---- Exterior Neumann problem")
    # Formulation: u(x) = S[sigma](x), match du/dn(gamma) = dSn[sigma](gamma)
    xn = S["Xncart"][:,:,0]
    yn = S["Xncart"][:,:,1]
    zn = S["Xncart"][:,:,2]
    trgN_sphere = jnp.column_stack([jnp.reshape(xn,-1), jnp.reshape(yn,-1), jnp.reshape(zn,-1)])
    BC_flux = compute_flux(trg_sphere, trgN_sphere, ptsrc, force) 
    BC_flux = jnp.reshape(BC_flux, x.shape)
    # DIRECT solve
    sig_direct = bio_onsurf_direct_solve(
        bc_pot=BC_flux,
        sh=sh,
        sl_scal=0.,
        dl_scal=0.,
        sgn=sgn,
        dsl_scal=1.0
    )
    bc_check_direct = bio_onsurf_apply(sig_direct, sh, 0., 0., sgn, 1.0)
    resid_direct = jnp.linalg.norm(bc_check_direct - BC_flux)
    jax.debug.print("Residual of DIRECT solve: {a}", a=resid_direct)
    # Accuracy
    xtrg = Strg["Xcart"][:,:,0] 
    ytrg = Strg["Xcart"][:,:,1]
    ztrg = Strg["Xcart"][:,:,2]
    trg_sphere2 = jnp.column_stack([jnp.reshape(xtrg,-1), jnp.reshape(ytrg,-1), jnp.reshape(ztrg,-1)])
    true_pot = compute_potential(trg_sphere2, ptsrc, force)
    true_pot = jnp.real(true_pot)

    S = set_density(S, sig_direct)
    Ksig_direct = bio_offsurf_apply_1sph(Strg, shtrg, S, sh, 1.0, 0.)
    Ksig_direct = jnp.real(jnp.reshape(Ksig_direct,(-1,1)))

    diff_direct = jnp.max(true_pot - Ksig_direct) / jnp.max(true_pot)
    jax.debug.print("Max relative error of order {lmax} solver at target radius {Rtrg} for direct solver is {d}", lmax=lmax, Rtrg=Rtrg, d=diff_direct)



    # Target -- interior
    print("\n Manufactured solutions test Laplace 3D solver on the unit sphere ---- Interior Dirichlet problem")
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

    S = set_density(S, sig_direct)
    Ksig_direct = bio_offsurf_apply_1sph(Strg, shtrg, S, sh, sl_scal, dl_scal)
    Ksig_direct = jnp.real(jnp.reshape(Ksig_direct,(-1,1)))

    diff_direct = jnp.max(true_pot - Ksig_direct) / jnp.max(true_pot)
    jax.debug.print("Max relative error of order {lmax} solver at target radius {Rtrg} for direct solver is {d}", lmax=lmax, Rtrg=Rtrg, d=diff_direct)

    # Visualization
    z_slices = [-0.5, 0.0, 0.5]
    Ng = 60
    gx = jnp.linspace(-radius, radius, Ng)
    gy = jnp.linspace(-radius, radius, Ng)
    X, Y = jnp.meshgrid(gx, gy, indexing="xy")     # (Ng, Ng)

    pot_slices = []
    for z0 in z_slices:
        Z = jnp.full_like(X, z0)
        trg_grid = jnp.column_stack([X.ravel(), Y.ravel(), Z.ravel()])  # (Ntrg, 3)
        sep_trg = separate_target(trg_grid, S, 0.1)                     # (Ntrg,) bool, far/near
        Ksig_far = bio_offsurf_apply(trg_grid, S, sh, sl_scal, dl_scal, True)
        Ksig_near = bio_offsurf_apply(trg_grid, S, sh, sl_scal, dl_scal, False)
        Ksig_mix = jnp.where(sep_trg[:, None], Ksig_far, Ksig_near)     # (Ntrg, 1)
        K = jnp.real(Ksig_mix).reshape(X.shape)
        rr = jnp.sqrt(X**2 + Y**2 + z0**2)
        K = jnp.where(rr < radius, K, jnp.nan)                          # mask outside sphere
        pot_slices.append(np.asarray(K))

    Xn, Yn = np.asarray(X), np.asarray(Y)
    finite = np.concatenate([P[np.isfinite(P)] for P in pot_slices])
    vmin, vmax = float(finite.min()), float(finite.max())

    # 3D visual: each cross section as a filled contour at its own z offset.
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    for z0, P in zip(z_slices, pot_slices):
        ax.contourf(Xn, Yn, P, zdir="z", offset=z0, levels=40, vmin=vmin, vmax=vmax, cmap="viridis")
    # source-sphere surface drawn from the actual quadrature grid S["Xcart"], bolder lines
    sx = np.asarray(S["Xcart"][:, :, 0])
    sy = np.asarray(S["Xcart"][:, :, 1])
    sz = np.asarray(S["Xcart"][:, :, 2])
    ax.plot_wireframe(sx, sy, sz, color="gray", alpha=0.5, linewidth=1.0)
    ax.set(xlabel="x", ylabel="y", zlabel="z",
           title="Interior Laplace layer potential -- cross sections")
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin, vmax)); sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.6, label="Re potential")
    fig.savefig("vis/Lap3d_interior_slices_3d.png", dpi=150)
    plt.close(fig)

    # 2D companion: one contourf panel per slice, shared color scale.
    fig2, axes = plt.subplots(1, len(z_slices), figsize=(4*len(z_slices), 4))
    for axk, z0, P in zip(axes, z_slices, pot_slices):
        cf = axk.contourf(Xn, Yn, P, levels=40, vmin=vmin, vmax=vmax, cmap="viridis")
        rad_z = np.sqrt(max(radius**2 - z0**2, 0.0))
        circ = plt.Circle((0, 0), rad_z, fill=False, color="k", linewidth=0.8)
        axk.add_patch(circ)
        axk.set(aspect="equal", xlabel="x", ylabel="y", title=f"z = {z0}")
    sm2 = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin, vmax)); sm2.set_array([])
    fig2.colorbar(sm2, ax=axes, shrink=0.8, label="Re potential")
    fig2.savefig("vis/Lap3d_interior_slices_2d.png", dpi=150)
    plt.close(fig2)
