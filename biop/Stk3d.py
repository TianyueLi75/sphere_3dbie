"""
Stokes Operator Class:
    SL and DL operators on a sphere, using spectra and solid harmonics
    TODO:
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
# import lineax as lx
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

def qst2vwx_cplx(qlm: jax.Array, slm: jax.Array, tlm: jax.Array, sh: shtns_jax.sht) -> tuple([jax.Array, jax.Array, jax.Array]):
    """
    Transform coefficients in Q,S,T basis (used in SHTns)
        to the V,W,X basis (diagonalizing basis for Stokes LP operators)
    """
    
    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64)
    vlm = (l_vals * slm - qlm) / (2.0*l_vals + 1.0)
    wlm = ((l_vals + 1.0) * slm + qlm) / (2.0*l_vals + 1.0)
    xlm = -tlm
    return vlm, wlm, xlm

def vwx2qst_cplx(vlm: jax.Array, wlm: jax.Array, xlm: jax.Array, sh: shtns_jax.sht) -> tuple([jax.Array, jax.Array, jax.Array]):
    """
    Transform coefficients in V,W,X basis (diagonalizing basis for Stokes LP operators)
        to the Q,S,T basis (used in SHTns)
    """
    
    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64)
    slm = vlm + wlm
    qlm = l_vals * (wlm - vlm) - vlm
    tlm = -xlm
    return qlm, slm, tlm

def qst2vwx(qlm: jax.Array, slm: jax.Array, tlm: jax.Array, sh: shtns_jax.sht) -> tuple([jax.Array, jax.Array, jax.Array]):
    """
    Transform coefficients in Q,S,T basis (used in SHTns)
        to the V,W,X basis (diagonalizing basis for Stokes LP operators)
    Real valued function -- use Real valued truncated qst and vwx only
    """
    
    l_vals = jnp.asarray(sh.l, dtype=jnp.float64)
    vlm = (l_vals * slm - qlm) / (2.0*l_vals + 1.0)
    wlm = ((l_vals + 1.0) * slm + qlm) / (2.0*l_vals + 1.0)
    xlm = -tlm
    return vlm, wlm, xlm

def vwx2qst(vlm: jax.Array, wlm: jax.Array, xlm: jax.Array, sh: shtns_jax.sht) -> tuple([jax.Array, jax.Array, jax.Array]):
    """
    Transform coefficients in V,W,X basis (diagonalizing basis for Stokes LP operators)
        to the Q,S,T basis (used in SHTns)
    Real valued function -- use Real valued truncated qst and vwx only
    """
    
    l_vals = jnp.asarray(sh.l, dtype=jnp.float64)
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

def sig_xyz2vwx_cplx(sigma_x: jax.Array, sigma_y: jax.Array, sigma_z: jax.Array, theta: jax.Array, phi: jax.Array, sh: shtns_jax.sht) -> tuple([jax.Array,jax.Array,jax.Array]):
    """
    Transforms vector fields in Cartesian coordinates  
        into its coefficients in V,W,X basis
    """

    sigma_r, sigma_t, sigma_p = cart2sph(sigma_x,sigma_y,sigma_z,theta,phi)
    sigma_rtp = vec_stack(sigma_r, sigma_t, sigma_p)
    qstlm = sh.analys_vec_cplx_jax(sigma_rtp)
    qlm, slm, tlm = vec_distr(qstlm) 
    vlm, wlm, xlm = qst2vwx_cplx(qlm, slm, tlm, sh)
    return vlm, wlm, xlm

def sig_vwx2xyz_cplx(vlm: jax.Array, wlm: jax.Array, xlm: jax.Array, theta: jax.Array, phi: jax.Array, sh: shtns_jax.sht) -> tuple([jax.Array, jax.Array, jax.Array]):
    """
    Transforms coefficients in V,W,X basis
        into its Cartesian coordinates
    """

    qlm, slm, tlm = vwx2qst_cplx(vlm,wlm,xlm,sh)
    qstlm = vec_stack(qlm, slm, tlm)
    vrtp = sh.synth_vec_cplx_jax(qstlm)
    vr, vt, vp = vec_distr(vrtp)
    vx, vy, vz = sph2cart(vr, vt, vp, theta, phi)
    return vx, vy, vz

def sig_xyz2vwx(sigma_x: jax.Array, sigma_y: jax.Array, sigma_z: jax.Array, theta: jax.Array, phi: jax.Array, sh: shtns_jax.sht) -> tuple([jax.Array,jax.Array,jax.Array]):
    """
    Transforms vector fields in Cartesian coordinates  
        into its coefficients in V,W,X basis
    Real valued function -- use Real valued truncated qst and vwx only
    """

    sigma_r, sigma_t, sigma_p = cart2sph(sigma_x,sigma_y,sigma_z,theta,phi)
    sigma_rtp = vec_stack(sigma_r, sigma_t, sigma_p)
    qstlm = sh.analys_vec_jax(sigma_rtp)
    qlm, slm, tlm = vec_distr(qstlm) 
    vlm, wlm, xlm = qst2vwx(qlm, slm, tlm, sh)
    return vlm, wlm, xlm

def sig_vwx2xyz(vlm: jax.Array, wlm: jax.Array, xlm: jax.Array, theta: jax.Array, phi: jax.Array, sh: shtns_jax.sht) -> tuple([jax.Array, jax.Array, jax.Array]):
    """
    Transforms coefficients in V,W,X basis
        into its Cartesian coordinates
    Real valued function -- use Real valued truncated qst and vwx only
    """

    qlm, slm, tlm = vwx2qst(vlm,wlm,xlm,sh)
    qstlm = vec_stack(qlm, slm, tlm)
    vrtp = sh.synth_vec_jax(qstlm)
    vr, vt, vp = vec_distr(vrtp)
    vx, vy, vz = sph2cart(vr, vt, vp, theta, phi)
    return vx, vy, vz

# ============================
# === Spectra ================
# ============================
def Stk3d_sl_VWX_diag_cplx(sh: shtns_jax.sht) -> tuple([jax.Array, jax.Array, jax.Array]):
    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64)
    diag_V = l_vals / (2.0*l_vals + 1.0) / (2.0*l_vals + 3.0)
    diag_W = (l_vals + 1.0) / (2.0*l_vals + 1.0) / (2.0*l_vals - 1.0)
    diag_X = 1.0 / (2.0*l_vals + 1.0)
    return diag_V, diag_W, diag_X

def Stk3d_dl_VWX_diag_cplx(sh: shtns_jax.sht) -> tuple([jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]):
    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64)

    diag_V_ext = (2.0*l_vals*l_vals + 4*l_vals + 3) / (2.0*l_vals + 1.0) / (2.0*l_vals + 3.0)
    diag_W_ext = 2.0*(l_vals + 1.0)*(l_vals - 1.0) / (2.0*l_vals + 1.0) / (2.0*l_vals - 1.0)
    diag_X_ext = (l_vals - 1.0) / (2.0*l_vals + 1.0)

    diag_V_int = -2.0*l_vals*(l_vals + 2) / (2.0*l_vals + 1.0) / (2.0*l_vals + 3.0)
    diag_W_int = -(2.0*l_vals*l_vals + 1.0) / (2.0*l_vals + 1.0) / (2.0*l_vals - 1.0)
    diag_X_int = -(l_vals + 2.0) / (2.0*l_vals + 1.0)

    return diag_V_ext, diag_W_ext, diag_X_ext, diag_V_int, diag_W_int, diag_X_int

def Stk3d_dsl_VWX_diag_cplx(sh: shtns_jax.sht) -> tuple([jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]):
    # dSL is the DL diagonal with the exterior/interior triples swapped.
    diag_V_ext, diag_W_ext, diag_X_ext, diag_V_int, diag_W_int, diag_X_int = Stk3d_dl_VWX_diag_cplx(sh)
    return diag_V_int, diag_W_int, diag_X_int, diag_V_ext, diag_W_ext, diag_X_ext

def diag_W2V_cplx(sh: shtns_jax.sht, which: str) -> jax.Array:
    """Exterior W->V off-surface coupling coefficient of the SL ('sl') or DL ('dl')
    solid-harmonic scaling, in the V/W/X diagonalizing basis."""
    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64)
    if which == "sl":
        return l_vals / (4.0*l_vals + 2.0)
    return 2.0*l_vals*(l_vals - 1.0) / (4.0*l_vals + 2.0)          # dl

def diag_V2W_cplx(sh: shtns_jax.sht, which: str) -> jax.Array:
    """Interior V->W off-surface coupling coefficient of the SL ('sl') or DL ('dl')
    solid-harmonic scaling, in the V/W/X diagonalizing basis."""
    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64)
    if which == "sl":
        return (l_vals + 1.0) / (4.0*l_vals + 2.0)
    return (l_vals + 1.0)*(l_vals + 2.0) / (2.0*l_vals + 1.0)      # dl

def Stk3d_sl_VWX_diag(sh: shtns_jax.sht) -> tuple([jax.Array, jax.Array, jax.Array]):
    l_vals = jnp.asarray(sh.l, dtype=jnp.float64)
    diag_V = l_vals / (2.0*l_vals + 1.0) / (2.0*l_vals + 3.0)
    diag_W = (l_vals + 1.0) / (2.0*l_vals + 1.0) / (2.0*l_vals - 1.0)
    diag_X = 1.0 / (2.0*l_vals + 1.0)
    return diag_V, diag_W, diag_X

def Stk3d_dl_VWX_diag(sh: shtns_jax.sht) -> tuple([jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]):
    l_vals = jnp.asarray(sh.l, dtype=jnp.float64)

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
    l_vals = jnp.asarray(sh.l, dtype=jnp.float64)
    if which == "sl":
        return l_vals / (4.0*l_vals + 2.0)
    return 2.0*l_vals*(l_vals - 1.0) / (4.0*l_vals + 2.0)          # dl

def diag_V2W(sh: shtns_jax.sht, which: str) -> jax.Array:
    """Interior V->W off-surface coupling coefficient of the SL ('sl') or DL ('dl')
    solid-harmonic scaling, in the V/W/X diagonalizing basis."""
    l_vals = jnp.asarray(sh.l, dtype=jnp.float64)
    if which == "sl":
        return (l_vals + 1.0) / (4.0*l_vals + 2.0)
    return (l_vals + 1.0)*(l_vals + 2.0) / (2.0*l_vals + 1.0)      # dl



# ============================
# === Operators ==============
# === In spectral domain =====
# ============================

STK_SL_FAR_PREFAC = 1.0 / 8.0 / jnp.pi
STK_DL_FAR_PREFAC = 6.0 / 8.0 / jnp.pi

@partial(jax.jit, static_argnames=["sh", "tile", "terms"])
def _stk_far_kernel_cplx(trg: jax.Array, sig_vwx: jax.Array, sl_scal, dl_scal,
                    Y: jax.Array, w: jax.Array, a, Xc: jax.Array,
                    th: jax.Array, ph: jax.Array,
                    sh: shtns_jax.sht, tile: int, terms: tuple) -> jax.Array:
    """
    Fused smooth-quadrature ("far") evaluation of  K = sl*SL + dl*DL  (Stokeslet + stresslet)
    of the source sphere described by <Y>, <w>, <a>, <Xc> (see sphere.far_src_geom) with
    vector density VWX coefficients <sig_vwx> ((3, nlm)), at targets <trg> (Ntrg x 3).
    Returns the velocity: Ntrg x 3.  <terms> selects which operators are built (static, so
    the SL-only / DL-only wrappers pay nothing for the other); the scalings stay traced.

    With x_i = trg_i - Xc, r = x_i - Y_j, g_j = w_j sigma_j and n_j = Y_j/a exactly (sphere),
        d2 = |x_i|^2 - 2 x_i.Y_j + a^2,   r.n_j = (|x_i|^2 - a^2 - d2)/(2a)
    so the stresslet's radial factor collapses to
        M_ij = invd5*(r.n) = [ (|x_i|^2 - a^2) invd5 - invd3 ] / (2a)
    and, with  coef = sl*pref_sl*invd3 + dl*pref_dl*M,  both operators share ONE pair matrix:
        u_i = sl*pref_sl * sum_j invd g_j  +  sum_j coef_ij (r.g_j) r
    Expanding r.g_j = x_i.g_j - Y_j.g_j and r = x_i - Y_j turns the second sum into
    contractions of <coef> against source-only column blocks (g_j, Y_j.g_j, g_j (x) Y_j,
    (Y_j.g_j) Y_j), reassembled with the per-target factor x_i. So NO pair-shaped array
    depends on the density: only invd and coef are Ntrg x Nsrc, and both are REAL. That
    removes the two 3-wide complex Ntrg x Nsrc x 3 temporaries, all complex pair arithmetic,
    and the ~10 divisions per pair of the previous formulation (one rsqrt remains; note the
    old d**3 / d**5 were already lowered to multiplies, so it is the divisions and the pair
    traffic that cost, not pow()).
    Measured ~12-19x faster per call than the previous per-operator kernels (lmax 32-100,
    container geometry), which makes a full suspension solve 6-10x faster.
    """
    want_sl, want_dl = ("sl" in terms), ("dl" in terms)
    sx, sy, sz = sig_vwx2xyz_cplx(sig_vwx[0], sig_vwx[1], sig_vwx[2], th, ph, sh)
    g = jnp.stack([sx, sy, sz], axis=-1).reshape(-1, 3) * w[:, None]   # (Nsrc, 3) weighted
    a2 = a * a
    Nsrc = Y.shape[0]

    # Source-only column blocks for the <coef> contraction (16 complex -> 32 real columns):
    #   0:3 g_jk | 3 Y_j.g_j | 4:13 g_jm Y_jk | 13:16 (Y_j.g_j) Y_jk
    b = jnp.sum(Y * g, axis=1)                                        # (Nsrc,)
    Z = jnp.concatenate([g, b[:, None],
                         (g[:, :, None] * Y[:, None, :]).reshape(Nsrc, 9),
                         b[:, None] * Y], axis=1)                     # (Nsrc, 16)
    Zr = jnp.concatenate([jnp.real(Z), jnp.imag(Z)], axis=1)          # (Nsrc, 32)
    if want_sl:
        Gr = jnp.concatenate([jnp.real(g), jnp.imag(g)], axis=1)      # (Nsrc, 6)

    def body(tt):
        X = tt - Xc                                                   # (nt, 3)
        px = jnp.sum(X * X, axis=1)[:, None]                          # (nt, 1)
        d2 = jnp.maximum(px + a2 - 2.0 * pair_dot(X, Y), 0.0)         # (nt, Nsrc)
        invd = jax.lax.rsqrt(d2)
        invd2 = invd * invd                                           # (= 1/d2, no divide)
        invd3 = invd * invd2
        coef = (sl_scal * STK_SL_FAR_PREFAC) * invd3 if want_sl else 0.0
        if want_dl:
            M = ((px - a2) * (invd3 * invd2) - invd3) / (2.0 * a)
            coef = coef + (dl_scal * STK_DL_FAR_PREFAC) * M

        R = coef @ Zr                                                 # (nt, 32)
        Zo = R[:, :16] + 1j * R[:, 16:]                                # (nt, 16)
        Cg = Zo[:, 0:3]                       # sum_j coef g_j
        cb = Zo[:, 3]                         # sum_j coef (Y_j.g_j)
        Tg = Zo[:, 4:13].reshape(-1, 3, 3)    # sum_j coef g_jm Y_jk
        cbY = Zo[:, 13:16]                    # sum_j coef (Y_j.g_j) Y_jk
        # sum_j coef (r.g_j) r  =  x_i (x_i.Cg - cb) - (x_i.Tg - cbY)
        u = X * (jnp.sum(X * Cg, axis=1) - cb)[:, None] \
            - (jnp.einsum("im,imk->ik", X, Tg) - cbY)
        if want_sl:
            Sg = invd @ Gr                                            # (nt, 6)
            u = u + (sl_scal * STK_SL_FAR_PREFAC) * (Sg[:, :3] + 1j * Sg[:, 3:])
        return u

    return far_tile_map(body, (trg,), tile)

def _stk_far_cplx(trg: jax.Array, sig_vwx: jax.Array, S: SphereDict, sh: shtns_jax.sht,
             terms: tuple, sl_scal=0.0, dl_scal=0.0, tile: int = None) -> jax.Array:
    """Eager wrapper of _stk_far_kernel_cplx: pulls the source-centred geometry out of <S> and
    picks the target tile size, then calls the jitted kernel."""
    Y, w, a, Xc = far_src_geom(S, sh)
    return _stk_far_kernel_cplx(trg, sig_vwx, sl_scal, dl_scal, Y, w, a, Xc,
                           S["Xsph"][:, :, 0], S["Xsph"][:, :, 1],
                           sh=sh, tile=far_tile_size(Y.shape[0], tile), terms=terms)

@partial(jax.jit, static_argnames=["sh", "tile", "terms"])
def _stk_far_kernel(trg: jax.Array, sig_vwx: jax.Array, sl_scal, dl_scal,
                    Y: jax.Array, w: jax.Array, a, Xc: jax.Array,
                    th: jax.Array, ph: jax.Array,
                    sh: shtns_jax.sht, tile: int, terms: tuple) -> jax.Array:
    """Real/truncated-layout counterpart of _stk_far_kernel_cplx. The density VWX coefficients
    <sig_vwx> are the real (m>=0) layout (length nlm), synthesized back to the source grid with
    the real sig_vwx2xyz -> float64 grid density. Because the density is REAL, the complex
    real/imag column split of the cplx kernel collapses: the contraction matrix Z has 16 real
    columns (not 32), one real matmul per tile, and the returned velocity is float64 (Ntrg x 3)."""
    want_sl, want_dl = ("sl" in terms), ("dl" in terms)
    sx, sy, sz = sig_vwx2xyz(sig_vwx[0], sig_vwx[1], sig_vwx[2], th, ph, sh)   # float64 grid
    g = jnp.stack([sx, sy, sz], axis=-1).reshape(-1, 3) * w[:, None]   # (Nsrc, 3) real, weighted
    a2 = a * a
    Nsrc = Y.shape[0]

    # Source-only column blocks for the <coef> contraction (16 real columns; no imag half):
    #   0:3 g_jk | 3 Y_j.g_j | 4:13 g_jm Y_jk | 13:16 (Y_j.g_j) Y_jk
    b = jnp.sum(Y * g, axis=1)                                        # (Nsrc,)
    Z = jnp.concatenate([g, b[:, None],
                         (g[:, :, None] * Y[:, None, :]).reshape(Nsrc, 9),
                         b[:, None] * Y], axis=1)                     # (Nsrc, 16) real
    if want_sl:
        G = g                                                        # (Nsrc, 3) real

    def body(tt):
        X = tt - Xc                                                   # (nt, 3)
        px = jnp.sum(X * X, axis=1)[:, None]                          # (nt, 1)
        d2 = jnp.maximum(px + a2 - 2.0 * pair_dot(X, Y), 0.0)         # (nt, Nsrc)
        invd = jax.lax.rsqrt(d2)
        invd2 = invd * invd
        invd3 = invd * invd2
        coef = (sl_scal * STK_SL_FAR_PREFAC) * invd3 if want_sl else 0.0
        if want_dl:
            M = ((px - a2) * (invd3 * invd2) - invd3) / (2.0 * a)
            coef = coef + (dl_scal * STK_DL_FAR_PREFAC) * M

        R = coef @ Z                                                 # (nt, 16) real
        Cg = R[:, 0:3]                        # sum_j coef g_j
        cb = R[:, 3]                          # sum_j coef (Y_j.g_j)
        Tg = R[:, 4:13].reshape(-1, 3, 3)     # sum_j coef g_jm Y_jk
        cbY = R[:, 13:16]                     # sum_j coef (Y_j.g_j) Y_jk
        # sum_j coef (r.g_j) r  =  x_i (x_i.Cg - cb) - (x_i.Tg - cbY)
        u = X * (jnp.sum(X * Cg, axis=1) - cb)[:, None] \
            - (jnp.einsum("im,imk->ik", X, Tg) - cbY)
        if want_sl:
            u = u + (sl_scal * STK_SL_FAR_PREFAC) * (invd @ G)       # (nt, 3)
        return u

    return far_tile_map(body, (trg,), tile)

def _stk_far(trg: jax.Array, sig_vwx: jax.Array, S: SphereDict, sh: shtns_jax.sht,
             terms: tuple, sl_scal=0.0, dl_scal=0.0, tile: int = None) -> jax.Array:
    """Real/truncated-layout eager wrapper of _stk_far_kernel (returns float64 velocity)."""
    Y, w, a, Xc = far_src_geom(S, sh)
    return _stk_far_kernel(trg, sig_vwx, sl_scal, dl_scal, Y, w, a, Xc,
                           S["Xsph"][:, :, 0], S["Xsph"][:, :, 1],
                           sh=sh, tile=far_tile_size(Y.shape[0], tile), terms=terms)

def Stk3d_sl_far(trg: jax.Array, sig_vwx: jax.Array, S: SphereDict, sh: shtns_jax.sht) -> jax.Array:
    """
    Off-surface evaluation of the Stokes SL (Stokeslet) velocity with the vector
    density coefficients <sig_vwx> (stacked VWX, shape (3, nlm))
        with <trg>: Ntrg x 3
        using the smooth surface quadrature (accurate only for targets well
        away from the surface). The direct quadrature needs the grid-space density, so
        the VWX coefficients are first synthesized back to the Cartesian source grid
        (inverse decomposition), then the direct matrix integral is applied.
    Returns the velocity at the targets: Ntrg x 3.

    Stokeslet kernel (matches compute_field): with r = trg - src, d = |r|,
        u_j(x) = (1/8pi) sum_src [ sigma_j / d + r_j (r.sigma) / d^3 ] * w.
    Thin wrapper over the fused _stk_far kernel (see there for the quadrature).
    """
    assert trg.shape[1] == 3
    return _stk_far(trg, sig_vwx, S, sh, ("sl",), sl_scal=1.0)

def Stk3d_dl_far(trg: jax.Array, sig_vwx: jax.Array, S: SphereDict, sh: shtns_jax.sht) -> jax.Array:
    """
    Off-surface evaluation of the Stokes DL (stresslet) velocity with the vector
    density coefficients <sig_vwx> (stacked VWX, shape (3, nlm))
        with <trg>: Ntrg x 3
        using the smooth surface quadrature (accurate only for targets well
        away from the surface). The VWX coefficients are synthesized to the Cartesian
        source grid first, then the direct matrix integral is applied.
    Returns the velocity at the targets: Ntrg x 3.

    Stresslet kernel T_ijk = 6 r_i r_j r_k / d^5. With r = trg - src, d = |r|,
    unit outward normal n = S["Xncart"] and density sigma,
        u_j(x) = (6/8pi) sum_src [ (r.sigma)(r.n) / d^5 ] r_j * w.
    Thin wrapper over the fused _stk_far kernel (see there for the quadrature).
    """
    assert trg.shape[1] == 3
    return _stk_far(trg, sig_vwx, S, sh, ("dl",), dl_scal=1.0)


# ============================
# === Point and shoot ========
# ============================

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

def _latlm_maps_cplx(sh: shtns_jax.sht) -> tuple:
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

def _stk_latitude_ring_cplx(qlm: jax.Array, slm: jax.Array, tlm: jax.Array,
                       cos_src, sh: shtns_jax.sht) -> tuple:
    """Single-ring FFT synthesis: complex Q/S/T coefficient vectors (each (nlm_cplx,),
    already radius-scaled) evaluated at ONE ring latitude cos_src (scalar) over all nphi
    longitudes. Reuses the vetted real FFT path sht.SHqst_to_lat_jax by splitting each
    complex coefficient vector into the real-layout coefficients of its real (G) and
    imaginary (H) parts. Returns (vr, vt, vp), each (nphi,) complex, in the source-centered
    spherical basis. _stk_latitude_cplx vmaps this over rings."""
    kpos, kneg, parity = _latlm_maps_cplx(sh)

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

def _stk_latitude_ring(qlm: jax.Array, slm: jax.Array, tlm: jax.Array,
                       cos_src, sh: shtns_jax.sht) -> tuple:
    """Single-ring FFT synthesis for the REAL/truncated layout: real-layout Q/S/T
    coefficient vectors (each (nlm,), already radius-scaled) evaluated at ONE ring latitude
    cos_src (scalar) over all nphi longitudes. Because a real-valued field's coefficients are
    ALREADY the real-layout coefficients, the complex G/H split of _stk_latitude_ring_cplx
    (the _latlm_maps gather + two SHqst_to_lat_jax calls + complex recombine) collapses to a
    SINGLE direct SHqst_to_lat_jax call -- no _latlm_maps, half the synthesis work. Returns
    (vr, vt, vp), each (nphi,) real (float64), in the source-centered spherical basis."""
    c = jnp.asarray(cos_src, dtype=jnp.float64)
    Vr, Vt, Vp = sh.SHqst_to_lat_jax(qlm, slm, tlm, c, sh.nphi)   # each (nphi,) float64
    return Vr, Vt, Vp

def _ps_scale_coeffs_cplx(sh: shtns_jax.sht, exterior: bool, which: str) -> tuple:
    """Ring-INDEPENDENT VWX diagonal + coupling symbols for operator 'which' ('sl'/'dl') on the
    exterior/interior branch. Depend only on sh.zl and the static branch, so precomputed ONCE per
    evaluator (in point_n_shoot_evaluator's eager setup) and closed over by the jitted _core --
    hoisted out of the per-ring vmap. (Under jit these fold to constants anyway; hoisting keeps
    the jaxpr small.) Returns (diag_V, diag_W, diag_X, coup)."""
    if which == "sl":
        diag_V, diag_W, diag_X = Stk3d_sl_VWX_diag_cplx(sh)
    else:
        (dVe, dWe, dXe, dVi, dWi, dXi) = Stk3d_dl_VWX_diag_cplx(sh)
        diag_V, diag_W, diag_X = (dVe, dWe, dXe) if exterior else (dVi, dWi, dXi)
    coup = diag_W2V_cplx(sh, which) if exterior else diag_V2W_cplx(sh, which)
    return diag_V, diag_W, diag_X, coup

def _ps_scale_coeffs(sh: shtns_jax.sht, exterior: bool, which: str) -> tuple:
    """Real/truncated-layout counterpart of _ps_scale_coeffs_cplx: ring-INDEPENDENT VWX
    diagonal + coupling symbols (length nlm, per-mode degree from sh.l) for operator 'which'
    on the exterior/interior branch. Returns (diag_V, diag_W, diag_X, coup)."""
    if which == "sl":
        diag_V, diag_W, diag_X = Stk3d_sl_VWX_diag(sh)
    else:
        (dVe, dWe, dXe, dVi, dWi, dXi) = Stk3d_dl_VWX_diag(sh)
        diag_V, diag_W, diag_X = (dVe, dWe, dXe) if exterior else (dVi, dWi, dXi)
    coup = diag_W2V(sh, which) if exterior else diag_V2W(sh, which)
    return diag_V, diag_W, diag_X, coup

# Same for Real or Cplx transforms
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

def _ps_ring_core_cplx(vlm: jax.Array, wlm: jax.Array, xlm: jax.Array,
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
    qlm_K, slm_K, tlm_K = vwx2qst_cplx(vK, wK, xK, sh_eval)                  # VWX -> QST (nlm_e,)
    # latitude FFT synthesis at this ring: source-centered spherical, (nphi_e,)
    vr, vt, vp = _stk_latitude_ring_cplx(qlm_K, slm_K, tlm_K, cos_src_j, sh_eval)
    # source-centered spherical -> Cartesian (rotated frame) -> target-local spherical
    uxR, uyR, uzR = sph2cart(vr, vt, vp, theta_src_j, phi_ring)
    return cart2sph(uxR, uyR, uzR, th_e_j, phi_ring)

def _ps_ring_core(vlm: jax.Array, wlm: jax.Array, xlm: jax.Array,
                  pw_V_j: jax.Array, rho_j, cos_src_j, theta_src_j, th_e_j,
                  phi_ring: jax.Array, exterior: bool,
                  a: float, sl_scal, dl_scal, sh_eval: shtns_jax.sht,
                  sl_coeffs: tuple, dl_coeffs: tuple) -> tuple:
    """Real/truncated-layout counterpart of _ps_ring_core_cplx. Identical radial-scaling and
    reframing logic, but the per-ring VWX->QST and latitude synthesis run in the real layout
    (real vwx2qst + the direct single-call _stk_latitude_ring), so no complex G/H split. All
    coefficient slices are length nlm_e (real layout)."""
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
    qlm_K, slm_K, tlm_K = vwx2qst(vK, wK, xK, sh_eval)                       # VWX -> QST (nlm_e,)
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

def point_n_shoot_evaluator_cplx(Strg: SphereDict, shtrg: shtns_jax.sht, S: SphereDict, sh: shtns_jax.sht,
                            near: bool = False):
    """
    Build a jitted point-and-shoot evaluator for a FIXED source/target geometry.
    Returns a callable apply(sig_vwx, sl_scal, dl_scal) -> velocity VWX coefficients
    (3, nlm_trg) on the target grid, for the combined Stokes layer potential
    K = sl_scal*SL + dl_scal*DL of source sphere
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
    sl_coeffs = _ps_scale_coeffs_cplx(sh_eval, exterior, "sl")
    dl_coeffs = _ps_scale_coeffs_cplx(sh_eval, exterior, "dl")
    _latlm_maps_cplx(sh_eval)   # populate the (sh-static) latlm cache eagerly, so _core reads
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

    def _core(sig_vwx: jax.Array, sl_scal, dl_scal) -> jax.Array:
        # ---- source density VWX coefficients -> Q/S/T coefficients (cplx layout) ----
        qlm_s, slm_s, tlm_s = vwx2qst_cplx(sig_vwx[0], sig_vwx[1], sig_vwx[2], sh)
        qst_src = vec_stack(qlm_s, slm_s, tlm_s)   # (3, nlm_src)

        # ---- STAGE 1: rotate the target center onto +z. #2: rotate all three scalar
        #      potentials in ONE vmapped FFI call over the stacked (3, nlm_src) array
        #      (was three separate apply_cplx_jax calls + per-component re-stacking). ----
        qstR = _pad3(jax.vmap(rot_fwd.apply_cplx_jax)(qst_src), nlm_e)   # (3, nlm_e)

        # ---- STAGE 2: per-ring radial scaling + latitude FFT + reframe to target-local
        #      spherical, one ring at a time, batched over rings with vmap (sh_eval grid) ----
        vlm, wlm, xlm = qst2vwx_cplx(qstR[0], qstR[1], qstR[2], sh_eval)  # rotated source VWX (nlm_e,)
        # #5: out_axes=1 places the mapped (ring) axis at position 1, so each output is built
        # directly as (nphi_e, ntheta_e) -- no post-hoc .T transpose of the ring-major result.
        vr_e, vt_e, vp_e = jax.vmap(
            _ps_ring_core_cplx,
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
        vlm_g, wlm_g, xlm_g = qst2vwx_cplx(qst_g[0], qst_g[1], qst_g[2], shtrg)
        return jnp.stack([vlm_g, wlm_g, xlm_g], axis=0)   # (3, nlm_t) target-basis VWX coeffs

    return jax.jit(_core)

def point_n_shoot_cplx(Strg: SphereDict, shtrg: shtns_jax.sht, sig_vwx: jax.Array, S: SphereDict, sh: shtns_jax.sht,
                  sl_scal: float, dl_scal: float, near: bool = False) -> jax.Array:
    """
    Point-and-shoot (move-pole) evaluation of the combined Stokes layer potential
        K = sl_scal*SL + dl_scal*DL
    of source sphere <S> (density VWX coefficients <sig_vwx>, shape (3, nlm)) at the surface
    grid of target sphere <Strg>.
    Thin wrapper over point_n_shoot_evaluator: fetches (or builds) the jitted evaluator for
    this geometry from a module cache, then applies it to <sig_vwx>. Repeated calls with
    the same geometry (e.g. every GMRES iteration of the suspension matvec) reuse the same
    compiled kernel. See point_n_shoot_evaluator for the algorithm and clear_point_n_shoot_cache
    to release cached kernels. Returns the velocity VWX coefficients on Strg's grid:
    (3, nlm_trg) complex.
    """
    key = _ps_geom_key(Strg, shtrg, S, sh) + ("cplx",)   # tag layout so real/cplx don't collide
    evaluator = _PS_EVALUATOR_CACHE.get(key)
    if evaluator is None:
        evaluator = point_n_shoot_evaluator_cplx(Strg, shtrg, S, sh, near=near)
        _PS_EVALUATOR_CACHE[key] = evaluator
    return evaluator(sig_vwx, sl_scal, dl_scal)

def _real_layout_remap(sh_from: shtns_jax.sht, sh_to: shtns_jax.sht) -> jax.Array:
    """Index map to reindex a real/truncated-layout coefficient vector from sh_from's (l,m)
    packing to sh_to's, by MATCHING (degree, order). Returns int64 (nlm_to,) with entries into
    sh_from's coeff array, or the value sh_from.nlm (a zero sentinel) where sh_to has an (l,m)
    mode absent from sh_from. The real layout is m-block ordered (per m: l=m..lmax), so growing
    or shrinking lmax is NOT a tail concat/slice -- this (l,m) match is required. Cached on
    sh_to, keyed by sh_from.lmax."""
    cache = getattr(sh_to, "_ps_real_remap", None)
    if cache is None:
        cache = {}; sh_to._ps_real_remap = cache
    key = int(sh_from.lmax)
    if key in cache:
        return cache[key]
    lf = np.asarray(sh_from.l, dtype=np.int64); mf = np.asarray(sh_from.m, dtype=np.int64)
    lt = np.asarray(sh_to.l, dtype=np.int64); mt = np.asarray(sh_to.m, dtype=np.int64)
    nfrom = int(sh_from.nlm)
    pos = {(int(l), int(m)): i for i, (l, m) in enumerate(zip(lf, mf))}
    idx = np.fromiter((pos.get((int(lt[j]), int(mt[j])), nfrom) for j in range(lt.shape[0])),
                      dtype=np.int64, count=lt.shape[0])
    out = jnp.asarray(idx)
    cache[key] = out
    return out

def _pad3_real(zzz: jax.Array, idx: jax.Array) -> jax.Array:
    """Reindex a stacked real-layout array (3, nlm_from) to (3, nlm_to) via idx from
    _real_layout_remap; the sentinel index nlm_from gathers a zero column. Handles zero-pad
    (grow lmax) and truncation (shrink lmax) correctly for the m-block real layout."""
    zc = jnp.concatenate([zzz, jnp.zeros((zzz.shape[0], 1), dtype=zzz.dtype)], axis=1)
    return zc[:, idx]

def point_n_shoot_evaluator(Strg: SphereDict, shtrg: shtns_jax.sht, S: SphereDict, sh: shtns_jax.sht,
                            near: bool = False):
    """
    Real/truncated-layout counterpart of point_n_shoot_evaluator_cplx. Same 3-stage
    move-pole algorithm (Corona & Veerapaneni 2018), but every coefficient array is the
    real (m>=0) layout of length nlm: rotations use rotation.apply_real_jax, degrees come
    from sh.l, the VWX<->QST changes of basis are the real qst2vwx/vwx2qst, the final
    analysis is analys_vec_jax, and the per-ring latitude synthesis uses the direct
    single-call _stk_latitude_ring (no complex G/H split / _latlm_maps). Returns a jitted
    apply(sig_vwx, sl_scal, dl_scal) -> (3, nlm_trg) target-basis VWX coefficients.

    Cross-lmax padding (source lmax != eval/target lmax) uses the (l,m)-matching
    _real_layout_remap / _pad3_real, NOT a tail concat -- the real layout is m-block ordered
    (unlike the cplx l(l+1)+m layout, where a tail concat is valid).
    """
    if S["lmax"] != sh.lmax:
        print("S lmax does not match sht's lmax, reform sht.")
        sh = shtns_jax.sht(S["lmax"], S["lmax"])
    if Strg["lmax"] != shtrg.lmax:
        print("Strg lmax does not match sht_trg's lmax, reform sht_trg.")
        shtrg = shtns_jax.sht(Strg["lmax"], Strg["lmax"])

    a = float(S["r"]); R_t = float(Strg["r"])
    t_vec = jnp.asarray(Strg["Xc"], dtype=jnp.float64) - jnp.asarray(S["Xc"], dtype=jnp.float64)
    sh_eval = sh if sh.lmax >= shtrg.lmax else shtrg
    rot_fwd, rot_inv, d = _ps_rotation(t_vec, sh.lmax, sh_eval.lmax)

    if abs(d - R_t) > a:
        exterior = True
    elif d + R_t < a:
        exterior = False
    else:
        raise ValueError(
            f"point_n_shoot requires the target sphere to be wholly exterior or "
            f"interior to the source (non-overlapping); got d={d}, a={a}, R_t={R_t}.")

    # ---- geometry-only constants (computed once; the jitted core closes over them) ----
    nlm_e = sh_eval.nlm; nlm_t = shtrg.nlm
    l_vals = jnp.asarray(sh_eval.l, dtype=jnp.float64)         # (nlm_e,) real-layout degrees
    theta_e = jnp.arccos(jnp.asarray(sh_eval.cos_theta))       # (ntheta_e,)
    r_j, cos_src = _ps_target_rings(d, R_t, theta_e)           # each (ntheta_e,)
    rho_ring = r_j / a                                         # (ntheta_e,) source-centered r/a
    theta_src_ring = jnp.arccos(cos_src)                       # (ntheta_e,) source-centered theta
    th_e_ring = theta_e                                        # (ntheta_e,) target-local theta
    ph_ring_1d = jnp.arange(sh_eval.nphi) * (2.0*jnp.pi/sh_eval.nphi)  # (nphi_e,) longitudes
    if exterior:
        pw_V_ring = rho_ring[:, None] ** (-l_vals - 2.0)       # (ntheta_e, nlm_e)
    else:
        pw_V_ring = rho_ring[:, None] ** (l_vals + 1.0)
    sl_coeffs = _ps_scale_coeffs(sh_eval, exterior, "sl")
    dl_coeffs = _ps_scale_coeffs(sh_eval, exterior, "dl")

    # (l,m)-matching reindex maps for the real layout (precomputed once, closed over by _core):
    #   stage 1: rotated source coeffs (nlm of sh) -> eval grid layout (nlm_e)
    #   stage 3: rotated eval coeffs (nlm_e) -> target layout (nlm_t)
    idx_src2eval = _real_layout_remap(sh, sh_eval)     # (nlm_e,)
    idx_eval2trg = _real_layout_remap(sh_eval, shtrg)  # (nlm_t,)

    def _core(sig_vwx: jax.Array, sl_scal, dl_scal) -> jax.Array:
        # ---- source density VWX coefficients -> Q/S/T coefficients (real layout) ----
        qlm_s, slm_s, tlm_s = vwx2qst(sig_vwx[0], sig_vwx[1], sig_vwx[2], sh)
        qst_src = vec_stack(qlm_s, slm_s, tlm_s)   # (3, nlm_src)

        # ---- STAGE 1: rotate the target center onto +z (real-layout Wigner-D apply) ----
        qstR = _pad3_real(jax.vmap(rot_fwd.apply_real_jax)(qst_src), idx_src2eval)   # (3, nlm_e)

        # ---- STAGE 2: per-ring radial scaling + latitude FFT + reframe to target-local ----
        vlm, wlm, xlm = qst2vwx(qstR[0], qstR[1], qstR[2], sh_eval)  # rotated source VWX (nlm_e,)
        vr_e, vt_e, vp_e = jax.vmap(
            _ps_ring_core,
            in_axes=(None, None, None, 0, 0, 0, 0, 0,
                     None, None, None, None, None, None, None, None),
            out_axes=1)(
            vlm, wlm, xlm, pw_V_ring, rho_ring, cos_src, theta_src_ring, th_e_ring,
            ph_ring_1d, exterior, a, sl_scal, dl_scal, sh_eval,
            sl_coeffs, dl_coeffs)

        # ---- STAGE 3: rotate the sampled field back, band-limit to the target grid. ----
        qst_R = sh_eval.analys_vec_jax(vec_stack(vr_e, vt_e, vp_e))   # (3, nlm_e)
        qst_g = _pad3_real(jax.vmap(rot_inv.apply_real_jax)(qst_R), idx_eval2trg)  # (3, nlm_t)
        vlm_g, wlm_g, xlm_g = qst2vwx(qst_g[0], qst_g[1], qst_g[2], shtrg)
        return jnp.stack([vlm_g, wlm_g, xlm_g], axis=0)   # (3, nlm_t) target-basis VWX coeffs

    return jax.jit(_core)

def point_n_shoot(Strg: SphereDict, shtrg: shtns_jax.sht, sig_vwx: jax.Array, S: SphereDict, sh: shtns_jax.sht,
                  sl_scal: float, dl_scal: float, near: bool = False) -> jax.Array:
    """Real/truncated-layout point-and-shoot (see point_n_shoot_cplx for the algorithm).
    Density coefficients <sig_vwx> are the real (m>=0) layout, shape (3, nlm). Returns the
    velocity VWX coefficients on Strg's grid: (3, nlm_trg)."""
    key = _ps_geom_key(Strg, shtrg, S, sh) + ("real",)   # tag layout so real/cplx don't collide
    evaluator = _PS_EVALUATOR_CACHE.get(key)
    if evaluator is None:
        evaluator = point_n_shoot_evaluator(Strg, shtrg, S, sh, near=near)
        _PS_EVALUATOR_CACHE[key] = evaluator
    return evaluator(sig_vwx, sl_scal, dl_scal)

def Stk3d_dl_point_and_shoot(Strg: SphereDict, shtrg: shtns_jax.sht, sig_vwx: jax.Array, S: SphereDict, sh: shtns_jax.sht) -> jax.Array:
    """DL-only real/truncated-layout point-and-shoot (thin wrapper over point_n_shoot)."""
    return point_n_shoot(Strg, shtrg, sig_vwx, S, sh, 0.0, 1.0, near=False)

def Stk3d_dl_point_and_shoot_cplx(Strg: SphereDict, shtrg: shtns_jax.sht, sig_vwx: jax.Array, S: SphereDict, sh: shtns_jax.sht) -> jax.Array:
    """DL-only point-and-shoot (thin wrapper over point_n_shoot)."""
    return point_n_shoot_cplx(Strg, shtrg, sig_vwx, S, sh, 0.0, 1.0, near=False)




# ============================
# === Eval ===================
# ============================
def _stk_trg_sph(trg: jax.Array, S: SphereDict) -> tuple([jax.Array, jax.Array, jax.Array]):
    """Spherical coordinates (dr, theta, phi) of targets <trg> relative to S["Xc"]."""
    trg_dx = trg[:,0] - S["Xc"][0]
    trg_dy = trg[:,1] - S["Xc"][1]
    trg_dz = trg[:,2] - S["Xc"][2]
    trg_dr = jnp.sqrt(trg_dx*trg_dx + trg_dy*trg_dy + trg_dz*trg_dz)
    trg_phi = jnp.atan2(trg_dy, trg_dx)
    trg_theta = jnp.acos(trg_dz / trg_dr)
    return trg_dr, trg_theta, trg_phi

NEAR_TILE_BYTES = 24 << 20   # memory budget for ONE (tile, nlm_cplx) complex128 block
_NEAR_PAD = 64               # chunk lengths are rounded up to this, to bucket compiled shapes

# Canonical sht per (lmax, zl), so that spheres sharing a resolution share ONE compiled kernel.
_NEAR_SH_CANON: Dict[tuple, Any] = {}

def _near_sh_canon(sh: shtns_jax.sht) -> shtns_jax.sht:
    """The sht that _stk_near_kernel should be keyed on. That kernel is a static_argnames jit,
    so it recompiles per sht OBJECT -- and quadr_suspension builds a separate sht per sphere,
    so a suspension whose spheres share an lmax would compile it once per sphere (measured: 3.64 s
    vs 0.65 s of first-call cost for a 5-sphere, 64-target evaluation). But the kernel reads
    NOTHING from sh except lmax and zl -- directly, and through vwx2qst and the *_VWX_diag /
    diag_W2V / diag_V2W symbol helpers -- and makes no FFI call, so (lmax, zl) fully determines
    the compiled code and equivalent shts can share it. (_stk_far_kernel cannot do this: it calls
    sh.synth_vec_cplx_jax, which is bound to its own sht.)"""
    key = getattr(sh, "_stk_near_sh_key", None)
    if key is None:
        key = (int(sh.lmax), np.asarray(sh.zl, dtype=np.int64).tobytes())
        sh._stk_near_sh_key = key
    return _NEAR_SH_CANON.setdefault(key, sh)

def _stk_near_tile(sh: shtns_jax.sht, tile: int = None) -> int:
    """Targets per chunk of the near evaluation, always a multiple of _NEAR_PAD (so that the
    padded target list splits into chunks of reusable shapes). The radially scaled coefficients
    are (tile, nlm_cplx) complex and several are live at once -- 47 MB per array per 2800 targets
    at lmax 32, 571 MB at lmax 64 -- so the chunk is set from a memory budget. (Unlike the far
    kernel's tile, which is picked for gemm blocking on pair-shaped arrays; here the per-target
    work is elementwise + a gather, so only memory matters.)"""
    if tile is None:
        tile = NEAR_TILE_BYTES // (16 * max(int(sh.nlm_cplx), 1))
    return int(max(1, int(tile) // _NEAR_PAD)) * _NEAR_PAD

def _stk_near_symbols_cplx(sh: shtns_jax.sht, sl_scal, dl_scal, a, exterior: bool) -> tuple:
    """Radius-independent VWX diagonal + coupling symbols of the COMBINED operator
    K = sl_scal*SL + dl_scal*DL, on the exterior or interior radial branch (complex/full
    layout, length nlm_cplx). The scaling is linear in these symbols, so pre-combining them
    here lets the per-target radial scaling run ONCE instead of once per operator.
    <a> rides along in the SL weight (the SL block carries the source radius; DL is
    scale-invariant). Returns (diag_V, diag_W, diag_X, coup).

    Note the interior V->W coupling enters with OPPOSITE signs for the two operators
    (rho^{l+1} - rho^{l-1} for SL, its negative for DL), so only <coup> takes the sign --
    never the three diagonals."""
    diag_V_sl, diag_W_sl, diag_X_sl = Stk3d_sl_VWX_diag_cplx(sh)
    dVe, dWe, dXe, dVi, dWi, dXi = Stk3d_dl_VWX_diag_cplx(sh)
    s = sl_scal * a
    if exterior:
        return (s * diag_V_sl + dl_scal * dVe,
                s * diag_W_sl + dl_scal * dWe,
                s * diag_X_sl + dl_scal * dXe,
                s * diag_W2V_cplx(sh, "sl") + dl_scal * diag_W2V_cplx(sh, "dl"))
    return (s * diag_V_sl + dl_scal * dVi,
            s * diag_W_sl + dl_scal * dWi,
            s * diag_X_sl + dl_scal * dXi,
            s * diag_V2W_cplx(sh, "sl") - dl_scal * diag_V2W_cplx(sh, "dl"))

def _stk_near_symbols(sh: shtns_jax.sht, sl_scal, dl_scal, a, exterior: bool) -> tuple:
    """Real/truncated-layout counterpart of _stk_near_symbols_cplx (symbols length nlm, from
    the real diag/coupling twins). Returns (diag_V, diag_W, diag_X, coup)."""
    diag_V_sl, diag_W_sl, diag_X_sl = Stk3d_sl_VWX_diag(sh)
    dVe, dWe, dXe, dVi, dWi, dXi = Stk3d_dl_VWX_diag(sh)
    s = sl_scal * a
    if exterior:
        return (s * diag_V_sl + dl_scal * dVe,
                s * diag_W_sl + dl_scal * dWe,
                s * diag_X_sl + dl_scal * dXe,
                s * diag_W2V(sh, "sl") + dl_scal * diag_W2V(sh, "dl"))
    return (s * diag_V_sl + dl_scal * dVi,
            s * diag_W_sl + dl_scal * dWi,
            s * diag_X_sl + dl_scal * dXi,
            s * diag_V2W(sh, "sl") - dl_scal * diag_V2W(sh, "dl"))

@partial(jax.jit, static_argnames=["sh"])
def _stk_near_kernel_cplx(trg_dr: jax.Array, sig_vwx: jax.Array, sl_scal, dl_scal, a,
                     sh: shtns_jax.sht) -> tuple:
    """
    Per-target solid-harmonic scaling of the combined K = sl_scal*SL + dl_scal*DL applied to
    the density VWX coefficients <sig_vwx> ((3, nlm)), for targets at source-centred radii
    <trg_dr> (Ntrg,). Returns the per-target Q/S/T coefficients (each Ntrg x nlm_cplx) ready
    for point synthesis. The near counterpart of _stk_far_kernel; two structural savings over
    the previous per-operator Stk3d_sl / Stk3d_dl:

    - ONE scaling pass instead of two. _stk_near_symbols pre-combines the SL and DL symbols
      per radial branch (the scaling is linear in them), so the ~14 (Ntrg, nlm) temporaries
      per operator collapse to one fused chain.
    - O(lmax) transcendental powers per target instead of O(nlm). The solid-harmonic radial
      power depends only on the DEGREE l, so the leading power is evaluated over degrees
      0..lmax and gathered onto the mode axis (a mode of degree l gets the same float exponent
      either way), and the other two follow by multiplies: rho^{-l} = rho^{-l-2} rho^2 etc.
      Two power arrays in total, where the old pair of kernels evaluated 16 (Ntrg, nlm) ones.

    Both radial branches are still evaluated and selected with one jnp.where on trg_dr > a,
    exactly as before, so a target list may straddle the source radius.

    Accuracy vs the old pair of kernels: <= 1.5e-15 on every geometry tested except a
    deep-interior one (source radius 3, rho down to 0.03), where it is 3e-14. That outlier is
    the <a> fold, not the fused symbols or the derived powers: rescaling the same geometry to
    a = 1, which makes the fold exact, drops it to 9.8e-16 (the derived powers alone contribute
    ~5e-16). It is a rounding-order change on an interior expansion spanning ~25 decades, i.e.
    an intrinsically ill-conditioned synthesis -- on that case old and new sit equidistant
    (6.02e-13) from the smooth-quadrature reference, so the difference is far below the method's
    own error there, and at higher lmax the new result is marginally closer.
    """
    vlm, wlm, xlm = sig_vwx[0], sig_vwx[1], sig_vwx[2]
    rho = (trg_dr / a)[:, None]                                  # (Ntrg, 1)
    zl = jnp.asarray(sh.zl, dtype=jnp.int32)                     # degree of each cplx mode
    deg = jnp.arange(int(sh.lmax) + 1, dtype=jnp.float64)        # 0 .. lmax

    pV_ext = (rho ** (-deg - 2.0))[:, zl]                        # rho^{-l-2}
    pW_ext = pV_ext * rho * rho                                  # rho^{-l}
    pX_ext = pV_ext * rho                                        # rho^{-l-1}
    pV_int = (rho ** (deg + 1.0))[:, zl]                         # rho^{l+1}
    pW_int = pV_int / (rho * rho)                                # rho^{l-1}
    pX_int = pV_int / rho                                        # rho^{l}

    dV_e, dW_e, dX_e, coup_e = _stk_near_symbols_cplx(sh, sl_scal, dl_scal, a, True)
    dV_i, dW_i, dX_i, coup_i = _stk_near_symbols_cplx(sh, sl_scal, dl_scal, a, False)
    v_ext = pV_ext * dV_e * vlm + (pV_ext - pW_ext) * coup_e * wlm     # W -> V coupling
    w_ext = pW_ext * dW_e * wlm
    x_ext = pX_ext * dX_e * xlm
    v_int = pV_int * dV_i * vlm
    w_int = pW_int * dW_i * wlm + (pV_int - pW_int) * coup_i * vlm     # V -> W coupling
    x_int = pX_int * dX_i * xlm

    is_ext = (trg_dr > a)[:, None]
    return vwx2qst_cplx(jnp.where(is_ext, v_ext, v_int),
                   jnp.where(is_ext, w_ext, w_int),
                   jnp.where(is_ext, x_ext, x_int), sh)

@partial(jax.jit, static_argnames=["sh"])
def _stk_near_kernel(trg_dr: jax.Array, sig_vwx: jax.Array, sl_scal, dl_scal, a,
                     sh: shtns_jax.sht) -> tuple:
    """Real/truncated-layout counterpart of _stk_near_kernel_cplx. Same per-target
    solid-harmonic scaling of K = sl_scal*SL + dl_scal*DL on the density VWX coefficients
    <sig_vwx> ((3, nlm), real layout), but the per-mode degree is sh.l, the symbols come from
    the real _stk_near_symbols, and the VWX->QST change of basis is the real vwx2qst. Returns
    the per-target real-layout Q/S/T coefficients (each Ntrg x nlm) for real point synthesis."""
    vlm, wlm, xlm = sig_vwx[0], sig_vwx[1], sig_vwx[2]
    rho = (trg_dr / a)[:, None]                                  # (Ntrg, 1)
    lr = jnp.asarray(sh.l, dtype=jnp.int32)                      # degree of each real mode
    deg = jnp.arange(int(sh.lmax) + 1, dtype=jnp.float64)        # 0 .. lmax

    pV_ext = (rho ** (-deg - 2.0))[:, lr]                        # rho^{-l-2}
    pW_ext = pV_ext * rho * rho                                  # rho^{-l}
    pX_ext = pV_ext * rho                                        # rho^{-l-1}
    pV_int = (rho ** (deg + 1.0))[:, lr]                         # rho^{l+1}
    pW_int = pV_int / (rho * rho)                                # rho^{l-1}
    pX_int = pV_int / rho                                        # rho^{l}

    dV_e, dW_e, dX_e, coup_e = _stk_near_symbols(sh, sl_scal, dl_scal, a, True)
    dV_i, dW_i, dX_i, coup_i = _stk_near_symbols(sh, sl_scal, dl_scal, a, False)
    v_ext = pV_ext * dV_e * vlm + (pV_ext - pW_ext) * coup_e * wlm     # W -> V coupling
    w_ext = pW_ext * dW_e * wlm
    x_ext = pX_ext * dX_e * xlm
    v_int = pV_int * dV_i * vlm
    w_int = pW_int * dW_i * wlm + (pV_int - pW_int) * coup_i * vlm     # V -> W coupling
    x_int = pX_int * dX_i * xlm

    is_ext = (trg_dr > a)[:, None]
    return vwx2qst(jnp.where(is_ext, v_ext, v_int),
                   jnp.where(is_ext, w_ext, w_int),
                   jnp.where(is_ext, x_ext, x_int), sh)

def _stk_near_cplx(trg: jax.Array, sig_vwx: jax.Array, S: SphereDict, sh: shtns_jax.sht,
              sl_scal=0.0, dl_scal=0.0, tile: int = None) -> jax.Array:
    """
    Fused spectral ("near") evaluation of K = sl_scal*SL + dl_scal*DL at arbitrary targets
    <trg> (Ntrg x 3) from source <S> with density VWX coefficients <sig_vwx> ((3, nlm)).
    Returns the Cartesian velocity: Ntrg x 3. Accurate for targets close to the surface,
    where the smooth quadrature (_stk_far) is not; eager wrapper of _stk_near_kernel.

    The point synthesis itself is the irreducibly eager part: sh.SHqst_to_point_cplx is a
    synchronous C evaluator and every target carries its OWN scaled coefficients, so it is
    called once per target. Everything around that call is hoisted out of the loop -- the
    scaling into the jitted kernel, the spherical->Cartesian rotation into a single vectorized
    sph2cart. That matters more than it looks: previously sph2cart (a jnp function) was called
    INSIDE the loop and the result assembled with jnp.array(list_of_tuples), so ~92% of the
    loop's wall time was per-target JAX dispatch and only ~1.5% was the C synthesis.

    Targets are processed in chunks (see _stk_near_tile) to bound the (Ntrg, nlm) coefficient
    blocks. The target list is first padded up to a multiple of _NEAR_PAD, so that EVERY
    shape downstream -- the geometry, the jitted kernel, the closing sph2cart -- is reused
    across nearby target counts. That matters because those surrounding ops run eagerly, and
    eager JAX compiles each primitive per shape: with an unbucketed count, a caller whose
    target count varies (bio_offsurf_apply(far=None), whose near subset follows the geometry)
    pays a few hundred ms of warm-up on every call. Cost of the padding: at most _NEAR_PAD-1
    wasted rows of scaling (the C synthesis is never run on them).

    Measured 13-24x faster than the previous per-operator Stk3d_sl + Stk3d_dl pair (lmax 16-36,
    5 geometries), and 4.6-8.0x on the off-surface field evaluation of the container-obstacle
    benchmark.

    NOTE for later -- there is another ~5x sitting inside SHqst_to_point_cplx that this does not
    take. Per call (shtns_jax.py:169) it re-derives sin(th)dS/dth, sin(th)dT/dth, i m S and i m T
    from sh._dtheta_cplx_op()'s cached gather indices, then makes 5 scalar SH_to_point_cplx C
    calls. Those derivations are fixed-index gathers and constant multiplies, linear along nlm
    and independent of the target, so they could be vectorized into _stk_near_kernel over the
    whole (Ntrg, nlm) block, leaving only the raw C calls in the loop: measured 5.5x (lmax 32) to
    7.3x (lmax 16) on that stage alone. Doing so means duplicating the four vt/vp assembly lines
    here, so it is better done behind a public batched entry point in shtns_jax (see the module
    TODO) than by reaching into the underscore helper from this file.
    """
    assert trg.shape[1] == 3
    if S["lmax"] != sh.lmax:
        print("S lmax does not match sht's lmax, reform sht.")
        sh = shtns_jax.sht(S["lmax"], S["lmax"])

    Nt = trg.shape[0]
    if Nt == 0:
        return jnp.zeros((0, 3), dtype=jnp.complex128)
    a = jnp.asarray(S["r"], dtype=jnp.float64)
    tile = _stk_near_tile(sh, tile)          # a multiple of _NEAR_PAD

    n_tot = -(-Nt // _NEAR_PAD) * _NEAR_PAD  # bucketed length; every chunk below is a
    if n_tot > Nt:                           # multiple of _NEAR_PAD, so shapes are reused
        trg = jnp.concatenate([trg, jnp.broadcast_to(trg[-1], (n_tot - Nt, 3))])
    trg_dr, trg_theta, trg_phi = _stk_trg_sph(trg, S)

    # Host-side geometry for the C loop. As before, everything the loop touches is
    # materialized up front: per-element transfers inside the loop race the synchronous C
    # call and can segfault.
    cost_h = np.cos(np.asarray(trg_theta, dtype=np.float64))
    phi_h = np.asarray(trg_phi, dtype=np.float64)
    vr = np.zeros(n_tot, dtype=np.complex128)
    vt = np.zeros(n_tot, dtype=np.complex128)
    vp = np.zeros(n_tot, dtype=np.complex128)

    sh_k = _near_sh_canon(sh)      # jit key only; the C synthesis below uses the caller's sh
    for start in range(0, n_tot, tile):
        stop = min(start + tile, n_tot)
        qlm, slm, tlm = _stk_near_kernel_cplx(trg_dr[start:stop], sig_vwx,
                                         sl_scal, dl_scal, a, sh=sh_k)
        qn = np.asarray(qlm, dtype=np.complex128)
        sn = np.asarray(slm, dtype=np.complex128)
        tn = np.asarray(tlm, dtype=np.complex128)
        for i in range(min(stop, Nt) - start):        # skip the padding rows
            j = start + i
            vr[j], vt[j], vp[j] = sh.SHqst_to_point_cplx(
                qn[i], sn[i], tn[i], float(cost_h[j]), float(phi_h[j]))

    vx, vy, vz = sph2cart(jnp.asarray(vr), jnp.asarray(vt), jnp.asarray(vp),
                          trg_theta, trg_phi)
    return jnp.stack([vx, vy, vz], axis=1)[:Nt]

def _stk_near(trg: jax.Array, sig_vwx: jax.Array, S: SphereDict, sh: shtns_jax.sht,
              sl_scal=0.0, dl_scal=0.0, tile: int = None) -> jax.Array:
    """Real/truncated-layout counterpart of _stk_near_cplx. The density VWX coefficients
    <sig_vwx> are the real (m>=0) layout ((3, nlm)); the per-target scaling runs in the real
    layout (_stk_near_kernel), and the per-point synthesis uses the base real-layout vector
    point evaluator sh.SHqst_to_point (real Q/S/T -> real vr/vt/vp), so no complex embed is
    needed. Returns the Cartesian velocity (Ntrg x 3, float64)."""
    assert trg.shape[1] == 3
    if S["lmax"] != sh.lmax:
        print("S lmax does not match sht's lmax, reform sht.")
        sh = shtns_jax.sht(S["lmax"], S["lmax"])

    Nt = trg.shape[0]
    if Nt == 0:
        return jnp.zeros((0, 3), dtype=jnp.float64)
    a = jnp.asarray(S["r"], dtype=jnp.float64)
    tile = _stk_near_tile(sh, tile)          # a multiple of _NEAR_PAD

    n_tot = -(-Nt // _NEAR_PAD) * _NEAR_PAD  # bucketed length; every chunk is a multiple of _NEAR_PAD
    if n_tot > Nt:
        trg = jnp.concatenate([trg, jnp.broadcast_to(trg[-1], (n_tot - Nt, 3))])
    trg_dr, trg_theta, trg_phi = _stk_trg_sph(trg, S)

    cost_h = np.cos(np.asarray(trg_theta, dtype=np.float64))
    phi_h = np.asarray(trg_phi, dtype=np.float64)
    vr = np.zeros(n_tot, dtype=np.float64)
    vt = np.zeros(n_tot, dtype=np.float64)
    vp = np.zeros(n_tot, dtype=np.float64)

    sh_k = _near_sh_canon(sh)      # jit key only; the C synthesis below uses the caller's sh
    for start in range(0, n_tot, tile):
        stop = min(start + tile, n_tot)
        qlm, slm, tlm = _stk_near_kernel(trg_dr[start:stop], sig_vwx,
                                         sl_scal, dl_scal, a, sh=sh_k)
        qn = np.asarray(qlm, dtype=np.complex128)   # real-layout Q/S/T (m>=0), still complex128
        sn = np.asarray(slm, dtype=np.complex128)
        tn = np.asarray(tlm, dtype=np.complex128)
        for i in range(min(stop, Nt) - start):        # skip the padding rows
            j = start + i
            vr[j], vt[j], vp[j] = sh.SHqst_to_point(
                qn[i], sn[i], tn[i], float(cost_h[j]), float(phi_h[j]))

    vx, vy, vz = sph2cart(jnp.asarray(vr), jnp.asarray(vt), jnp.asarray(vp),
                          trg_theta, trg_phi)
    return jnp.stack([vx, vy, vz], axis=1)[:Nt]

def Stk3d_sl(trg: jax.Array, sig_vwx: jax.Array, S: SphereDict, sh: shtns_jax.sht) -> jax.Array:
    """
    Off-surface evaluation of the Stokes SL velocity at arbitrary points <trg>: Ntrg x 3
        From source <S> with density VWX coefficients <sig_vwx> ((3, nlm)), source uses <sh>.
    Spectral solid-harmonic scaling, but per-target
    radius and point synthesis.  Returns Ntrg x 3.
    Thin wrapper over the fused _stk_near kernel (see there for the scaling).
    """
    assert trg.shape[1] == 3
    return _stk_near(trg, sig_vwx, S, sh, sl_scal=1.0, dl_scal=0.0)

def Stk3d_dl(trg: jax.Array, sig_vwx: jax.Array, S: SphereDict, sh: shtns_jax.sht) -> jax.Array:
    """
    Off-surface evaluation of the Stokes DL velocity at arbitrary points <trg>: Ntrg x 3
        From source <S> with density VWX coefficients <sig_vwx> ((3, nlm)), source uses <sh>.
    Spectral solid-harmonic scaling, but per-target
    radius and point synthesis.  Returns Ntrg x 3.
    Thin wrapper over the fused _stk_near kernel (see there for the scaling).
    """
    assert trg.shape[1] == 3
    return _stk_near(trg, sig_vwx, S, sh, sl_scal=0.0, dl_scal=1.0)

def bio_offsurf_apply_cplx(trg: jax.Array, sig_vwx: jax.Array, S: SphereDict, sh: shtns_jax.sht, sl_scal: float, dl_scal: float, far: bool = None, sep_eta: float = 1e-1) -> jax.Array:
    """
    Complex/full-layout off-surface evaluation of the KL formulation of <S> with density VWX
    coefficients <sig_vwx> ((3, nlm_cplx)) at arbitrary target <trg>: Ntrg x 3.

    Routes each target to the accuracy-appropriate kernel:
        far == None (default): split trg into far/near via separate_target and evaluate each
                               kernel on ITS OWN targets (far: smooth quadrature; near: spectral).
        far == True          : force the smooth quadrature for every target.
        far == False         : force the spectral synthesis for every target.
    """
    if far is True:
        return _stk_far_cplx(trg, sig_vwx, S, sh, ("sl", "dl"), sl_scal=sl_scal, dl_scal=dl_scal)
    if far is False:
        return _stk_near_cplx(trg, sig_vwx, S, sh, sl_scal=sl_scal, dl_scal=dl_scal)

    sep_far = np.asarray(separate_target(trg, S, sep_eta))
    idx_far = np.flatnonzero(sep_far)
    idx_near = np.flatnonzero(~sep_far)
    Ksigma = jnp.zeros((trg.shape[0], 3), dtype=jnp.complex128)
    if idx_far.size:
        Ksigma = Ksigma.at[idx_far].set(
            _stk_far_cplx(trg[idx_far], sig_vwx, S, sh, ("sl", "dl"), sl_scal=sl_scal, dl_scal=dl_scal))
    if idx_near.size:
        Ksigma = Ksigma.at[idx_near].set(
            _stk_near_cplx(trg[idx_near], sig_vwx, S, sh, sl_scal=sl_scal, dl_scal=dl_scal))
    return Ksigma

def bio_offsurf_apply(trg: jax.Array, sig_vwx: jax.Array, S: SphereDict, sh: shtns_jax.sht, sl_scal: float, dl_scal: float, far: bool = None, sep_eta: float = 1e-1) -> jax.Array:
    """
    Real/truncated-layout off-surface evaluation. Density VWX coefficients <sig_vwx> are the
    real (m>=0) layout ((3, nlm)); targets <trg>: Ntrg x 3; returns point values (Ntrg x 3,
    float64).

    Both branches are now NATIVE real: the FAR (smooth-quadrature) branch uses real _stk_far,
    and the NEAR (spectral) branch uses real _stk_near -- its per-point synthesis is the base
    real-layout sh.SHqst_to_point, so no real->complex embed is needed. far routing as in the
    cplx twin.
    """
    if far is True:
        return _stk_far(trg, sig_vwx, S, sh, ("sl", "dl"), sl_scal=sl_scal, dl_scal=dl_scal)
    if far is False:
        return _stk_near(trg, sig_vwx, S, sh, sl_scal=sl_scal, dl_scal=dl_scal)

    sep_far = np.asarray(separate_target(trg, S, sep_eta))
    idx_far = np.flatnonzero(sep_far)
    idx_near = np.flatnonzero(~sep_far)
    Ksigma = jnp.zeros((trg.shape[0], 3), dtype=jnp.float64)
    if idx_far.size:
        Ksigma = Ksigma.at[idx_far].set(
            _stk_far(trg[idx_far], sig_vwx, S, sh, ("sl", "dl"), sl_scal=sl_scal, dl_scal=dl_scal))
    if idx_near.size:
        Ksigma = Ksigma.at[idx_near].set(
            _stk_near(trg[idx_near], sig_vwx, S, sh, sl_scal=sl_scal, dl_scal=dl_scal))
    return Ksigma

@partial(jax.jit, static_argnames=["sh"])
def bio_onsurf_apply_cplx(sig_vwx: jax.Array, sh: shtns_jax.sht, sl_scal: float, dl_scal: float, sgn: float, dsl_scal: float = 0., radius: float = 1.0) -> jax.Array:
    """
    Apply the combined DL potential operator K = sl_scal * SL + dl_scal * DL
        to the density with VWX (diagonalizing-basis) coefficients <sig_vwx> ((3, nlm))
        taking into account the DL jump condition with sign <sgn>.
    Returns the resulting function's VWX coefficients ((3, nlm)), so the on-surface
    self-apply is a pure per-component diagonal multiply (COB is the caller's responsibility).
    The SL block scales by the source-sphere <radius> (DL/dSL/jump are scale-invariant).
    """

    vlm, wlm, xlm = sig_vwx[0], sig_vwx[1], sig_vwx[2]

    diag_V, diag_W, diag_X = Stk3d_sl_VWX_diag_cplx(sh)
    vlm_SL_sigma = radius * diag_V * vlm
    wlm_SL_sigma = radius * diag_W * wlm
    xlm_SL_sigma = radius * diag_X * xlm

    diag_V_ext, diag_W_ext, diag_X_ext, diag_V_int, diag_W_int, diag_X_int = Stk3d_dl_VWX_diag_cplx(sh)
    vlm_DL_sigma = 0.5*(diag_V_int + diag_V_ext) * vlm
    wlm_DL_sigma = 0.5*(diag_W_int + diag_W_ext) * wlm
    xlm_DL_sigma = 0.5*(diag_X_int + diag_X_ext) * xlm

    diag_V_ext, diag_W_ext, diag_X_ext, diag_V_int, diag_W_int, diag_X_int = Stk3d_dsl_VWX_diag_cplx(sh)
    vlm_dSL_sigma = 0.5*(diag_V_int + diag_V_ext) * vlm
    wlm_dSL_sigma = 0.5*(diag_W_int + diag_W_ext) * wlm
    xlm_dSL_sigma = 0.5*(diag_X_int + diag_X_ext) * xlm

    # The jump condition is a multiple of the identity, hence diagonal in the VWX basis too.
    jump = 0.5 * sgn * dl_scal + 0.5 * (-1*sgn) * dsl_scal
    vlm_op = sl_scal * vlm_SL_sigma + dl_scal * vlm_DL_sigma + dsl_scal * vlm_dSL_sigma + jump * vlm
    wlm_op = sl_scal * wlm_SL_sigma + dl_scal * wlm_DL_sigma + dsl_scal * wlm_dSL_sigma + jump * wlm
    xlm_op = sl_scal * xlm_SL_sigma + dl_scal * xlm_DL_sigma + dsl_scal * xlm_dSL_sigma + jump * xlm

    return jnp.stack([vlm_op, wlm_op, xlm_op], axis=0)

@partial(jax.jit, static_argnames=["sh"])
def stokes_onsurf_direct_solve_cplx(sig_vwx_bc: jax.Array, sh: shtns_jax.sht, sl_scal: float, dl_scal: float, sgn: float, dsl_scal: float = 0., radius: float = 1.0) -> jax.Array:
    """
    Directly solves the Stokes BIO equation using the VWX diagonal property. The right-hand
    side and returned density are both given by their VWX (diagonalizing-basis) coefficients
    (<sig_vwx_bc> in, sig_vwx out, each (3, nlm)); the solve is a per-component diagonal division.
    The SL block scales by the source-sphere <radius> (DL/dSL/jump are scale-invariant).
    """

    vlm_bc, wlm_bc, xlm_bc = sig_vwx_bc[0], sig_vwx_bc[1], sig_vwx_bc[2]
    diag_V_sl, diag_W_sl, diag_X_sl = Stk3d_sl_VWX_diag_cplx(sh)
    diag_V_ext, diag_W_ext, diag_X_ext, diag_V_int, diag_W_int, diag_X_int = Stk3d_dl_VWX_diag_cplx(sh)
    diag_V_dl = 0.5 * (diag_V_int + diag_V_ext)
    diag_W_dl = 0.5 * (diag_W_int + diag_W_ext)
    diag_X_dl = 0.5 * (diag_X_int + diag_X_ext)
    diag_V_ext, diag_W_ext, diag_X_ext, diag_V_int, diag_W_int, diag_X_int = Stk3d_dsl_VWX_diag_cplx(sh)
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

    return jnp.stack([vlm_sigma, wlm_sigma, xlm_sigma], axis=0)

@partial(jax.jit, static_argnames=["sh"])
def bio_onsurf_apply(sig_vwx: jax.Array, sh: shtns_jax.sht, sl_scal: float, dl_scal: float, sgn: float, dsl_scal: float = 0., radius: float = 1.0) -> jax.Array:
    """
    Apply the combined DL potential operator K = sl_scal * SL + dl_scal * DL
        to the density with VWX (diagonalizing-basis) coefficients <sig_vwx> ((3, nlm))
        taking into account the DL jump condition with sign <sgn>.
    Returns the resulting function's VWX coefficients ((3, nlm)), so the on-surface
    self-apply is a pure per-component diagonal multiply (COB is the caller's responsibility).
    The SL block scales by the source-sphere <radius> (DL/dSL/jump are scale-invariant).
    """

    vlm, wlm, xlm = sig_vwx[0], sig_vwx[1], sig_vwx[2]

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

    # The jump condition is a multiple of the identity, hence diagonal in the VWX basis too.
    jump = 0.5 * sgn * dl_scal + 0.5 * (-1*sgn) * dsl_scal
    vlm_op = sl_scal * vlm_SL_sigma + dl_scal * vlm_DL_sigma + dsl_scal * vlm_dSL_sigma + jump * vlm
    wlm_op = sl_scal * wlm_SL_sigma + dl_scal * wlm_DL_sigma + dsl_scal * wlm_dSL_sigma + jump * wlm
    xlm_op = sl_scal * xlm_SL_sigma + dl_scal * xlm_DL_sigma + dsl_scal * xlm_dSL_sigma + jump * xlm

    return jnp.stack([vlm_op, wlm_op, xlm_op], axis=0)

@partial(jax.jit, static_argnames=["sh"])
def stokes_onsurf_direct_solve(sig_vwx_bc: jax.Array, sh: shtns_jax.sht, sl_scal: float, dl_scal: float, sgn: float, dsl_scal: float = 0., radius: float = 1.0) -> jax.Array:
    """
    Directly solves the Stokes BIO equation using the VWX diagonal property. The right-hand
    side and returned density are both given by their VWX (diagonalizing-basis) coefficients
    (<sig_vwx_bc> in, sig_vwx out, each (3, nlm)); the solve is a per-component diagonal division.
    The SL block scales by the source-sphere <radius> (DL/dSL/jump are scale-invariant).
    """

    vlm_bc, wlm_bc, xlm_bc = sig_vwx_bc[0], sig_vwx_bc[1], sig_vwx_bc[2]
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

    return jnp.stack([vlm_sigma, wlm_sigma, xlm_sigma], axis=0)


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

    # DIRECT solve -- densities are handled in the VWX (diagonalizing) basis: COB the
    # boundary data with sig_xyz2vwx before the solve, invert with sig_vwx2xyz before the error.

    # COMPLEX TRANSFORM
    vwx_bc = jnp.stack(sig_xyz2vwx_cplx(BC_pot[:,:,0], BC_pot[:,:,1], BC_pot[:,:,2], theta, phi, sh), axis=0)
    vwx_sig = stokes_onsurf_direct_solve_cplx(vwx_bc, sh, sl_scal, dl_scal, sgn)
    vwx_bc_check = bio_onsurf_apply_cplx(vwx_sig, sh, sl_scal, dl_scal, sgn)
    resid_direct = jnp.linalg.norm(vwx_bc_check - vwx_bc)
    print("Residual of DIRECT solve = {a}".format(a=resid_direct))

    # Accuracy
    xtrg = Strg["Xcart"][:,:,0]
    ytrg = Strg["Xcart"][:,:,1]
    ztrg = Strg["Xcart"][:,:,2]
    theta_trg = Strg["Xsph"][:,:,0]
    phi_trg = Strg["Xsph"][:,:,1]
    trg_sphere2 = jnp.column_stack([jnp.reshape(xtrg,-1), jnp.reshape(ytrg,-1), jnp.reshape(ztrg,-1)])
    true_field = compute_field(trg_sphere2, ptsrc, force)
    true_field = jnp.reshape(true_field, Strg["Xcart"].shape)
    true_field = jnp.real(true_field)

    vwx_out = point_n_shoot_cplx(Strg, shtrg, vwx_sig, S, sh, sl_scal, dl_scal)
    vx, vy, vz = sig_vwx2xyz_cplx(vwx_out[0], vwx_out[1], vwx_out[2], theta_trg, phi_trg, shtrg)
    Ksig_direct = jnp.real(jnp.stack([vx, vy, vz], axis=2))

    diff_direct = jnp.max(true_field - Ksig_direct) / jnp.max(true_field)
    print("Max relative error of order {lmax} solver at target radius {Rtrg} for direct solver is {d}".format(lmax=lmax, Rtrg=Rtrg, d=diff_direct))


    # REAL TRANSFORMS -- the real/truncated analysis takes float64 spatial data, and the
    # manufactured field is real (compute_field stores it as complex128), so pass its real part.
    BC_pot_re = jnp.real(BC_pot)
    vwx_bc = jnp.stack(sig_xyz2vwx(BC_pot_re[:,:,0], BC_pot_re[:,:,1], BC_pot_re[:,:,2], theta, phi, sh), axis=0)
    vwx_sig = stokes_onsurf_direct_solve(vwx_bc, sh, sl_scal, dl_scal, sgn)
    vwx_bc_check = bio_onsurf_apply(vwx_sig, sh, sl_scal, dl_scal, sgn)
    resid_direct = jnp.linalg.norm(vwx_bc_check - vwx_bc)
    print("Residual of DIRECT solve = {a}".format(a=resid_direct))

    # Accuracy
    xtrg = Strg["Xcart"][:,:,0]
    ytrg = Strg["Xcart"][:,:,1]
    ztrg = Strg["Xcart"][:,:,2]
    theta_trg = Strg["Xsph"][:,:,0]
    phi_trg = Strg["Xsph"][:,:,1]
    trg_sphere2 = jnp.column_stack([jnp.reshape(xtrg,-1), jnp.reshape(ytrg,-1), jnp.reshape(ztrg,-1)])
    true_field = compute_field(trg_sphere2, ptsrc, force)
    true_field = jnp.reshape(true_field, Strg["Xcart"].shape)
    true_field = jnp.real(true_field)

    vwx_out = point_n_shoot(Strg, shtrg, vwx_sig, S, sh, sl_scal, dl_scal)
    vx, vy, vz = sig_vwx2xyz(vwx_out[0], vwx_out[1], vwx_out[2], theta_trg, phi_trg, shtrg)
    Ksig_direct = jnp.real(jnp.stack([vx, vy, vz], axis=2))

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
    vwx_bc = jnp.stack(sig_xyz2vwx_cplx(BC_flux[:,:,0], BC_flux[:,:,1], BC_flux[:,:,2], theta, phi, sh), axis=0)
    vwx_sig = stokes_onsurf_direct_solve_cplx(
        vwx_bc,
        sh=sh,
        sl_scal=0.,
        dl_scal=0.,
        sgn=sgn,
        dsl_scal=1.0
    )
    vwx_bc_check = bio_onsurf_apply_cplx(vwx_sig, sh, 0., 0., sgn, 1.0)
    resid_direct = jnp.linalg.norm(vwx_bc_check - vwx_bc)
    jax.debug.print("Residual of DIRECT solve: {a}", a=resid_direct)
    # Accuracy
    xtrg = Strg["Xcart"][:,:,0]
    ytrg = Strg["Xcart"][:,:,1]
    ztrg = Strg["Xcart"][:,:,2]
    theta_trg = Strg["Xsph"][:,:,0]
    phi_trg = Strg["Xsph"][:,:,1]
    trg_sphere2 = jnp.column_stack([jnp.reshape(xtrg,-1), jnp.reshape(ytrg,-1), jnp.reshape(ztrg,-1)])
    true_field = compute_field(trg_sphere2, ptsrc, force)
    true_field = jnp.reshape(true_field, Strg["Xcart"].shape)
    true_field = jnp.real(true_field)

    vwx_out = point_n_shoot_cplx(Strg, shtrg, vwx_sig, S, sh, 1.0, 0.)
    vx, vy, vz = sig_vwx2xyz_cplx(vwx_out[0], vwx_out[1], vwx_out[2], theta_trg, phi_trg, shtrg)
    Ksig_direct = jnp.real(jnp.stack([vx, vy, vz], axis=2))

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
    vwx_bc = jnp.stack(sig_xyz2vwx_cplx(BC_pot[:,:,0], BC_pot[:,:,1], BC_pot[:,:,2], theta, phi, sh), axis=0)
    vwx_sig = stokes_onsurf_direct_solve_cplx(vwx_bc, sh, sl_scal, dl_scal, sgn)
    vwx_bc_check = bio_onsurf_apply_cplx(vwx_sig, sh, sl_scal, dl_scal, sgn)
    resid_diag = jnp.linalg.norm(vwx_bc_check - vwx_bc)
    print("Residual of DIRECT solve = {a}".format(a=resid_diag))

    # Accuracy
    xtrg = Strg["Xcart"][:,:,0]
    ytrg = Strg["Xcart"][:,:,1]
    ztrg = Strg["Xcart"][:,:,2]
    theta_trg = Strg["Xsph"][:,:,0]
    phi_trg = Strg["Xsph"][:,:,1]
    trg_sphere2 = jnp.column_stack([jnp.reshape(xtrg,-1), jnp.reshape(ytrg,-1), jnp.reshape(ztrg,-1)])
    true_field = compute_field(trg_sphere2, ptsrc, force)
    true_field = jnp.reshape(true_field, Strg["Xcart"].shape)
    true_field = jnp.real(true_field)

    vwx_out = point_n_shoot_cplx(Strg, shtrg, vwx_sig, S, sh, sl_scal, dl_scal)
    vx, vy, vz = sig_vwx2xyz_cplx(vwx_out[0], vwx_out[1], vwx_out[2], theta_trg, phi_trg, shtrg)
    Ksig_direct = jnp.real(jnp.stack([vx, vy, vz], axis=2))

    diff_direct = jnp.max(true_field - Ksig_direct) / jnp.max(true_field)
    print("Max relative error of order {lmax} solver at target radius {Rtrg} for direct solver is {d}".format(lmax=lmax, Rtrg=Rtrg, d=diff_direct))

    # Arbitrary-point spectral eval (bio_offsurf_apply) vs point_n_shoot (to_lat_jax) to check
    # far eval formulas. point_n_shoot returns target-basis VWX coeffs, so synthesize them to
    # Cartesian point values (inverse COB) before comparing against the point-valued far eval.
    K_pns = jnp.real(jnp.reshape(Ksig_direct, (-1, 3)))
    K_far = jnp.real(bio_offsurf_apply_cplx(trg_sphere2, vwx_sig, S, sh, sl_scal, dl_scal, far=True))
    err_far = jnp.max(jnp.abs(K_far - K_pns)) / jnp.max(jnp.abs(K_pns))
    jax.debug.print("Max relative error of bio_offsurf_apply (far eval) vs point_n_shoot at radius {Rtrg} = {e}", Rtrg=Rtrg, e=err_far)

    #  ...  to check SHqst_to_point_cplx
    K_pt = jnp.real(bio_offsurf_apply_cplx(trg_sphere2, vwx_sig, S, sh, sl_scal, dl_scal, far=False))
    err_pt = jnp.max(jnp.abs(K_pt - K_pns)) / jnp.max(jnp.abs(K_pns))
    jax.debug.print("Max relative error of bio_offsurf_apply (point eval) vs point_n_shoot at radius {Rtrg} = {e}", Rtrg=Rtrg, e=err_pt)

