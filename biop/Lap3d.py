"""
Laplace Operator Class:
    SL and DL operators on a sphere, using spectra and solid harmonics
    Tests at single concentric sphere surface.

    TODO:
        SH to point evaluation function needs to be jax-enabled
        Allow on-surface evaluation in bio_offsurf_apply()
        onsurf_diag_solve() l=0 currently set to BC values. Throw exception instead?
        Point and shoot algorithm 

    NOTE: 
        (Jun 17, 2026) added SL traction far eval and on-surface eval, for far and for Neumann BC formulation. No near formula for SL traction yet.
        (Jun 22, 2026) Vectorized SH_to_point_cplx
        (Aug 11, 2026) Deprecated cplx transforms, use truncated real-valued transforms only.
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

def Lap3d_sl_diag_cplx(sh: shtns_jax.sht) -> jax.Array:
    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64)
    diag = 1.0 / (2.0 * l_vals + 1.0)
    return diag

def Lap3d_sl_diag(sh: shtns_jax.sht) -> jax.Array:
    l_vals = jnp.asarray(sh.l, dtype=jnp.float64)
    diag = 1.0 / (2.0 * l_vals + 1.0)
    return diag

def Lap3d_dl_diag_cplx(sh: shtns_jax.sht) -> tuple([jax.Array, jax.Array]):
    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64)
    diag_ext = l_vals / (2.0 * l_vals + 1.0)
    diag_int = - (l_vals + 1) / (2.0 * l_vals + 1.0)
    return diag_ext, diag_int

def Lap3d_dl_diag(sh: shtns_jax.sht) -> tuple([jax.Array, jax.Array]):
    l_vals = jnp.asarray(sh.l, dtype=jnp.float64)
    diag_ext = l_vals / (2.0 * l_vals + 1.0)
    diag_int = - (l_vals + 1) / (2.0 * l_vals + 1.0)
    return diag_ext, diag_int

def Lap3d_dsl_diag_cplx(sh: shtns_jax.sht) -> tuple([jax.Array, jax.Array]):
    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64)
    diag_int = l_vals / (2.0 * l_vals + 1.0)
    diag_ext = - (l_vals + 1) / (2.0 * l_vals + 1.0)
    return diag_ext, diag_int

def Lap3d_dsl_diag(sh: shtns_jax.sht) -> tuple([jax.Array, jax.Array]):
    l_vals = jnp.asarray(sh.l, dtype=jnp.float64)
    diag_int = l_vals / (2.0 * l_vals + 1.0)
    diag_ext = - (l_vals + 1) / (2.0 * l_vals + 1.0)
    return diag_ext, diag_int

LAP_FAR_PREFAC = 1.0 / 4.0 / jnp.pi

@partial(jax.jit, static_argnames=["sh", "tile", "terms"])
def _lap_far_kernel_cplx(trg: jax.Array, qlm_sigma: jax.Array, trgN: jax.Array,
                    sl_scal, dl_scal, dsl_scal,
                    Y: jax.Array, w: jax.Array, a, Xc: jax.Array,
                    sh: shtns_jax.sht, tile: int, terms: tuple) -> jax.Array:
    """
    Fused smooth-quadrature ("far") evaluation of  K = sl*SL + dl*DL + dsl*dSL  of the
    source sphere described by <Y>, <w>, <a>, <Xc> (see sphere.far_src_geom) with scalar
    density coefficients <qlm_sigma>, at targets <trg> (Ntrg x 3) with target normals
    <trgN> (only read by the dSL term).  Returns Ntrg x 1 potential values.
    <terms> selects which operators are built at all (static, so an unused operator costs
    nothing); the scalings stay traced.

    All three kernels are dense sums over the Ntrg x Nsrc pairs, so the whole cost is in how
    much per-pair work is done. Writing r = x_i - Y_j with x_i = trg_i - Xc, and using that
    the source normal on a sphere is exactly n_j = Y_j / a:

        d2 = |x_i|^2 - 2 x_i.Y_j + a^2        (one K=3 gemm; no Ntrg x Nsrc x 3 temporary)
        r.n_j = (x_i.Y_j - a^2)/a = (|x_i|^2 - a^2 - d2)/(2a)

    so the DL radial factor collapses onto quantities SL already needs:
        invd3*(r.n) = [ (|x_i|^2 - a^2) invd3 - invd ] / (2a).
    Both operators are then just contractions of the two pair matrices invd, invd3 against
    the source-side density columns -- no pow(), no sqrt beyond one rsqrt, and every
    pair-shaped array stays REAL (the complex density enters only through the narrow column
    blocks, real and imaginary parts carried side by side, since the sum is linear in it).
    dSL keeps a per-target normal so its r.n does not collapse, but
        sum_j invd3 (Y_j.n_x_i) g_j = n_x_i . sum_j invd3 Y_j g_j
    turns that into three extra source columns rather than another pair matrix.
    """
    f = sh.synth_cplx_jax(qlm_sigma).reshape(-1)   # coeffs -> grid, once
    g = w * f                                      # (Nsrc,) weighted density
    a2 = a * a
    want_sl, want_dl, want_dsl = ("sl" in terms), ("dl" in terms), ("dsl" in terms)
    # Source column blocks, real/imag side by side (the density is complex but the sum is
    # linear in it, so it rides along as extra columns and every pair array stays real).
    Z1 = jnp.stack([jnp.real(g), jnp.imag(g)], axis=1)              # (Nsrc, 2)
    if want_dsl:
        gY = g[:, None] * Y                                         # (Nsrc, 3)
        ZY = jnp.concatenate([jnp.real(gY), jnp.imag(gY)], axis=1)  # (Nsrc, 6)

    def body(tt, nn):
        X = tt - Xc                                                 # (nt, 3)
        px = jnp.sum(X * X, axis=1)[:, None]                        # (nt, 1)
        d2 = jnp.maximum(px + a2 - 2.0 * pair_dot(X, Y), 0.0)        # (nt, Nsrc)
        invd = jax.lax.rsqrt(d2)
        if want_dl or want_dsl:
            invd3 = invd * invd * invd
        # SL, DL and the (r.n_x)-free part of dSL are all  sum_j <radial factor> g_j, so they
        # collapse into ONE pair matrix -- the per-target factors multiply the radial powers,
        # not the contraction result.
        pair = 0.0
        if want_sl:
            pair = pair + sl_scal * invd
        if want_dl:
            pair = pair + dl_scal * (((px - a2) / (2.0 * a)) * invd3 - invd / (2.0 * a))
        if want_dsl:
            pair = pair - dsl_scal * jnp.sum(X * nn, axis=1)[:, None] * invd3
        R = (LAP_FAR_PREFAC * pair) @ Z1
        out = R[:, 0] + 1j * R[:, 1]
        if want_dsl:
            # the remaining dSL piece: sum_j invd3 (Y_j.n_x_i) g_j = n_x_i . sum_j invd3 Y_j g_j
            RY = invd3 @ ZY
            V3 = RY[:, 0:3] + 1j * RY[:, 3:6]
            out = out + (dsl_scal * LAP_FAR_PREFAC) * jnp.sum(nn * V3, axis=1)
        return out[:, None]

    if trgN is None:
        trgN = jnp.zeros_like(trg)
    return far_tile_map(body, (trg, trgN), tile)

@partial(jax.jit, static_argnames=["sh", "tile", "terms"])
def _lap_far_kernel(trg: jax.Array, qlm_sigma: jax.Array, trgN: jax.Array,
                    sl_scal, dl_scal, dsl_scal,
                    Y: jax.Array, w: jax.Array, a, Xc: jax.Array,
                    sh: shtns_jax.sht, tile: int, terms: tuple) -> jax.Array:
    """Real/truncated-layout counterpart of _lap_far_kernel_cplx. The density is REAL here
    (sh.synth_jax returns a float64 grid), so the complex real/imag column split collapses
    and every array stays real. See _lap_far_kernel_cplx for the quadrature derivation."""
    f = sh.synth_jax(qlm_sigma).reshape(-1)        # coeffs -> real grid, once
    g = w * f                                      # (Nsrc,) real weighted density
    a2 = a * a
    want_sl, want_dl, want_dsl = ("sl" in terms), ("dl" in terms), ("dsl" in terms)
    if want_dsl:
        gY = g[:, None] * Y                                         # (Nsrc, 3) real

    def body(tt, nn):
        X = tt - Xc                                                 # (nt, 3)
        px = jnp.sum(X * X, axis=1)[:, None]                        # (nt, 1)
        d2 = jnp.maximum(px + a2 - 2.0 * pair_dot(X, Y), 0.0)        # (nt, Nsrc)
        invd = jax.lax.rsqrt(d2)
        if want_dl or want_dsl:
            invd3 = invd * invd * invd
        pair = 0.0
        if want_sl:
            pair = pair + sl_scal * invd
        if want_dl:
            pair = pair + dl_scal * (((px - a2) / (2.0 * a)) * invd3 - invd / (2.0 * a))
        if want_dsl:
            pair = pair - dsl_scal * jnp.sum(X * nn, axis=1)[:, None] * invd3
        R = (LAP_FAR_PREFAC * pair) @ g[:, None]                     # (nt, 1)
        out = R[:, 0]
        if want_dsl:
            RY = invd3 @ gY                                          # (nt, 3)
            out = out + (dsl_scal * LAP_FAR_PREFAC) * jnp.sum(nn * RY, axis=1)
        return out[:, None]

    if trgN is None:
        trgN = jnp.zeros_like(trg)
    return far_tile_map(body, (trg, trgN), tile)

def _lap_far_cplx(trg: jax.Array, qlm_sigma: jax.Array, S: SphereDict, sh: shtns_jax.sht,
             terms: tuple, sl_scal=0.0, dl_scal=0.0, dsl_scal=0.0,
             trgN: jax.Array = None, tile: int = None) -> jax.Array:
    """Eager wrapper of _lap_far_kernel_cplx: pulls the source-centred geometry out of <S> and
    picks the target tile size, then calls the jitted kernel."""
    Y, w, a, Xc = far_src_geom(S, sh)
    return _lap_far_kernel_cplx(trg, qlm_sigma, trgN, sl_scal, dl_scal, dsl_scal, Y, w, a, Xc,
                           sh=sh, tile=far_tile_size(Y.shape[0], tile), terms=terms)

def _lap_far(trg: jax.Array, qlm_sigma: jax.Array, S: SphereDict, sh: shtns_jax.sht,
             terms: tuple, sl_scal=0.0, dl_scal=0.0, dsl_scal=0.0,
             trgN: jax.Array = None, tile: int = None) -> jax.Array:
    """Eager wrapper of _lap_far_kernel: pulls the source-centred geometry out of <S> and
    picks the target tile size, then calls the jitted kernel."""
    Y, w, a, Xc = far_src_geom(S, sh)
    return _lap_far_kernel(trg, qlm_sigma, trgN, sl_scal, dl_scal, dsl_scal, Y, w, a, Xc,
                           sh=sh, tile=far_tile_size(Y.shape[0], tile), terms=terms)

def Lap3d_sl_far(trg: jax.Array, qlm_sigma: jax.Array, S:SphereDict, sh: shtns_jax.sht) -> jax.Array:
    """
    Off-surface evaluation of the Lap3d SL potential with density coefficients <qlm_sigma>
        with <trg>: Ntrg x 3
        using the smooth surface quadrature (accurate only for targets well
        away from the surface).
    The direct quadrature needs the grid-space density, so the coefficients are first
    synthesized back to the (nphi, ntheta) grid (inverse decomposition), then the direct
    matrix integral is applied.
    Returns the potential at the targets: Ntrg x 1.
    Thin wrapper over the fused _lap_far kernel (see there for the quadrature).
    """
    assert trg.shape[1] == 3
    return _lap_far(trg, qlm_sigma, S, sh, ("sl",), sl_scal=1.0)

def Lap3d_dl_far(trg: jax.Array, qlm_sigma: jax.Array, S:SphereDict, sh: shtns_jax.sht) -> jax.Array:
    """
    Off-surface evaluation of the Lap3d DL potential with density coefficients <qlm_sigma>
        with <trg>: Ntrg x 3
        using the smooth surface quadrature (accurate only for targets well
        away from the surface). The coefficients are synthesized to the grid first,
        then the direct matrix integral is applied.
    Returns the potential at the targets: Ntrg x 1.

    The DL kernel is d/dn_y (1/(4 pi |x-y|)) = (1/4pi) (x-y).n_y / |x-y|^3.
    With r = trg - src = x - y, this is prefac * (r.n) / |r|^3.
    Thin wrapper over the fused _lap_far kernel (see there for the quadrature).
    """
    assert trg.shape[1] == 3
    return _lap_far(trg, qlm_sigma, S, sh, ("dl",), dl_scal=1.0)

def Lap3d_dsl_far(trg: jax.Array, trgN: jax.Array, qlm_sigma: jax.Array, S:SphereDict, sh: shtns_jax.sht) -> jax.Array:
    """
    Off-surface evaluation of the surface-normal derviatve of the Lap3d SL potential ("T")
        with <trg>: Ntrg x 3
        against target normal <trgN>: Ntrg x 3
        using the smooth surface quadrature (accurate only for targets well
        away from the surface). Density given by coefficients <qlm_sigma>, synthesized
        to the grid first, then the direct matrix integral is applied.
    Returns the traction at the targets: Ntrg x 1.

    The SL traction kernel is -(1/4pi) (x-y).n_x / |x-y|^3.
    With r = trg - src = x - y
    Thin wrapper over the fused _lap_far kernel (see there for the quadrature).
    """
    assert trg.shape[1] == 3 and trgN.shape[1] == 3
    return _lap_far(trg, qlm_sigma, S, sh, ("dsl",), dsl_scal=1.0, trgN=trgN)

def Lap3d_sl_cplx(trg: jax.Array, qlm_sigma: jax.Array, S: SphereDict, sh: shtns_jax.sht) -> jax.Array:
    """
    Off-surface evaluation at points <trg>: Ntrg x 3
        From source <S> with density coefficients <qlm_sigma> in the SH basis
        source uses SHT object <sh>
    """

    assert trg.shape[1] == 3

    if S["lmax"] != sh.lmax:
        print("S lmax does not match sht's lmax, reform sht.")
        sh = shtns_jax.sht(S["lmax"], S["lmax"])

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
    diag = Lap3d_sl_diag_cplx(sh)
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

    return SL_sigma

def Lap3d_sl(trg: jax.Array, qlm_sigma: jax.Array, S: SphereDict, sh: shtns_jax.sht) -> jax.Array:
    """
    Real/truncated-layout counterpart of Lap3d_sl_cplx.
    Off-surface evaluation at points <trg>: Ntrg x 3
        From source <S> with density coefficients <qlm_sigma> in the real SH basis
        source uses SHT object <sh>
    """

    assert trg.shape[1] == 3

    if S["lmax"] != sh.lmax:
        print("S lmax does not match sht's lmax, reform sht.")
        sh = shtns_jax.sht(S["lmax"], S["lmax"])

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
    l_vals = jnp.asarray(sh.l, dtype=jnp.float64)
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
        [sh.SH_to_point(qlm_SL_sigma[trg_i, :], trg_costheta[trg_i], trg_phi[trg_i])
         for trg_i in range(trg_theta.shape[0])],
        dtype=jnp.float64,
    )[:, None]

    return SL_sigma

def Lap3d_dl_cplx(trg: jax.Array, qlm_sigma: jax.Array, S: SphereDict, sh: shtns_jax.sht) -> jax.Array:
    """
    Off-surface evaluation at points <trg>: Ntrg x 3
        From source <S> with density coefficients <qlm_sigma> in the SH basis
        source uses SHT object <sh>
    """

    assert trg.shape[1] == 3

    if S["lmax"] != sh.lmax:
        print("S lmax does not match sht's lmax, reform sht.")
        sh = shtns_jax.sht(S["lmax"])

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
    [diag_ext, diag_int] = Lap3d_dl_diag_cplx(sh)
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

def Lap3d_dl(trg: jax.Array, qlm_sigma: jax.Array, S: SphereDict, sh: shtns_jax.sht) -> jax.Array:
    """
    Real/truncated-layout counterpart of Lap3d_dl_cplx.
    Off-surface evaluation at points <trg>: Ntrg x 3
        From source <S> with density coefficients <qlm_sigma> in the real SH basis
        source uses SHT object <sh>
    """

    assert trg.shape[1] == 3

    if S["lmax"] != sh.lmax:
        print("S lmax does not match sht's lmax, reform sht.")
        sh = shtns_jax.sht(S["lmax"])

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
    l_vals = jnp.asarray(sh.l, dtype=jnp.float64)
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
        [sh.SH_to_point(qlm_DL_sigma[trg_i, :], trg_costheta[trg_i], trg_phi[trg_i])
         for trg_i in range(trg_theta.shape[0])],
        dtype=jnp.float64,
    )[:, None]

    return DL_sigma

def Lap3d_dsl_cplx(trg: jax.Array, trgN: jax.Array, qlm_sigma: jax.Array, S: SphereDict, sh: shtns_jax.sht) -> jax.Array:
    """
    Off-surface evaluation at points <trg>: Ntrg x 3 with target normal <trgN>: Ntrg x 3
        From source <S> with density coefficients <qlm_sigma> in the SH basis
        source uses SHT object <sh>
    """

    assert trg.shape[1] == 3 and trgN.shape[1] == 3

    if S["lmax"] != sh.lmax:
        print("S lmax does not match sht's lmax, reform sht.")
        sh = shtns_jax.sht(S["lmax"])

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
    [diag_ext, diag_int] = Lap3d_dsl_diag_cplx(sh)
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

def Lap3d_dsl(trg: jax.Array, trgN: jax.Array, qlm_sigma: jax.Array, S: SphereDict, sh: shtns_jax.sht) -> jax.Array:
    """
    Real/truncated-layout counterpart of Lap3d_dsl_cplx.
    Off-surface evaluation at points <trg>: Ntrg x 3 with target normal <trgN>: Ntrg x 3
        From source <S> with density coefficients <qlm_sigma> in the real SH basis
        source uses SHT object <sh>
    """

    assert trg.shape[1] == 3 and trgN.shape[1] == 3

    if S["lmax"] != sh.lmax:
        print("S lmax does not match sht's lmax, reform sht.")
        sh = shtns_jax.sht(S["lmax"])

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
    l_vals = jnp.asarray(sh.l, dtype=jnp.float64)
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
        [sh.SH_to_point(qlm_T_sigma[trg_i, :], trg_costheta[trg_i], trg_phi[trg_i])
         for trg_i in range(trg_theta.shape[0])],
        dtype=jnp.float64,
    )[:, None]

    return T_sigma

@partial(jax.jit, static_argnames=["sh", "shtrg", "exterior"])
def _lap_sl_1sph_kernel_cplx(qlm_sigma: jax.Array, a: float, trg_dr: float,
                        sh: shtns_jax.sht, shtrg: shtns_jax.sht, exterior: bool) -> jax.Array:
    """Jitted core of Lap3d_sl_r_1sph. Takes the source density coefficients <qlm_sigma>
    and returns the SL potential coefficients on the target grid (in the SH basis).
    <exterior> (the trg_dr > a branch) and the nlm pad/truncate are compile-time
    constants (static args)."""
    rho = trg_dr / a          # solid harmonics as if src sphere were unit; SL scales by a
    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64)
    diag = Lap3d_sl_diag_cplx(sh)   # 1sph kernel is complex-layout (nlm_cplx / sh.zl)
    rpowers = rho ** (-l_vals - 1) if exterior else rho ** (l_vals)
    qlm_SL_sigma = a * rpowers * qlm_sigma * diag

    # Interpolate to new grid by padding or truncating coefficients to Strg['lmax']
    nlm_src = sh.nlm_cplx
    nlm_trg = shtrg.nlm_cplx
    if nlm_trg > nlm_src:
        qlm_SL_sigma = jnp.pad(qlm_SL_sigma, (0, nlm_trg - nlm_src), constant_values=0)
    elif nlm_trg < nlm_src:
        qlm_SL_sigma = qlm_SL_sigma[:nlm_trg]

    return qlm_SL_sigma

def Lap3d_sl_r_1sph_cplx(Strg: SphereDict, shtrg: shtns_jax.sht, qlm_sigma: jax.Array, S: SphereDict, sh: shtns_jax.sht) -> jax.Array:
    """
    Off-surface evaluation at a spherical grid of targets
        From source <S> with density coefficients <qlm_sigma> (SH basis), source uses <sh>
        To target <Strg>, target uses SHT object <shtrg>
    Returns the SL potential coefficients on the target grid (SH basis).
    """

    if S["lmax"] != sh.lmax:
        print("S lmax does not match sht's lmax, reform sht.")
        sh = shtns_jax.sht(S["lmax"], S["lmax"])
    if Strg["lmax"] != shtrg.lmax:
        print("Strg lmax does not match sht_trg's lmax, reform sht_trg.")
        shtrg = shtns_jax.sht(Strg["lmax"], Strg["lmax"])

    exterior = bool(Strg["r"] > S["r"])
    return _lap_sl_1sph_kernel_cplx(qlm_sigma, S["r"], Strg["r"], sh, shtrg, exterior)

@partial(jax.jit, static_argnames=["sh", "shtrg", "exterior"])
def _lap_dl_1sph_kernel_cplx(qlm_sigma: jax.Array, a: float, trg_dr: float,
                        sh: shtns_jax.sht, shtrg: shtns_jax.sht, exterior: bool) -> jax.Array:
    """Jitted core of Lap3d_dl_r_1sph. Takes the source density coefficients <qlm_sigma>
    and returns the DL potential coefficients on the target grid (in the SH basis).
    <exterior> (the trg_dr > a branch) and the nlm pad/truncate are compile-time
    constants (static args)."""
    rho = trg_dr / a          # solid harmonics as if src sphere were unit; DL is scale-invariant
    l_vals = jnp.asarray(sh.zl, dtype=jnp.float64)
    [diag_ext, diag_int] = Lap3d_dl_diag_cplx(sh)   # 1sph kernel is complex-layout (nlm_cplx / sh.zl)
    if exterior:
        qlm_DL_sigma = (rho ** (-l_vals - 1)) * qlm_sigma * diag_ext
    else:
        qlm_DL_sigma = (rho ** (l_vals)) * qlm_sigma * diag_int

    # Interpolate to new grid by padding or truncating coefficients to Strg['lmax']
    nlm_src = sh.nlm_cplx
    nlm_trg = shtrg.nlm_cplx
    if nlm_trg > nlm_src:
        qlm_DL_sigma = jnp.pad(qlm_DL_sigma, (0, nlm_trg - nlm_src), constant_values=0)
    elif nlm_trg < nlm_src:
        qlm_DL_sigma = qlm_DL_sigma[:nlm_trg]

    return qlm_DL_sigma

def Lap3d_dl_r_1sph_cplx(Strg: SphereDict, shtrg: shtns_jax.sht, qlm_sigma: jax.Array, S: SphereDict, sh: shtns_jax.sht) -> jax.Array:
    """
    Off-surface evaluation at a spherical grid of targets
        From source <S> with density coefficients <qlm_sigma> (SH basis), source uses <sh>
        To target <Strg>, target uses SHT object <shtrg>
    Returns the DL potential coefficients on the target grid (SH basis).
    """

    if S["lmax"] != sh.lmax:
        print("S lmax does not match sht's lmax, reform sht.")
        sh = shtns_jax.sht(S["lmax"], S["lmax"])
    if Strg["lmax"] != shtrg.lmax:
        print("Strg lmax does not match sht_trg's lmax, reform sht_trg.")
        shtrg = shtns_jax.sht(Strg["lmax"], Strg["lmax"])

    exterior = bool(Strg["r"] > S["r"])
    return _lap_dl_1sph_kernel_cplx(qlm_sigma, S["r"], Strg["r"], sh, shtrg, exterior)

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
def bio_diag_apply_cplx(qlm_sigma: jax.Array, sh: shtns_jax.sht, sl_scal: float, dl_scal: float, dsl_scal: float = 0., radius: float = 1.0) -> jax.Array:
    """
    Apply the combined DL potential operator K = sl_scal * SL + dl_scal * DL
        to the density with coefficients <qlm_sigma> in SH basis
    Returns the resulting function coefficients <qlm_KL_sigma> in SH basis,
        where DL is evaluated in the P.V. sense.
    Added dSL, the traction kernel for singular layer potential. Formulation should be either a combination of SL+DL, or T, not both.
    The SL block scales by the source-sphere <radius> (DL/dSL are scale-invariant).
    """

    sl_diag = radius * Lap3d_sl_diag_cplx(sh)
    [dl_diag_ext, dl_diag_int] = Lap3d_dl_diag_cplx(sh)
    dl_diag = 0.5*(dl_diag_ext + dl_diag_int)
    [dsl_diag_ext, dsl_diag_int] = Lap3d_dsl_diag_cplx(sh)
    dsl_diag = 0.5*(dsl_diag_ext + dsl_diag_int)
    qlm_SL_sigma = sl_diag * qlm_sigma
    qlm_DL_sigma = dl_diag * qlm_sigma
    qlm_T_sigma = dsl_diag * qlm_sigma
    qlm_KL_sigma = dl_scal * qlm_DL_sigma + sl_scal * qlm_SL_sigma + dsl_scal * qlm_T_sigma
    return qlm_KL_sigma

@partial(jax.jit, static_argnames=["sh"])
def bio_diag_apply(qlm_sigma: jax.Array, sh: shtns_jax.sht, sl_scal: float, dl_scal: float, dsl_scal: float = 0., radius: float = 1.0) -> jax.Array:
    """
    Real/truncated-layout counterpart of bio_diag_apply_cplx.
    Apply the combined DL potential operator K = sl_scal * SL + dl_scal * DL
        to the density with coefficients <qlm_sigma> in the real SH basis
    Returns the resulting function coefficients <qlm_KL_sigma> in the real SH basis,
        where DL is evaluated in the P.V. sense.
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
def bio_onsurf_apply_cplx(qlm_sigma: jax.Array, sh: shtns_jax.sht, sl_scal: float, dl_scal: float, sgn: float, dsl_scal: float = 0., radius: float = 1.0) -> jax.Array:
    """
    Apply the combined DL potential operator K = sl_scal * SL + dl_scal * DL
        to the density with coefficients <qlm_sigma> in the SH (diagonalizing) basis
        taking into account the DL jump condition with sign <sgn>.
    Returns the resulting function coefficients in the SH basis, so the on-surface
    self-apply is a pure diagonal multiply (COB is the caller's responsibility).
    Traction kernel should have opposite sign to DL, but only one should appear per formulation.
    The SL block scales by the source-sphere <radius> (DL/dSL/jump are scale-invariant).
    """
    # assert dl_scal * dsl_scal == 0 # check that only one is nonzero. -- ran into jit compile problem

    qlm_KL_sigma = bio_diag_apply_cplx(qlm_sigma, sh, sl_scal, dl_scal, dsl_scal, radius)
    # The jump condition is a multiple of the identity, hence diagonal in the SH basis too.
    qlm_jump = 0.5 * dl_scal * sgn * qlm_sigma + 0.5 * dsl_scal * (-1*sgn) * qlm_sigma
    return qlm_KL_sigma + qlm_jump

@partial(jax.jit, static_argnames=["sh"])
def bio_onsurf_apply(qlm_sigma: jax.Array, sh: shtns_jax.sht, sl_scal: float, dl_scal: float, sgn: float, dsl_scal: float = 0., radius: float = 1.0) -> jax.Array:
    """
    Real/truncated-layout counterpart of bio_onsurf_apply_cplx.
    Apply the combined DL potential operator K = sl_scal * SL + dl_scal * DL
        to the density with coefficients <qlm_sigma> in the real SH (diagonalizing) basis
        taking into account the DL jump condition with sign <sgn>.
    The SL block scales by the source-sphere <radius> (DL/dSL/jump are scale-invariant).
    """
    # assert dl_scal * dsl_scal == 0 # check that only one is nonzero. -- ran into jit compile problem

    qlm_KL_sigma = bio_diag_apply(qlm_sigma, sh, sl_scal, dl_scal, dsl_scal, radius)
    # The jump condition is a multiple of the identity, hence diagonal in the SH basis too.
    qlm_jump = 0.5 * dl_scal * sgn * qlm_sigma + 0.5 * dsl_scal * (-1*sgn) * qlm_sigma
    return qlm_KL_sigma + qlm_jump

@partial(jax.jit, static_argnames=["sh"])
def bio_onsurf_direct_solve_cplx(qlm_bc: jax.Array, sh: shtns_jax.sht, sl_scal: float, dl_scal: float, sgn: float, dsl_scal: float = 0., radius: float = 1.0) -> jax.Array:
    """
    Directly solves the BIO equation in the spectral domain.
    Equation: [0.5 * dl_scal * sgn * I + KL] sigma = bc, with the right-hand side and the
    returned density both given by their coefficients in the SH (diagonalizing) basis
    (<qlm_bc> in, qlm_sigma out); the solve is a diagonal division.

    Note: The l=0 mode is in the null space and is set to the BC value.
    The SL block scales by the source-sphere <radius> (DL/dSL/jump are scale-invariant).
    """
    # assert dl_scal * dsl_scal == 0

    sl_diag = radius * Lap3d_sl_diag_cplx(sh)
    [dl_diag_ext, dl_diag_int] = Lap3d_dl_diag_cplx(sh)
    dl_diag = 0.5 * (dl_diag_ext + dl_diag_int)
    [dsl_diag_ext, dsl_diag_int] = Lap3d_dsl_diag_cplx(sh)
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

    return qlm_sigma

@partial(jax.jit, static_argnames=["sh"])
def bio_onsurf_direct_solve(qlm_bc: jax.Array, sh: shtns_jax.sht, sl_scal: float, dl_scal: float, sgn: float, dsl_scal: float = 0., radius: float = 1.0) -> jax.Array:
    """
    Real/truncated-layout counterpart of bio_onsurf_direct_solve_cplx.
    Directly solves the BIO equation in the spectral domain.
    Equation: [0.5 * dl_scal * sgn * I + KL] sigma = bc, with <qlm_bc> in and qlm_sigma out
    both given by their coefficients in the real SH (diagonalizing) basis; a diagonal division.

    Note: The l=0 mode is in the null space and is set to the BC value.
    The SL block scales by the source-sphere <radius> (DL/dSL/jump are scale-invariant).
    """
    # assert dl_scal * dsl_scal == 0

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

    return qlm_sigma

def bio_offsurf_apply_cplx(trg: jax.Array, qlm_sigma: jax.Array, S: SphereDict, sh: shtns_jax.sht, sl_scal: float, dl_scal: float, far: bool = False) -> jax.Array:
    """
    Evaluate the KL formulation of <S> with density coefficients <qlm_sigma> (SH basis)
    at target points <trg>. Returns point values (Ntrg x 1).
    """
    if not far:
        SLsigma = Lap3d_sl_cplx(trg, qlm_sigma, S, sh)
        DLsigma = Lap3d_dl_cplx(trg, qlm_sigma, S, sh)
        return sl_scal * SLsigma + dl_scal * DLsigma
    # One fused pass over the Ntrg x Nsrc pairs (and one synth) for both operators.
    return _lap_far_cplx(trg, qlm_sigma, S, sh, ("sl", "dl"), sl_scal=sl_scal, dl_scal=dl_scal)

def bio_offsurf_apply(trg: jax.Array, qlm_sigma: jax.Array, S: SphereDict, sh: shtns_jax.sht, sl_scal: float, dl_scal: float, far: bool = False) -> jax.Array:
    """
    Real/truncated-layout counterpart of bio_offsurf_apply_cplx.
    Evaluate the KL formulation of <S> with density coefficients <qlm_sigma> (real SH basis)
    at target points <trg>. Returns point values (Ntrg x 1).
    """
    if not far:
        SLsigma = Lap3d_sl(trg, qlm_sigma, S, sh)
        DLsigma = Lap3d_dl(trg, qlm_sigma, S, sh)
        return sl_scal * SLsigma + dl_scal * DLsigma
    # One fused pass over the Ntrg x Nsrc pairs (and one synth) for both operators.
    return _lap_far(trg, qlm_sigma, S, sh, ("sl", "dl"), sl_scal=sl_scal, dl_scal=dl_scal)

def bio_offsurf_apply_1sph(Strg: SphereDict, shtrg: shtns_jax.sht, qlm_sigma: jax.Array, S: SphereDict, sh: shtns_jax.sht, sl_scal: float, dl_scal: float) -> jax.Array:
    """
    Evaluate the KL formulation of <S> with density coefficients <qlm_sigma> (SH basis)
    at a spherical grid of targets on <Strg>. Returns the potential coefficients on the
    target grid (SH basis).
    """

    SLsigma = Lap3d_sl_r_1sph_cplx(Strg, shtrg, qlm_sigma, S, sh)
    DLsigma = Lap3d_dl_r_1sph_cplx(Strg, shtrg, qlm_sigma, S, sh)
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

    # DIRECT solve -- densities are handled in the SH (diagonalizing) basis: COB the
    # boundary data with analys before the solve, invert with synth before the error.
    qlm_bc = sh.analys_cplx_jax(BC_pot)
    qlm_sig = bio_onsurf_direct_solve_cplx(
        qlm_bc,
        sh=sh,
        sl_scal=sl_scal,
        dl_scal=dl_scal,
        sgn=sgn
    )
    qlm_bc_check = bio_onsurf_apply_cplx(qlm_sig, sh, sl_scal, dl_scal, sgn)
    resid_direct = jnp.linalg.norm(qlm_bc_check - qlm_bc)
    jax.debug.print("Residual of DIRECT solve: {a}", a=resid_direct)

    # Accuracy
    xtrg = Strg["Xcart"][:,:,0]
    ytrg = Strg["Xcart"][:,:,1]
    ztrg = Strg["Xcart"][:,:,2]
    trg_sphere2 = jnp.column_stack([jnp.reshape(xtrg,-1), jnp.reshape(ytrg,-1), jnp.reshape(ztrg,-1)])
    true_pot = compute_potential(trg_sphere2, ptsrc, force)
    true_pot = jnp.real(true_pot)

    qlm_out = bio_offsurf_apply_1sph(Strg, shtrg, qlm_sig, S, sh, sl_scal, dl_scal)
    Ksig_direct = jnp.real(jnp.reshape(shtrg.synth_cplx_jax(qlm_out), (-1, 1)))

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
    qlm_bc = sh.analys_cplx_jax(BC_flux)
    qlm_sig = bio_onsurf_direct_solve_cplx(
        qlm_bc,
        sh=sh,
        sl_scal=0.,
        dl_scal=0.,
        sgn=sgn,
        dsl_scal=1.0
    )
    qlm_bc_check = bio_onsurf_apply_cplx(qlm_sig, sh, 0., 0., sgn, 1.0)
    resid_direct = jnp.linalg.norm(qlm_bc_check - qlm_bc)
    jax.debug.print("Residual of DIRECT solve: {a}", a=resid_direct)

    # Accuracy
    qlm_out = bio_offsurf_apply_1sph(Strg, shtrg, qlm_sig, S, sh, 1.0, 0.)
    Ksig_direct = jnp.real(jnp.reshape(shtrg.synth_cplx_jax(qlm_out), (-1, 1)))

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
    qlm_bc = sh.analys_cplx_jax(BC_pot)
    qlm_sig = bio_onsurf_direct_solve_cplx(
        qlm_bc,
        sh=sh,
        sl_scal=sl_scal,
        dl_scal=dl_scal,
        sgn=sgn
    )
    qlm_bc_check = bio_onsurf_apply_cplx(qlm_sig, sh, sl_scal, dl_scal, sgn)
    resid_direct = jnp.linalg.norm(qlm_bc_check - qlm_bc)
    jax.debug.print("Residual of DIRECT solve: {a}", a=resid_direct)

    # Accuracy
    xtrg = Strg["Xcart"][:,:,0]
    ytrg = Strg["Xcart"][:,:,1]
    ztrg = Strg["Xcart"][:,:,2]
    trg_sphere2 = jnp.column_stack([jnp.reshape(xtrg,-1), jnp.reshape(ytrg,-1), jnp.reshape(ztrg,-1)])
    true_pot = compute_potential(trg_sphere2, ptsrc, force)
    true_pot = jnp.real(true_pot)

    qlm_out = bio_offsurf_apply_1sph(Strg, shtrg, qlm_sig, S, sh, sl_scal, dl_scal)
    Ksig_direct = jnp.real(jnp.reshape(shtrg.synth_cplx_jax(qlm_out), (-1, 1)))

    diff_direct = jnp.max(true_pot - Ksig_direct) / jnp.max(true_pot)
    jax.debug.print("Max relative error of order {lmax} solver at target radius {Rtrg} for direct solver is {d}", lmax=lmax, Rtrg=Rtrg, d=diff_direct)
