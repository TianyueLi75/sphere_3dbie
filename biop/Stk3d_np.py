"""
Numpy-only twin of the Stk3d operators used inside the Stk3d_onsurf_solve_spla matvec.

TEMPORARY DIAGNOSTIC module. Every function here is a pure-numpy / plain-shtns mirror of a
counterpart in biop/Stk3d.py, with all JAX removed:
  - jnp.*                    -> np.*
  - @jax.jit / jax.jit(...)  -> plain functions / closures
  - jax.vmap over rings      -> an explicit numpy loop over target rings
  - shtns_jax *_jax methods  -> the inherited plain shtns C methods (validated to match the
                                _jax variants to ~machine precision):
        analys_vec_cplx_jax(x)      -> spat_cplx_to_SHqst(Vr,Vt,Vp, Qlm,Slm,Tlm)   (out-args)
        synth_vec_cplx_jax(x)       -> SHqst_to_spat_cplx(Qlm,Slm,Tlm, Vr,Vt,Vp)   (out-args)
        SHqst_to_lat_jax(Q,S,T,c,n) -> SHqst_to_lat(Q,S,T,c, Vr,Vt,Vp)             (out-args)
        rotation.apply_cplx_jax(q)  -> rotation.apply_cplx(q)

The sh objects are the SAME shtns_jax.sht instances used by the JAX path (shtns_jax.sht
subclasses shtns.sht), so no object swap is needed -- we just call the plain methods on them.
This module is selected by Stk3d_onsurf_solve_spla(..., numpy=True); the JAX Stk3d.py is left
untouched so the lineax solver and off-surface field/VTK eval keep working.
"""

import numpy as np
import shtns
import shtns_jax


# ---------------------------------------------------------------------------
# pure coordinate / basis transforms (jnp -> np)
# ---------------------------------------------------------------------------
def cart2sph_np(Vx, Vy, Vz, theta, phi):
    Vr = Vx * np.sin(theta) * np.cos(phi) + Vy * np.sin(theta) * np.sin(phi) + Vz * np.cos(theta)
    Vt = Vx * np.cos(theta) * np.cos(phi) + Vy * np.cos(theta) * np.sin(phi) - Vz * np.sin(theta)
    Vp = -Vx * np.sin(phi) + Vy * np.cos(phi)
    return Vr, Vt, Vp


def sph2cart_np(Vr, Vt, Vp, theta, phi):
    Vx = Vr * np.sin(theta) * np.cos(phi) + Vt * np.cos(theta) * np.cos(phi) - Vp * np.sin(phi)
    Vy = Vr * np.sin(theta) * np.sin(phi) + Vt * np.cos(theta) * np.sin(phi) + Vp * np.cos(phi)
    Vz = Vr * np.cos(theta) - Vt * np.sin(theta)
    return Vx, Vy, Vz


def qst2vwx_np(qlm, slm, tlm, sh):
    l = np.asarray(sh.zl, dtype=np.float64)
    vlm = (l * slm - qlm) / (2.0 * l + 1.0)
    wlm = ((l + 1.0) * slm + qlm) / (2.0 * l + 1.0)
    xlm = -tlm
    return vlm, wlm, xlm


def vwx2qst_np(vlm, wlm, xlm, sh):
    l = np.asarray(sh.zl, dtype=np.float64)
    slm = vlm + wlm
    qlm = l * (wlm - vlm) - vlm
    tlm = -xlm
    return qlm, slm, tlm


def Stk3d_sl_VWX_diag_np(sh):
    l = np.asarray(sh.zl, dtype=np.float64)
    diag_V = l / (2.0 * l + 1.0) / (2.0 * l + 3.0)
    diag_W = (l + 1.0) / (2.0 * l + 1.0) / (2.0 * l - 1.0)
    diag_X = 1.0 / (2.0 * l + 1.0)
    return diag_V, diag_W, diag_X


def Stk3d_dl_VWX_diag_np(sh):
    l = np.asarray(sh.zl, dtype=np.float64)
    diag_V_ext = (2.0 * l * l + 4 * l + 3) / (2.0 * l + 1.0) / (2.0 * l + 3.0)
    diag_W_ext = 2.0 * (l + 1.0) * (l - 1.0) / (2.0 * l + 1.0) / (2.0 * l - 1.0)
    diag_X_ext = (l - 1.0) / (2.0 * l + 1.0)
    diag_V_int = -2.0 * l * (l + 2) / (2.0 * l + 1.0) / (2.0 * l + 3.0)
    diag_W_int = -(2.0 * l * l + 1.0) / (2.0 * l + 1.0) / (2.0 * l - 1.0)
    diag_X_int = -(l + 2.0) / (2.0 * l + 1.0)
    return diag_V_ext, diag_W_ext, diag_X_ext, diag_V_int, diag_W_int, diag_X_int


def Stk3d_dsl_VWX_diag_np(sh):
    dVe, dWe, dXe, dVi, dWi, dXi = Stk3d_dl_VWX_diag_np(sh)
    return dVi, dWi, dXi, dVe, dWe, dXe


def diag_W2V_np(sh, which):
    l = np.asarray(sh.zl, dtype=np.float64)
    if which == "sl":
        return l / (4.0 * l + 2.0)
    return 2.0 * l * (l - 1.0) / (4.0 * l + 2.0)


def diag_V2W_np(sh, which):
    l = np.asarray(sh.zl, dtype=np.float64)
    if which == "sl":
        return (l + 1.0) / (4.0 * l + 2.0)
    return (l + 1.0) * (l + 2.0) / (2.0 * l + 1.0)


# ---------------------------------------------------------------------------
# plain-shtns vector transforms (preallocated output args)
# ---------------------------------------------------------------------------
def _analys_vec_cplx_np(sh, Vr, Vt, Vp):
    """spatial cplx (each spat_shape) -> QST cplx coeffs (each nlm_cplx). Mirrors
    sh.analys_vec_cplx_jax via the plain sh.spat_cplx_to_SHqst (fills preallocated outputs)."""
    Q = np.zeros(sh.nlm_cplx, dtype=np.complex128)
    S = np.zeros(sh.nlm_cplx, dtype=np.complex128)
    T = np.zeros(sh.nlm_cplx, dtype=np.complex128)
    sh.spat_cplx_to_SHqst(np.ascontiguousarray(Vr, dtype=np.complex128),
                          np.ascontiguousarray(Vt, dtype=np.complex128),
                          np.ascontiguousarray(Vp, dtype=np.complex128), Q, S, T)
    return Q, S, T


def _synth_vec_cplx_np(sh, Qlm, Slm, Tlm):
    """QST cplx coeffs (each nlm_cplx) -> spatial cplx (each spat_shape). Mirrors
    sh.synth_vec_cplx_jax via the plain sh.SHqst_to_spat_cplx (fills preallocated outputs)."""
    Vr = np.zeros(sh.spat_shape, dtype=np.complex128)
    Vt = np.zeros(sh.spat_shape, dtype=np.complex128)
    Vp = np.zeros(sh.spat_shape, dtype=np.complex128)
    sh.SHqst_to_spat_cplx(np.ascontiguousarray(Qlm, dtype=np.complex128),
                          np.ascontiguousarray(Slm, dtype=np.complex128),
                          np.ascontiguousarray(Tlm, dtype=np.complex128), Vr, Vt, Vp)
    return Vr, Vt, Vp


def sig_xyz2vwx_np(sigma_x, sigma_y, sigma_z, theta, phi, sh):
    sr, st, sp = cart2sph_np(sigma_x, sigma_y, sigma_z, theta, phi)
    qlm, slm, tlm = _analys_vec_cplx_np(sh, sr, st, sp)
    return qst2vwx_np(qlm, slm, tlm, sh)


def sig_vwx2xyz_np(vlm, wlm, xlm, theta, phi, sh):
    qlm, slm, tlm = vwx2qst_np(vlm, wlm, xlm, sh)
    vr, vt, vp = _synth_vec_cplx_np(sh, qlm, slm, tlm)
    return sph2cart_np(vr, vt, vp, theta, phi)


# ---------------------------------------------------------------------------
# self block + preconditioner
# ---------------------------------------------------------------------------
def bio_onsurf_apply_np(sigma_tens, theta, phi, sh, sl_scal, dl_scal, sgn, dsl_scal=0., radius=1.0):
    sigma_x = sigma_tens[:, :, 0]
    sigma_y = sigma_tens[:, :, 1]
    sigma_z = sigma_tens[:, :, 2]
    vlm, wlm, xlm = sig_xyz2vwx_np(sigma_x, sigma_y, sigma_z, theta, phi, sh)

    dV, dW, dX = Stk3d_sl_VWX_diag_np(sh)
    vlm_SL = radius * dV * vlm
    wlm_SL = radius * dW * wlm
    xlm_SL = radius * dX * xlm

    dVe, dWe, dXe, dVi, dWi, dXi = Stk3d_dl_VWX_diag_np(sh)
    vlm_DL = 0.5 * (dVi + dVe) * vlm
    wlm_DL = 0.5 * (dWi + dWe) * wlm
    xlm_DL = 0.5 * (dXi + dXe) * xlm

    dVe, dWe, dXe, dVi, dWi, dXi = Stk3d_dsl_VWX_diag_np(sh)
    vlm_dSL = 0.5 * (dVi + dVe) * vlm
    wlm_dSL = 0.5 * (dWi + dWe) * wlm
    xlm_dSL = 0.5 * (dXi + dXe) * xlm

    vlm_op = sl_scal * vlm_SL + dl_scal * vlm_DL + dsl_scal * vlm_dSL
    wlm_op = sl_scal * wlm_SL + dl_scal * wlm_DL + dsl_scal * wlm_dSL
    xlm_op = sl_scal * xlm_SL + dl_scal * xlm_DL + dsl_scal * xlm_dSL

    vx, vy, vz = sig_vwx2xyz_np(vlm_op, wlm_op, xlm_op, theta, phi, sh)
    vx = vx + 0.5 * sgn * dl_scal * sigma_x + 0.5 * (-1 * sgn) * dsl_scal * sigma_x
    vy = vy + 0.5 * sgn * dl_scal * sigma_y + 0.5 * (-1 * sgn) * dsl_scal * sigma_y
    vz = vz + 0.5 * sgn * dl_scal * sigma_z + 0.5 * (-1 * sgn) * dsl_scal * sigma_z
    return np.stack([vx, vy, vz], axis=2)


def stokes_onsurf_direct_solve_np(bc_vec, theta, phi, sh, sl_scal, dl_scal, sgn, dsl_scal=0., radius=1.0):
    vlm_bc, wlm_bc, xlm_bc = sig_xyz2vwx_np(bc_vec[:, :, 0], bc_vec[:, :, 1], bc_vec[:, :, 2], theta, phi, sh)
    dV_sl, dW_sl, dX_sl = Stk3d_sl_VWX_diag_np(sh)
    dVe, dWe, dXe, dVi, dWi, dXi = Stk3d_dl_VWX_diag_np(sh)
    dV_dl = 0.5 * (dVi + dVe); dW_dl = 0.5 * (dWi + dWe); dX_dl = 0.5 * (dXi + dXe)
    dVe, dWe, dXe, dVi, dWi, dXi = Stk3d_dsl_VWX_diag_np(sh)
    dV_dsl = 0.5 * (dVi + dVe); dW_dsl = 0.5 * (dWi + dWe); dX_dsl = 0.5 * (dXi + dXe)

    op_diag_V = (0.5 * dl_scal * sgn) + (dl_scal * dV_dl) + (radius * sl_scal * dV_sl) + (0.5 * dsl_scal * (-1 * sgn)) + (dsl_scal * dV_dsl)
    op_diag_W = (0.5 * dl_scal * sgn) + (dl_scal * dW_dl) + (radius * sl_scal * dW_sl) + (0.5 * dsl_scal * (-1 * sgn)) + (dsl_scal * dW_dsl)
    op_diag_X = (0.5 * dl_scal * sgn) + (dl_scal * dX_dl) + (radius * sl_scal * dX_sl) + (0.5 * dsl_scal * (-1 * sgn)) + (dsl_scal * dX_dsl)

    eps = 1e-14

    def safe_div(bc_lm, op_diag):
        safe = np.where(np.abs(op_diag) > eps, op_diag, 1.0 + 0j)
        res = bc_lm / safe
        return np.where(np.abs(op_diag) <= eps, bc_lm, res)

    vlm_sigma = safe_div(vlm_bc, op_diag_V)
    wlm_sigma = safe_div(wlm_bc, op_diag_W)
    xlm_sigma = safe_div(xlm_bc, op_diag_X)

    sig_x, sig_y, sig_z = sig_vwx2xyz_np(vlm_sigma, wlm_sigma, xlm_sigma, theta, phi, sh)
    return np.stack([sig_x, sig_y, sig_z], axis=-1)


# ---------------------------------------------------------------------------
# point-and-shoot near evaluator (numpy, ring loop instead of jax.vmap)
# ---------------------------------------------------------------------------
def _latlm_maps_np(sh):
    """Cached (kpos, kneg, parity) for the cplx->real-layout G/H split. Cached under a
    separate attr (_ps_latlm_maps_np) so it never collides with the JAX _ps_latlm_maps."""
    cached = getattr(sh, "_ps_latlm_maps_np", None)
    if cached is not None:
        return cached
    lr = np.asarray(sh.l, dtype=np.int64)
    mr = np.asarray(sh.m, dtype=np.int64)
    kpos = (lr * (lr + 1) + mr).astype(np.int64)
    kneg = (lr * (lr + 1) - mr).astype(np.int64)
    parity = ((-1.0) ** mr).astype(np.float64)
    maps = (kpos, kneg, parity)
    sh._ps_latlm_maps_np = maps
    return maps


def _stk_latitude_ring_np(qlm, slm, tlm, cos_src, sh):
    """Single-ring FFT synthesis (numpy). cplx QST (each nlm_cplx) at one latitude cos_src
    over all nphi longitudes -> (vr, vt, vp) each (nphi,) complex, source-centered basis."""
    kpos, kneg, parity = _latlm_maps_np(sh)

    def split(z):
        zp = z[kpos]; zn = z[kneg]
        aG = 0.5 * (zp + parity * np.conj(zn))
        aH = (zp - parity * np.conj(zn)) / 2j
        return np.ascontiguousarray(aG, dtype=np.complex128), np.ascontiguousarray(aH, dtype=np.complex128)

    Qg, Qh = split(qlm); Sg, Sh = split(slm); Tg, Th = split(tlm)
    c = float(cos_src)
    VrG = np.zeros(sh.nphi, dtype=np.float64); VtG = np.zeros(sh.nphi, dtype=np.float64); VpG = np.zeros(sh.nphi, dtype=np.float64)
    sh.SHqst_to_lat(Qg, Sg, Tg, c, VrG, VtG, VpG)
    VrH = np.zeros(sh.nphi, dtype=np.float64); VtH = np.zeros(sh.nphi, dtype=np.float64); VpH = np.zeros(sh.nphi, dtype=np.float64)
    sh.SHqst_to_lat(Qh, Sh, Th, c, VrH, VtH, VpH)
    return VrG + 1j * VrH, VtG + 1j * VtH, VpG + 1j * VpH


def _ps_scale_coeffs(sh, exterior, which):
    """Ring-INDEPENDENT VWX diagonal + coupling symbols for operator 'which' ('sl'/'dl') on
    the exterior/interior branch. Depends only on sh.zl and the (static) branch, so this is
    computed ONCE per evaluator and reused across all target rings (was previously recomputed
    twice per ring inside _ps_scale_vwx_np -- the dominant redundant cost in the profile)."""
    if which == "sl":
        dV, dW, dX = Stk3d_sl_VWX_diag_np(sh)
    else:
        dVe, dWe, dXe, dVi, dWi, dXi = Stk3d_dl_VWX_diag_np(sh)
        dV, dW, dX = (dVe, dWe, dXe) if exterior else (dVi, dWi, dXi)
    coup = diag_W2V_np(sh, which) if exterior else diag_V2W_np(sh, which)
    return dV, dW, dX, coup


def _ps_scale_vwx_np(vlm, wlm, xlm, rho, l_vals, exterior, which, coeffs):
    """Per-ring solid-harmonic radial scaling (V/W/X basis) using precomputed <coeffs>
    (dV, dW, dX, coup) from _ps_scale_coeffs. Only the rho powers are ring-dependent."""
    dV, dW, dX, coup = coeffs
    if exterior:
        c = (rho ** (-l_vals - 2.0) - rho ** (-l_vals)) * coup * wlm
        vlm_o = rho ** (-l_vals - 2.0) * dV * vlm + c
        wlm_o = rho ** (-l_vals) * dW * wlm
        xlm_o = rho ** (-l_vals - 1.0) * dX * xlm
    else:
        sgn = 1.0 if which == "sl" else -1.0
        c = sgn * (rho ** (l_vals + 1.0) - rho ** (l_vals - 1.0)) * coup * vlm
        vlm_o = rho ** (l_vals + 1.0) * dV * vlm
        wlm_o = rho ** (l_vals - 1.0) * dW * wlm + c
        xlm_o = rho ** (l_vals) * dX * xlm
    return vlm_o, wlm_o, xlm_o


def _ps_ring_core_np(vlm, wlm, xlm, rho_j, cos_src_j, theta_src_j, th_e_j,
                     phi_ring, l_vals, exterior, a, sl_scal, dl_scal, sh_eval,
                     sl_coeffs, dl_coeffs):
    vSL, wSL, xSL = _ps_scale_vwx_np(vlm, wlm, xlm, rho_j, l_vals, exterior, "sl", sl_coeffs)
    vDL, wDL, xDL = _ps_scale_vwx_np(vlm, wlm, xlm, rho_j, l_vals, exterior, "dl", dl_coeffs)
    vK = sl_scal * a * vSL + dl_scal * vDL
    wK = sl_scal * a * wSL + dl_scal * wDL
    xK = sl_scal * a * xSL + dl_scal * xDL
    qlm_K, slm_K, tlm_K = vwx2qst_np(vK, wK, xK, sh_eval)
    vr, vt, vp = _stk_latitude_ring_np(qlm_K, slm_K, tlm_K, cos_src_j, sh_eval)
    uxR, uyR, uzR = sph2cart_np(vr, vt, vp, theta_src_j, phi_ring)
    return cart2sph_np(uxR, uyR, uzR, th_e_j, phi_ring)


def _ps_rotation_np(t_vec, lmax_src, lmax_trg):
    """Numpy twin of Stk3d._ps_rotation: build shtns rotation objects (plain apply_cplx is
    called later). Returns (rot_fwd, rot_inv, d)."""
    t = np.asarray(t_vec, dtype=np.float64).reshape(3)
    d = np.linalg.norm(t)
    that = t / d
    beta = np.arccos(np.clip(that[2], -1.0, 1.0))
    axis = np.array([that[1], -that[0], 0.0])
    nrm = np.linalg.norm(axis)
    axis = np.array([1.0, 0.0, 0.0]) if nrm < 1e-14 else axis / nrm
    beta = float(beta); ax = (float(axis[0]), float(axis[1]), float(axis[2]))
    rot_fwd = shtns_jax.rotation(lmax_src, lmax_src, 0)
    rot_fwd.set_angle_axis(beta, *ax)
    rot_inv = shtns_jax.rotation(lmax_trg, lmax_trg, 0)
    rot_inv.set_angle_axis(-beta, *ax)
    return rot_fwd, rot_inv, float(d)


def _ps_target_rings_np(d, R_t, theta_std):
    theta_std = np.asarray(theta_std, dtype=np.float64)
    ct = np.cos(theta_std); st = np.sin(theta_std)
    z = d + R_t * ct
    r = np.sqrt((R_t * st) ** 2 + z * z)
    return r, z / r


def point_n_shoot_evaluator_np(Strg, shtrg, S, sh, near=False):
    """Numpy twin of Stk3d.point_n_shoot_evaluator. Same 3-stage move-pole algorithm, but the
    returned closure is a plain Python function (no jax.jit) and stage 2 loops over target rings
    (replacing jax.vmap over _ps_ring_core). Stages 1/3 use the plain rotation.apply_cplx."""
    if S["lmax"] != sh.lmax:
        sh = shtns_jax.sht(S["lmax"], S["lmax"])
    if Strg["lmax"] != shtrg.lmax:
        shtrg = shtns_jax.sht(Strg["lmax"], Strg["lmax"])

    a = float(S["r"]); R_t = float(Strg["r"])
    t_vec = np.asarray(Strg["Xc"], dtype=np.float64) - np.asarray(S["Xc"], dtype=np.float64)
    sh_eval = sh if sh.lmax >= shtrg.lmax else shtrg
    rot_fwd, rot_inv, d = _ps_rotation_np(t_vec, sh.lmax, sh_eval.lmax)

    if abs(d - R_t) > a:
        exterior = True
    elif d + R_t < a:
        exterior = False
    else:
        raise ValueError(
            f"point_n_shoot requires the target sphere wholly exterior or interior to the "
            f"source (non-overlapping); got d={d}, a={a}, R_t={R_t}.")

    th_s = np.asarray(S["Xsph"][:, :, 0], dtype=np.float64); ph_s = np.asarray(S["Xsph"][:, :, 1], dtype=np.float64)
    th_t = np.asarray(Strg["Xsph"][:, :, 0], dtype=np.float64); ph_t = np.asarray(Strg["Xsph"][:, :, 1], dtype=np.float64)
    nlm_e = sh_eval.nlm_cplx; nlm_t = shtrg.nlm_cplx
    l_vals = np.asarray(sh_eval.zl, dtype=np.float64)
    theta_e = np.arccos(np.asarray(sh_eval.cos_theta, dtype=np.float64))
    r_j, cos_src = _ps_target_rings_np(d, R_t, theta_e)
    rho_ring = r_j / a
    theta_src_ring = np.arccos(cos_src)
    th_e_ring = theta_e
    ph_ring_1d = np.arange(sh_eval.nphi) * (2.0 * np.pi / sh_eval.nphi)
    ntheta_e = theta_e.shape[0]
    nphi_e = sh_eval.nphi
    _latlm_maps_np(sh_eval)
    # Ring-independent VWX scaling symbols: computed ONCE here (depend only on sh_eval.zl and
    # the static exterior branch), reused across every ring in _core (removes the per-ring diag
    # recomputation that dominated the numpy profile).
    sl_coeffs = _ps_scale_coeffs(sh_eval, exterior, "sl")
    dl_coeffs = _ps_scale_coeffs(sh_eval, exterior, "dl")

    def _pad(z, n):
        return np.concatenate([z, np.zeros(n - z.shape[0], dtype=np.complex128)]) if n > z.shape[0] else z[:n]

    def _core(sigma_cart, sl_scal, dl_scal):
        sigma_cart = np.asarray(sigma_cart, dtype=np.complex128)
        # source density -> Q/S/T cplx coefficients
        sr, st_s, sp_s = cart2sph_np(sigma_cart[:, :, 0], sigma_cart[:, :, 1], sigma_cart[:, :, 2], th_s, ph_s)
        qst0, qst1, qst2 = _analys_vec_cplx_np(sh, sr, st_s, sp_s)

        # STAGE 1: rotate each scalar potential so the target center -> +z
        qR = _pad(rot_fwd.apply_cplx(qst0), nlm_e)
        sR = _pad(rot_fwd.apply_cplx(qst1), nlm_e)
        tR = _pad(rot_fwd.apply_cplx(qst2), nlm_e)

        # STAGE 2: per-ring radial scaling + latitude FFT (loop over rings)
        vlm, wlm, xlm = qst2vwx_np(qR, sR, tR, sh_eval)
        vr_e = np.empty((ntheta_e, nphi_e), dtype=np.complex128)
        vt_e = np.empty((ntheta_e, nphi_e), dtype=np.complex128)
        vp_e = np.empty((ntheta_e, nphi_e), dtype=np.complex128)
        for j in range(ntheta_e):
            rj, tj, tsj, thj = float(rho_ring[j]), float(cos_src[j]), float(theta_src_ring[j]), float(th_e_ring[j])
            vrj, vtj, vpj = _ps_ring_core_np(vlm, wlm, xlm, rj, tj, tsj, thj,
                                             ph_ring_1d, l_vals, exterior, a, sl_scal, dl_scal, sh_eval,
                                             sl_coeffs, dl_coeffs)
            vr_e[j] = vrj; vt_e[j] = vtj; vp_e[j] = vpj
        # ring-major (ntheta_e, nphi_e) -> (nphi_e, ntheta_e)
        vr_e = vr_e.T; vt_e = vt_e.T; vp_e = vp_e.T

        # STAGE 3: rotate the sampled field back, band-limit to the target grid
        qR2, sR2, tR2 = _analys_vec_cplx_np(sh_eval, vr_e, vt_e, vp_e)
        qg = _pad(rot_inv.apply_cplx(qR2), nlm_t)
        sg = _pad(rot_inv.apply_cplx(sR2), nlm_t)
        tg = _pad(rot_inv.apply_cplx(tR2), nlm_t)
        vr_g, vt_g, vp_g = _synth_vec_cplx_np(shtrg, qg, sg, tg)
        ux, uy, uz = sph2cart_np(vr_g, vt_g, vp_g, th_t, ph_t)
        return np.stack([ux, uy, uz], axis=2)

    return _core


# ---------------------------------------------------------------------------
# far smooth-quadrature (numpy dense Nystrom sums)
# ---------------------------------------------------------------------------
def Stk3d_sl_far_np(trg, S, sh):
    trg = np.asarray(trg, dtype=np.float64)
    assert trg.shape[1] == 3
    grid_shape = S["Xcart"].shape[:2]
    ysrc = np.asarray(S["Xcart"], dtype=np.float64).reshape(-1, 3)
    fsrc = np.asarray(S["Sigma"], dtype=np.complex128).reshape(-1, 3)
    wts = np.broadcast_to(np.asarray(sh._grid_weights(), dtype=np.float64), grid_shape).reshape(-1) * float(S["r"]) ** 2
    r = trg[:, None, :] - ysrc[None, :, :]
    d = np.linalg.norm(r, axis=2)
    rdotf = np.sum(r * fsrc[None, :, :], axis=2)
    term = fsrc[None, :, :] / d[..., None] + r * (rdotf / d ** 3)[..., None]
    prefac = 1.0 / 8.0 / np.pi
    return prefac * np.sum(term * wts[None, :, None], axis=1)


def Stk3d_dl_far_np(trg, S, sh):
    trg = np.asarray(trg, dtype=np.float64)
    assert trg.shape[1] == 3
    grid_shape = S["Xcart"].shape[:2]
    ysrc = np.asarray(S["Xcart"], dtype=np.float64).reshape(-1, 3)
    fsrc = np.asarray(S["Sigma"], dtype=np.complex128).reshape(-1, 3)
    nsrc = np.asarray(S["Xncart"], dtype=np.float64).reshape(-1, 3)
    wts = np.broadcast_to(np.asarray(sh._grid_weights(), dtype=np.float64), grid_shape).reshape(-1) * float(S["r"]) ** 2
    r = trg[:, None, :] - ysrc[None, :, :]
    d = np.linalg.norm(r, axis=2)
    rdotn = np.sum(r * nsrc[None, :, :], axis=2)
    rdots = np.sum(r * fsrc[None, :, :], axis=2)
    prefac = 6.0 / 8.0 / np.pi
    return prefac * np.sum(r * (rdotn * rdots / d ** 5 * wts[None, :])[..., None], axis=1)


def bio_offsurf_apply_np(trg, S, sh, sl_scal, dl_scal, far=True):
    """Numpy far-only twin of Stk3d.bio_offsurf_apply. Only far=True is supported (the matvec
    far evaluators always force the smooth quadrature)."""
    if far is not True:
        raise NotImplementedError("Stk3d_np.bio_offsurf_apply_np only supports far=True")
    SLsigma = Stk3d_sl_far_np(trg, S, sh)
    DLsigma = Stk3d_dl_far_np(trg, S, sh)
    return sl_scal * SLsigma + dl_scal * DLsigma
