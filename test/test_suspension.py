"""Coupled multi-sphere suspension tests (moved from suspension.py __main__).

The suspension coupled solvers operate in the real/truncated SH layout only (the complex/full
layout was deprecated in the real-transform migration), so -- unlike the single-sphere biop
tests -- these exercise the truncated transforms exclusively. TEST 1/2 are manufactured-solution
accuracy checks against exact point-singularity fields; TEST 3 is a squirmer-in-container solve
with no closed form, checked for GMRES convergence.
"""

import jax.numpy as jnp
import pytest

from suspension import (build_suspension, quadr_suspension,
                        Lap3d_onsurf_solve, Stk3d_onsurf_solve_spla)
from biop import Lap3d, Stk3d

SL, DL = 1.0, 1.0
SEP_ETA = 0.0001                                  # tiny -> every off-diagonal pair is FAR
_CHK = jnp.array([[6., 1., 0.5], [1.5, 5., -2.], [-4., -3., 2.]])   # well separated from both spheres


def test_two_sphere_exterior_laplace():
    """TEST 1: two well-separated unit spheres, one interior point charge each; coupled
    exterior Laplace BIE, checked against the exact potential at far exterior points."""
    lmax = 32
    centers = jnp.array([[0., 0., 0.], [3., 0., 0.]])
    radii = jnp.array([1.0, 1.0])
    Sp = build_suspension(centers, radii, SEP_ETA)
    Sp, sh_lst = quadr_suspension(Sp, jnp.array([lmax, lmax]))
    Ns = Sp["Ns"]
    sl_lst = [SL] * Ns; dl_lst = [DL] * Ns; sgn_lst = [1.0] * Ns

    ptsrc = jnp.array([[0.1, 0.3, 0.15], [2.8, 0.1, -0.15]])
    force = jnp.array([[1.0], [-0.7]])
    dsp = Sp["Nnodes_dsp"]
    bc = jnp.zeros((int(dsp[-1]),), dtype=jnp.complex128)
    for s in range(Ns):
        nodes = Sp["spheres_lst"][s]["Xcart"].reshape(-1, 3)
        bc = bc.at[int(dsp[s]):int(dsp[s + 1])].set(
            Lap3d.compute_potential(nodes, ptsrc, force).reshape(-1))

    sigma, info, resid = Lap3d_onsurf_solve(bc, Sp, sh_lst, sl_lst, dl_lst, sgn_lst)
    assert info == 0, f"GMRES did not converge (info={info})"
    assert float(resid) < 1e-10, f"residual too large: {float(resid):.3e}"

    true_pot = jnp.real(Lap3d.compute_potential(_CHK, ptsrc, force))
    approx = jnp.zeros((_CHK.shape[0], 1), dtype=jnp.float64)
    for s in range(Ns):
        grid = Sp["spheres_lst"][s]["Xcart"].shape[:2]
        qlm_s = sh_lst[s].analys_jax(jnp.real(sigma[int(dsp[s]):int(dsp[s + 1])]).reshape(grid))
        approx = approx + Lap3d.bio_offsurf_apply(_CHK, qlm_s, Sp["spheres_lst"][s], sh_lst[s], SL, DL)
    rel = float(jnp.max(jnp.abs(true_pot - approx)) / jnp.max(jnp.abs(true_pot)))
    assert rel < 1e-9, f"relative error too large: {rel:.3e}"


def test_two_sphere_exterior_stokes():
    """TEST 2: two well-separated spheres of unequal radius (exercises solid-harmonic radius
    scaling), one interior point Stokeslet each; coupled exterior Stokes BIE (scipy GMRES),
    checked against the exact velocity at far exterior points."""
    lmax = 24
    centers = jnp.array([[0., 0., 0.], [3., 0., 0.]])
    radii = jnp.array([1.0, 0.5])
    Sp = build_suspension(centers, radii, SEP_ETA)
    Sp, sh_lst = quadr_suspension(Sp, jnp.array([lmax, lmax]))
    Ns = Sp["Ns"]
    sl_lst = [SL] * Ns; dl_lst = [DL] * Ns; sgn_lst = [1.0] * Ns

    ptsrc = jnp.array([[0.1, 0.3, 0.15], [2.95, 0.1, 0.05]])
    force = jnp.array([[1.0, 0.5, -0.3], [-0.7, 0.2, 0.4]])
    dsp = Sp["Nnodes_dsp"]
    bc = jnp.zeros((3 * int(dsp[-1]),), dtype=jnp.complex128)
    for s in range(Ns):
        nodes = Sp["spheres_lst"][s]["Xcart"].reshape(-1, 3)
        bc = bc.at[3 * int(dsp[s]):3 * int(dsp[s + 1])].set(
            Stk3d.compute_field(nodes, ptsrc, force).reshape(-1))

    Nnodes = dsp[-1].item()
    sigma, t_solve, niter, info, resid = Stk3d_onsurf_solve_spla(
        bc, Sp, Ns, Nnodes, sh_lst, sl_lst, dl_lst, sgn_lst)
    assert info == 0, f"GMRES did not converge (info={info})"
    assert float(resid) < 1e-8, f"residual too large: {float(resid):.3e}"

    true_vel = jnp.real(Stk3d.compute_field(_CHK, ptsrc, force))
    approx = jnp.zeros((_CHK.shape[0], 3), dtype=jnp.complex128)
    for s in range(Ns):
        s_sph = Sp["spheres_lst"][s]
        nphi, ntheta = s_sph["Xcart"].shape[:2]
        sig_s = sigma[3 * int(dsp[s]):3 * int(dsp[s + 1])].reshape(nphi, ntheta, 3)
        vwx_s = jnp.stack(Stk3d.sig_xyz2vwx(sig_s[:, :, 0], sig_s[:, :, 1], sig_s[:, :, 2],
                                            s_sph["Xsph"][:, :, 0], s_sph["Xsph"][:, :, 1], sh_lst[s]))
        approx = approx + Stk3d.bio_offsurf_apply(_CHK, vwx_s, s_sph, sh_lst[s], SL, DL, far=True)
    rel = float(jnp.max(jnp.abs(true_vel - jnp.real(approx))) / jnp.max(jnp.abs(true_vel)))
    assert rel < 1e-7, f"relative error too large: {rel:.3e}"


def test_obstacle_in_container_stokes():
    """TEST 3: a slip obstacle (squirmer, exterior sgn=+1) inside a no-slip container
    (interior sgn=-1). No closed-form solution, so this checks that the coupled combined-field
    system converges under the scipy block-Jacobi-preconditioned GMRES."""
    lmax = 48
    centers = jnp.array([[0., 0., 0.], [0.3, 0.1, -0.05]])
    radii = jnp.array([1.0, 0.2])
    Sp = build_suspension(centers, radii, SEP_ETA)
    Sp, sh_lst = quadr_suspension(Sp, jnp.array([lmax, lmax]))
    Ns = Sp["Ns"]
    sl_lst = [SL, SL]; dl_lst = [DL, DL]; sgn_lst = [-1.0, 1.0]

    dsp = Sp["Nnodes_dsp"]
    bc = jnp.zeros((3 * int(dsp[-1]),), dtype=jnp.complex128)     # no-slip container (sphere 0)
    U = 1.0
    vslip = lambda t: jnp.sin(t) * 3. / 2. * U                   # squirmer tangential surface speed
    obst = Sp["spheres_lst"][1]
    th1, ph1 = obst["Xsph"][:, :, 0], obst["Xsph"][:, :, 1]
    zeros1 = jnp.zeros_like(th1)
    sx, sy, sz = Stk3d.sph2cart(zeros1, vslip(th1), zeros1, th1, ph1)
    bc = bc.at[3 * int(dsp[1]):3 * int(dsp[2])].set(jnp.stack([sx, sy, sz], axis=2).reshape(-1))

    Nnodes = dsp[-1].item()
    sigma, t_solve, niter, info, resid = Stk3d_onsurf_solve_spla(
        bc, Sp, Ns, Nnodes, sh_lst, sl_lst, dl_lst, sgn_lst)
    assert info == 0, f"GMRES did not converge (info={info})"
    assert float(resid) < 1e-9, f"residual too large: {float(resid):.3e}"
