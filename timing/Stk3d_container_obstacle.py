"""Timing + self-convergence benchmark for the container-obstacle Stokes problem.

Problem (suspension TEST 3): a squirming obstacle (tangential slip) inside a no-slip
spherical container. There is no closed-form solution, so accuracy is assessed by
*self-convergence*: the velocity field evaluated at a fixed cloud of interior check points
should converge as lmax increases. For each lmax we also time the coupled GMRES solve and the
off-surface field evaluation.

Run (from repo root, CPU venv):
    python timing/Stk3d_container_obstacle.py
"""

import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from suspension import build_suspension, quadr_suspension, Stk3d_onsurf_solve
from sphere import set_density
from biop import Stk3d

jax.config.update("jax_enable_x64", True)


class IterationCounter:
    """Counts GMRES iterations (passed as the scipy gmres callback)."""
    def __init__(self):
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1


# Fixed problem geometry / parameters (matches suspension.py TEST 3).
CENTERS = jnp.array([[0., 0., 0.], [0.3, 0.1, -0.05]])
RADII = jnp.array([1.0, 0.2])
SEP_ETA = 0.1
SL_SCAL = 1.0
DL_SCAL = 1.0
SGN_LST = [-1.0, 1.0]   # interior container, exterior obstacle
U = 1.0
vslip_mag = lambda theta: jnp.sin(theta) * 3. / 2. * U   # squirmer tangential surface speed


def test(lmax: int, chk: jax.Array):
    """One run at resolution <lmax>; returns timings + the velocity at fixed points <chk>."""
    Sp = build_suspension(CENTERS, RADII, SEP_ETA)
    Sp, sh_lst = quadr_suspension(Sp, jnp.array([lmax, lmax]))
    Ns = Sp["Ns"]
    sl_lst = [SL_SCAL] * Ns
    dl_lst = [DL_SCAL] * Ns
    dsp = Sp["Nnodes_dsp"]

    # Squirmer slip BC: container no-slip (zero), obstacle tangential slip.
    bc = jnp.zeros((3 * int(dsp[-1]),), dtype=jnp.complex128)
    obst = Sp["spheres_lst"][1]
    th1, ph1 = obst["Xsph"][:, :, 0], obst["Xsph"][:, :, 1]
    zeros1 = jnp.zeros_like(th1)
    sx, sy, sz = Stk3d.sph2cart(zeros1, vslip_mag(th1), zeros1, th1, ph1)
    bc = bc.at[3 * int(dsp[1]):3 * int(dsp[2])].set(jnp.stack([sx, sy, sz], axis=2).reshape(-1))

    # Warmup: compile the jitted self-block / preconditioner kernels for this lmax's shapes
    # with a single (untimed) GMRES iteration, so they are excluded from the timed solve.
    Stk3d_onsurf_solve(bc, Sp, sh_lst, sl_lst, dl_lst, SGN_LST, maxiter=1)

    # Time the coupled solve, counting GMRES iterations.
    counter = IterationCounter()
    t0 = time.time()
    sigma, info, resid = Stk3d_onsurf_solve(bc, Sp, sh_lst, sl_lst, dl_lst, SGN_LST, callback=counter)
    t_solve = time.time() - t0
    iters = counter.count

    # Time the off-surface field evaluation at the fixed check points.
    t0 = time.time()
    approx = jnp.zeros((chk.shape[0], 3), dtype=jnp.complex128)
    for s in range(Ns):
        nphi, ntheta = Sp["spheres_lst"][s]["Xcart"].shape[:2]
        sig_s = sigma[3 * int(dsp[s]):3 * int(dsp[s + 1])].reshape(nphi, ntheta, 3)
        s_sph = set_density(Sp["spheres_lst"][s], sig_s[:, :, 0], sig_s[:, :, 1], sig_s[:, :, 2])
        approx = approx + Stk3d.bio_offsurf_apply(chk, s_sph, sh_lst[s], SL_SCAL, DL_SCAL)
    u_chk = np.real(np.asarray(approx))
    t_eval = time.time() - t0

    return t_solve, t_eval, float(resid), int(info), int(iters), u_chk


def make_check_points(n=64, seed=0):
    """Fixed cloud of points interior to the container and clear of the obstacle (for accurate
    spectral evaluation), identical across all lmax."""
    rng = np.random.default_rng(seed)
    R0 = float(RADII[0])
    c1 = np.asarray(CENTERS[1])
    r1 = float(RADII[1])
    pts = []
    while len(pts) < n:
        p = rng.uniform(-R0, R0, size=3)
        if np.linalg.norm(p) < 0.9 * R0 and np.linalg.norm(p - c1) > 1.3 * r1:
            pts.append(p)
    return jnp.asarray(np.array(pts))


if __name__ == "__main__":
    # lmax_list = [2**n for n in range(2, 13)]
    lmax_list = [2**n for n in range(2, 8)]
    chk = make_check_points()

    t_solve = np.zeros(len(lmax_list))
    t_eval = np.zeros(len(lmax_list))
    iters_list = np.zeros(len(lmax_list), dtype=int)
    u_all = []
    print(f"{'lmax':>5} {'info':>5} {'iters':>6} {'residual':>12} {'t_solve(s)':>12} {'t_eval(s)':>12}", flush=True)
    for i, lmax in enumerate(lmax_list):
        print(f"  ... starting lmax={lmax}", flush=True)
        ts, te, resid, info, iters, u_chk = test(int(lmax), chk)
        t_solve[i], t_eval[i], iters_list[i] = ts, te, iters
        u_all.append(u_chk)
        print(f"{lmax:>5d} {info:>5d} {iters:>6d} {resid:>12.3e} {ts:>12.3f} {te:>12.3f}", flush=True)

    # Self-convergence: compare each field to the finest-lmax reference.
    u_ref = u_all[-1]
    denom = np.max(np.abs(u_ref))
    err = np.array([np.max(np.abs(u_all[i] - u_ref)) / denom for i in range(len(lmax_list))])
    print("\nSelf-convergence vs lmax =", lmax_list[-1], "(reference):")
    for i in range(len(lmax_list) - 1):
        print(f"  lmax={lmax_list[i]:>3d}: max rel diff = {err[i]:.3e}")

    here = os.path.dirname(os.path.abspath(__file__))

    # Timing plot.
    plt.figure()
    plt.plot(lmax_list, t_solve, "k*-", label="coupled solve")
    plt.plot(lmax_list, t_eval, "ko-", label="off-surf eval")
    plt.xlabel("lmax"); plt.ylabel("Time (s)")
    plt.title("Container-obstacle Stokes: timing")
    plt.legend()
    plt.savefig(os.path.join(here, "./plots/Stk3d_container_obstacle_timing.png"), dpi=150)

    # Self-convergence plot (drop the reference point, which is 0 by construction).
    plt.figure()
    plt.semilogy(lmax_list[:-1], err[:-1], "k*-")
    plt.xlabel("lmax"); plt.ylabel(f"max rel diff vs lmax={lmax_list[-1]}")
    plt.title("Container-obstacle Stokes: self-convergence")
    plt.savefig(os.path.join(here, "./plots/Stk3d_container_obstacle_convergence.png"), dpi=150)

    print("\nWrote Stk3d_container_obstacle_timing.png and Stk3d_container_obstacle_convergence.png to", here)
