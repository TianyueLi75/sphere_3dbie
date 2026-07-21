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

# Cap OpenMP threads BEFORE numpy/shtns load the OpenMP runtime. The eager (C shtns)
# off-surface eval spawns an OpenMP pool that oversubscribes against XLA/LLVM's own
# compilation threads and segfaults inside JAX's MLIR lowering. The safe thread count is
# machine-dependent (e.g. 8 already crashes here), so pin to 1 unless the caller overrides.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from suspension import build_suspension, quadr_suspension, Stk3d_onsurf_solve, Stk3d_onsurf_solve_spla
from sphere import set_density
from biop import Stk3d

jax.config.update("jax_enable_x64", True)


class IterationCounter:
    """Counts GMRES iterations (passed as the scipy gmres callback)."""
    def __init__(self):
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1


def test(lmax: int, chk: jax.Array):
    # Fixed problem geometry / parameters (matches suspension.py TEST 3).
    CENTERS = jnp.array([[0., 0., 0.], [0.3, 0.1, -0.05]])
    RADII = jnp.array([1.0, 0.2])
    SEP_ETA = 0.1
    SL_SCAL = 1.0
    DL_SCAL = 1.0
    SGN_LST = [-1.0, 1.0]   # interior container, exterior obstacle
    U = 1.0
    vslip_mag = lambda theta: jnp.sin(theta) * 3. / 2. * U   # squirmer tangential surface speed

    """One run at resolution <lmax>; returns timings + the velocity at fixed points <chk>."""
    Sp = build_suspension(CENTERS, RADII, SEP_ETA)
    Sp, sh_lst = quadr_suspension(Sp, jnp.array([lmax, lmax]))
    Ns = Sp["Ns"]
    sl_lst = [SL_SCAL] * Ns
    dl_lst = [DL_SCAL] * Ns
    dsp = Sp["Nnodes_dsp"]
    Nnodes = dsp[-1].item()

    # Squirmer slip BC: container no-slip (zero), obstacle tangential slip.
    bc = jnp.zeros((3 * int(dsp[-1]),), dtype=jnp.complex128)
    obst = Sp["spheres_lst"][1]
    th1, ph1 = obst["Xsph"][:, :, 0], obst["Xsph"][:, :, 1]
    zeros1 = jnp.zeros_like(th1)
    sx, sy, sz = Stk3d.sph2cart(zeros1, vslip_mag(th1), zeros1, th1, ph1)
    bc = bc.at[3 * int(dsp[1]):3 * int(dsp[2])].set(jnp.stack([sx, sy, sz], axis=2).reshape(-1))

    # Warmup: compile the jitted self-block / preconditioner kernels for this lmax's shapes
    # with a single (untimed) GMRES iteration, so they are excluded from the timed solve.
    Stk3d_onsurf_solve(bc, Sp, Ns, Nnodes, sh_lst, sl_lst, dl_lst, SGN_LST, maxiter=1)

    sigma, t_solve, iters, info, resid = Stk3d_onsurf_solve(bc, Sp, Ns, Nnodes, sh_lst, sl_lst, dl_lst, SGN_LST)

    # Time the off-surface field evaluation at the fixed check points.
    t0 = time.time()
    approx = jnp.zeros((chk.shape[0], 3), dtype=jnp.complex128)
    for s in range(Ns):
        nphi, ntheta = Sp["spheres_lst"][s]["Xcart"].shape[:2]
        sig_s = sigma[3 * int(dsp[s]): 3 * int(dsp[s + 1])].reshape(nphi, ntheta, 3)
        s_sph = set_density(Sp["spheres_lst"][s], sig_s[:, :, 0], sig_s[:, :, 1], sig_s[:, :, 2])
        approx = approx + Stk3d.bio_offsurf_apply(chk, s_sph, sh_lst[s], sl_lst[s], dl_lst[s])
    u_chk = np.real(np.asarray(approx))
    t_eval = time.time() - t0

    return t_solve, t_eval, float(resid), int(info), int(iters), u_chk


def plot_timing_convergence(lmax_list, t_solve, t_eval, err, title, path):
    """Combined loglog figure: timing on the left axis, self-convergence relative
    difference on the right axis, sharing a log lmax axis. The reference point
    (finest lmax, zero diff by construction) is dropped from the error curve."""
    fig, ax1 = plt.subplots()
    l1 = ax1.loglog(lmax_list, t_solve, 'k+-', label="coupled solve")
    l2 = ax1.loglog(lmax_list, t_eval, 'ko-', label="off-surf eval")
    ax1.set_xlabel("lmax"); ax1.set_ylabel("Time (s)")

    ax2 = ax1.twinx()
    l3 = ax2.loglog(lmax_list[:-1], err[:-1], 'r*--',
                    label=f"max rel diff vs lmax={lmax_list[-1]}")
    ax2.set_ylabel(f"max rel diff vs lmax={lmax_list[-1]}")

    lines = l1 + l2 + l3
    ax1.legend(lines, [ln.get_label() for ln in lines])
    ax1.set_title(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, format="svg", dpi=150)
    plt.close(fig)


def make_check_points(n=64, seed=0):
    # Fixed problem geometry / parameters (matches suspension.py TEST 3).
    CENTERS = jnp.array([[0., 0., 0.], [0.3, 0.1, -0.05]])
    RADII = jnp.array([1.0, 0.2])

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
    lmax_list = [2**n for n in range(2, 10)]
    # lmax_list = [2**n for n in range(2, 5)]
    chk = make_check_points()

    t_solve = np.zeros(len(lmax_list))
    t_periter_solve = np.zeros(len(lmax_list))
    t_eval = np.zeros(len(lmax_list))
    iters_list = np.zeros(len(lmax_list), dtype=int)
    u_all = []
    print(f"{'lmax':>5} {'info':>5} {'iters':>6} {'residual':>12} {'t_solve(s)':>12} {'t/iter(s)':>12} {'t_eval(s)':>12}", flush=True)
    for i, lmax in enumerate(lmax_list):
        print(f"  ... starting lmax={lmax}", flush=True)
        ts, te, resid, info, iters, u_chk = test(int(lmax), chk)
        t_solve[i], t_eval[i], iters_list[i] = ts, te, iters
        u_all.append(u_chk)
        print(f"{lmax:>5d} {info:>5d} {iters:>6d} {resid:>12.3e} {ts:>12.3f} {ts/iters:>12} {te:>12.3f}", flush=True)
        t_periter_solve[i] = ts/iters

    # Self-convergence: compare each field to the finest-lmax reference.
    u_ref = u_all[-1]
    denom = np.max(np.abs(u_ref))
    err = np.array([np.max(np.abs(u_all[i] - u_ref)) / denom for i in range(len(lmax_list))])
    print("\nSelf-convergence vs lmax =", lmax_list[-1], "(reference):")
    for i in range(len(lmax_list) - 1):
        print(f"  lmax={lmax_list[i]:>3d}: max rel diff = {err[i]:.3e}")

    here = os.path.dirname(os.path.abspath(__file__))

    plot_timing_convergence(lmax_list, t_periter_solve, t_eval, err,
                            "Container-obstacle Stokes: timing + self-convergence",
                            os.path.join(here, "./plots/Stk3d_container_obstacle.svg"))
    print("\nWrote Stk3d_container_obstacle.svg to", here)


    '''
 lmax  info  iters     residual   t_solve(s)    t/iter(s)    t_eval(s)
    4     0      4    2.155e-14        0.045 0.0111619234085083        3.113
    8     0      4    1.616e-14        0.099 0.024657130241394043        2.175
   16     0      4    1.524e-14        0.236 0.05894827842712402        2.267
   32     0      4    1.162e-14        0.586 0.14652353525161743        2.407
   64     0      4    9.199e-15        2.387 0.5968244075775146        2.582
  128     0      4    1.359e-14       15.913 3.9781848192214966        3.306
  256     0      6    1.590e-14      195.266 32.544267535209656        4.695
  512     0     18    1.139e-14     5974.050 331.89166649182636       14.345

  Self-convergence vs lmax = 512 (reference):
  lmax=  4: max rel diff = 3.355e-02
  lmax=  8: max rel diff = 4.532e-04
  lmax= 16: max rel diff = 1.381e-08
  lmax= 32: max rel diff = 3.369e-15
  lmax= 64: max rel diff = 4.115e-15
  lmax=128: max rel diff = 2.297e-15
  lmax=256: max rel diff = 2.144e-15
    '''
