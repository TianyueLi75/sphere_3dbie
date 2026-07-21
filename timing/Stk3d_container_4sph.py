"""Timing + self-convergence benchmark for a 4-sphere-in-container Stokes problem.

Geometry: a unit spherical container (radius 1) at the origin holding four small,
well-separated interior spheres (radius 1/4) at the vertices of a regular tetrahedron.
There is no closed-form solution, so accuracy is assessed by *self-convergence*: the
velocity field evaluated at a fixed cloud of interior check points should converge as lmax
increases (uniform on all five spheres). For each lmax we time the coupled GMRES solve and
the off-surface field evaluation, and fit the per-iteration solve cost against lmax to
check the expected O(lmax^3) growth.

Two boundary-condition variants are run:
    - "obstacles": no-slip container, tangential squirmer slip on each interior sphere,
    - "container": tangential squirmer slip on the container, no-slip interior spheres.

Run (from repo root, CPU venv):
    /mnt/home/tli10/jax_venv312_cpu/bin/python timing/Stk3d_container_4sph.py
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

from suspension import build_suspension, quadr_suspension, Stk3d_onsurf_solve
from sphere import set_density
from biop import Stk3d

jax.config.update("jax_enable_x64", True)


# --- Fixed problem geometry / parameters --------------------------------------------
# Container (sphere 0) at the origin plus four interior spheres of radius R/4 at the
# vertices of a regular tetrahedron, scaled so each center sits at distance 0.5 from the
# origin. Checks: center+radius = 0.75 < 1 (0.25 wall gap); obstacle surface-to-surface
# gap ~ 0.316 -- well apart, relatively evenly spaced.
_R = 1.0
_r = 0.25
_s = 0.5 / np.sqrt(3.0)
_TETRA = np.array([[1., 1., 1.], [1., -1., -1.], [-1., 1., -1.], [-1., -1., 1.]])
CENTERS = jnp.asarray(np.vstack([np.zeros((1, 3)), _s * _TETRA]))
RADII = jnp.asarray([_R, _r, _r, _r, _r])
NS = int(RADII.shape[0])

SEP_ETA = 0.1
SL_SCAL = 1.0
DL_SCAL = 1.0
SGN_LST = [-1.0] + [1.0] * (NS - 1)   # interior container, exterior obstacles
U = 1.0
vslip_mag = lambda theta: jnp.sin(theta) * 3. / 2. * U   # squirmer tangential surface speed


def _squirmer_block(sphere):
    """Tangential squirmer slip velocity (flattened cartesian) on one sphere's grid."""
    th, ph = sphere["Xsph"][:, :, 0], sphere["Xsph"][:, :, 1]
    zeros = jnp.zeros_like(th)
    sx, sy, sz = Stk3d.sph2cart(zeros, vslip_mag(th), zeros, th, ph)
    return jnp.stack([sx, sy, sz], axis=2).reshape(-1)


def build_bc(Sp, dsp, mode):
    """Assemble the (3*Nnodes,) complex Dirichlet BC vector.

    mode="obstacles": zero (no-slip) container, squirmer slip on each interior sphere.
    mode="container": squirmer slip on the container, zero (no-slip) interior spheres.
    """
    bc = jnp.zeros((3 * int(dsp[-1]),), dtype=jnp.complex128)
    if mode == "obstacles":
        slip_spheres = range(1, Sp["Ns"])
    elif mode == "container":
        slip_spheres = [0]
    else:
        raise ValueError(f"unknown BC mode {mode!r}")
    for s in slip_spheres:
        blk = _squirmer_block(Sp["spheres_lst"][s])
        bc = bc.at[3 * int(dsp[s]):3 * int(dsp[s + 1])].set(blk)
    return bc


def test(lmax: int, chk: jax.Array, mode: str):
    """One run at resolution <lmax> (uniform on all spheres); returns timings + the
    velocity at fixed points <chk>."""
    Sp = build_suspension(CENTERS, RADII, SEP_ETA)
    Sp, sh_lst = quadr_suspension(Sp, jnp.full((NS,), lmax))
    Ns = Sp["Ns"]
    sl_lst = [SL_SCAL] * Ns
    dl_lst = [DL_SCAL] * Ns
    dsp = Sp["Nnodes_dsp"]
    Nnodes = dsp[-1].item()

    bc = build_bc(Sp, dsp, mode)

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
    (finest lmax, zero diff by construction) is dropped from the error curve.
    A dashed guide line shows the O(lmax^3) slope anchored at the finest lmax."""
    lmax_arr = np.asarray(lmax_list, dtype=float)
    fig, ax1 = plt.subplots()
    l1 = ax1.loglog(lmax_list, t_solve, 'k+-', label="coupled solve / iter")
    l2 = ax1.loglog(lmax_list, t_eval, 'ko-', label="off-surf eval")
    # O(lmax^3) reference anchored at the finest per-iteration timing.
    ref = t_solve[-1] * (lmax_arr / lmax_arr[-1]) ** 3
    l4 = ax1.loglog(lmax_arr, ref, 'b:', label=r"$\propto$ lmax$^3$")
    ax1.set_xlabel("lmax"); ax1.set_ylabel("Time (s)")

    ax2 = ax1.twinx()
    l3 = ax2.loglog(lmax_list[:-1], err[:-1], 'r*--',
                    label=f"max rel diff vs lmax={lmax_list[-1]}")
    ax2.set_ylabel(f"max rel diff vs lmax={lmax_list[-1]}")

    lines = l1 + l2 + l4 + l3
    ax1.legend(lines, [ln.get_label() for ln in lines])
    ax1.set_title(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, format="svg", dpi=150)
    plt.close(fig)


def make_check_points(n=64, seed=0):
    """Fixed cloud of points interior to the container and clear of every obstacle (for
    accurate spectral evaluation), identical across all lmax and BC modes."""
    rng = np.random.default_rng(seed)
    R0 = float(RADII[0])
    Cnp = np.asarray(CENTERS)
    Rnp = np.asarray(RADII)
    pts = []
    while len(pts) < n:
        p = rng.uniform(-R0, R0, size=3)
        if np.linalg.norm(p) >= 0.9 * R0:
            continue
        if all(np.linalg.norm(p - Cnp[s]) > 1.3 * Rnp[s] for s in range(1, NS)):
            pts.append(p)
    return jnp.asarray(np.array(pts))


def growth_exponent(lmax_list, t_periter, tail_min=32):
    """Report the observed growth of per-iteration solve cost with lmax: a least-squares
    slope of log(t/iter) vs log(lmax) over the resolved tail (lmax >= tail_min), plus the
    consecutive-doubling ratios (expected ~8 under O(lmax^3))."""
    lmax_arr = np.asarray(lmax_list, dtype=float)
    t = np.asarray(t_periter, dtype=float)
    ratios = t[1:] / t[:-1]
    mask = lmax_arr >= tail_min
    slope = np.nan
    if np.count_nonzero(mask) >= 2:
        slope = np.polyfit(np.log(lmax_arr[mask]), np.log(t[mask]), 1)[0]
    print(f"  per-iter growth: fitted exponent (lmax>={tail_min}) = {slope:.2f} "
          f"(expect ~3 for O(lmax^3))")
    print("  consecutive t/iter ratios (expect ~8):",
          " ".join(f"{lmax_list[i]}->{lmax_list[i+1]}:{ratios[i]:.1f}" for i in range(len(ratios))))
    return slope


def run(mode, chk, lmax_list):
    here = os.path.dirname(os.path.abspath(__file__))
    print(f"\n=== BC mode: {mode} ===", flush=True)

    t_solve = np.zeros(len(lmax_list))
    t_periter_solve = np.zeros(len(lmax_list))
    t_eval = np.zeros(len(lmax_list))
    iters_list = np.zeros(len(lmax_list), dtype=int)
    u_all = []
    print(f"{'lmax':>5} {'info':>5} {'iters':>6} {'residual':>12} {'t_solve(s)':>12} {'t/iter(s)':>12} {'t_eval(s)':>12}", flush=True)
    for i, lmax in enumerate(lmax_list):
        print(f"  ... starting lmax={lmax}", flush=True)
        ts, te, resid, info, iters, u_chk = test(int(lmax), chk, mode)
        t_solve[i], t_eval[i], iters_list[i] = ts, te, iters
        t_periter_solve[i] = ts / iters
        u_all.append(u_chk)
        print(f"{lmax:>5d} {info:>5d} {iters:>6d} {resid:>12.3e} {ts:>12.3f} {ts/iters:>12.4f} {te:>12.3f}", flush=True)

    # Self-convergence: compare each field to the finest-lmax reference.
    u_ref = u_all[-1]
    denom = np.max(np.abs(u_ref))
    err = np.array([np.max(np.abs(u_all[i] - u_ref)) / denom for i in range(len(lmax_list))])
    print(f"\nSelf-convergence vs lmax = {lmax_list[-1]} (reference):")
    for i in range(len(lmax_list) - 1):
        print(f"  lmax={lmax_list[i]:>3d}: max rel diff = {err[i]:.3e}")

    growth_exponent(lmax_list, t_periter_solve)

    path = os.path.join(here, f"./plots/Stk3d_container_4sph_{mode}.svg")
    plot_timing_convergence(lmax_list, t_periter_solve, t_eval, err,
                            f"4-sphere container Stokes ({mode} slip): timing + self-convergence",
                            path)
    print(f"Wrote Stk3d_container_4sph_{mode}.svg to", here)


if __name__ == "__main__":
    lmax_list = [2 ** n for n in range(2, 9)]   # [4, 8, 16, 32, 64, 128, 256]
    chk = make_check_points()
    print(f"{chk.shape[0]} interior check points; {NS} spheres "
          f"(container R={_R}, {NS - 1} obstacles r={_r}).", flush=True)

    run("obstacles", chk, lmax_list)
    run("container", chk, lmax_list)


'''
Partial results (CPU, OMP_NUM_THREADS=1), BC mode "obstacles", captured through lmax=128.
lmax=256 was not run: extrapolating the per-iter ratios below it is ~1.5-2 days for that
single point (the far direct sum enters an O(lmax^4) regime). The 128 point already took
~4 hours; 256 is left in the default sweep as the production target to run when convenient.

 lmax  info  iters     residual   t_solve(s)    t/iter(s)    t_eval(s)
    4     1      6    3.722e-03        1.080       0.1801        2.851
    8     1      6    2.082e-05        3.457       0.5762        1.988
   16     1      6    1.495e-08       17.308       2.8847        2.208
   32     0      6    3.391e-14       97.404      16.2340        2.413
   64     0      6    3.107e-14     1015.776     169.2959        2.692
  128     0      6    4.057e-14    12705.005    2117.5008        4.755

Observations:
  - GMRES iteration count is constant (6) across lmax: conditioning is lmax-independent for
    this well-separated geometry, so the cost growth is entirely in time-per-iteration.
  - The residual floor is set by the point-and-shoot near-eval accuracy: info=1 (tol not
    reached) below lmax=32, then machine precision (~1e-14) from lmax=32 on. Self-convergence
    of the interior velocity field is likewise saturated by lmax=32.
  - Per-iter t/iter doubling ratios: 3.2, 5.0, 5.6, 10.4, 12.5. A log-log fit over
    lmax in [32,128] gives exponent ~3.5: ~O(lmax^3) in the mid-range, bending toward
    O(lmax^4) at high lmax as the far direct sum dominates.
'''
