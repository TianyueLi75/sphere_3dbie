"""Component-level profiler for the coupled Stokes suspension solve.

`Stk3d_onsurf_solve` runs a matrix-free GMRES over `Stk3d_onsurf_apply`, whose per-iteration
cost is a sum of three cross-evaluation strategies with DIFFERENT lmax scaling:

  - self blocks (Stk3d.bio_onsurf_apply)          -- shtns analysis/synthesis, O(lmax^3)
  - near cross  (point-and-shoot, ps_evals)       -- Wigner rotations + per-ring FFT, O(lmax^3)
  - far  cross  (direct Nystrom sum, far_evals)   -- dense Ntrg x Nsrc pair sum, O(lmax^4)

plus the block-Jacobi preconditioner (O(lmax^3)/sphere) and a GMRES iteration count that
itself grows with lmax. Grid size is Nnodes = nphi*ntheta = O(lmax^2), so N = 3*Nnodes.

This harness attributes wall time to each component and fits its empirical lmax-slope, so the
O(lmax^3) point-and-shoot path and the O(lmax^4) far direct sum can be compared directly. It
sweeps TWO geometries side by side:

  - "far":  two well-separated unit spheres  -> both cross pairs FAR  (exercises the direct sum)
  - "near": container + obstacle             -> both cross pairs NEAR (exercises point-and-shoot)

The far O(lmax^4) sum blows up in memory/time, so its sweep is capped below the near sweep.

Run (from repo root, CPU venv):
    /mnt/home/tli10/jax_venv312_cpu/bin/python timing/Stk3d_solve_profile.py [quick]

`quick` runs a short sweep for a smoke test / decomposition sanity check.

Note on the solve metric: the scipy-driven Stk3d_onsurf_solve is used for niters/resid. Its t_solve includes a
one-time XLA compile (no warmup) -- that confound is measured separately below as
first-call vs steady-state matvec time, so the clean per-iteration cost comes from the warmed
component timings, not from t_solve/niters.
"""

import os

# Cap OpenMP threads BEFORE numpy/shtns load (eager shtns segfaults under XLA MLIR lowering
# otherwise -- see timing/Stk3d_container_obstacle.py).
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from suspension import (build_suspension, quadr_suspension, separate_spheres,
                        build_ps_evaluators, build_far_evaluators, Stk3d_onsurf_apply,
                        Stk3d_onsurf_solve, _block_bounds3)
from biop import Stk3d

jax.config.update("jax_enable_x64", True)


# ----------------------------------------------------------------------------------------
# problem setups: each returns everything the matvec / solve needs at a given lmax
# ----------------------------------------------------------------------------------------
def setup_far(lmax: int):
    """Two well-separated unit spheres (both cross pairs FAR). Exterior Stokeslet BC from one
    interior point force per sphere (suspension.py __main__ TEST 2 geometry)."""
    centers = jnp.array([[0., 0., 0.], [3., 0., 0.]])
    radii = jnp.array([1.0, 1.0])
    Sp = build_suspension(centers, radii, sep_eta=0.01)
    Sp, sh_lst = quadr_suspension(Sp, jnp.array([lmax, lmax]))
    Ns = Sp["Ns"]
    sl_lst, dl_lst, sgn_lst = [1.0] * Ns, [1.0] * Ns, [1.0] * Ns

    ptsrc = jnp.array([[0.1, 0.3, 0.15], [2.95, 0.1, 0.05]])
    force = jnp.array([[1.0, 0.5, -0.3], [-0.7, 0.2, 0.4]])
    dsp = Sp["Nnodes_dsp"]
    bc = jnp.zeros((3 * int(dsp[-1]),), dtype=jnp.complex128)
    for s in range(Ns):
        nodes = Sp["spheres_lst"][s]["Xcart"].reshape(-1, 3)
        vel = Stk3d.compute_field(nodes, ptsrc, force)
        bc = bc.at[3 * int(dsp[s]):3 * int(dsp[s + 1])].set(vel.reshape(-1))
    return Sp, Ns, sh_lst, sl_lst, dl_lst, sgn_lst, bc, int(dsp[-1])


def setup_near(lmax: int):
    """Container + interior obstacle (both cross pairs NEAR). Squirmer slip BC on the obstacle,
    no-slip container (timing/Stk3d_container_obstacle.py geometry)."""
    centers = jnp.array([[0., 0., 0.], [0.3, 0.1, -0.05]])
    radii = jnp.array([1.0, 0.2])
    Sp = build_suspension(centers, radii, sep_eta=0.1)
    Sp, sh_lst = quadr_suspension(Sp, jnp.array([lmax, lmax]))
    Ns = Sp["Ns"]
    sl_lst, dl_lst, sgn_lst = [1.0] * Ns, [1.0] * Ns, [-1.0, 1.0]

    U = 1.0
    vslip_mag = lambda theta: jnp.sin(theta) * 3. / 2. * U
    dsp = Sp["Nnodes_dsp"]
    bc = jnp.zeros((3 * int(dsp[-1]),), dtype=jnp.complex128)
    obst = Sp["spheres_lst"][1]
    th1, ph1 = obst["Xsph"][:, :, 0], obst["Xsph"][:, :, 1]
    z1 = jnp.zeros_like(th1)
    sx, sy, sz = Stk3d.sph2cart(z1, vslip_mag(th1), z1, th1, ph1)
    bc = bc.at[3 * int(dsp[1]):3 * int(dsp[2])].set(jnp.stack([sx, sy, sz], axis=2).reshape(-1))
    return Sp, Ns, sh_lst, sl_lst, dl_lst, sgn_lst, bc, int(dsp[-1])


SETUPS = {"far": setup_far, "near": setup_near}


# ----------------------------------------------------------------------------------------
# timing helpers
# ----------------------------------------------------------------------------------------
def time_call(fn, arg, repeats: int = 5):
    """Warm up a (jitted) fn once, then return the mean wall time over `repeats` calls, each
    forced to completion with block_until_ready. Returns NaN if fn is None (empty component)."""
    if fn is None:
        return float("nan")
    jax.block_until_ready(fn(arg))          # compile + warm
    t0 = time.time()
    for _ in range(repeats):
        jax.block_until_ready(fn(arg))
    return (time.time() - t0) / repeats


def time_first_vs_steady(fn, arg):
    """Time the FIRST call (XLA compile + run) vs a subsequent steady call. The gap is the
    one-time compile cost that Stk3d_onsurf_solve pays under its (un-warmed) timer."""
    t0 = time.time(); jax.block_until_ready(fn(arg)); t_first = time.time() - t0
    t0 = time.time(); jax.block_until_ready(fn(arg)); t_steady = time.time() - t0
    return t_first, t_steady


# ----------------------------------------------------------------------------------------
# component matvec pieces (each mirrors exactly what Stk3d_onsurf_apply does for that term)
# ----------------------------------------------------------------------------------------
def make_components(Sp, Ns, sh_lst, sl_lst, dl_lst, sgn_lst, ps_evals, far_evals):
    """Build jitted callables for each isolated cost center + the full matvec. All operate in
    COEFFICIENT space (matching Stk3d_onsurf_apply): the test vector is the concatenation of each
    sphere's stacked-VWX coeffs. Empty cross dicts (no near/far pairs) yield None (reported NaN).
    Returns (components, Ncoef)."""
    spheres = Sp["spheres_lst"]
    bounds = _block_bounds3(Sp, Ns)          # grid ranges (used by the far scatter)
    Ngrid = bounds[-1][1]
    cb, c0 = [], 0
    for s in range(Ns):
        n = 3 * sh_lst[s].nlm; cb.append((c0, c0 + n)); c0 += n
    Ncoef = c0

    def _blocks(c):   # flat coeff vector -> list of per-sphere (3, nlm) VWX blocks
        return [c[cb[s][0]:cb[s][1]].reshape(3, sh_lst[s].nlm) for s in range(Ns)]

    @jax.jit
    def self_only(c):
        vb = _blocks(c)
        return jnp.concatenate([
            Stk3d.bio_onsurf_apply(vb[t], sh_lst[t], sl_lst[t], dl_lst[t], sgn_lst[t],
                                   radius=spheres[t]["r"]).reshape(-1)
            for t in range(Ns)])

    near_only = None
    if ps_evals:
        @jax.jit
        def near_only(c):
            vb = _blocks(c); out = []
            for tind in range(Ns):
                acc = jnp.zeros((3, sh_lst[tind].nlm), dtype=jnp.complex128)
                for sind in range(Ns):
                    if (tind, sind) in ps_evals:
                        acc = acc + ps_evals[(tind, sind)](vb[sind], sl_lst[sind], dl_lst[sind])
                out.append(acc.reshape(-1))
            return jnp.concatenate(out)

    far_only = None
    if far_evals:
        @jax.jit
        def far_only(c):
            vb = _blocks(c)
            u_all = jnp.concatenate([
                far_evals[sind][0](vb[sind], sl_lst[sind], dl_lst[sind]).reshape(-1)
                for sind in far_evals])
            dest_all = jnp.concatenate([dest for (_, dest) in far_evals.values()])
            far_grid = jnp.zeros(Ngrid, dtype=jnp.float64).at[dest_all].add(u_all)
            out = []
            for tind in range(Ns):
                t_sph = spheres[tind]; nphi_t, ntheta_t = t_sph["Xcart"].shape[:2]
                th_t, ph_t = t_sph["Xsph"][:, :, 0], t_sph["Xsph"][:, :, 1]
                gt = far_grid[bounds[tind][0]:bounds[tind][1]].reshape(nphi_t, ntheta_t, 3)
                out.append(jnp.stack(Stk3d.sig_xyz2vwx(
                    gt[:, :, 0], gt[:, :, 1], gt[:, :, 2], th_t, ph_t, sh_lst[tind])).reshape(-1))
            return jnp.concatenate(out)

    @jax.jit
    def precond(c):
        vb = _blocks(c); out = []
        for sind in range(Ns):
            if float(sgn_lst[sind]) < 0.0:      # interior block: identity (see solve docstring)
                out.append(vb[sind].reshape(-1)); continue
            vz = Stk3d.stokes_onsurf_direct_solve(
                vb[sind], sh_lst[sind], sl_lst[sind], dl_lst[sind], sgn_lst[sind], radius=spheres[sind]["r"])
            out.append(vz.reshape(-1))
        return jnp.concatenate(out)

    @jax.jit
    def full_matvec(c):
        return Stk3d_onsurf_apply(c, Sp, Ns, sh_lst, sl_lst, dl_lst, sgn_lst, ps_evals, far_evals)

    return {"self": self_only, "near": near_only, "far": far_only,
            "precond": precond, "matvec": full_matvec}, Ncoef


# ----------------------------------------------------------------------------------------
# one (geometry, lmax) measurement
# ----------------------------------------------------------------------------------------
def profile_one(geom: str, lmax: int, repeats: int = 3, do_solve: bool = True):
    setup = SETUPS[geom]
    Sp, Ns, sh_lst, sl_lst, dl_lst, sgn_lst, bc, Nnodes = setup(lmax)

    # eager build (rotation C-objects, geometry keys, evaluator construction) -- once per solve.
    # The returned evaluators are jitted; their compile is captured later in the component
    # warmups / first-call timing, not here.
    t0 = time.time()
    sep_mat = separate_spheres(Sp)
    ps_evals = build_ps_evaluators(Sp, Ns, sh_lst, sep_mat)
    far_evals = build_far_evaluators(Sp, Ns, sh_lst, sep_mat)
    t_build = time.time() - t0

    comps, N = make_components(Sp, Ns, sh_lst, sl_lst, dl_lst, sgn_lst, ps_evals, far_evals)

    rng = np.random.default_rng(0)
    sigma = jnp.asarray(rng.standard_normal(N) + 1j * rng.standard_normal(N),
                        dtype=jnp.complex128)

    times = {name: time_call(fn, sigma, repeats) for name, fn in comps.items()}
    # compile confound: fresh (un-warmed) jit of the full matvec, first vs steady call.
    fresh = jax.jit(lambda x: Stk3d_onsurf_apply(x, Sp, Ns, sh_lst, sl_lst, dl_lst, sgn_lst,
                                                 ps_evals, far_evals))
    t_first, t_steady = time_first_vs_steady(fresh, sigma)

    row = {"lmax": lmax, "Nnodes": Nnodes, "N": N, "t_build": t_build,
           "t_first": t_first, "t_compile": t_first - t_steady, **times}

    if do_solve:
        _, t_solve, niters, info, resid = Stk3d_onsurf_solve(
            bc, Sp, Ns, Nnodes, sh_lst, sl_lst, dl_lst, sgn_lst)
        row.update({"t_solve": t_solve, "niters": int(niters), "info": int(info),
                    "resid": float(resid)})
    else:
        row.update({"t_solve": float("nan"), "niters": 0, "info": -1, "resid": float("nan")})
    return row


# ----------------------------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------------------------
COMP_KEYS = ["self", "near", "far", "precond", "matvec"]


def fit_slope(lmax_list, vals):
    """Asymptotic log-log slope over the finite points (upper half of the sweep)."""
    x = np.asarray(lmax_list, float); y = np.asarray(vals, float)
    m = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x, y = x[m], y[m]
    if x.size < 2:
        return float("nan")
    k = max(2, x.size // 2)          # fit the largest-lmax half for the asymptotic rate
    return float(np.polyfit(np.log(x[-k:]), np.log(y[-k:]), 1)[0])


def print_table(geom, rows):
    print(f"\n===== geometry: {geom} =====")
    hdr = (f"{'lmax':>5} {'Nnodes':>9} {'iters':>6} {'t_build':>9} {'self':>9} {'near':>9} "
           f"{'far':>10} {'precond':>9} {'matvec':>9} {'compile':>9} {'t_solve':>9} {'resid':>10}")
    print(hdr)
    for r in rows:
        print(f"{r['lmax']:>5d} {r['Nnodes']:>9d} {r['niters']:>6d} {r['t_build']:>9.3f} "
              f"{r['self']:>9.4f} {r['near']:>9.4f} {r['far']:>10.4f} {r['precond']:>9.4f} "
              f"{r['matvec']:>9.4f} {r['t_compile']:>9.3f} {r['t_solve']:>9.3f} "
              f"{r['resid']:>10.2e}")
    lmx = [r["lmax"] for r in rows]
    print("  asymptotic log-log slopes:")
    for k in COMP_KEYS + ["t_solve"]:
        print(f"    {k:>8}: {fit_slope(lmx, [r[k] for r in rows]):.2f}")


def plot_geometry(geom, rows, path):
    lmx = [r["lmax"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    styles = {"self": "C0o-", "near": "C1s-", "far": "C2^-",
              "precond": "C3d-", "matvec": "C4x-"}
    for k in COMP_KEYS:
        vals = [r[k] for r in rows]
        if not any(np.isfinite(v) and v > 0 for v in vals):
            continue
        ax.loglog(lmx, vals, styles[k], label=f"{k} (slope {fit_slope(lmx, vals):.2f})")
    ts = [r["t_solve"] for r in rows]
    if any(np.isfinite(v) and v > 0 for v in ts):
        ax.loglog(lmx, ts, "k+--", label=f"t_solve (slope {fit_slope(lmx, ts):.2f})")
    ax.set_xlabel("lmax"); ax.set_ylabel("time (s)")
    ax.set_title(f"Stk3d solve component scaling: {geom}")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, format="svg"); plt.close(fig)


def plot_comparison(rows_far, rows_near, path):
    """Direct O(lmax^4) far direct-sum vs O(lmax^3) point-and-shoot overlay + iteration count."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    lf = [r["lmax"] for r in rows_far]; far = [r["far"] for r in rows_far]
    ln = [r["lmax"] for r in rows_near]; near = [r["near"] for r in rows_near]
    ax.loglog(lf, far, "C2^-", label=f"far direct sum (slope {fit_slope(lf, far):.2f})")
    ax.loglog(ln, near, "C1s-", label=f"point-and-shoot (slope {fit_slope(ln, near):.2f})")
    ax.set_xlabel("lmax"); ax.set_ylabel("per-matvec cross-term time (s)")
    ax.set_title("Far direct-sum vs point-and-shoot"); ax.legend(); ax.grid(True, which="both", alpha=0.3)

    for rows, lab, sty in ((rows_far, "far", "C2^-"), (rows_near, "near", "C1s-")):
        lmx = [r["lmax"] for r in rows]; it = [r["niters"] for r in rows]
        if any(i > 0 for i in it):
            ax2.semilogx(lmx, it, sty, label=f"{lab} geometry")
    ax2.set_xlabel("lmax"); ax2.set_ylabel("GMRES iterations")
    ax2.set_title("Iteration count vs lmax"); ax2.legend(); ax2.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, format="svg"); plt.close(fig)


# ----------------------------------------------------------------------------------------
if __name__ == "__main__":
    quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
    if quick:
        sweeps = {"far": [4, 8, 16, 32], "near": [4, 8, 16, 32]}
        solve_cap = {"far": 32, "near": 32}
    else:
        # far O(lmax^4) capped below near to stay tractable in memory/time. The far solve is
        # skipped above solve_cap: each far matvec is ~minutes there and the manufactured BC is
        # anyway under-resolved at low lmax (hits maxiter), so t_solve is not informative -- the
        # component (far-only) timing is what carries the O(lmax^4) scaling result.
        sweeps = {"far": [8, 16, 32, 64, 128],
                  "near": [8, 16, 32, 64, 128, 256, 512]}
        solve_cap = {"far": 64, "near": 512}

    here = os.path.dirname(os.path.abspath(__file__))
    results = {}
    for geom, lmax_list in sweeps.items():
        rows = []
        for lmax in lmax_list:
            print(f"  ... {geom} lmax={lmax}", flush=True)
            rows.append(profile_one(geom, int(lmax), do_solve=(lmax <= solve_cap[geom])))
        results[geom] = rows
        print_table(geom, rows)
        plot_geometry(geom, rows, os.path.join(here, f"plots/Stk3d_solve_profile_{geom}.svg"))

    plot_comparison(results["far"], results["near"],
                    os.path.join(here, "plots/Stk3d_solve_profile_compare.svg"))
    print("\nWrote plots to", os.path.join(here, "plots"))
