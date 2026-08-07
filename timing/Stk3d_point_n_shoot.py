"""
Benchmark: point-and-shoot (move-pole) sphere-to-sphere Stokes evaluation vs the
single-point spectral evaluator.

For two non-concentric spheres with a random surface density on the source, evaluate the
combined layer potential K = SL + DL at the target sphere's grid three ways:
  - point_n_shoot               : FFT-accelerated point-and-shoot (Corona-Veerapaneni 2018)
  - bio_offsurf_apply(far=False): single-point spectral eval (SHqst_to_point_cplx loop)  [baseline]
  - bio_offsurf_apply(far=True) : smooth far-field quadrature (only accurate when far)

Reports, over an lmax sweep: the relative error of point_n_shoot vs the (exact, spectral)
single-point baseline -- the correctness gate -- and the wall-clock of each evaluator.
Expectation (paper Fig. 3): point_n_shoot scales ~O(p^3 log p) and pulls away from the
O(p^4) single-point loop as p grows.
"""
import os
import sys
import time

import numpy as np
import jax
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt

from sphere import build_sphere, quadr_sphere, set_density
from biop import Stk3d


def _timed(fn, *args, **kw):
    """Warm up once (compile / first-call overhead), then time a second call."""
    out = fn(*args, **kw)
    jax.block_until_ready(out)
    t0 = time.time()
    out = fn(*args, **kw)
    jax.block_until_ready(out)
    return out, time.time() - t0


def test(lmax, src_c=(0., 0., 0.), src_r=1.0, trg_c=(3.0, 0., 0.), trg_r=0.5, seed=0):
    sl, dl = 1.0, 1.0
    S = build_sphere(jnp.array(src_c), float(src_r)); S, sh = quadr_sphere(S, lmax)
    Strg = build_sphere(jnp.array(trg_c), float(trg_r)); Strg, shtrg = quadr_sphere(Strg, lmax)

    # Random (real) vector surface density on the source.
    rng = np.random.default_rng(seed)
    nphi, ntheta = S["Xcart"].shape[:2]
    sig = rng.standard_normal((nphi, ntheta, 3)) + 0j
    # density -> VWX coefficients (the coeff-basis evaluator input)
    vwx = jnp.stack(Stk3d.sig_xyz2vwx(jnp.asarray(sig[:, :, 0]), jnp.asarray(sig[:, :, 1]),
                                      jnp.asarray(sig[:, :, 2]), S["Xsph"][:, :, 0], S["Xsph"][:, :, 1], sh))
    trg = Strg["Xcart"].reshape(-1, 3)

    base, t_base = _timed(lambda: Stk3d.bio_offsurf_apply(trg, vwx, S, sh, sl, dl, far=False))
    base = np.asarray(base)
    pns_vwx, t_pns = _timed(lambda: Stk3d.point_n_shoot(Strg, shtrg, vwx, S, sh, sl, dl, near=False))
    # point_n_shoot returns target-basis VWX coeffs; recompose to Cartesian points to compare
    pvx, pvy, pvz = Stk3d.sig_vwx2xyz(pns_vwx[0], pns_vwx[1], pns_vwx[2],
                                      Strg["Xsph"][:, :, 0], Strg["Xsph"][:, :, 1], shtrg)
    pns = np.asarray(jnp.stack([pvx, pvy, pvz], axis=2)).reshape(-1, 3)
    far, t_far = _timed(lambda: Stk3d.bio_offsurf_apply(trg, vwx, S, sh, sl, dl, far=True))
    far = np.asarray(far)

    denom = np.max(np.abs(base))
    err_pns = float(np.max(np.abs(pns - base)) / denom)
    err_far = float(np.max(np.abs(far - base)) / denom)
    return t_pns, t_base, t_far, err_pns, err_far


def plot_timing_convergence(lmax_list, t_pns, t_base, t_far, err_pns, title, path):
    """Combined loglog figure: timing of each evaluator on the left axis, the
    point-and-shoot relative error (vs the exact single-point baseline) on the
    right axis, sharing a log lmax axis."""
    fig, ax1 = plt.subplots()
    l1 = ax1.loglog(lmax_list, t_pns, 'bo-', label="point-and-shoot")
    l2 = ax1.loglog(lmax_list, t_base, 'k+-', label="single-point eval")
    l3 = ax1.loglog(lmax_list, t_far, 'gs-', label="far quadrature")
    ax1.set_xlabel("lmax"); ax1.set_ylabel("Time (s)")

    ax2 = ax1.twinx()
    l4 = ax2.loglog(lmax_list, err_pns, 'r*--', label="rel err (P&S vs single-pt)")
    ax2.set_ylabel("max relative error vs single-point eval")

    lines = l1 + l2 + l3 + l4
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="upper left", fontsize=8)
    ax1.set_title(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, format="svg", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    lmax_list = [4, 8, 12, 16, 24, 32, 48, 64]
    Np = len(lmax_list)
    Tpns = np.zeros(Np); Tbase = np.zeros(Np); Tfar = np.zeros(Np)
    Epns = np.zeros(Np); Efar = np.zeros(Np)

    print("Two well-separated spheres: source r=1 @ origin, target r=0.5 @ (3,0,0)")
    print(f"{'lmax':>5} {'err_pns':>11} {'err_far':>11} {'t_pns(s)':>10} {'t_base(s)':>10} "
          f"{'t_far(s)':>10} {'speedup':>8}", flush=True)
    for i, lmax in enumerate(lmax_list):
        tp, tb, tf, ep, ef = test(lmax)
        Tpns[i], Tbase[i], Tfar[i], Epns[i], Efar[i] = tp, tb, tf, ep, ef
        print(f"{lmax:>5d} {ep:>11.3e} {ef:>11.3e} {tp:>10.4f} {tb:>10.4f} {tf:>10.4f} "
              f"{tb/tp:>7.1f}x", flush=True)

    plot_timing_convergence(lmax_list, Tpns, Tbase, Tfar, Epns,
                            "Stokes sphere-to-sphere: point-and-shoot vs single-point eval",
                            os.path.join(here, "plots", "Stk3d_point_n_shoot.svg"))
    print("Wrote plots/Stk3d_point_n_shoot.svg to", here)
