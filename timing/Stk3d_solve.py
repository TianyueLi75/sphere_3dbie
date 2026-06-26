import jax
import jax.numpy as jnp
import numpy as np
import lineax as lx
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sphere import *
from biop.Stk3d import *
import shtns
import shtns_jax
import time
import matplotlib.pyplot as plt

class IterationCounter:
    def __init__(self):
        self.count = 0
    def __call__(self, rk=None):
        self.count += 1

def test(lmax: int, exterior: bool = True):
    # Geometry setup
    center = jnp.array([0.,0.,0.])
    radius = 1.
    S = build_sphere(center, radius)
    S, sh = quadr_sphere(S, lmax)
    # BIOp parameter
    sl_scal = 1.0
    dl_scal = 1.0
    
    if exterior:
        Rtrg = radius * 1.025
        sgn = 1.0 
    else:
        Rtrg = radius * 0.975
        sgn = -1.0 
    Strg = build_sphere(center, Rtrg)
    lmax_trg = 40 # Fix target size
    Strg, shtrg = quadr_sphere(Strg, lmax_trg)

    # Manufactured solutions test
    if exterior:
        ptsrc = jnp.array([[0.1,0.3,0.15],[-0.35,0.2,0.]]) # shifted source to avoid constant potential on all of S
        force = jnp.array([[1,1,1],[-1,0,0]])
    else:
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

    sig_direct = stokes_onsurf_direct_solve(BC_pot, theta, phi, sh, sl_scal, dl_scal, sgn)
    jax.block_until_ready(sig_direct)   # warmup: finish compile before timing
    tstart = time.time()
    sig_direct = stokes_onsurf_direct_solve(BC_pot, theta, phi, sh, sl_scal, dl_scal, sgn)
    jax.block_until_ready(sig_direct)
    tend = time.time()
    time_direct = tend - tstart
    bc_check_direct = bio_onsurf_apply(sig_direct, theta, phi, sh, sl_scal, dl_scal, sgn)
    resid_direct = jnp.linalg.norm(bc_check_direct - BC_pot)
    print("Residual of DIRECT solve = {a}".format(a=resid_direct))

    # Accuracy
    xtrg = Strg["Xcart"][:,:,0] 
    ytrg = Strg["Xcart"][:,:,1]
    ztrg = Strg["Xcart"][:,:,2]
    trg_sphere2 = jnp.column_stack([jnp.reshape(xtrg,-1), jnp.reshape(ytrg,-1), jnp.reshape(ztrg,-1)])
    true_field = compute_field(trg_sphere2, ptsrc, force) 
    true_field = jnp.reshape(true_field, Strg["Xcart"].shape)
    true_field = jnp.real(true_field)

    S = set_density(S, sig_direct[:,:,0], sig_direct[:,:,1], sig_direct[:,:,2])
    Ksig_direct = bio_offsurf_apply_1sph(Strg, shtrg, S, sh, sl_scal, dl_scal)  # warmup
    jax.block_until_ready(Ksig_direct)
    tstart = time.time()
    Ksig_direct = bio_offsurf_apply_1sph(Strg, shtrg, S, sh, sl_scal, dl_scal)
    jax.block_until_ready(Ksig_direct)
    tend = time.time()
    time_eval = tend - tstart
    Ksig_direct = jnp.real(Ksig_direct)

    err_direct = jnp.max(jnp.abs(true_field - Ksig_direct)) / jnp.max(jnp.abs(true_field))
    print("Max relative error of order {lmax} solver at target radius {Rtrg} for direct solver is {d}".format(lmax=lmax, Rtrg=Rtrg, d=err_direct))

    return time_direct, time_eval, float(err_direct)

def plot_timing_convergence(lmax_list, t_solve, t_eval, err, title, path):
    """Combined loglog figure: timing on the left axis, relative error on the
    right axis, sharing a log lmax axis."""
    fig, ax1 = plt.subplots()
    l1 = ax1.loglog(lmax_list, t_solve, 'k+-', label="direct solve")
    l2 = ax1.loglog(lmax_list, t_eval, 'ko-', label="off-surf eval")
    ax1.set_xlabel("lmax"); ax1.set_ylabel("Time (s)")

    ax2 = ax1.twinx()
    l3 = ax2.loglog(lmax_list, err, 'r*--', label="rel error vs exact")
    ax2.set_ylabel("max relative error vs exact")

    lines = l1 + l2 + l3
    ax1.legend(lines, [ln.get_label() for ln in lines])
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(path, format="svg", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    lmax_list = [2**n for n in range(2,13)]
    Np = len(lmax_list)

    # Exterior problem
    Tsolve_diag = np.zeros((Np,))
    Teval = np.zeros((Np,))
    Edirect = np.zeros((Np,))
    print(f"{'lmax':>5} {'err_direct':>12} {'t_direct(s)':>12} {'t_eval(s)':>12}", flush=True)
    for li in range(Np):
        print(f"  ... starting lmax={lmax_list[li]}", flush=True)
        t1, t2, e1 = test(lmax_list[li])
        Tsolve_diag[li], Teval[li] = t1, t2
        Edirect[li] = e1
        print(f"{lmax_list[li]:>5d} {e1:>12.3e} {t1:>12.4f} {t2:>12.4f}", flush=True)

    plot_timing_convergence(lmax_list, Tsolve_diag, Teval, Edirect,
                            "Stk3d manufactured solution, exterior",
                            os.path.join(here, './plots/Stk3d_exterior1sph.svg'))
    print("Wrote Stk3d_exterior1sph.svg to", here)


    # Interior problem
    Tsolve_diag = np.zeros((Np,))
    Teval = np.zeros((Np,))
    Edirect = np.zeros((Np,))
    print(f"{'lmax':>5} {'err_direct':>12} {'t_direct(s)':>12} {'t_eval(s)':>12}", flush=True)
    for li in range(Np):
        print(f"  ... starting lmax={lmax_list[li]}", flush=True)
        t1, t2, e1 = test(lmax_list[li], False)
        Tsolve_diag[li], Teval[li] = t1, t2
        Edirect[li] = e1
        print(f"{lmax_list[li]:>5d} {e1:>12.3e} {t1:>12.4f} {t2:>12.4f}", flush=True)

    plot_timing_convergence(lmax_list, Tsolve_diag, Teval, Edirect,
                            "Stk3d manufactured solution, interior",
                            os.path.join(here, './plots/Stk3d_interior1sph.svg'))
    print("Wrote Stk3d_interior1sph.svg to", here)
