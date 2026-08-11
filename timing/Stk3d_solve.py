import jax
import jax.numpy as jnp
import numpy as np
# import lineax as lx
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
    
    # Non-concentric target sphere so point_n_shoot (to_lat_jax) applies.
    if exterior:
        trg_center = jnp.array([3.,0.,0.])   # wholly exterior to S (d - Rtrg = 2 > a = 1)
        Rtrg = radius
        sgn = 1.0
    else:
        trg_center = jnp.array([0.2,0.,0.])  # wholly interior to S (d + Rtrg = 0.7 < a = 1)
        Rtrg = radius * 0.5
        sgn = -1.0
    Strg = build_sphere(trg_center, Rtrg)
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

    # densities live in the VWX (diagonalizing) basis: COB the BC before the solve.
    # Real/truncated sig_xyz2vwx takes float64 grid; the manufactured field is real (compute_field
    # stores it complex128), so pass its real part.
    BC_pot_re = jnp.real(BC_pot)
    vwx_bc = jnp.stack(sig_xyz2vwx(BC_pot_re[:, :, 0], BC_pot_re[:, :, 1], BC_pot_re[:, :, 2], theta, phi, sh))
    sig_direct = stokes_onsurf_direct_solve(vwx_bc, sh, sl_scal, dl_scal, sgn)
    jax.block_until_ready(sig_direct)   # warmup: finish compile before timing
    tstart = time.time()
    sig_direct = stokes_onsurf_direct_solve(vwx_bc, sh, sl_scal, dl_scal, sgn)
    jax.block_until_ready(sig_direct)
    tend = time.time()
    time_direct = tend - tstart
    bc_check_direct = bio_onsurf_apply(sig_direct, sh, sl_scal, dl_scal, sgn)
    resid_direct = jnp.linalg.norm(bc_check_direct - vwx_bc)
    print("Residual of DIRECT solve = {a}".format(a=resid_direct))

    # Accuracy
    xtrg = Strg["Xcart"][:,:,0] 
    ytrg = Strg["Xcart"][:,:,1]
    ztrg = Strg["Xcart"][:,:,2]
    trg_sphere2 = jnp.column_stack([jnp.reshape(xtrg,-1), jnp.reshape(ytrg,-1), jnp.reshape(ztrg,-1)])
    true_field = compute_field(trg_sphere2, ptsrc, force) 
    true_field = jnp.reshape(true_field, Strg["Xcart"].shape)
    true_field = jnp.real(true_field)

    # sig_direct are the source VWX coefficients; point_n_shoot takes/returns coefficients.
    Ksig_vwx = point_n_shoot(Strg, shtrg, sig_direct, S, sh, sl_scal, dl_scal)  # warmup
    jax.block_until_ready(Ksig_vwx)
    tstart = time.time()
    Ksig_vwx = point_n_shoot(Strg, shtrg, sig_direct, S, sh, sl_scal, dl_scal)
    jax.block_until_ready(Ksig_vwx)
    tend = time.time()
    time_eval = tend - tstart
    # point_n_shoot returns target-basis VWX coeffs; recompose to Cartesian for the error
    vx, vy, vz = sig_vwx2xyz(Ksig_vwx[0], Ksig_vwx[1], Ksig_vwx[2],
                             Strg["Xsph"][:, :, 0], Strg["Xsph"][:, :, 1], shtrg)
    Ksig_direct = jnp.real(jnp.stack([vx, vy, vz], axis=2))

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
    lmax_list = [2**n for n in range(2,11)]
    Np = len(lmax_list)
    # 'real_' tag so the real/truncated-transform plots do not overwrite stored complex results.
    plotname_prefix = os.path.join(here, './plots/Stk3d_1sph_real_')
    if shtns_jax.CUDA_AVAILABLE:
        plotname_prefix += 'gpu_'
    else:
        plotname_prefix += 'cpu_'

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
                            plotname_prefix+'exterior.svg')
    print("Wrote"+ plotname_prefix +"exterior.svg to ", here)


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
                            plotname_prefix+'interior.svg')
    print("Wrote"+ plotname_prefix +"interior.svg to ", here)
