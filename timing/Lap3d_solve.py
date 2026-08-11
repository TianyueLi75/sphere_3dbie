import jax
import jax.numpy as jnp
import numpy as np
import lineax as lx
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sphere import *
from biop.Lap3d import *
import shtns
import shtns_jax
import time
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

def test(lmax: int):
    # Geometry setup
    center = jnp.array([0.,0.,0.])
    radius = 1.
    S = build_sphere(center, radius)
    S, sh = quadr_sphere(S, lmax)
    # BIOp parameter
    sl_scal = 1.0
    dl_scal = 1.0
    
    Rtrg = radius * 1.025
    sgn = 1.0
    # A small fixed set of exterior check points on the sphere of radius Rtrg. The concentric
    # 1sph evaluator was removed (mirroring Stokes), so accuracy is checked with the supported
    # point evaluator bio_offsurf_apply on a handful of points -- keeps the eager near-eval cheap.
    _dirs = jnp.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                       [1., 1., 1.], [1., -1., 0.5], [-1., 0.5, -1.]])
    chk = Rtrg * _dirs / jnp.linalg.norm(_dirs, axis=1, keepdims=True)
    
    # Manufactured solutions test
    ptsrc = jnp.array([[0.1,0.3,0.15]]) 
    force = jnp.ones((1,1)) 
    
    x = S["Xcart"][:,:,0]
    y = S["Xcart"][:,:,1]
    z = S["Xcart"][:,:,2]
    trg_sphere = jnp.column_stack([jnp.reshape(x,-1), jnp.reshape(y,-1), jnp.reshape(z,-1)])
    BC_pot = compute_potential(trg_sphere, ptsrc, force)
    BC_pot = jnp.reshape(BC_pot, x.shape)

    # GMRES solve -- operates in the SH (diagonalizing) basis: COB the BC to coefficients, solve
    # the coeff->coeff operator, keep the density as coefficients (recomposed only for the error).
    qlm_bc = sh.analys_cplx_jax(BC_pot)
    struct = jax.eval_shape(lambda: jnp.zeros((sh.nlm_cplx,), dtype=jnp.complex128))
    LapK_apply = partial(
        bio_onsurf_apply_cplx,   # accuracy eval below (bio_offsurf_apply_cplx) is cplx; keep whole test cplx
        sh=sh,
        sl_scal=sl_scal,
        dl_scal=dl_scal,
        sgn=sgn
    )
    gmres_func = lx.FunctionLinearOperator(LapK_apply, struct)
    solver = lx.GMRES(rtol=1e-10, atol=1e-12, max_steps=200)
    opts = {"y0": jnp.zeros((sh.nlm_cplx,), dtype=jnp.complex128)}

    solution = lx.linear_solve(gmres_func, qlm_bc, solver=solver, options=opts)
    jax.block_until_ready(solution.value)   # warmup: finish compile before timing
    tstart = time.time()
    solution = lx.linear_solve(gmres_func, qlm_bc, solver=solver, options=opts)
    jax.block_until_ready(solution.value)
    tend = time.time()
    time_gmres = tend - tstart

    sig_gmres = solution.value                   # SH coefficients
    stats = solution.stats
    bc_check = bio_onsurf_apply_cplx(sig_gmres, sh, sl_scal, dl_scal, sgn)
    resid_gmres = jnp.linalg.norm(bc_check - qlm_bc)
    jax.debug.print("Residual of GMRES solve = {a}, number of iterations = {b}", a=resid_gmres, b=stats["num_steps"])

    # DIRECT solve (coeff -> coeff)
    sig_direct = bio_onsurf_direct_solve_cplx(qlm_bc, sh=sh, sl_scal=sl_scal, dl_scal=dl_scal, sgn=sgn)
    jax.block_until_ready(sig_direct)   # warmup: finish compile before timing
    tstart = time.time()
    sig_direct = bio_onsurf_direct_solve_cplx(qlm_bc, sh=sh, sl_scal=sl_scal, dl_scal=dl_scal, sgn=sgn)
    jax.block_until_ready(sig_direct)
    tend = time.time()
    time_direct = tend - tstart

    bc_check_direct = bio_onsurf_apply_cplx(sig_direct, sh, sl_scal, dl_scal, sgn)
    resid_direct = jnp.linalg.norm(bc_check_direct - qlm_bc)
    jax.debug.print("Residual of DIRECT solve: {a}", a=resid_direct)

    # Accuracy at the exterior check points (radius Rtrg).
    true_pot = jnp.real(compute_potential(chk, ptsrc, force))

    Ksig_gmres = bio_offsurf_apply_cplx(chk, sig_gmres, S, sh, sl_scal, dl_scal, far=False)
    jax.block_until_ready(Ksig_gmres)   # warmup eval: finish compile before the timed direct eval
    Ksig_gmres = jnp.real(jnp.reshape(Ksig_gmres, (-1, 1)))

    tstart = time.time()
    Ksig_direct = bio_offsurf_apply_cplx(chk, sig_direct, S, sh, sl_scal, dl_scal, far=False)
    jax.block_until_ready(Ksig_direct)
    tend = time.time()
    time_eval = tend - tstart
    Ksig_direct = jnp.real(jnp.reshape(Ksig_direct, (-1, 1)))

    err_gmres = jnp.max(jnp.abs(true_pot - Ksig_gmres)) / jnp.max(jnp.abs(true_pot))
    err_direct = jnp.max(jnp.abs(true_pot - Ksig_direct)) / jnp.max(jnp.abs(true_pot))
    jax.debug.print("Max relative error of order {lmax} solver at target radius {Rtrg} for GMRES solver is {d1}, for direct solver is {d2}", lmax=lmax, Rtrg=Rtrg, d1=err_gmres, d2=err_direct)

    return time_gmres, time_direct, time_eval, float(err_gmres), float(err_direct)

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    lmax_list = list(range(4, 101, 8))   # manufactured-solution sweep, lmax in [4, 100]
    Np = len(lmax_list)
    Tsolve = np.zeros((Np,))
    Tsolve_diag = np.zeros((Np,))
    Teval = np.zeros((Np,))
    Egmres = np.zeros((Np,))
    Edirect = np.zeros((Np,))
    print(f"{'lmax':>5} {'err_gmres':>12} {'err_direct':>12} {'t_gmres(s)':>12} {'t_direct(s)':>12} {'t_eval(s)':>12}", flush=True)
    for li in range(Np):
        print(f"  ... starting lmax={lmax_list[li]}", flush=True)
        t1, t2, t3, e1, e2 = test(lmax_list[li])
        Tsolve[li], Tsolve_diag[li], Teval[li] = t1, t2, t3
        Egmres[li], Edirect[li] = e1, e2
        print(f"{lmax_list[li]:>5d} {e1:>12.3e} {e2:>12.3e} {t1:>12.4f} {t2:>12.4f} {t3:>12.4f}", flush=True)

    # Timing plot.
    plt.figure()
    plt.plot(lmax_list, Tsolve, 'k*-', label="gmres")
    plt.plot(lmax_list, Tsolve_diag, 'k+-', label="direct")
    plt.plot(lmax_list, Teval, 'ko-', label="off-surf eval")
    plt.xlabel("lmax"); plt.ylabel("Time (s)")
    plt.title("Lap3d manufactured solution: timing")
    plt.legend()
    plt.savefig(os.path.join(here, './plots/Lap3d_timing.svg'), format="svg", dpi=150)

    # Convergence plot (true-solution relative error).
    plt.figure()
    plt.semilogy(lmax_list, Egmres, 'k*-', label="gmres")
    plt.semilogy(lmax_list, Edirect, 'r+--', label="direct")
    plt.xlabel("lmax"); plt.ylabel("max relative error vs exact")
    plt.title("Lap3d manufactured solution: convergence")
    plt.legend()
    plt.savefig(os.path.join(here, './plots/Lap3d_convergence.svg'), format="svg", dpi=150)
    print("Wrote Lap3d_timing.svg and Lap3d_convergence.svg to", here)
