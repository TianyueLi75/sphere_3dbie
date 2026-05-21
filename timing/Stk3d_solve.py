import numpy as jnp
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
    Strg = build_sphere(center, Rtrg)
    lmax_trg = 40 # Fix target size
    Strg, shtrg = quadr_sphere(Strg, lmax_trg)

    # Manufactured solutions test
    ptsrc = jnp.array([[0.1,0.3,0.15],[-0.35,0.2,0.]]) # shifted source to avoid constant potential on all of S
    force = jnp.array([[1,1,1],[-1,0,0]])
    
    x = S["Xcart"][:,:,0]
    y = S["Xcart"][:,:,1]
    z = S["Xcart"][:,:,2]
    theta = S["Xsph"][:,:,0]
    phi = S["Xsph"][:,:,1]
    trg_sphere = jnp.column_stack([jnp.reshape(x,-1), jnp.reshape(y,-1), jnp.reshape(z,-1)])
    BC_pot = compute_field(trg_sphere, ptsrc, force)
    BC_pot = jnp.reshape(BC_pot, S["Xcart"].shape)
    
    # GMRES solve
    StkK_apply = partial(
        bio_onsurf_apply,
        theta = theta,
        phi = phi,
        sh=sh,
        sl_scal=sl_scal, 
        dl_scal=dl_scal, 
        sgn=sgn
    )
    gmres_func = lx.FunctionLinearOperator(
        StkK_apply, jax.eval_shape(lambda: jnp.zeros(S["Xcart"].shape, dtype=jnp.complex128))
    )
    solver = lx.GMRES(rtol=1e-10, atol=1e-12, max_steps=200)

    solution = lx.linear_solve(
        gmres_func,
        BC_pot,
        solver=solver,
        options={"y0": jnp.zeros(S["Xcart"].shape, dtype=jnp.complex128)},
    )
    
    
    tstart = time.time()
    solution = lx.linear_solve(
        gmres_func,
        BC_pot,
        solver=solver,
        options={"y0": jnp.zeros(S["Xcart"].shape, dtype=jnp.complex128)},
    )
    tend = time.time()
    time_gmres = tend - tstart
    sig_gmres = solution.value 
    stats = solution.stats
    # Manually check residual
    bc_check = bio_onsurf_apply(sig_gmres, theta, phi, sh, sl_scal, dl_scal, sgn)
    resid_gmres = jnp.linalg.norm(bc_check - BC_pot)
    jax.debug.print("Residual of GMRES solve = {a}, number of iterations = {b}", a=resid_gmres, b=stats["num_steps"])

    sig_direct = stokes_onsurf_direct_solve(BC_pot, theta, phi, sh, sl_scal, dl_scal, sgn)
    tstart = time.time()
    sig_direct = stokes_onsurf_direct_solve(BC_pot, theta, phi, sh, sl_scal, dl_scal, sgn)
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

    S = set_density(S, sig_gmres[:,:,0], sig_gmres[:,:,1], sig_gmres[:,:,2])
    Ksig_gmres = bio_offsurf_apply_1sph(Strg, shtrg, S, sh, sl_scal, dl_scal)
    Ksig_gmres = jnp.real(Ksig_gmres)

    S = set_density(S, sig_direct[:,:,0], sig_direct[:,:,1], sig_direct[:,:,2])
    tstart = time.time()
    Ksig_direct = bio_offsurf_apply_1sph(Strg, shtrg, S, sh, sl_scal, dl_scal)
    tend = time.time()
    time_eval = tend - tstart
    Ksig_direct = jnp.real(Ksig_direct)

    diff_gmres = jnp.max(true_field - Ksig_gmres) / jnp.max(true_field)
    diff_direct = jnp.max(true_field - Ksig_direct) / jnp.max(true_field)
    print("Max relative error of order {lmax} solver at target radius {Rtrg} for GMRES solver is {d1}, for direct solver is {d2}".format(lmax=lmax, Rtrg=Rtrg, d1=diff_gmres, d2=diff_direct))

    return time_gmres, time_direct, time_eval

if __name__ == "__main__":
    # pmin = 16
    # pmax = 1000
    # pstep = 50
    pmin = 4
    pmax = 100
    pstep = 25
    lmax_list = jnp.arange(pmin, pmax, pstep, dtype = int)
    Np = len(lmax_list)
    Tsolve = np.zeros((Np,))
    Tsolve_diag = np.zeros((Np,))
    Teval = np.zeros((Np,))
    for li in range(Np):
        t1, t2, t3 = test(lmax_list[li])
        Tsolve[li] = t1
        Tsolve_diag[li] = t2
        Teval[li] = t3

    plt.plot(lmax_list, Tsolve, 'k*', label="gmres")
    plt.plot(lmax_list, Tsolve_diag, 'k+', label="direct")
    plt.plot(lmax_list, Teval, 'ko', label="off-surf eval")
    plt.xlabel("lmax")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.savefig('Stk3d_timing_2.png')
