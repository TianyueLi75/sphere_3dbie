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
import time
import matplotlib.pyplot as plt

jax.clear_caches()
jax.config.update("jax_enable_x64", True)  # support float64

def test(lmax: int):
    # Geometry setup
    center = jnp.array([0.,0.,0.])
    radius = 1.
    S = build_sphere(center, radius)
    S, sh = quadr_sphere(S, lmax)
    # Lap op far
    ext = True
    if ext:
        Rtrg = radius * 1.025
        sgn = 1.0 # exterior problem, sgn = +1
    else:
        Rtrg = radius * 0.5
        sgn = -1.0
    Strg = build_sphere(center, Rtrg)
    lmax_trg = 40 # Fix target size
    Strg, shtrg = quadr_sphere(Strg, lmax_trg)
    # BIOp parameter
    sl_scal = 1.0
    dl_scal = 1.0
    # Source for manufactured solutions
    ptsrc = jnp.array([[0.1,0.3,0.15]]) # shifted source to avoid constant potential on all of S
    force = jnp.ones((1,1)) 
    # Compute Boundary Conditions
    x = S["Xcart"][:,:,0]
    y = S["Xcart"][:,:,1]
    z = S["Xcart"][:,:,2]
    trg_sphere = jnp.column_stack([jnp.reshape(x,-1), jnp.reshape(y,-1), jnp.reshape(z,-1)])
    BC_pot = compute_potential(trg_sphere, ptsrc, force)
    BC_pot = jnp.reshape(BC_pot, x.shape)
    # BIO and gmres operator; solve
    LapK_apply = partial(
        bio_onsurf_apply,
        sh=sh,
        sl_scal=1.0, 
        dl_scal=1.0, 
        sgn=sgn
    )
    gmres_func = lx.FunctionLinearOperator(
        LapK_apply, jax.eval_shape(lambda: jnp.zeros(x.shape, dtype=jnp.complex128))
    )
    solver = lx.GMRES(rtol=1e-10, atol=1e-12, max_steps=200)
    solution = lx.linear_solve(
        gmres_func,
        BC_pot,
        solver=solver,
        options={"y0": jnp.zeros(x.shape, dtype=jnp.complex128)},
    )
    time_solver_start = time.time()
    solution = lx.linear_solve(
        gmres_func,
        BC_pot,
        solver=solver,
        options={"y0": jnp.zeros(x.shape, dtype=jnp.complex128)},
    )
    time_solver_end = time.time()
    time_solver = time_solver_end - time_solver_start

    sig_fromBC = solution.value # This will have the ntheta x nphi grid size
    stats = solution.stats
    # Manually check residual
    bc_check = bio_onsurf_apply(sig_fromBC, sh, sl_scal, dl_scal, sgn)
    resid_gmres = jnp.linalg.norm(bc_check - BC_pot)
    jax.debug.print("Checking residual of solve: {a}, number of iterations needed: {b}", a=resid_gmres, b=stats["num_steps"])


    # Compare with true solution at target sphere
    xtrg = Strg["Xcart"][:,:,0] 
    ytrg = Strg["Xcart"][:,:,1]
    ztrg = Strg["Xcart"][:,:,2]
    trg_sphere2 = jnp.column_stack([jnp.reshape(xtrg,-1), jnp.reshape(ytrg,-1), jnp.reshape(ztrg,-1)])
    S = set_density(S, sig_fromBC)
    Ksigma = bio_offsurf_apply(trg_sphere2, S, sh, sl_scal, dl_scal) # trg_sphere2 reshaped to be Ntrg x 3, so output is Ntrg x 1.

    time_eval_start = time.time()
    Ksigma = bio_offsurf_apply(trg_sphere2, S, sh, sl_scal, dl_scal) # trg_sphere2 reshaped to be Ntrg x 3, so output is Ntrg x 1.
    Ksigma.block_until_ready()
    time_eval_end = time.time()
    time_eval = time_eval_end - time_eval_start

    true_pot = compute_potential(trg_sphere2, ptsrc, force) # true_pot also computed as Ntrg x 1
    # For scalar electric potential calculation, only real values
    Ksigma = jnp.real(Ksigma)
    true_pot = jnp.real(true_pot)
    diff = jnp.max(true_pot - Ksigma) / jnp.max(true_pot )
    jax.debug.print("At target sphere Rtrg = {a}, max relative error from true potential using lmax = {b} is {c}", a=Rtrg, b=lmax, c=diff)

    return time_solver, time_eval

if __name__ == "__main__":
    pmin = 4
    pmax = 1000
    pstep = 25
    # pmin = 4
    # pmax = 64
    # pstep = 4
    lmax_list = jnp.arange(pmin, pmax, pstep, dtype = int)
    Np = len(lmax_list)
    Tsolve = np.zeros((Np,))
    Teval = np.zeros((Np,))
    for li in range(Np):
        t1, t2 = test(lmax_list[li])
        Tsolve[li] = t1
        Teval[li] = t2

    plt.plot(lmax_list, Tsolve, 'k*', label="gmres")
    plt.plot(lmax_list, Teval, 'b*', label="off-surf eval")
    plt.legend()
    plt.savefig('Lap3d_timing.png')
