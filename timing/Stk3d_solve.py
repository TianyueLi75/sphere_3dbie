import numpy as np
import scipy
import scipy.sparse.linalg
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sphere_np import *
from biop.Stk3d import *
import shtns
import time
import matplotlib.pyplot as plt

def test(lmax: int):
    # Geometry setup
    center = np.array([0.,0.,0.])
    radius = 1.
    S = build_sphere(center, radius)
    S, sh = quadr_sphere(S, lmax)
    # Stk op far
    ext = False
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
    # Sources for manufactured solutions
    if ext:
        ptsrc = np.array([[0.1,0.3,0.15],[-0.35,0.2,0.]]) # shifted source to avoid constant potential on all of S
        # force = np.array([[1,1,1],[-1,-1,-1]])
        force = np.array([[1,1,1],[-1,0,0]])
    else:
        ptsrc = np.array([[1.3,1.75,-2],[-1.3,-1.75,2]])
        force = np.array([[1,1,1],[-1,-1,-1]]) # Need to have net force zero over the sphere 
    # Compute Boundary Conditions
    x = S["Xcart"][:,:,0]
    y = S["Xcart"][:,:,1]
    z = S["Xcart"][:,:,2]
    theta = S["Xsph"][:,:,0]
    phi = S["Xsph"][:,:,1]
    trg_sphere = np.column_stack([np.reshape(x,-1), np.reshape(y,-1), np.reshape(z,-1)])
    BC_pot = compute_field(trg_sphere, ptsrc, force)
    BC_pot = np.reshape(BC_pot, S["Xcart"].shape)
    # BIO and gmres operator; solve
    StkK_apply = partial(
        bio_onsurf_apply,
        theta = theta,
        phi = phi,
        sh=sh,
        sl_scal=sl_scal, 
        dl_scal=dl_scal, 
        sgn=sgn
    )
    total_dofs = S["Xcart"].size
    gmres_func = scipy.sparse.linalg.LinearOperator((total_dofs, total_dofs), \
                                                    matvec=StkK_apply, \
                                                    dtype=np.complex128)
    time_solver_start = time.time()
    x, info = scipy.sparse.linalg.gmres(gmres_func, BC_pot.flatten(), x0=np.zeros(total_dofs, dtype=np.complex128), \
                                        atol = 1e-14, rtol = 1e-13, maxiter=200)
    time_solver_end = time.time()
    time_solver = time_solver_end - time_solver_start

    sig_fromBC = x.reshape(theta.shape[0], theta.shape[1], 3)
    # Manually check residual
    bc_check = bio_onsurf_apply(x, theta, phi, sh, sl_scal, dl_scal, sgn)
    resid_gmres = np.linalg.norm(bc_check - BC_pot.flatten())
    print("Checking residual of solve: {a}, exitcode (0:successful): {b}".format(a=resid_gmres, b=info))

    # Compare with true solution at target sphere
    S = set_density(S, sig_fromBC[:,:,0], sig_fromBC[:,:,1], sig_fromBC[:,:,2])
    time_eval_start = time.time()
    Ksigma = bio_offsurf_apply_1sph(Strg, shtrg, S, sh, sl_scal, dl_scal) 
    time_eval_end = time.time()
    time_eval = time_eval_end - time_eval_start

    xtrg = Strg["Xcart"][:,:,0] 
    ytrg = Strg["Xcart"][:,:,1]
    ztrg = Strg["Xcart"][:,:,2]
    trg_sphere2 = np.column_stack([np.reshape(xtrg,-1), np.reshape(ytrg,-1), np.reshape(ztrg,-1)])
    true_field = compute_field(trg_sphere2, ptsrc, force) 
    true_field = np.reshape(true_field, Strg["Xcart"].shape)
    Ksigma = np.real(Ksigma)
    true_field = np.real(true_field)
    diff = np.max(true_field - Ksigma) / np.max(true_field)
    print("At target sphere Rtrg = {a}, max true field is {d}, relative error from true field using lmax = {b} is {c}".format(a=Rtrg, d=np.max(true_field), b=lmax, c=diff))

    return time_solver, time_eval

if __name__ == "__main__":
    pmin = 16
    pmax = 1000
    pstep = 50
    # pmin = 4
    # pmax = 40
    # pstep = 4
    lmax_list = np.arange(pmin, pmax, pstep, dtype = int)
    Np = len(lmax_list)
    Tsolve = np.zeros((Np,))
    Teval = np.zeros((Np,))
    for li in range(Np):
        t1, t2 = test(lmax_list[li])
        Tsolve[li] = t1
        Teval[li] = t2

    plt.plot(lmax_list, Tsolve, 'k*', label="gmres")
    plt.plot(lmax_list, Teval, 'b*', label="off-surf eval")
    plt.xlabel("lmax")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.savefig('Stk3d_timing_int.png')
