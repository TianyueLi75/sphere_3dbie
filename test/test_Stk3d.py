"""Stokes single-sphere manufactured-solution tests (moved from biop/Stk3d.py __main__).

Each solve is run through the real/truncated VSHT functions (float64 grids).
Interior point Stokeslets produce a field that is a valid Stokes flow on the solution side.
We solve the on-surface BIE (spectral direct solve in the VWX diagonalizing basis), check the
self-consistency residual, and evaluate off-surface with the arbitrary-point far evaluator
(bio_offsurf_apply far=True), comparing against the exact manufactured velocity on a
non-concentric target sphere.

Far eval only: point_n_shoot (move-pole) uses shtns Wigner-D rotations, which have no GPU
kernel, so it is skipped here to keep these tests GPU-runnable.
"""

import jax.numpy as jnp

from sphere import build_sphere, quadr_sphere
from biop import Stk3d

LMAX = 36
SL, DL = 1.0, 1.0


def _fns():
    """Function bundle for the real/truncated transform layout."""
    return dict(
        xyz2vwx=Stk3d.sig_xyz2vwx,
        vwx2xyz=Stk3d.sig_vwx2xyz,
        direct_solve=Stk3d.stokes_onsurf_direct_solve,
        onsurf_apply=Stk3d.bio_onsurf_apply,
        offsurf=Stk3d.bio_offsurf_apply,
    )


def _sphere(center, radius, lmax=LMAX):
    S = build_sphere(jnp.asarray(center, dtype=float), radius)
    return quadr_sphere(S, lmax)


def _to_vwx(fns, field, theta, phi, sh):
    """Cartesian surface field (nphi, ntheta, 3) -> stacked VWX coeffs (3, nlm).

    The real/truncated sig_xyz2vwx takes a float64 grid; the manufactured field is stored
    complex128 (imag ~0), so pass its real part."""
    f = jnp.real(field)
    return jnp.stack(fns["xyz2vwx"](f[:, :, 0], f[:, :, 1], f[:, :, 2], theta, phi, sh), axis=0)


def test_exterior_dirichlet():
    """Exterior Dirichlet, SL+DL (sgn=+1); target sphere wholly exterior to the source."""
    fns = _fns()
    S, sh = _sphere([0., 0., 0.], 1.0)
    Strg, _ = _sphere([3., 0., 0.], 1.0)     # d - Rtrg = 2 > a = 1 -> wholly exterior
    theta, phi = S["Xsph"][:, :, 0], S["Xsph"][:, :, 1]
    ptsrc = jnp.array([[0.1, 0.3, 0.15], [-0.35, 0.2, 0.]])
    force = jnp.array([[1., 1., 1.], [-1., 0., 0.]])

    BC = Stk3d.compute_field(S["Xcart"].reshape(-1, 3), ptsrc, force).reshape(S["Xcart"].shape)
    vwx_bc = _to_vwx(fns, BC, theta, phi, sh)
    vwx_sig = fns["direct_solve"](vwx_bc, sh, SL, DL, 1.0)

    resid = float(jnp.linalg.norm(fns["onsurf_apply"](vwx_sig, sh, SL, DL, 1.0) - vwx_bc))
    assert resid < 1e-10, f"residual too large: {resid:.3e}"

    trg = Strg["Xcart"].reshape(-1, 3)
    true = jnp.real(Stk3d.compute_field(trg, ptsrc, force))
    approx = jnp.real(fns["offsurf"](trg, vwx_sig, S, sh, SL, DL, far=True))
    rel = float(jnp.max(jnp.abs(true - approx)) / jnp.max(jnp.abs(true)))
    assert rel < 1e-7, f"relative error too large: {rel:.3e}"


def test_exterior_neumann():
    """Exterior Neumann (traction BC): pure single-layer (dSL) formulation."""
    fns = _fns()
    S, sh = _sphere([0., 0., 0.], 1.0)
    Strg, _ = _sphere([3., 0., 0.], 1.0)
    theta, phi = S["Xsph"][:, :, 0], S["Xsph"][:, :, 1]
    ptsrc = jnp.array([[0.1, 0.3, 0.15], [-0.35, 0.2, 0.]])
    force = jnp.array([[1., 1., 1.], [-1., 0., 0.]])

    nodes = S["Xcart"].reshape(-1, 3)
    nodesN = S["Xncart"].reshape(-1, 3)
    BC = Stk3d.compute_traction(nodes, nodesN, ptsrc, force).reshape(S["Xcart"].shape)
    vwx_bc = _to_vwx(fns, BC, theta, phi, sh)
    vwx_sig = fns["direct_solve"](vwx_bc, sh=sh, sl_scal=0., dl_scal=0., sgn=1.0, dsl_scal=1.0)

    resid = float(jnp.linalg.norm(fns["onsurf_apply"](vwx_sig, sh, 0., 0., 1.0, 1.0) - vwx_bc))
    assert resid < 1e-10, f"residual too large: {resid:.3e}"

    trg = Strg["Xcart"].reshape(-1, 3)
    true = jnp.real(Stk3d.compute_field(trg, ptsrc, force))
    approx = jnp.real(fns["offsurf"](trg, vwx_sig, S, sh, 1.0, 0., far=True))   # SL only
    rel = float(jnp.max(jnp.abs(true - approx)) / jnp.max(jnp.abs(true)))
    assert rel < 1e-7, f"relative error too large: {rel:.3e}"


def test_interior_dirichlet():
    """Interior Dirichlet, SL+DL (sgn=-1); target sphere wholly interior to the source."""
    fns = _fns()
    S, sh = _sphere([0., 0., 0.], 1.0)
    Strg, _ = _sphere([0.2, 0., 0.], 0.5)    # d + Rtrg = 0.7 < a = 1 -> wholly interior
    theta, phi = S["Xsph"][:, :, 0], S["Xsph"][:, :, 1]
    ptsrc = jnp.array([[1.3, 1.75, -2.], [-1.3, -1., 2.32]])
    force = jnp.array([[1., -0.93, 1.25], [-0.2, 1.37, 0.]])

    BC = Stk3d.compute_field(S["Xcart"].reshape(-1, 3), ptsrc, force).reshape(S["Xcart"].shape)
    vwx_bc = _to_vwx(fns, BC, theta, phi, sh)
    vwx_sig = fns["direct_solve"](vwx_bc, sh, SL, DL, -1.0)

    resid = float(jnp.linalg.norm(fns["onsurf_apply"](vwx_sig, sh, SL, DL, -1.0) - vwx_bc))
    assert resid < 1e-10, f"residual too large: {resid:.3e}"

    trg = Strg["Xcart"].reshape(-1, 3)
    true = jnp.real(Stk3d.compute_field(trg, ptsrc, force))
    approx = jnp.real(fns["offsurf"](trg, vwx_sig, S, sh, SL, DL, far=True))
    rel = float(jnp.max(jnp.abs(true - approx)) / jnp.max(jnp.abs(true)))
    assert rel < 1e-7, f"relative error too large: {rel:.3e}"
