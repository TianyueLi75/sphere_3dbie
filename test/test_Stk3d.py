"""Stokes single-sphere manufactured-solution tests (moved from biop/Stk3d.py __main__).

Each solve is run through BOTH transform layouts:
  - "cplx": the complex/full VSHT functions (*_cplx)
  - "real": the real/truncated VSHT functions (float64 grids)
Interior point Stokeslets produce a field that is a valid Stokes flow on the solution side.
We solve the on-surface BIE (spectral direct solve in the VWX diagonalizing basis), check the
self-consistency residual, and evaluate off-surface with the move-pole point_n_shoot evaluator,
comparing against the exact manufactured velocity on a non-concentric target sphere. The interior
case additionally cross-checks the arbitrary-point evaluators (bio_offsurf_apply far / point) vs
point_n_shoot.
"""

import jax.numpy as jnp
import pytest

from sphere import build_sphere, quadr_sphere
from biop import Stk3d

LMAX = 36
SL, DL = 1.0, 1.0


def _fns(transform):
    """Layout-specific function bundle for the requested transform."""
    sfx = "_cplx" if transform == "cplx" else ""
    return dict(
        xyz2vwx=getattr(Stk3d, "sig_xyz2vwx" + sfx),
        vwx2xyz=getattr(Stk3d, "sig_vwx2xyz" + sfx),
        direct_solve=getattr(Stk3d, "stokes_onsurf_direct_solve" + sfx),
        onsurf_apply=getattr(Stk3d, "bio_onsurf_apply" + sfx),
        pns=getattr(Stk3d, "point_n_shoot" + sfx),
        offsurf=getattr(Stk3d, "bio_offsurf_apply" + sfx),
    )


def _sphere(center, radius, lmax=LMAX):
    S = build_sphere(jnp.asarray(center, dtype=float), radius)
    return quadr_sphere(S, lmax)


def _to_vwx(fns, field, theta, phi, sh, transform):
    """Cartesian surface field (nphi, ntheta, 3) -> stacked VWX coeffs (3, nlm).

    The real/truncated sig_xyz2vwx takes a float64 grid; the manufactured field is stored
    complex128 (imag ~0), so pass its real part on the real path."""
    f = field if transform == "cplx" else jnp.real(field)
    return jnp.stack(fns["xyz2vwx"](f[:, :, 0], f[:, :, 1], f[:, :, 2], theta, phi, sh), axis=0)


def _synth_points(fns, vwx_out, Strg, shtrg):
    """point_n_shoot target-basis VWX coeffs -> Cartesian point values on the target sphere."""
    theta_t, phi_t = Strg["Xsph"][:, :, 0], Strg["Xsph"][:, :, 1]
    vx, vy, vz = fns["vwx2xyz"](vwx_out[0], vwx_out[1], vwx_out[2], theta_t, phi_t, shtrg)
    return jnp.real(jnp.stack([vx, vy, vz], axis=2))


@pytest.mark.parametrize("transform", ["cplx", "real"])
def test_exterior_dirichlet(transform):
    """Exterior Dirichlet, SL+DL (sgn=+1); target sphere wholly exterior to the source."""
    fns = _fns(transform)
    S, sh = _sphere([0., 0., 0.], 1.0)
    Strg, shtrg = _sphere([3., 0., 0.], 1.0)     # d - Rtrg = 2 > a = 1 -> wholly exterior
    theta, phi = S["Xsph"][:, :, 0], S["Xsph"][:, :, 1]
    ptsrc = jnp.array([[0.1, 0.3, 0.15], [-0.35, 0.2, 0.]])
    force = jnp.array([[1., 1., 1.], [-1., 0., 0.]])

    BC = Stk3d.compute_field(S["Xcart"].reshape(-1, 3), ptsrc, force).reshape(S["Xcart"].shape)
    vwx_bc = _to_vwx(fns, BC, theta, phi, sh, transform)
    vwx_sig = fns["direct_solve"](vwx_bc, sh, SL, DL, 1.0)

    resid = float(jnp.linalg.norm(fns["onsurf_apply"](vwx_sig, sh, SL, DL, 1.0) - vwx_bc))
    assert resid < 1e-10, f"residual too large: {resid:.3e}"

    true = jnp.real(Stk3d.compute_field(Strg["Xcart"].reshape(-1, 3), ptsrc, force).reshape(Strg["Xcart"].shape))
    approx = _synth_points(fns, fns["pns"](Strg, shtrg, vwx_sig, S, sh, SL, DL), Strg, shtrg)
    rel = float(jnp.max(jnp.abs(true - approx)) / jnp.max(jnp.abs(true)))
    assert rel < 1e-9, f"relative error too large: {rel:.3e}"


@pytest.mark.parametrize("transform", ["cplx", "real"])
def test_exterior_neumann(transform):
    """Exterior Neumann (traction BC): pure single-layer (dSL) formulation."""
    fns = _fns(transform)
    S, sh = _sphere([0., 0., 0.], 1.0)
    Strg, shtrg = _sphere([3., 0., 0.], 1.0)
    theta, phi = S["Xsph"][:, :, 0], S["Xsph"][:, :, 1]
    ptsrc = jnp.array([[0.1, 0.3, 0.15], [-0.35, 0.2, 0.]])
    force = jnp.array([[1., 1., 1.], [-1., 0., 0.]])

    nodes = S["Xcart"].reshape(-1, 3)
    nodesN = S["Xncart"].reshape(-1, 3)
    BC = Stk3d.compute_traction(nodes, nodesN, ptsrc, force).reshape(S["Xcart"].shape)
    vwx_bc = _to_vwx(fns, BC, theta, phi, sh, transform)
    vwx_sig = fns["direct_solve"](vwx_bc, sh=sh, sl_scal=0., dl_scal=0., sgn=1.0, dsl_scal=1.0)

    resid = float(jnp.linalg.norm(fns["onsurf_apply"](vwx_sig, sh, 0., 0., 1.0, 1.0) - vwx_bc))
    assert resid < 1e-10, f"residual too large: {resid:.3e}"

    true = jnp.real(Stk3d.compute_field(Strg["Xcart"].reshape(-1, 3), ptsrc, force).reshape(Strg["Xcart"].shape))
    approx = _synth_points(fns, fns["pns"](Strg, shtrg, vwx_sig, S, sh, 1.0, 0.), Strg, shtrg)   # SL only
    rel = float(jnp.max(jnp.abs(true - approx)) / jnp.max(jnp.abs(true)))
    assert rel < 1e-8, f"relative error too large: {rel:.3e}"


@pytest.mark.parametrize("transform", ["cplx", "real"])
def test_interior_dirichlet(transform):
    """Interior Dirichlet, SL+DL (sgn=-1); target sphere wholly interior to the source.

    Also cross-checks the arbitrary-point evaluators (bio_offsurf_apply far / point) against
    the move-pole point_n_shoot result synthesized to the same points.
    """
    fns = _fns(transform)
    S, sh = _sphere([0., 0., 0.], 1.0)
    Strg, shtrg = _sphere([0.2, 0., 0.], 0.5)    # d + Rtrg = 0.7 < a = 1 -> wholly interior
    theta, phi = S["Xsph"][:, :, 0], S["Xsph"][:, :, 1]
    ptsrc = jnp.array([[1.3, 1.75, -2.], [-1.3, -1., 2.32]])
    force = jnp.array([[1., -0.93, 1.25], [-0.2, 1.37, 0.]])

    BC = Stk3d.compute_field(S["Xcart"].reshape(-1, 3), ptsrc, force).reshape(S["Xcart"].shape)
    vwx_bc = _to_vwx(fns, BC, theta, phi, sh, transform)
    vwx_sig = fns["direct_solve"](vwx_bc, sh, SL, DL, -1.0)

    resid = float(jnp.linalg.norm(fns["onsurf_apply"](vwx_sig, sh, SL, DL, -1.0) - vwx_bc))
    assert resid < 1e-10, f"residual too large: {resid:.3e}"

    true = jnp.real(Stk3d.compute_field(Strg["Xcart"].reshape(-1, 3), ptsrc, force).reshape(Strg["Xcart"].shape))
    K_pns = _synth_points(fns, fns["pns"](Strg, shtrg, vwx_sig, S, sh, SL, DL), Strg, shtrg)
    rel = float(jnp.max(jnp.abs(true - K_pns)) / jnp.max(jnp.abs(true)))
    assert rel < 1e-9, f"relative error too large: {rel:.3e}"

    # Arbitrary-point far / point evaluators vs point_n_shoot at the same target points.
    trg = Strg["Xcart"].reshape(-1, 3)
    K_pns_pts = jnp.real(K_pns.reshape(-1, 3))
    K_far = jnp.real(fns["offsurf"](trg, vwx_sig, S, sh, SL, DL, far=True))
    err_far = float(jnp.max(jnp.abs(K_far - K_pns_pts)) / jnp.max(jnp.abs(K_pns_pts)))
    assert err_far < 1e-7, f"far-eval vs point_n_shoot error too large: {err_far:.3e}"

    K_pt = jnp.real(fns["offsurf"](trg, vwx_sig, S, sh, SL, DL, far=False))
    err_pt = float(jnp.max(jnp.abs(K_pt - K_pns_pts)) / jnp.max(jnp.abs(K_pns_pts)))
    assert err_pt < 1e-9, f"point-eval vs point_n_shoot error too large: {err_pt:.3e}"
