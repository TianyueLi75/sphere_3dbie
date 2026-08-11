"""Laplace single-sphere manufactured-solution tests (moved from biop/Lap3d.py __main__).

Each solve is run through BOTH transform layouts:
  - "cplx": the complex/full SH functions (*_cplx, sh.analys_cplx_jax)
  - "real": the real/truncated SH functions (sh.analys_jax, float64 grids)
A point charge placed on the singular side of the surface produces a field that is
harmonic on the solution side; we solve the on-surface BIE (spectral direct solve),
check the self-consistency residual, and check the off-surface potential against the
exact manufactured potential at check points.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from sphere import build_sphere, quadr_sphere
from biop import Lap3d

LMAX = 36
SL, DL = 1.0, 1.0

# A handful of check directions (unit vectors) scaled to the target radius.
_DIRS = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                  [1., 1., 1.], [1., -1., 0.5], [-1., 0.5, -1.]])
_DIRS = jnp.asarray(_DIRS / np.linalg.norm(_DIRS, axis=1, keepdims=True))


def _analys(sh, grid, transform):
    """Boundary data grid -> SH coefficients in the requested layout."""
    if transform == "cplx":
        return sh.analys_cplx_jax(grid)
    return sh.analys_jax(jnp.real(grid))          # real/truncated analysis takes float64


def _fns(transform):
    """(direct_solve, onsurf_apply, offsurf_apply) for the requested layout."""
    sfx = "_cplx" if transform == "cplx" else ""
    return (getattr(Lap3d, "bio_onsurf_direct_solve" + sfx),
            getattr(Lap3d, "bio_onsurf_apply" + sfx),
            getattr(Lap3d, "bio_offsurf_apply" + sfx))


def _unit_sphere(lmax=LMAX):
    S = build_sphere(jnp.array([0., 0., 0.]), 1.0)
    S, sh = quadr_sphere(S, lmax)
    return S, sh


@pytest.mark.parametrize("transform", ["cplx", "real"])
def test_exterior_dirichlet(transform):
    """Exterior Dirichlet: interior point charge, SL+DL formulation (sgn=+1)."""
    direct_solve, onsurf_apply, offsurf_apply = _fns(transform)
    S, sh = _unit_sphere()
    ptsrc = jnp.array([[0.1, 0.3, 0.15]])      # inside S -> harmonic exterior
    force = jnp.ones((1, 1))

    BC = Lap3d.compute_potential(S["Xcart"].reshape(-1, 3), ptsrc, force).reshape(S["Xcart"].shape[:2])
    qlm_bc = _analys(sh, BC, transform)
    qlm_sig = direct_solve(qlm_bc, sh=sh, sl_scal=SL, dl_scal=DL, sgn=1.0)

    resid = float(jnp.linalg.norm(onsurf_apply(qlm_sig, sh, SL, DL, 1.0) - qlm_bc))
    assert resid < 1e-10, f"residual too large: {resid:.3e}"

    chk = 1.5 * _DIRS                            # exterior check points
    true_pot = jnp.real(Lap3d.compute_potential(chk, ptsrc, force))
    approx = jnp.real(offsurf_apply(chk, qlm_sig, S, sh, SL, DL, far=False))
    rel = float(jnp.max(jnp.abs(true_pot - approx)) / jnp.max(jnp.abs(true_pot)))
    assert rel < 1e-9, f"relative error too large: {rel:.3e}"


@pytest.mark.parametrize("transform", ["cplx", "real"])
def test_exterior_neumann(transform):
    """Exterior Neumann: match du/dn with a pure single-layer (dSL) formulation."""
    direct_solve, onsurf_apply, offsurf_apply = _fns(transform)
    S, sh = _unit_sphere()
    ptsrc = jnp.array([[0.1, 0.3, 0.15]])
    force = jnp.ones((1, 1))

    nodes = S["Xcart"].reshape(-1, 3)
    nodesN = S["Xncart"].reshape(-1, 3)
    BC = Lap3d.compute_flux(nodes, nodesN, ptsrc, force).reshape(S["Xcart"].shape[:2])
    qlm_bc = _analys(sh, BC, transform)
    qlm_sig = direct_solve(qlm_bc, sh=sh, sl_scal=0., dl_scal=0., sgn=1.0, dsl_scal=1.0)

    resid = float(jnp.linalg.norm(onsurf_apply(qlm_sig, sh, 0., 0., 1.0, 1.0) - qlm_bc))
    assert resid < 1e-10, f"residual too large: {resid:.3e}"

    # The solution is represented as a single layer, so evaluate SL only.
    chk = 1.5 * _DIRS
    true_pot = jnp.real(Lap3d.compute_potential(chk, ptsrc, force))
    approx = jnp.real(offsurf_apply(chk, qlm_sig, S, sh, 1.0, 0., far=False))
    rel = float(jnp.max(jnp.abs(true_pot - approx)) / jnp.max(jnp.abs(true_pot)))
    assert rel < 1e-8, f"relative error too large: {rel:.3e}"


@pytest.mark.parametrize("transform", ["cplx", "real"])
def test_interior_dirichlet(transform):
    """Interior Dirichlet: exterior point charges, SL+DL formulation (sgn=-1)."""
    direct_solve, onsurf_apply, offsurf_apply = _fns(transform)
    S, sh = _unit_sphere()
    ptsrc = jnp.array([[1.5, 2., 1.5], [-1.5, -2., -1.5]])   # outside S -> harmonic interior
    force = jnp.array([[1.], [-1.]])

    BC = Lap3d.compute_potential(S["Xcart"].reshape(-1, 3), ptsrc, force).reshape(S["Xcart"].shape[:2])
    qlm_bc = _analys(sh, BC, transform)
    qlm_sig = direct_solve(qlm_bc, sh=sh, sl_scal=SL, dl_scal=DL, sgn=-1.0)

    resid = float(jnp.linalg.norm(onsurf_apply(qlm_sig, sh, SL, DL, -1.0) - qlm_bc))
    assert resid < 1e-10, f"residual too large: {resid:.3e}"

    chk = 0.5 * _DIRS                            # interior check points
    true_pot = jnp.real(Lap3d.compute_potential(chk, ptsrc, force))
    approx = jnp.real(offsurf_apply(chk, qlm_sig, S, sh, SL, DL, far=False))
    rel = float(jnp.max(jnp.abs(true_pot - approx)) / jnp.max(jnp.abs(true_pot)))
    assert rel < 1e-9, f"relative error too large: {rel:.3e}"
