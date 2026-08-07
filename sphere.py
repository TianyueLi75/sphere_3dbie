"""
Sphere Class: 
    a dictionary object containing the 
    geometry and density information on a sphere
"""

from typing import Dict, Any, Tuple

import jax
import jax.numpy as jnp
import os
import sys

import shtns
import shtns_jax
jax.config.update("jax_enable_x64", True)  # support float64

SphereDict = Dict[str, Any]

def build_sphere(center: jax.Array, radius: float) -> SphereDict:
    """ 
    create a sphere dictionary with given center and radius. 
    """

    return {
        "Xc": center,
        "r": radius,
        "lmax": int(-1), 
        "Xcart": jnp.array([]), # Npts x 3
        "Xncart": jnp.array([]),
        "Xsph": jnp.array([]), # Npts x 2, theta and phi grids.
        "Sigma": jnp.array([])
    }

def quadr_sphere(S: SphereDict, lmax: int) -> Tuple[SphereDict, shtns_jax.sht]:
    """
    Add a regular spherical grid for the sphere surface.
    """

    S["lmax"] = lmax
    
    sh = shtns_jax.sht(int(lmax))
    ntheta, nphi = sh.set_grid(flags=shtns.SHT_ALLOW_GPU + shtns.SHT_THETA_CONTIGUOUS) 
    thetas = jnp.arccos(sh.cos_theta)
    phis = jnp.linspace(0, 2 * jnp.pi, nphi, endpoint=False)
    phi_grid, theta_grid = jnp.meshgrid(phis, thetas, indexing="ij")

    radius = S["r"]
    center = S["Xc"]

    x = center[0] + radius * jnp.sin(theta_grid) * jnp.cos(phi_grid)
    y = center[1] + radius * jnp.sin(theta_grid) * jnp.sin(phi_grid)
    z = center[2] + radius * jnp.cos(theta_grid)
    S["Xcart"] = jnp.dstack([x,y,z])

    xn = jnp.sin(theta_grid) * jnp.cos(phi_grid)
    yn = jnp.sin(theta_grid) * jnp.sin(phi_grid)
    zn = jnp.cos(theta_grid)
    S["Xncart"] = jnp.dstack([xn,yn,zn])

    S["Xsph"] = jnp.dstack([theta_grid, phi_grid])

    return S, sh

def set_density(S: SphereDict, sig_x: jax.Array, sig_y: jax.Array = jnp.array([]), sig_z: jax.Array = jnp.array([])) -> SphereDict:
    """
    Add surface density to Sphere object
    """

    assert sig_x.shape[0] > 0 
    assert S["Xcart"].shape[0] > 0
    assert S["Xcart"].shape[0] == sig_x.shape[0] and S["Xcart"].shape[1] == sig_x.shape[1]
    
    if len(sig_y) == 0:
        S["Sigma"] = jnp.dstack([sig_x])
    else:
        assert sig_y.shape[0] == sig_x.shape[0] and sig_z.shape[0] == sig_x.shape[0]
        assert sig_y.shape[1] == sig_x.shape[1] and sig_z.shape[1] == sig_x.shape[1]
        S["Sigma"] = jnp.dstack([sig_x, sig_y, sig_z])
    
    return S

# ---------------------------------------------------------------------------
# Smooth-quadrature ("far") evaluation helpers, shared by biop/Lap3d.py and biop/Stk3d.py.
# The far kernels are dense Nystrom sums over Ntrg x Nsrc pairs, so they all need the same
# source-centred geometry and the same bounded-memory target tiling.
# ---------------------------------------------------------------------------
FAR_TILE_BYTES = 43 << 20     # memory budget for ONE pair-shaped (tile, Nsrc) float64 block
FAR_TILE_MIN = 96             # ... clamped: below this the gemms lose blocking efficiency,
FAR_TILE_MAX = 256            # ... above it the pair blocks stop being cache-resident

def far_src_geom(S: SphereDict, sh: shtns_jax.sht) -> Tuple[jax.Array, jax.Array, Any, jax.Array]:
    """
    Source-centred quadrature geometry of sphere <S>, as consumed by the far kernels:
        Y  : (Nsrc, 3) surface nodes RELATIVE to the sphere centre, so |Y_j| = r and the
             outward unit normal is exactly n_j = Y_j / r -- which is what lets the far
             kernels get r.n_src for free out of the same dot product that gives d^2.
             Built as r * Xncart rather than Xcart - Xc: the normals are unit to 2e-16 by
             construction, whereas re-subtracting the centre loses digits for a small
             sphere far from the origin (up to ~4e-14 in |Y|/r for |Xc| ~ 100 r).
        w  : (Nsrc,) quadrature weights, Gauss-Legendre x uniform-in-phi (sh._grid_weights)
             including the r^2 surface Jacobian.
        r, Xc : the radius and centre.
    Working relative to Xc is also what keeps the Gram-form squared distance
    (|x|^2 - 2 x.Y + r^2) accurate: the cancellation is then bounded by the same
    target/source separation the smooth quadrature already requires.
    """
    grid_shape = S["Xcart"].shape[:2]
    Xc = jnp.asarray(S["Xc"], dtype=jnp.float64)
    Y = (S["r"] * S["Xncart"]).reshape(-1, 3)
    w = jnp.broadcast_to(sh._grid_weights(), grid_shape).reshape(-1) * S["r"] ** 2
    return Y, w, S["r"], Xc

def pair_dot(X: jax.Array, Y: jax.Array) -> jax.Array:
    """x_i . Y_j for every target/source pair: (Ntrg,3) x (Nsrc,3) -> (Ntrg, Nsrc).
    Written as three broadcast products rather than X @ Y.T on purpose. A dot's operands
    must be materialized by XLA, so X @ Y.T both writes a whole extra pair-shaped buffer and
    runs an Eigen contraction over a length-3 inner dimension; as three broadcasts it fuses
    into whatever consumes d^2 and is never materialized. Measured ~4x on the far kernels."""
    return X[:, 0:1] * Y[:, 0] + X[:, 1:2] * Y[:, 1] + X[:, 2:3] * Y[:, 2]

def far_tile_size(Nsrc: int, tile: int = None) -> int:
    """Number of targets handled per tile of a far sum: <tile> if given, else the largest
    tile whose (tile, Nsrc) float64 pair block fits in FAR_TILE_BYTES, clamped to
    [FAR_TILE_MIN, FAR_TILE_MAX].
    Measured on the Stokes far kernel (lmax 16..128, Nsrc 1e3..4e4): 256 is the optimum over
    most of that range and the optimum drifts down to ~96-128 once Nsrc ~ 4e4, which is what
    this budget-with-clamp reproduces (within ~10% of the per-size optimum throughout).
    Tiles far from it are slower, not just heavier -- 1024 costs ~2x at lmax 32."""
    if tile is not None:
        return int(tile)
    budget = FAR_TILE_BYTES // (8 * max(int(Nsrc), 1))
    return int(min(FAR_TILE_MAX, max(FAR_TILE_MIN, budget)))

def far_tile_map(body, per_trg: tuple, tile: int) -> jax.Array:
    """
    Evaluate <body>(*tiles) over tiles of <tile> targets and concatenate the result, so a
    far sum's peak memory is O(tile * Nsrc) instead of O(Ntrg * Nsrc).
    <per_trg> is a tuple of (Ntrg, k) per-target arrays (targets, and target normals where
    the kernel needs them); all are tiled consistently. <body> maps them to (tile, ncol).
    Targets are padded up to a whole number of tiles by repeating the last one (a finite
    dummy target, dropped afterwards) so the traced shapes stay static; jax.lax.map keeps
    the tiles sequential rather than unrolling them into the jaxpr.
    """
    N = per_trg[0].shape[0]
    if not tile or N <= tile:
        return body(*per_trg)
    n_tiles = -(-N // tile)
    pad = n_tiles * tile - N
    if pad:
        per_trg = tuple(jnp.concatenate([v, jnp.broadcast_to(v[-1], (pad,) + v.shape[1:])])
                        for v in per_trg)
    tiled = tuple(v.reshape(n_tiles, tile, *v.shape[1:]) for v in per_trg)
    out = jax.lax.map(lambda args: body(*args), tiled)
    return out.reshape(-1, out.shape[-1])[:N]

def separate_target(trg: jax.Array, S: SphereDict, sep_eta: float) -> jax.Array:
    """
    Given a list of targets and min separation distance, 
    return an array of booleans (0,1) 
    whether each target is outside sep_eta and 
    is therefore far from sphere S.
    """
    # reshape target to Ntrg x 3
    if trg.shape[1] != 3:
        if trg.shape[0] != 3:
            raise Exception("target should be a Ntrg x 3 array.")
        else:
            trg = jnp.transpose(trg)

    trg_dc = jnp.linalg.norm(trg - S["Xc"], axis=1) # per-target distance to sphere center
    sep_vec = (trg_dc - S["r"]) > sep_eta * S["r"] # far if the surface gap exceeds sep_eta * r

    return sep_vec
