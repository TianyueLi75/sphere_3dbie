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
    ntheta, nphi = sh.set_grid(flags=shtns.SHT_THETA_CONTIGUOUS) # TODO: allow GPU
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
