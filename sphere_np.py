"""
Numpy version of the Sphere Class: 
    for vectorized and non-jaxed SHTns code
"""

from typing import Dict, Any, Tuple

import os
import sys
import numpy as np
import shtns
import shtns_jax

SphereDict = Dict[str, Any]

def build_sphere(center: np.ndarray, radius: float) -> SphereDict:
    """ 
    create a sphere dictionary with given center and radius. 
    """

    return {
        "Xc": center,
        "r": radius,
        "lmax": int(-1), 
        "Xcart": np.array([]), # Npts x 3
        "Xncart": np.array([]),
        "Xsph": np.array([]), # Npts x 2, theta and phi grids.
        "Sigma": np.array([])
    }

def quadr_sphere(S: SphereDict, lmax: int) -> Tuple[SphereDict, shtns_jax.sht]:
    """
    Add a regular spherical grid for the sphere surface.
    """

    S["lmax"] = lmax
    
    sh = shtns_jax.sht(int(lmax))
    ntheta, nphi = sh.set_grid(flags=shtns.SHT_THETA_CONTIGUOUS) # TODO: allow GPU
    thetas = np.arccos(sh.cos_theta)
    phis = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
    phi_grid, theta_grid = np.meshgrid(phis, thetas, indexing="ij")

    radius = S["r"]
    center = S["Xc"]

    x = center[0] + radius * np.sin(theta_grid) * np.cos(phi_grid)
    y = center[1] + radius * np.sin(theta_grid) * np.sin(phi_grid)
    z = center[2] + radius * np.cos(theta_grid)
    S["Xcart"] = np.dstack([x,y,z])

    xn = radius * np.sin(theta_grid) * np.cos(phi_grid)
    yn = radius * np.sin(theta_grid) * np.sin(phi_grid)
    zn = radius * np.cos(theta_grid)
    S["Xncart"] = np.dstack([xn,yn,zn])

    S["Xsph"] = np.dstack([theta_grid, phi_grid])

    return S, sh

def set_density(S: SphereDict, sig_x: np.ndarray, sig_y: np.ndarray = np.array([]), sig_z: np.ndarray = np.array([])) -> SphereDict:
    """
    Add surface density to Sphere object
    """
    
    assert sig_x.shape[0] > 0 
    assert S["Xcart"].shape[0] > 0
    assert S["Xcart"].shape[0] == sig_x.shape[0] and S["Xcart"].shape[1] == sig_x.shape[1]
    
    if len(sig_y) == 0:
        S["Sigma"] = np.dstack([sig_x])
    else:
        assert sig_y.shape[0] == sig_x.shape[0] and sig_z.shape[0] == sig_x.shape[0]
        assert sig_y.shape[1] == sig_x.shape[1] and sig_z.shape[1] == sig_x.shape[1]
        S["Sigma"] = np.dstack([sig_x, sig_y, sig_z])
    
    return S


