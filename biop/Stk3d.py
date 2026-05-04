from typing import Dict, Any, Tuple
from functools import partial

import scipy, time
import numpy as np
import scipy.sparse.linalg
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from sphere import *
from sphere_np import *
import shtns

SphereDict = Dict[str, Any]

def cart2sph(Vx: np.ndarray, Vy: np.ndarray, Vz: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> tuple([np.ndarray, np.ndarray, np.ndarray]):
    # Transform vector field components from Cartesian to spherical coordinates
    Vr = Vx * np.sin(theta) * np.cos(phi) + Vy * np.sin(theta) * np.sin(phi) + Vz * np.cos(theta)
    Vtheta = Vx * np.cos(theta) * np.cos(phi) + Vy * np.cos(theta) * np.sin(phi) - Vz * np.sin(theta)
    Vphi = -Vx * np.sin(phi) + Vy * np.cos(phi)
    return Vr, Vtheta, Vphi

def sph2cart(Vr: np.ndarray, Vtheta: np.ndarray, Vphi: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> tuple([np.ndarray, np.ndarray, np.ndarray]):
    # Transform vector field components from spherical to Cartesian coordinates
    Vx = Vr * np.sin(theta) * np.cos(phi) + Vtheta * np.cos(theta) * np.cos(phi) - Vphi * np.sin(phi)
    Vy = Vr * np.sin(theta) * np.sin(phi) + Vtheta * np.cos(theta) * np.sin(phi) + Vphi * np.cos(phi)
    Vz = Vr * np.cos(theta) - Vtheta * np.sin(theta)
    return Vx, Vy, Vz

def qst2vwx(qlm: np.ndarray, slm: np.ndarray, tlm: np.ndarray, sh: shtns.sht) -> tuple([np.ndarray, np.ndarray, np.ndarray]):
    l_vals = np.asarray(sh.zl, dtype=np.float64)
    vlm = (l_vals * slm - qlm) / (2.0*l_vals + 1.0)
    wlm = ((l_vals + 1.0) * slm + qlm) / (2.0*l_vals + 1.0)
    xlm = -tlm
    return vlm, wlm, xlm

def vwx2qst(vlm: np.ndarray, wlm: np.ndarray, xlm: np.ndarray, sh: shtns.sht) -> tuple([np.ndarray, np.ndarray, np.ndarray]):
    l_vals = np.asarray(sh.zl, dtype=np.float64)
    slm = vlm + wlm
    qlm = l_vals * (wlm - vlm) - vlm
    tlm = -xlm
    return qlm, slm, tlm

def Stk3d_sl_VWX_diag(sh: shtns.sht) -> tuple([np.ndarray, np.ndarray, np.ndarray]):
    l_vals = np.asarray(sh.zl, dtype=np.float64)
    diag_V = l_vals / (2.0*l_vals + 1.0) / (2.0*l_vals + 3.0)
    diag_W = (l_vals + 1.0) / (2.0*l_vals + 1.0) / (2.0*l_vals - 1.0)
    diag_X = 1.0 / (2.0*l_vals + 1.0)

    return diag_V, diag_W, diag_X

def Stk3d_dl_VWX_diag(sh: shtns.sht) -> tuple([np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
    l_vals = np.asarray(sh.zl, dtype=np.float64)

    diag_V_ext = (2.0*l_vals*l_vals + 4*l_vals + 3) / (2.0*l_vals + 1.0) / (2.0*l_vals + 3.0)
    diag_W_ext = 2.0*(l_vals + 1.0)*(l_vals - 1.0) / (2.0*l_vals + 1.0) / (2.0*l_vals - 1.0)
    diag_X_ext = (l_vals - 1.0) / (2.0*l_vals + 1.0)

    diag_V_int = -2.0*l_vals*(l_vals + 2) / (2.0*l_vals + 1.0) / (2.0*l_vals + 3.0)
    diag_W_int = -(2.0*l_vals*l_vals + 1.0) / (2.0*l_vals + 1.0) / (2.0*l_vals - 1.0)
    diag_X_int = -(l_vals + 2.0) / (2.0*l_vals + 1.0)

    return diag_V_ext, diag_W_ext, diag_X_ext, diag_V_int, diag_W_int, diag_X_int
    
# TODO: S traction

# UPDATE: Since have to evaluate on whole spheres of different radii, will take in an array of radii instead, and loop through.
# TODO: r should also be scaled, since formula r is relative to 1.
def Stk3d_sl_r(trg_r: np.ndarray, S: SphereDict, sh: shtns.sht) -> np.ndarray:
    if S["lmax"] != sh.lmax:
        print("S lmax does not match sht's lmax, reform sht.")
        sh = shtns.sht(S["lmax"], S["lmax"])

    Sigma_x = S["Sigma"][:,:,0] 
    Sigma_y = S["Sigma"][:,:,1] 
    Sigma_z = S["Sigma"][:,:,2] 
    theta = S["Xsph"][:,:,0]
    phi = S["Xsph"][:,:,1]
    vlm_sigma, wlm_sigma, xlm_sigma = sig_xyz2vwx(Sigma_x, Sigma_y, Sigma_z, theta, phi, sh)
    # # DEBUG
    # print(f"VWX coeff after Vana: W: {wlm_sigma}, v (=0): {np.linalg.norm(vlm_sigma)}, |x| (=0): {np.linalg.norm(xlm_sigma)}") # Checked

    SL_sigma = np.zeros((Sigma_x.shape[0], Sigma_x.shape[1], 3, trg_r.size), dtype=np.complex128)

    for ri in range(trg_r.size):
        trg_dr = trg_r[ri]

        l_vals = np.asarray(sh.zl, dtype=np.float64) # shape ((p+1)^2, )
        diag_V, diag_W, diag_X = Stk3d_sl_VWX_diag(sh) # Coeff for V-V, W-W, X-X matches for SL
        rpowers_V_ext = trg_dr ** (-l_vals-2.0) # Ntrg x Nlm
        rpowers_V_int = trg_dr ** (l_vals+1.0) 
        vlm_SL_sigma_ext = rpowers_V_ext * diag_V * vlm_sigma 
        vlm_SL_sigma_int = rpowers_V_int * diag_V * vlm_sigma

        rpowers_W_ext = trg_dr ** (-l_vals)
        rpowers_W_int = trg_dr ** (l_vals - 1.0)
        wlm_SL_sigma_ext = rpowers_W_ext * diag_W * wlm_sigma 
        wlm_SL_sigma_int = rpowers_W_int * diag_W * wlm_sigma 

        rpowers_X_ext = trg_dr ** (-l_vals - 1.0)
        rpowers_X_int = trg_dr ** (l_vals)
        xlm_SL_sigma_ext = rpowers_X_ext * diag_X * xlm_sigma
        xlm_SL_sigma_int = rpowers_X_int * diag_X * xlm_sigma

        diag_V2W_int = (l_vals+1.0) / (4.0*l_vals+2.0) # additional coeff for V<-W, W<-V
        diag_W2V_ext = l_vals / (4.0*l_vals+2.0) # additional coeff for V<-W, W<-V
        rpowers_V2W_int = trg_dr ** (l_vals+1.0) - trg_dr ** (l_vals - 1.0) # NOTE: TYPO IN PAPER
        rpowers_W2V_ext = trg_dr ** (-l_vals - 2.0) - trg_dr ** (-l_vals)
        V2Wlm_SL_sigma_int = rpowers_V2W_int * diag_V2W_int * vlm_sigma
        W2Vlm_SL_sigma_ext = rpowers_W2V_ext * diag_W2V_ext * wlm_sigma

        vlm_SL_sigma = vlm_SL_sigma_ext + W2Vlm_SL_sigma_ext if trg_dr > S['r'] else vlm_SL_sigma_int 
        wlm_SL_sigma = wlm_SL_sigma_ext if trg_dr > S['r'] else wlm_SL_sigma_int + V2Wlm_SL_sigma_int
        xlm_SL_sigma = xlm_SL_sigma_ext if trg_dr > S['r'] else xlm_SL_sigma_int

        val_x, val_y, val_z = sig_vwx2xyz(vlm_SL_sigma, wlm_SL_sigma, xlm_SL_sigma, theta, phi, sh)
        SL_sigma[:,:,:,ri] = np.stack([val_x, val_y, val_z], axis=2)

    return SL_sigma

def Stk3d_dl_r(trg_r: np.ndarray, S: SphereDict, sh: shtns.sht) -> np.ndarray:
    if S["lmax"] != sh.lmax:
        print("S lmax does not match sht's lmax, reform sht.")
        sh = shtns.sht(S["lmax"], S["lmax"])

    Sigma_x = S["Sigma"][:,:,0] 
    Sigma_y = S["Sigma"][:,:,1] 
    Sigma_z = S["Sigma"][:,:,2] 
    theta = S["Xsph"][:,:,0]
    phi = S["Xsph"][:,:,1]
    vlm_sigma, wlm_sigma, xlm_sigma = sig_xyz2vwx(Sigma_x, Sigma_y, Sigma_z, theta, phi, sh)

    DL_sigma = np.zeros((Sigma_x.shape[0], Sigma_x.shape[1], 3, trg_r.size), dtype=np.complex128)

    for ri in range(trg_r.size):
        trg_dr = trg_r[ri]

        l_vals = np.asarray(sh.zl, dtype=np.float64) # shape ((p+1)^2, )
        diag_V_ext, diag_W_ext, diag_X_ext, diag_V_int, diag_W_int, diag_X_int = Stk3d_dl_VWX_diag(sh) # Coeff for V-V, W-W, X-X matches for DL
        
        rpowers_V_ext = trg_dr ** (- l_vals - 2.0) # Ntrg x Nlm
        rpowers_V_int = trg_dr ** (l_vals + 1.0) 
        vlm_DL_sigma_ext = rpowers_V_ext * diag_V_ext * vlm_sigma # resulting Vnm coeff from sigma_Vnm terms (DL[V] = aV + bW)
        vlm_DL_sigma_int = rpowers_V_int * diag_V_int * vlm_sigma

        rpowers_W_ext = trg_dr ** (-l_vals)
        rpowers_W_int = trg_dr ** (l_vals - 1.0)
        wlm_DL_sigma_ext = rpowers_W_ext * diag_W_ext * wlm_sigma
        wlm_DL_sigma_int = rpowers_W_int * diag_W_int * wlm_sigma 

        rpowers_X_ext = trg_dr ** (-l_vals - 1.0)
        rpowers_X_int = trg_dr ** (l_vals)
        xlm_DL_sigma_ext = rpowers_X_ext * diag_X_ext * xlm_sigma
        xlm_DL_sigma_int = rpowers_X_int * diag_X_int * xlm_sigma

        diag_V2W_int = (l_vals+1.0) * (l_vals + 2.0) / (2.0*l_vals+1.0) # additional coeff for V<-W and W->V
        diag_W2V_ext = 2. * l_vals * (l_vals - 1.0) / (4.0*l_vals+2.0)
        rpowers_V2W_int = - trg_dr ** (l_vals + 1.0) + trg_dr ** (l_vals - 1.0)
        rpowers_W2V_ext = trg_dr ** (-l_vals - 2.0) - trg_dr ** (-l_vals) 
        V2Wlm_DL_sigma_int = rpowers_V2W_int * diag_V2W_int * vlm_sigma
        W2Vlm_DL_sigma_ext = rpowers_W2V_ext * diag_W2V_ext * wlm_sigma

        vlm_DL_sigma = vlm_DL_sigma_ext + W2Vlm_DL_sigma_ext if trg_dr > S['r'] else vlm_DL_sigma_int
        wlm_DL_sigma = wlm_DL_sigma_ext if trg_dr > S['r'] else wlm_DL_sigma_int + V2Wlm_DL_sigma_int
        xlm_DL_sigma = xlm_DL_sigma_ext if trg_dr > S['r'] else xlm_DL_sigma_int

        val_x, val_y, val_z = sig_vwx2xyz(vlm_DL_sigma, wlm_DL_sigma, xlm_DL_sigma, theta, phi, sh)
        DL_sigma[:,:,:,ri] = np.stack([val_x, val_y, val_z], axis=2)

    return DL_sigma

def Stk3d_sl_r_1sph(Strg: SphereDict, shtrg: shtns.sht, S: SphereDict, sh: shtns.sht) -> np.ndarray:
    if S["lmax"] != sh.lmax:
        print("S lmax does not match sht's lmax, reform sht.")
        sh = shtns.sht(S["lmax"], S["lmax"])

    Sigma_x = S["Sigma"][:,:,0] 
    Sigma_y = S["Sigma"][:,:,1] 
    Sigma_z = S["Sigma"][:,:,2] 
    theta = S["Xsph"][:,:,0]
    phi = S["Xsph"][:,:,1]
    vlm_sigma, wlm_sigma, xlm_sigma = sig_xyz2vwx(Sigma_x, Sigma_y, Sigma_z, theta, phi, sh)
    
    trg_dr = Strg["r"]
    l_vals = np.asarray(sh.zl, dtype=np.float64) # shape ((p+1)^2, )
    diag_V, diag_W, diag_X = Stk3d_sl_VWX_diag(sh) # Coeff for V-V, W-W, X-X matches for SL
    rpowers_V_ext = trg_dr ** (-l_vals-2.0) # Ntrg x Nlm
    rpowers_V_int = trg_dr ** (l_vals+1.0) 
    vlm_SL_sigma_ext = rpowers_V_ext * diag_V * vlm_sigma 
    vlm_SL_sigma_int = rpowers_V_int * diag_V * vlm_sigma

    rpowers_W_ext = trg_dr ** (-l_vals)
    rpowers_W_int = trg_dr ** (l_vals - 1.0)
    wlm_SL_sigma_ext = rpowers_W_ext * diag_W * wlm_sigma 
    wlm_SL_sigma_int = rpowers_W_int * diag_W * wlm_sigma 

    rpowers_X_ext = trg_dr ** (-l_vals - 1.0)
    rpowers_X_int = trg_dr ** (l_vals)
    xlm_SL_sigma_ext = rpowers_X_ext * diag_X * xlm_sigma
    xlm_SL_sigma_int = rpowers_X_int * diag_X * xlm_sigma

    diag_V2W_int = (l_vals+1.0) / (4.0*l_vals+2.0) # additional coeff for V<-W, W<-V
    diag_W2V_ext = l_vals / (4.0*l_vals+2.0) # additional coeff for V<-W, W<-V
    rpowers_V2W_int = trg_dr ** (l_vals+1.0) - trg_dr ** (l_vals - 1.0) # NOTE: TYPO IN PAPER
    rpowers_W2V_ext = trg_dr ** (-l_vals - 2.0) - trg_dr ** (-l_vals)
    V2Wlm_SL_sigma_int = rpowers_V2W_int * diag_V2W_int * vlm_sigma
    W2Vlm_SL_sigma_ext = rpowers_W2V_ext * diag_W2V_ext * wlm_sigma

    vlm_SL_sigma = vlm_SL_sigma_ext + W2Vlm_SL_sigma_ext if trg_dr > S['r'] else vlm_SL_sigma_int 
    wlm_SL_sigma = wlm_SL_sigma_ext if trg_dr > S['r'] else wlm_SL_sigma_int + V2Wlm_SL_sigma_int
    xlm_SL_sigma = xlm_SL_sigma_ext if trg_dr > S['r'] else xlm_SL_sigma_int

    theta_trg = Strg["Xsph"][:,:,0]
    phi_trg = Strg["Xsph"][:,:,1]
    # Interpolate to new grid by padding or truncating coefficients to Strg['lmax']
    nlm_src = sh.nlm_cplx
    nlm_trg = shtrg.nlm_cplx
    if nlm_trg > nlm_src:
        vlm_SL_sigma = np.pad(vlm_SL_sigma, (0, nlm_trg - nlm_src), constant_values=0)
        wlm_SL_sigma = np.pad(wlm_SL_sigma, (0, nlm_trg - nlm_src), constant_values=0)
        xlm_SL_sigma = np.pad(xlm_SL_sigma, (0, nlm_trg - nlm_src), constant_values=0)
    elif nlm_trg < nlm_src:
        vlm_SL_sigma = vlm_SL_sigma[:nlm_trg]
        wlm_SL_sigma = wlm_SL_sigma[:nlm_trg]
        xlm_SL_sigma = xlm_SL_sigma[:nlm_trg]
    val_x, val_y, val_z = sig_vwx2xyz(vlm_SL_sigma, wlm_SL_sigma, xlm_SL_sigma, theta_trg, phi_trg, shtrg)
    SL_sigma = np.stack([val_x, val_y, val_z], axis=2)

    return SL_sigma

def Stk3d_dl_r_1sph(Strg: SphereDict, shtrg: shtns.sht, S: SphereDict, sh: shtns.sht) -> np.ndarray:
    if S["lmax"] != sh.lmax:
        print("S lmax does not match sht's lmax, reform sht.")
        sh = shtns.sht(S["lmax"], S["lmax"])

    Sigma_x = S["Sigma"][:,:,0] 
    Sigma_y = S["Sigma"][:,:,1] 
    Sigma_z = S["Sigma"][:,:,2] 
    theta = S["Xsph"][:,:,0]
    phi = S["Xsph"][:,:,1]
    vlm_sigma, wlm_sigma, xlm_sigma = sig_xyz2vwx(Sigma_x, Sigma_y, Sigma_z, theta, phi, sh)

    trg_dr = Strg["r"]
    l_vals = np.asarray(sh.zl, dtype=np.float64) # shape ((p+1)^2, )
    diag_V_ext, diag_W_ext, diag_X_ext, diag_V_int, diag_W_int, diag_X_int = Stk3d_dl_VWX_diag(sh) # Coeff for V-V, W-W, X-X matches for DL
    
    rpowers_V_ext = trg_dr ** (- l_vals - 2.0) # Ntrg x Nlm
    rpowers_V_int = trg_dr ** (l_vals + 1.0) 
    vlm_DL_sigma_ext = rpowers_V_ext * diag_V_ext * vlm_sigma # resulting Vnm coeff from sigma_Vnm terms (DL[V] = aV + bW)
    vlm_DL_sigma_int = rpowers_V_int * diag_V_int * vlm_sigma

    rpowers_W_ext = trg_dr ** (-l_vals)
    rpowers_W_int = trg_dr ** (l_vals - 1.0)
    wlm_DL_sigma_ext = rpowers_W_ext * diag_W_ext * wlm_sigma
    wlm_DL_sigma_int = rpowers_W_int * diag_W_int * wlm_sigma 

    rpowers_X_ext = trg_dr ** (-l_vals - 1.0)
    rpowers_X_int = trg_dr ** (l_vals)
    xlm_DL_sigma_ext = rpowers_X_ext * diag_X_ext * xlm_sigma
    xlm_DL_sigma_int = rpowers_X_int * diag_X_int * xlm_sigma

    diag_V2W_int = (l_vals+1.0) * (l_vals + 2.0) / (2.0*l_vals+1.0) # additional coeff for V<-W and W->V
    diag_W2V_ext = 2. * l_vals * (l_vals - 1.0) / (4.0*l_vals+2.0)
    rpowers_V2W_int = - trg_dr ** (l_vals + 1.0) + trg_dr ** (l_vals - 1.0)
    rpowers_W2V_ext = trg_dr ** (-l_vals - 2.0) - trg_dr ** (-l_vals) 
    V2Wlm_DL_sigma_int = rpowers_V2W_int * diag_V2W_int * vlm_sigma
    W2Vlm_DL_sigma_ext = rpowers_W2V_ext * diag_W2V_ext * wlm_sigma

    vlm_DL_sigma = vlm_DL_sigma_ext + W2Vlm_DL_sigma_ext if trg_dr > S['r'] else vlm_DL_sigma_int
    wlm_DL_sigma = wlm_DL_sigma_ext if trg_dr > S['r'] else wlm_DL_sigma_int + V2Wlm_DL_sigma_int
    xlm_DL_sigma = xlm_DL_sigma_ext if trg_dr > S['r'] else xlm_DL_sigma_int

    theta_trg = Strg["Xsph"][:,:,0]
    phi_trg = Strg["Xsph"][:,:,1]
    # Interpolate to new grid by padding or truncating coefficients to Strg['lmax']
    nlm_src = sh.nlm_cplx
    nlm_trg = shtrg.nlm_cplx
    if nlm_trg > nlm_src:
        vlm_DL_sigma = np.pad(vlm_DL_sigma, (0, nlm_trg - nlm_src), constant_values=0)
        wlm_DL_sigma = np.pad(wlm_DL_sigma, (0, nlm_trg - nlm_src), constant_values=0)
        xlm_DL_sigma = np.pad(xlm_DL_sigma, (0, nlm_trg - nlm_src), constant_values=0)
    elif nlm_trg < nlm_src:
        vlm_DL_sigma = vlm_DL_sigma[:nlm_trg]
        wlm_DL_sigma = wlm_DL_sigma[:nlm_trg]
        xlm_DL_sigma = xlm_DL_sigma[:nlm_trg]
    val_x, val_y, val_z = sig_vwx2xyz(vlm_DL_sigma, wlm_DL_sigma, xlm_DL_sigma, theta_trg, phi_trg, shtrg)
    DL_sigma = np.stack([val_x, val_y, val_z], axis=2)

    return DL_sigma

def trueY00(theta,phi):
    rhat = np.sqrt(1./4./np.pi)
    that = 0.
    phat = 0.
    return sph2cart(np.array([rhat]),np.array([that]),np.array([phat]),np.array([theta]),np.array([phi]))

def trueG00(theta,phi):
    rhat = 0.
    that = 0.
    phat = 0.
    return sph2cart(np.array([rhat]),np.array([that]),np.array([phat]),np.array([theta]),np.array([phi]))

def trueX00(theta,phi):
    rhat = 0.
    that = 0.
    phat = 0.
    return sph2cart(np.array([rhat]),np.array([that]),np.array([phat]),np.array([theta]),np.array([phi]))

def trueY10(theta,phi):
    rhat = np.sqrt(3./4./np.pi) * np.cos(theta)
    that = 0.
    phat = 0.
    return sph2cart(np.array([rhat]),np.array([that]),np.array([phat]),np.array([theta]),np.array([phi]))

def trueG10(theta,phi):
    that = (-1) * np.sqrt(3./4./np.pi) * np.sin(theta)
    rhat = 0.
    phat = 0.
    return sph2cart(np.array([rhat]),np.array([that]),np.array([phat]),np.array([theta]),np.array([phi]))

def trueX10(theta,phi):
    phat = (-1) * np.sqrt(3./4./np.pi) * np.sin(theta)
    rhat = 0.
    that = 0.
    return sph2cart(np.array([rhat]),np.array([that]),np.array([phat]),np.array([theta]),np.array([phi]))

def trueY11(theta,phi):
    rhat = (-1) * np.sqrt(3./8./np.pi) * np.sin(theta) * np.exp(1j*phi)
    that = 0.
    phat = 0.
    return sph2cart(np.array([rhat]),np.array([that]),np.array([phat]),np.array([theta]),np.array([phi]))

def trueG11(theta,phi):
    rhat = 0.
    that = (-1) * np.sqrt(3./8./np.pi) * np.cos(theta) * np.exp(1j*phi)
    phat = (-1) * np.sqrt(3./8./np.pi) * 1j * np.exp(1j*phi)
    return sph2cart(np.array([rhat]),np.array([that]),np.array([phat]),np.array([theta]),np.array([phi]))

def trueX11(theta,phi):
    rhat = 0.
    phat = np.sqrt(3./8./np.pi) * (-1) * np.cos(theta) * np.exp(1j*phi)
    that = np.sqrt(3./8./np.pi) * 1j * np.exp(1j*phi)
    return sph2cart(np.array([rhat]),np.array([that]),np.array([phat]),np.array([theta]),np.array([phi]))

def trueY1m1(theta,phi):
    rhat = np.sqrt(3./8./np.pi) * np.sin(theta) * np.exp(-1j*phi)
    that = 0.
    phat = 0.
    return sph2cart(np.array([rhat]),np.array([that]),np.array([phat]),np.array([theta]),np.array([phi]))

def trueG1m1(theta,phi):
    rhat = 0.
    that = np.sqrt(3./8./np.pi) * np.cos(theta) * np.exp(-1j*phi)
    phat = np.sqrt(3./8./np.pi) * -1j * np.exp(-1j*phi)
    return sph2cart(np.array([rhat]),np.array([that]),np.array([phat]),np.array([theta]),np.array([phi]))

def trueX1m1(theta,phi):
    rhat = 0.
    phat = np.sqrt(3./8./np.pi) * np.cos(theta) * np.exp(-1j*phi)
    that = np.sqrt(3./8./np.pi) * 1j * np.exp(-1j*phi)
    return sph2cart(np.array([rhat]),np.array([that]),np.array([phat]),np.array([theta]),np.array([phi]))

def sig_xyz2vwx(sigma_x: np.ndarray, sigma_y: np.ndarray, sigma_z: np.ndarray, theta: np.ndarray, phi: np.ndarray, sh: shtns.sht) -> tuple([np.ndarray,np.ndarray,np.ndarray]):
    sigma_r, sigma_t, sigma_p = cart2sph(sigma_x,sigma_y,sigma_z,theta,phi)
    qlm, slm, tlm = sh.analys_cplx(sigma_r, sigma_t, sigma_p)
    vlm, wlm, xlm = qst2vwx(qlm, slm, tlm, sh)
    return vlm, wlm, xlm

def sig_vwx2xyz(vlm: np.ndarray, wlm: np.ndarray, xlm: np.ndarray, theta: np.ndarray, phi: np.ndarray, sh: shtns.sht) -> tuple([np.ndarray, np.ndarray, np.ndarray]):
    qlm, slm, tlm = vwx2qst(vlm,wlm,xlm,sh)
    vr, vt, vp = sh.synth_cplx(qlm, slm, tlm)
    vx, vy, vz = sph2cart(vr, vt, vp, theta, phi)
    return vx, vy, vz

def bio_onsurf_apply(sigma_tens: np.ndarray, theta: np.ndarray, phi: np.ndarray, sh: shtns.sht, sl_scal: float, dl_scal: float, sgn: float) -> np.ndarray:
    # Due to scipy...gmres requirement, input sigma_tens has to be a 1d array. reshape to get tensor
    sigma_tens = sigma_tens.reshape(theta.shape[0], theta.shape[1], 3)

    sigma_x = sigma_tens[:,:,0]
    sigma_y = sigma_tens[:,:,1]
    sigma_z = sigma_tens[:,:,2]
    vlm, wlm, xlm = sig_xyz2vwx(sigma_x, sigma_y, sigma_z, theta, phi, sh)

    diag_V, diag_W, diag_X = Stk3d_sl_VWX_diag(sh)
    vlm_SL_sigma = diag_V * vlm
    wlm_SL_sigma = diag_W * wlm
    xlm_SL_sigma = diag_X * xlm
    
    diag_V_ext, diag_W_ext, diag_X_ext, diag_V_int, diag_W_int, diag_X_int = Stk3d_dl_VWX_diag(sh)
    vlm_DL_sigma = 0.5*(diag_V_int + diag_V_ext) * vlm
    wlm_DL_sigma = 0.5*(diag_W_int + diag_W_ext) * wlm
    xlm_DL_sigma = 0.5*(diag_X_int + diag_X_ext) * xlm

    vlm_op = sl_scal * vlm_SL_sigma + dl_scal * vlm_DL_sigma
    wlm_op = sl_scal * wlm_SL_sigma + dl_scal * wlm_DL_sigma
    xlm_op = sl_scal * xlm_SL_sigma + dl_scal * xlm_DL_sigma

    vx,vy,vz = sig_vwx2xyz(vlm_op,wlm_op,xlm_op,theta,phi,sh)
    vx = vx + 0.5 * sgn * dl_scal * sigma_x
    vy = vy + 0.5 * sgn * dl_scal * sigma_y
    vz = vz + 0.5 * sgn * dl_scal * sigma_z
    V = np.stack([vx, vy, vz], axis=2)
    return V.flatten()

def stokes_onsurf_diag_solve(
    bc_vec: np.ndarray,        # Shape (Ntheta, Nphi, 3)
    theta: np.ndarray,
    phi: np.ndarray,
    sh: shtns.sht,
    sl_scal: float,
    dl_scal: float,
    sgn: float
) -> np.ndarray:
    """
    Directly solves the Stokes BIO equation using the VWX diagonal property.
    Equation: [0.5 * dl_scal * sgn * I + KL] sigma = bc_vec
    """
    # 1. Decompose the vector BC into V, W, X spectral coefficients
    # Expects bc_vec components at [..., 0], [..., 1], [..., 2]
    vlm_bc, wlm_bc, xlm_bc = sig_xyz2vwx(
        bc_vec[..., 0], bc_vec[..., 1], bc_vec[..., 2], 
        theta, phi, sh
    )

    # 2. Get the diagonal spectra for SL and DL
    diag_V_sl, diag_W_sl, diag_X_sl = Stk3d_sl_VWX_diag(sh)
    (diag_V_ext, diag_W_ext, diag_X_ext, 
     diag_V_int, diag_W_int, diag_X_int) = Stk3d_dl_VWX_diag(sh)

    # Principal Value of K = 0.5 * (Interior + Exterior)
    diag_V_k = 0.5 * (diag_V_int + diag_V_ext)
    diag_W_k = 0.5 * (diag_W_int + diag_W_ext)
    diag_X_k = 0.5 * (diag_X_int + diag_X_ext)

    # 3. Construct the full diagonal for each mode
    # Operator = (0.5 * dl_scal * sgn) * I + dl_scal * K + sl_scal * SL
    id_term = 0.5 * dl_scal * sgn
    
    op_diag_V = id_term + (dl_scal * diag_V_k) + (sl_scal * diag_V_sl)
    op_diag_W = id_term + (dl_scal * diag_W_k) + (sl_scal * diag_W_sl)
    op_diag_X = id_term + (dl_scal * diag_X_k) + (sl_scal * diag_X_sl)

    # 4. Solve in spectral space with null-space handling
    # Using a small epsilon to avoid division by zero in null spaces (like l=0 or l=1)
    eps = 1e-14
    
    def safe_div(bc_lm, op_diag):
        safe = np.where(np.abs(op_diag) > eps, op_diag, 1.0+0j)
        res = bc_lm / safe
        # Return BC value where diag is zero (null space)
        return np.where(np.abs(op_diag) <= eps, bc_lm, res)

    vlm_sigma = safe_div(vlm_bc, op_diag_V)
    wlm_sigma = safe_div(wlm_bc, op_diag_W)
    xlm_sigma = safe_div(xlm_bc, op_diag_X)

    # 5. Transform back to Cartesian physical space
    sig_x, sig_y, sig_z = sig_vwx2xyz(
        vlm_sigma, wlm_sigma, xlm_sigma, 
        theta, phi, sh
    )
    
    return np.stack([sig_x, sig_y, sig_z], axis=-1)

def bio_offsurf_apply(Rtrg_lst: np.ndarray, S: SphereDict, sh: shtns.sht, sl_scal: float, dl_scal: float) -> np.ndarray:
    SLsigma = Stk3d_sl_r(Rtrg_lst, S, sh) 
    DLsigma = Stk3d_dl_r(Rtrg_lst, S, sh) 
    Ksigma = sl_scal * SLsigma + dl_scal * DLsigma
    return Ksigma

# Need to accomodate different ordered spheres for target. leave the looping over differetn targets to outside, plug in one target sphere and one source.
def bio_offsurf_apply_1sph(Strg: SphereDict, shtrg: shtns.sht, S: SphereDict, sh: shtns.sht, sl_scal: float, dl_scal: float) -> np.ndarray:
    SLsigma = Stk3d_sl_r_1sph(Strg, shtrg, S, sh) 
    DLsigma = Stk3d_dl_r_1sph(Strg, shtrg, S, sh) 
    Ksigma = sl_scal * SLsigma + dl_scal * DLsigma
    return Ksigma

def compute_field(trg: np.ndarray, src: np.ndarray, force: np.ndarray) -> np.ndarray:
    assert trg.shape[1] == 3 and src.shape[1] == 3 and force.shape[1] == 3
    assert force.shape[0] == src.shape[0]

    srcx = src[:,0][None,:] # reshape source x,y,z into 1 x Nsrc 
    srcy = src[:,1][None,:]
    srcz = src[:,2][None,:]
    dx = trg[:,0][:,None] - srcx # size = Ntrg x Nsrc
    dy = trg[:,1][:,None] - srcy
    dz = trg[:,2][:,None] - srcz
    dr = np.sqrt(dx*dx + dy*dy + dz*dz)
    
    # Compute Stokeslet field
    r_vec = np.stack([dx, dy, dz], axis=-1)  # Ntrg x Nsrc x 3
    r_norm = dr[..., None]  # Ntrg x Nsrc x 1
    r_hat = r_vec / r_norm
    force_expanded = force[None, :, :]  # Ntrg x Nsrc x 3
    dot_prod = np.sum(force_expanded * r_vec, axis=-1, keepdims=True)  # Ntrg x Nsrc x 1
    u_contrib = (1/(8*np.pi)) * (force_expanded / r_norm + dot_prod * r_hat / (r_norm**2))
    u = np.sum(u_contrib, axis=1)  # Ntrg x 3
    
    return u.astype(np.complex128)
    
if __name__ == "__main__":
    # Make a sphere
    lmax = 36
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
    Strg, shtrg = quadr_sphere(Strg, lmax)
    # BIOp parameter
    sl_scal = 1.0
    dl_scal = 1.0

    # # Check sphere coordinates
    # x = S["Xcart"][:,:,0]
    # y = S["Xcart"][:,:,1]
    # z = S["Xcart"][:,:,2]
    theta = S["Xsph"][:,:,0]
    phi = S["Xsph"][:,:,1]
    # trg_sphere = jnp.column_stack([jnp.reshape(x,-1), jnp.reshape(y,-1), jnp.reshape(z,-1)])
    # print(f"surface nodes of S: {trg_sphere}")
    # Print theta and phi values to check
    # theta = S["Xsph"][:,:,0]
    # phi = S["Xsph"][:,:,1]
    # print(f"values of theta: {theta[0,:]}")
    # print(f"values of phi: {phi[:,0]}")

    # Define surface density as Y10 e_r or something else simple # CHECKED: SL, DL spectra of most bases up to l=2 checked
    # xlm = np.zeros((sh.nlm_cplx,), dtype=np.complex128)
    # vlm = np.zeros_like(xlm)
    # wlm = np.zeros_like(xlm)
    # xlm[1] = 1.0 
    # vlm[4] = 1.0 
    # wlm[4] = 1.0

    # Sigma_x, Sigma_y, Sigma_z = sig_vwx2xyz(vlm, wlm, xlm, theta, phi, sh)

    # qlm, slm, tlm = vwx2qst(vlm, wlm, xlm, sh)
    # TODO: changed two lines in shtns library
    # Sigma_r, Sigma_t, Sigma_p = sh.synth_cplx(qlm, slm, tlm) 
    # theta = S["Xsph"][:,:,0]
    # phi = S["Xsph"][:,:,1]
    # Sigma_x, Sigma_y, Sigma_z = sph2cart(Sigma_r, Sigma_t, Sigma_p, theta, phi)
    # X1m1exact_x, X1m1exact_y, X1m1exact_z = trueX1m1(theta,phi) # Also checked Y1_, G1_, all matches formulas.
    # print(f"Y11 formula - synth: x={np.linalg.norm(X1m1exact_x-Sigma_x)}, y={np.linalg.norm(X1m1exact_y-Sigma_y)}, z={np.linalg.norm(X1m1exact_z-Sigma_z)}")

    # qlm, slm, tlm = sh.analys_cplx(Sigma_r, Sigma_t, Sigma_p)
    # vlm, wlm, xlm = qst2vwx(qlm, slm, tlm, sh)
    # print(f"VWX coeff after Vana: |v| (=0): {np.linalg.norm(vlm)}, |w| (=0): {np.linalg.norm(wlm)}, x (x[1]=1): {xlm}") # Checked; gets back X10
    
    # # Evaluate at interior/exterior surfaces using VSH basis functions, and compare to analytical formulas with spectra multiplied.
    # S = set_density(S, Sigma_x, Sigma_y, Sigma_z)
    # xtrg = Strg["Xcart"][:,:,0]
    # ytrg = Strg["Xcart"][:,:,1]
    # ztrg = Strg["Xcart"][:,:,2]
    # trg_sphere2 = np.column_stack([np.reshape(xtrg,-1), np.reshape(ytrg,-1), np.reshape(ztrg,-1)])
    # SLsigma = Stk3d_sl_r(np.array([Rtrg]), S, sh) # Ntrg x 3
    # # expect: 1/(3*1)V11*r^2 + 2/6 W11*(-r^0 + r^2)
    # vlmexp = np.zeros_like(xlm)
    # wlmexp = np.zeros_like(xlm)
    # xlmexp = np.zeros_like(xlm)
    # vlmexp[4] = 2./10. * (Rtrg**(-4) - Rtrg**(-2))
    # wlmexp[4] = 3./5./3.*Rtrg**(-2.)
    # SLexp_x,SLexp_y,SLexp_z = sig_vwx2xyz(vlmexp, wlmexp, xlmexp, theta, phi, sh)
    # print(f"SL[W2-2] err at Rtrg = {Rtrg} is: {np.linalg.norm(SLexp_x-SLsigma[:,:,0,0])}, {np.linalg.norm(SLexp_y-SLsigma[:,:,1,0])}, {np.linalg.norm(SLexp_z-SLsigma[:,:,2,0])}")
    # DLsigma = Stk3d_dl_r(np.array([Rtrg]), S, sh) # Ntrg x 3
    # # expect: -2(3)/(3*5)V11*r^2 + 2*1/6 W11*(-r^2 + r^0)
    # vlmexp = np.zeros_like(xlm)
    # wlmexp = np.zeros_like(xlm)
    # xlmexp = np.zeros_like(xlm)
    # vlmexp[4] = -2.*2./(10.) * (Rtrg**(-4) - Rtrg**(-2))
    # wlmexp[4] = 2.*3.*1./5./3.* Rtrg**(-2.)
    # DLexp_x,DLexp_y,DLexp_z = sig_vwx2xyz(vlmexp, wlmexp, xlmexp, theta, phi, sh)
    # print(f"DL[W2-2] err at Rtrg = {Rtrg} is: {np.linalg.norm(DLexp_x-DLsigma[:,:,0,0])}, {np.linalg.norm(DLexp_y-DLsigma[:,:,1,0])}, {np.linalg.norm(DLexp_z-DLsigma[:,:,2,0])}")

    # TEST 2: Manufactored solutions with 1 stokeslet inside.
    if ext:
        ptsrc = np.array([[0.1,0.3,0.15],[-0.35,0.2,0.]]) # shifted source to avoid constant potential on all of S
        # force = np.array([[1,1,1],[-1,-1,-1]])
        force = np.array([[1,1,1],[-1,0,0]])
    else:
        ptsrc = np.array([[1.3,1.75,-2],[-1.3,-1.75,2]])
        force = np.array([[1,1,1],[-1,-1,-1]]) # Need to have net force zero over the sphere

    x = S["Xcart"][:,:,0]
    y = S["Xcart"][:,:,1]
    z = S["Xcart"][:,:,2]
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
    x, info = scipy.sparse.linalg.gmres(gmres_func, BC_pot.flatten(), x0=np.zeros(total_dofs, dtype=np.complex128), \
                                        atol = 1e-14, rtol = 1e-13, maxiter=200)
    sig_fromBC = x.reshape(theta.shape[0], theta.shape[1], 3)
    # Manually check residual
    bc_check = bio_onsurf_apply(x, theta, phi, sh, sl_scal, dl_scal, sgn)
    resid_gmres = np.linalg.norm(bc_check - BC_pot.flatten())
    print("Checking residual of solve: {a}, exitcode (0:successful): {b}".format(a=resid_gmres, b=info))

    sigma_solution = stokes_onsurf_diag_solve(BC_pot, theta, phi, sh, sl_scal, dl_scal, sgn)
    bc_check_diag = bio_onsurf_apply(sigma_solution.flatten(), theta, phi, sh, sl_scal, dl_scal, sgn)
    resid_diag = np.linalg.norm(bc_check_diag - BC_pot.flatten())
    print("Checking residual of direct solve: {a}".format(a=resid_diag))

    # Compare with true solution at target sphere
    xtrg = Strg["Xcart"][:,:,0] 
    ytrg = Strg["Xcart"][:,:,1]
    ztrg = Strg["Xcart"][:,:,2]
    trg_sphere2 = np.column_stack([np.reshape(xtrg,-1), np.reshape(ytrg,-1), np.reshape(ztrg,-1)])
    # S = set_density(S, sig_fromBC[:,:,0], sig_fromBC[:,:,1], sig_fromBC[:,:,2])
    S = set_density(S, sigma_solution[:,:,0], sigma_solution[:,:,1], sigma_solution[:,:,2])
    time_eval_start = time.time()
    Ksigma = bio_offsurf_apply_1sph(Strg, shtrg, S, sh, sl_scal, dl_scal) # trg_sphere2 reshaped to be Ntrg x 3, so output is Ntrg x 1.
    time_eval_end = time.time()
    print(f"Timing results off-surface eval: {time_eval_end - time_eval_start}")

    true_field = compute_field(trg_sphere2, ptsrc, force) # true_pot also computed as Ntrg x 1
    true_field = np.reshape(true_field, S["Xcart"].shape)
    # For scalar electric potential calculation, only real values
    Ksigma = np.real(Ksigma)
    true_field = np.real(true_field)
    diff = np.max(true_field - Ksigma) / np.max(true_field)
    print("At target sphere Rtrg = {a}, max true field is {d}, relative error from true field using lmax = {b} is {c}".format(a=Rtrg, d=np.max(true_field), b=lmax, c=diff))
