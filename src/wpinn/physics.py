"""Residuals of the column model (all dimensionless, t~ = t / T, h~ = h / L).

Richards in the coordinate xi = z' / H(t) (z' = z + d_top, H = zb - d_top), as in ../pinn/zb_inverse physics.py with
the column height H in place of zb (the top of the column, the top sensor, is fixed in time):
    R = dM/dt~|xi - (xi * Hdot / H) dM/dxi + (1 / H) dq~/dxi + S L / Ks
    M = Se(h) + ss~ h,   q~ = -K(h) ((1 / H) dh/dxi + 1),   H and Hdot = dH~/dt~ in tilde units
Root uptake S(depth, t, h) [1/s] at the absolute depth d_top - xi * H.
Water balance of a block [xi_a, xi_b] x [t0, t0 + dur]: the integral of z R over the block is the change of the water
in the layer minus what crosses its faces, so its mean over stratified points is the defect of the block's balance;
returned as a mean water content of the layer (theta* times the defect / layer thickness).
"""
import torch


def _grad(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]


def pde_residual(model, xi, t_days):
    soil = model.soil
    Td = model.T_days
    xi = xi.detach().requires_grad_(True)
    t = t_days.detach().requires_grad_(True)
    H = model.H_tilde(t)
    Hdot = Td * _grad(H, t)
    h = model.h_tilde(xi, t)
    depth = model.d_top - xi * H * soil.L                       # true depth (m), differentiable in xi and t
    layered = getattr(soil, 'layered', False)                  # layered soil: parameters at that depth
    M = soil.storage(h, depth) if layered else soil.storage(h)
    dM_dt = Td * _grad(M, t)
    dM_dxi = _grad(M, xi)
    dh_dxi = _grad(h, xi)
    q = -(soil.K(h, depth) if layered else soil.K(h)) * (dh_dxi / H + 1.0)
    dq_dxi = _grad(q, xi)
    R = dM_dt - (xi * Hdot / H) * dM_dxi + dq_dxi / H
    if model.sink is not None:
        R = R + model.sink(depth, t + model.t_off, h * soil.L) * (soil.L / soil.Ks)
    return R


def block_defect(model, xi, t_days, dur_days, dxi):
    """xi (n, nt, nx) and t_days (n, nt) stratified points of n blocks; dur_days, dxi (n, 1) their size in t and xi.
    Returns (n, 1): the defect of each block's water balance as a mean water content of the layer."""
    soil = model.soil
    n, nt, nx = xi.shape
    tt = t_days.unsqueeze(2).expand(-1, -1, nx).reshape(-1, 1)
    R = pde_residual(model, xi.reshape(-1, 1), tt).reshape(n, nt, nx)
    H = model.H_tilde(t_days.reshape(-1, 1)).reshape(n, nt)
    B = (H * dxi * R.mean(2)).mean(1, keepdim=True) * (dur_days / model.T_days)       # integral of z R, tilde units
    return soil.theta_star * B / (H.mean(1, keepdim=True) * dxi)
