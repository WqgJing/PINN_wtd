"""Variational posterior over the water-table spline coefficients and the saturated conductivity, for the PINN.

u = (c_1..c_m, z) with c the water-table spline coefficients (zb = zb_min + softplus(B . c)) and z = log(Ks / Ks_ref).
- GaussianPosterior: q(u) = N(mu, L L^T), full Cholesky L (diagonal kept positive through exp); reparameterised
  samples u = mu + L eps and their exact log q.
- WaterTableKsPrior: log p(u) = level prior on mean(c) + random-walk smoothness on the daily differences of c
  + normal prior on z. Its covariance (for contraction) is available in closed form.
- VarKsSoil: SoilVG whose conductivity is multiplied by k_factor = Ks / Ks_ref (time scaling and the sink's scale keep
  Ks_ref, so only K(h) in the flux changes).
- VIProfilePINN: ProfilePINN evaluated for ONE sample at a time: the sample's coefficients replace the trained path,
  and z is an extra network input, so the network represents the best profile for any (path, Ks) the posterior
  proposes. set_sample(c, z) before calling the losses.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import ProfilePINN, MLP
from .soil import SoilVG


class VarKsSoil(SoilVG):
    """SoilVG with K(h) multiplied by self.k_factor (a scalar tensor, default 1)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k_factor = 1.0

    def K(self, h):
        return super().K(h) * self.k_factor


class GaussianPosterior(nn.Module):
    def __init__(self, mean_init, sd_init):
        super().__init__()
        mean_init = torch.as_tensor(np.asarray(mean_init, dtype=float), dtype=torch.float64)
        sd_init = torch.as_tensor(np.asarray(sd_init, dtype=float), dtype=torch.float64)
        self.d = mean_init.numel()
        self.mu = nn.Parameter(mean_init.clone())
        self.log_diag = nn.Parameter(torch.log(sd_init.clone()))
        self.off = nn.Parameter(torch.zeros(self.d, self.d, dtype=torch.float64))
        self.register_buffer('tril_mask', torch.tril(torch.ones(self.d, self.d, dtype=torch.float64), diagonal=-1))

    def chol(self):
        return torch.diag(torch.exp(self.log_diag)) + self.off * self.tril_mask

    def sample(self, k, generator=None):
        """k reparameterised samples (k, d) and their log q (k,)."""
        eps = torch.randn(k, self.d, dtype=self.mu.dtype, device=self.mu.device, generator=generator)
        u = self.mu[None, :] + eps @ self.chol().T
        logq = (-0.5 * (eps ** 2).sum(1) - self.log_diag.sum() - 0.5 * self.d * math.log(2.0 * math.pi))
        return u, logq

    def covariance(self):
        L = self.chol()
        return L @ L.T


class WaterTableKsPrior(nn.Module):
    """Gaussian prior on u = (c_1..c_m, z): mean(c) ~ N(level_mean, level_sd^2), c_{k+1} - c_k ~ N(0, step_sd^2),
    z ~ N(0, z_sd^2) independent of c. Precision P assembled once; log p up to a constant."""

    def __init__(self, m, level_mean, level_sd, step_sd, z_sd):
        super().__init__()
        A = np.ones((1, m)) / m                                    # level
        D = np.diff(np.eye(m), axis=0)                             # (m-1, m) daily differences
        Pc = A.T @ A / level_sd ** 2 + D.T @ D / step_sd ** 2
        P = np.zeros((m + 1, m + 1))
        P[:m, :m] = Pc
        P[m, m] = 1.0 / z_sd ** 2
        mean = np.concatenate([np.full(m, level_mean), [0.0]])     # mean(c) = level_mean, differences 0
        self.register_buffer('P', torch.tensor(P, dtype=torch.float64))
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float64))
        self.m = m

    def log_prob(self, u):
        d = u - self.mean[None, :]
        return -0.5 * torch.einsum('ki,ij,kj->k', d, self.P, d)

    def covariance(self):
        return torch.linalg.inv(self.P)


class VIProfilePINN(ProfilePINN):
    """ProfilePINN for one posterior sample at a time: set_sample(c, z) sets the water-table coefficients (m,) and
    z = log(Ks / Ks_ref) (scalar); zb(t), the path features and the conductivity then follow that sample. The network
    takes z as one extra input. The base class's trainable coef is kept only as the start (frozen)."""

    def __init__(self, soil, *args, **kwargs):
        if not isinstance(soil, VarKsSoil):
            raise TypeError('VIProfilePINN needs a VarKsSoil')
        super().__init__(soil, *args, **kwargs)
        self.coef.requires_grad_(False)
        n_in = self.N.net[0].in_features + 1
        width, depth = self.N.net[0].out_features, sum(isinstance(l, nn.Linear) for l in self.N.net) - 1
        self.N = MLP(n_in, width, depth)
        self.cur_coef = self.coef.detach().clone()                # plain tensors (not parameters); set_sample
        self.cur_z = torch.zeros((), dtype=torch.float64)          # replaces them before every evaluation

    def set_sample(self, coef, z):
        self.cur_coef = coef
        self.cur_z = z
        self.soil.k_factor = torch.exp(z)

    def zb(self, t):
        return self.zb_min + F.softplus(self.spline.eval_torch(t, self.cur_coef))

    def features(self, xi, t):
        f = [2.0 * xi + 1.0, 2.0 * t / self.Tw - 1.0]
        ta = t + self.t_off
        wt = ta * self.omegas[None, :]
        f += [torch.sin(wt), torch.cos(wt)]
        f.append(self.path(t, self.cur_coef))
        if self.memories is not None:
            f.append(self.memories(ta))
        f.append(self.cur_z * torch.ones_like(t))
        return torch.cat(f, dim=1)
