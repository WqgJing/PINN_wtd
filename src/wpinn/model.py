"""PINN for the soil column between the top sensor and the water table; the water-table path is trained with it.

Coordinates: z' = z + d_top (z up, surface at z = 0), so the column is -H(t) <= z' <= 0 with H(t) = zb(t) - d_top,
and xi = z' / H in [-1, 0]: xi = 0 at the top sensor, xi = -1 at the water table.

    h~(xi, t) = (1 + xi) * [ -H(t)/L + N(xi, t, features) ]      (h~ = h / L)

- h = 0 at the water table for any weights and any path (factor 1 + xi).
- N = 0 is the hydrostatic (no-flow) profile of the current water table; N is the departure from it.
- No initial condition and no top flux are imposed: the sensors (the top one included) are data.
zb(t) = zb_min + softplus(B(t) . c) with trainable coefficients c (daily knots).
Inputs of N: xi, t, Fourier features of time, causal features of the path (CausalPathFeatures), and exponential
memories of known series (the top sensor's record and the transpiration), all built over the record before the window
so that the memories are not reset at the window start.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .spline import CausalPathFeatures


class MLP(nn.Module):
    def __init__(self, n_in, width, depth):
        super().__init__()
        layers, d = [], n_in
        for _ in range(depth):
            lin = nn.Linear(d, width)
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
            layers += [lin, nn.Tanh()]
            d = width
        last = nn.Linear(d, 1)
        nn.init.zeros_(last.weight)          # N = 0 at the start: training starts from the hydrostatic profile
        nn.init.zeros_(last.bias)
        layers.append(last)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ProfilePINN(nn.Module):
    def __init__(self, soil, spline, coef_init, zb_min, d_top, t_window_days, t_offset_days, sink=None, memories=None,
                 width=128, depth=6, fourier_periods_days=(1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
                 path_taus_days=(0.25, 1.0, 4.0, 16.0)):
        super().__init__()
        self.soil, self.spline = soil, spline
        self.zb_min, self.d_top = float(zb_min), float(d_top)
        self.Tw, self.t_off = float(t_window_days), float(t_offset_days)
        self.T_days = soil.T / 86400.0
        self.coef = nn.Parameter(torch.as_tensor(np.asarray(coef_init), dtype=torch.float64))
        self.path = CausalPathFeatures(spline, taus_days=path_taus_days)
        self.sink = sink                                  # RootSink on record time (t + t_offset), or None
        self.memories = memories                          # ForcingMemoryFeatures on record time, or None
        self.register_buffer('omegas', torch.tensor([2 * math.pi / p for p in fourier_periods_days],
                                                    dtype=torch.float64))
        n_in = 2 + 2 * len(fourier_periods_days) + self.path.n_out + (memories.n_out if memories is not None else 0)
        self.N = MLP(n_in, width, depth)

    # ---- water table ------------------------------------------------------------------------------------------
    def zb(self, t):
        """Water-table depth [m] at window time t (N, 1) [d]."""
        return self.zb_min + F.softplus(self.spline.eval_torch(t, self.coef))

    def H_tilde(self, t):
        """Column height (water table minus top sensor depth) / L."""
        return (self.zb(t) - self.d_top) / self.soil.L

    # ---- head -------------------------------------------------------------------------------------------------
    def features(self, xi, t):
        f = [2.0 * xi + 1.0, 2.0 * t / self.Tw - 1.0]
        ta = t + self.t_off                                  # record time: the diurnal phase of the forcing
        wt = ta * self.omegas[None, :]
        f += [torch.sin(wt), torch.cos(wt)]
        f.append(self.path(t, self.coef))
        if self.memories is not None:
            f.append(self.memories(ta))
        return torch.cat(f, dim=1)

    def h_tilde(self, xi, t):
        return (1.0 + xi) * (-self.H_tilde(t) + self.N(self.features(xi, t)))

    def xi_of_depth(self, depth_m, t):
        """xi of a point depth_m [m below the surface] at window time t (N, 1)."""
        return -(depth_m - self.d_top) / (self.zb(t) - self.d_top)

    def theta_at(self, depth_m, t):
        """Soil moisture at depth_m [m] (scalar or (N, 1)) and window time t (N, 1)."""
        return self.soil.theta(self.h_tilde(self.xi_of_depth(depth_m, t), t))
