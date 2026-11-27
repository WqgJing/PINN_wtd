"""Layered van Genuchten-Mualem soil for the PINN: parameters that change with depth below the surface.

Layers are given top-down with the depth of their top (the first at 0 m). Between layer k and k+1 every parameter
blends smoothly with w(z) = 0.5 (1 + tanh((z - top_{k+1}) / width)) - the same form as the FEM's layered soil
(fem_hetero.profile_factors), so a synthetic layered record can be matched exactly: theta_s, theta_r, alpha and n blend
linearly, Ks log-linearly.

All functions take h~ (h / L) and depth (m, positive down; scalar or tensor broadcastable with h). Scaling constants
(L, T, theta_star, Ks) are those of the TOP layer (the reference), so the PDE residual keeps its form:
    storage M = (theta(h, z) - theta_r,ref) / theta*_ref + ss~ h~,   K~ = K(h, z) / Ks_ref.
With a single layer every function equals SoilVG's (no smoothing differences: the same blend and |h| smoothing).
"""
import math

import numpy as np
import torch

from .soil import SoilVG


def _is_torch(x):
    return isinstance(x, torch.Tensor)


class LayeredSoilVG(SoilVG):
    layered = True

    def __init__(self, layers, width_m=0.02, L=2.0, l=0.5, Ss=1.0e-5, blend_width_m=0.02, abs_smooth_m=1.0e-3):
        """layers: list of dicts with top_m, theta_s, theta_r, alpha (1/m), n, Ks (m/s), top-down, first top_m = 0."""
        if not layers or float(layers[0]['top_m']) != 0.0:
            raise ValueError('the first layer must start at 0 m')
        tops = [float(x['top_m']) for x in layers]
        if any(b <= a for a, b in zip(tops, tops[1:])):
            raise ValueError('layer tops must increase')
        ref = layers[0]
        super().__init__(theta_s=ref['theta_s'], theta_r=ref['theta_r'], alpha=ref['alpha'], n=ref['n'], Ks=ref['Ks'],
                         L=L, l=l, Ss=Ss, blend_width_m=blend_width_m, abs_smooth_m=abs_smooth_m)
        self.tops = tops
        self.width = float(width_m)
        self.layer_params = [{k: float(x[k]) for k in ('theta_s', 'theta_r', 'alpha', 'n', 'Ks')} for x in layers]

    # ---- parameters at depth -------------------------------------------------------------------------------------
    def params(self, depth):
        """theta_s, theta_r, alpha~, n, m, Ks / Ks_ref at depth (same type and shape as depth)."""
        tt = _is_torch(depth)
        tanh = torch.tanh if tt else np.tanh
        p = self.layer_params
        ts, tr, a, n, lk = (p[0]['theta_s'], p[0]['theta_r'], p[0]['alpha'], p[0]['n'], math.log(p[0]['Ks']))
        zero = depth * 0.0
        ts, tr, a, n, lk = zero + ts, zero + tr, zero + a, zero + n, zero + lk
        for k in range(1, len(p)):
            w = 0.5 * (1.0 + tanh((depth - self.tops[k]) / self.width))
            ts = ts + w * (p[k]['theta_s'] - p[k - 1]['theta_s'])
            tr = tr + w * (p[k]['theta_r'] - p[k - 1]['theta_r'])
            a = a + w * (p[k]['alpha'] - p[k - 1]['alpha'])
            n = n + w * (p[k]['n'] - p[k - 1]['n'])
            lk = lk + w * (math.log(p[k]['Ks']) - math.log(p[k - 1]['Ks']))
        exp = torch.exp if tt else np.exp
        return ts, tr, a * self.L, n, 1.0 - 1.0 / n, exp(lk) / self.Ks

    # ---- VG-Mualem at depth (the base-class formulas with local parameters) --------------------------------------
    def _se_vg_local(self, h, a_t, n, m):
        return (1.0 + (a_t * self._sabs(h)) ** n) ** (-m)

    def _kr_local(self, se, m):
        x = 1.0 - se ** (1.0 / m)
        x = torch.clamp(x, min=1e-30) if _is_torch(x) else np.maximum(x, 1e-30)
        return se ** self.l * (1.0 - x ** m) ** 2

    def Se(self, h, depth=None):
        if depth is None:
            return super().Se(h)
        _, _, a_t, n, m, _ = self.params(depth)
        b = self._blend(h)
        return b + (1.0 - b) * self._se_vg_local(h, a_t, n, m)

    def theta(self, h, depth=None):
        if depth is None:
            return super().theta(h)
        ts, tr, _, _, _, _ = self.params(depth)
        return tr + (ts - tr) * self.Se(h, depth)

    def K(self, h, depth=None):
        if depth is None:
            return super().K(h)
        _, _, a_t, n, m, kf = self.params(depth)
        b = self._blend(h)
        return kf * (b + (1.0 - b) * self._kr_local(self._se_vg_local(h, a_t, n, m), m))

    def storage(self, h, depth=None):
        if depth is None:
            return super().storage(h)
        return (self.theta(h, depth) - self.theta_r) / self.theta_star + self.ss_t * h

    def h_from_theta(self, theta, depth=None):
        """VG inverse in metres at depth (scalar), ignoring the smoothing; for initial guesses only."""
        if depth is None:
            return super().h_from_theta(theta)
        ts, tr, a_t, n, m, _ = (float(v) for v in self.params(np.float64(depth)))
        se = np.clip((np.asarray(theta, dtype=float) - tr) / (ts - tr), 1e-9, 1 - 1e-12)
        return -((se ** (-1.0 / m) - 1.0) ** (1.0 / n)) / (a_t / self.L)


def layers_from_config(base, spec):
    """Layer list from a config spec {'width_m': w, 'layers': [{'top_m': 0.0, ...}, {'top_m': 0.3, 'alpha_factor': 0.5,
    'n': 2.0, 'Ks_factor': 0.3}, ...]}: each layer gives values directly (theta_s, theta_r, alpha, n, Ks) or as
    <name>_factor times the base soil's value; anything missing is the base soil's."""
    out = []
    for lay in spec['layers']:
        d = {'top_m': float(lay['top_m'])}
        for k in ('theta_s', 'theta_r', 'alpha', 'n', 'Ks'):
            if k in lay:
                d[k] = float(lay[k])
            else:
                d[k] = float(base[k]) * float(lay.get(f'{k}_factor', 1.0))
        out.append(d)
    return out
