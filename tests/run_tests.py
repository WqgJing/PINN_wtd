#!/usr/bin/env python
"""Unit tests of the model and the residuals (plain asserts; run through hpc/run_script_cpu.slurm)."""
import os
import sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from wpinn.model import ProfilePINN                    # noqa: E402
from wpinn.physics import pde_residual, block_defect   # noqa: E402
from wpinn.soil import SoilVG                          # noqa: E402
from wpinn.spline import CubicBSpline1D, ZbSplineNumpy  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
SOIL = SoilVG(theta_s=0.4193514611932402, theta_r=5.66e-09, alpha=0.822311364445854, n=2.7469217167092075,
              Ks=1.922838127196462e-06, L=2.0)
SP = CubicBSpline1D(0.0, 20.0, 1.0)
D_TOP = 0.05


def make(path_m, randomise=False):
    """path_m: water-table depth [m] at the knots' fit points (constant or array over 0..20 d)."""
    t = np.linspace(0, 20, 481)
    zb = np.full_like(t, path_m) if np.ndim(path_m) == 0 else np.interp(t, np.linspace(0, 20, len(path_m)), path_m)
    coef = ZbSplineNumpy.fit(SP, t, zb, 0.05).coef
    m = ProfilePINN(SOIL, SP, coef, 0.05, D_TOP, 20.0, 35.0, width=16, depth=2)
    if randomise:
        torch.nn.init.normal_(m.N.net[-1].weight, std=0.05)
        torch.nn.init.normal_(m.N.net[-1].bias, std=0.05)
    return m


def test_water_table_condition():
    m = make([0.9, 1.1, 0.8, 1.2], randomise=True)
    t = 20 * torch.rand(200, 1)
    h = m.h_tilde(-torch.ones_like(t), t)
    assert float(h.abs().max()) == 0.0, 'h must vanish at xi = -1'


def test_hydrostatic_sensor():
    m = make([0.9, 1.1, 0.8, 1.2])                       # N = 0 at initialisation
    t = 20 * torch.rand(100, 1)
    for d in (0.05, 0.2, 0.4, 0.6):
        th = m.theta_at(d, t)
        ref = SOIL.theta(-(m.zb(t) - d) / SOIL.L)
        assert float((th - ref).abs().max()) < 1e-12, f'hydrostatic theta at {d} m'


def test_rest_has_no_residual():
    m = make(1.0)                                        # constant water table, N = 0, no sink: hydrostatic rest
    xi, t = -torch.rand(500, 1), 20 * torch.rand(500, 1)
    R = pde_residual(m, xi, t)
    assert float(R.abs().max()) < 1e-10, f'rest residual {float(R.abs().max())}'
    n, nt, nx = 16, 8, 4
    bx = (-torch.rand(n, nt, nx), 18 * torch.rand(n, nt), torch.full((n, 1), 1.0), torch.full((n, 1), 0.3))
    assert float(block_defect(m, *bx).abs().max()) < 1e-10, 'rest block defect'


def test_residual_finite_differences():
    """Autograd residual against central differences of the same network (no sink)."""
    m = make([0.9, 1.1, 0.8, 1.2], randomise=True)
    xi = -0.1 - 0.8 * torch.rand(20, 1)
    t = 1.0 + 18 * torch.rand(20, 1)
    R = pde_residual(m, xi, t).detach()
    e_t, e_x = 1e-4, 1e-4
    Td, s = m.T_days, SOIL
    with torch.no_grad():
        Mf = lambda x, tt: s.storage(m.h_tilde(x, tt))
        H = lambda tt: m.H_tilde(tt)
        dM_dt = Td * (Mf(xi, t + e_t) - Mf(xi, t - e_t)) / (2 * e_t)
        Hdot = Td * (H(t + e_t) - H(t - e_t)) / (2 * e_t)
        dM_dxi = (Mf(xi + e_x, t) - Mf(xi - e_x, t)) / (2 * e_x)

        def q(x):
            dh = (m.h_tilde(x + e_x, t) - m.h_tilde(x - e_x, t)) / (2 * e_x)
            return -s.K(m.h_tilde(x, t)) * (dh / H(t) + 1.0)
        dq_dxi = (q(xi + e_x) - q(xi - e_x)) / (2 * e_x)
        R_fd = dM_dt - (xi * Hdot / H(t)) * dM_dxi + dq_dxi / H(t)
    err = float((R - R_fd).abs().max())
    scale = float(R.abs().max())
    assert err < 1e-4 * max(scale, 1e-3), f'finite differences: max |dR| {err:.3e}, max |R| {scale:.3e}'


def test_path_gradient():
    m = make([0.9, 1.1, 0.8, 1.2], randomise=True)
    t = 20 * torch.rand(50, 1)
    loss = (m.theta_at(0.6, t) ** 2).sum()
    loss.backward()
    assert float(m.coef.grad.abs().sum()) > 0.0, 'the data must reach the path coefficients'


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for f in tests:
        try:
            f()
            print(f'PASS {f.__name__}', flush=True)
        except AssertionError as e:
            failed += 1
            print(f'FAIL {f.__name__}: {e}', flush=True)
    print(f'{len(tests) - failed} passed, {failed} failed', flush=True)
    sys.exit(1 if failed else 0)
