#!/usr/bin/env python
"""Unit tests of the model and the residuals (plain asserts; run through hpc/run_script_cpu.slurm)."""
import math
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
from wpinn.vi import VarKsSoil, VIProfilePINN, GaussianPosterior, WaterTableKsPrior   # noqa: E402
from wpinn.soil_layered import LayeredSoilVG   # noqa: E402

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
SOIL = SoilVG(theta_s=0.4193514611932402, theta_r=5.66e-09, alpha=0.822311364445854, n=2.7469217167092075,
              Ks=1.922838127196462e-06, L=2.0)
SP = CubicBSpline1D(0.0, 20.0, 1.0)
D_TOP = 0.05


def make(path_m, randomise=False, head_form='shift'):
    """path_m: water-table depth [m] at the knots' fit points (constant or array over 0..20 d)."""
    t = np.linspace(0, 20, 481)
    if np.ndim(path_m) == 0:      # exactly constant (a least-squares fit is constant only to ~1e-8)
        coef = np.full(SP.n_coef, np.log(np.expm1(float(path_m) - 0.05)))
    else:
        coef = ZbSplineNumpy.fit(SP, t, np.interp(t, np.linspace(0, 20, len(path_m)), path_m), 0.05).coef
    m = ProfilePINN(SOIL, SP, coef, 0.05, D_TOP, 20.0, 35.0, width=16, depth=2, head_form=head_form)
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


def test_scale_form_matches_at_rest():
    """head_form scale: h = 0 at the water table, hydrostatic at N = 0, no residual at rest."""
    m = make([0.9, 1.1, 0.8, 1.2], randomise=True, head_form='scale')
    t = 20 * torch.rand(200, 1)
    assert float(m.h_tilde(-torch.ones_like(t), t).abs().max()) == 0.0, 'scale: h must vanish at xi = -1'
    m = make([0.9, 1.1, 0.8, 1.2], head_form='scale')
    for d in (0.05, 0.2, 0.4, 0.6):
        ref = SOIL.theta(-(m.zb(t) - d) / SOIL.L)
        assert float((m.theta_at(d, t) - ref).abs().max()) < 1e-12, f'scale: hydrostatic theta at {d} m'
    m = make(1.0, head_form='scale')
    xi, tt = -torch.rand(500, 1), 20 * torch.rand(500, 1)
    assert float(pde_residual(m, xi, tt).abs().max()) < 1e-10, 'scale: rest residual'


def test_scale_form_stays_unsaturated():
    """Large network outputs: the shift form can make h > 0 above the water table (saturated, theta flat, no gradient);
    the scale form keeps h < 0 there and the sensor still responds to the water table."""
    for form, expect_pos in (('shift', True), ('scale', False)):
        m = make(3.0, head_form=form)
        torch.nn.init.constant_(m.N.net[-1].bias, 3.0)   # far from the hydrostatic profile, towards wet
        xi, t = -0.05 - 0.9 * torch.rand(300, 1), 20 * torch.rand(300, 1)
        pos = bool((m.h_tilde(xi, t) > 0).any())
        assert pos == expect_pos, f'{form}: h > 0 above the water table is {pos}'
    m = make(3.0, head_form='scale')
    torch.nn.init.constant_(m.N.net[-1].bias, -3.0)      # very wet but unsaturated
    t = 20 * torch.rand(50, 1)
    m.theta_at(0.6, t).sum().backward()
    assert float(m.coef.grad.abs().sum()) > 0.0, 'scale: the sensor must still reach the path coefficients'


VSOIL = VarKsSoil(theta_s=0.4193514611932402, theta_r=5.66e-09, alpha=0.822311364445854, n=2.7469217167092075,
                  Ks=1.922838127196462e-06, L=2.0)


def make_vi(path_m, randomise=False):
    t = np.linspace(0, 20, 481)
    if np.ndim(path_m) == 0:
        coef = np.full(SP.n_coef, np.log(np.expm1(float(path_m) - 0.05)))
    else:
        coef = ZbSplineNumpy.fit(SP, t, np.interp(t, np.linspace(0, 20, len(path_m)), path_m), 0.05).coef
    m = VIProfilePINN(VSOIL, SP, coef, 0.05, D_TOP, 20.0, 35.0, width=16, depth=2, head_form='scale')
    if randomise:
        torch.nn.init.normal_(m.N.net[-1].weight, std=0.05)
        torch.nn.init.normal_(m.N.net[-1].bias, std=0.05)
    return m, torch.tensor(coef)


def test_vi_gaussian_logq():
    q = GaussianPosterior(np.arange(5.0), [0.3, 0.5, 1.0, 2.0, 0.1])
    with torch.no_grad():
        q.off.copy_(0.2 * torch.randn(5, 5))
    u, lq = q.sample(50)
    ref = torch.distributions.MultivariateNormal(q.mu, scale_tril=q.chol()).log_prob(u)
    assert float((lq - ref).abs().max()) < 1e-10, 'log q of the Gaussian posterior'
    pr = WaterTableKsPrior(4, 1.0, 2.0, 0.5, 1.0)
    ref = torch.distributions.MultivariateNormal(pr.mean, precision_matrix=pr.P).log_prob(u[:, :5])
    d = pr.log_prob(u[:, :5]) - ref
    assert float((d - d[0]).abs().max()) < 1e-10, 'prior log p up to a constant'


def test_vi_sample_drives_water_table_and_ks():
    m, coef = make_vi([0.9, 1.1, 0.8, 1.2])
    t = 20 * torch.rand(100, 1)
    c2 = coef + 0.3
    m.set_sample(c2, torch.tensor(math.log(3.0)))
    zb = 0.05 + torch.nn.functional.softplus(SP.eval_torch(t, c2))
    assert float((m.zb(t) - zb).abs().max()) < 1e-12, 'zb follows the sample'
    for d in (0.05, 0.2, 0.4, 0.6):
        ref = VSOIL.theta(-(zb - d) / VSOIL.L)
        assert float((m.theta_at(d, t) - ref).abs().max()) < 1e-12, f'hydrostatic theta at {d} m for the sample'
    h = -torch.rand(20, 1)
    assert float((VSOIL.K(h) - 3.0 * SoilVG.K(VSOIL, h)).abs().max()) < 1e-14, 'K scaled by Ks / Ks_ref'


def test_vi_rest_and_finite_differences():
    m, coef = make_vi(1.0)
    m.set_sample(coef, torch.tensor(0.7))
    xi, tt = -torch.rand(300, 1), 20 * torch.rand(300, 1)
    assert float(pde_residual(m, xi, tt).abs().max()) < 1e-10, 'rest residual with Ks changed'
    m, coef = make_vi([0.9, 1.1, 0.8, 1.2], randomise=True)
    m.set_sample(coef, torch.tensor(-0.9))
    xi = -0.1 - 0.8 * torch.rand(20, 1)
    t = 1.0 + 18 * torch.rand(20, 1)
    R = pde_residual(m, xi, t).detach()
    e_t, e_x, s = 1e-4, 1e-4, m.soil
    with torch.no_grad():
        Mf = lambda x, tt: s.storage(m.h_tilde(x, tt))
        H = lambda tt: m.H_tilde(tt)
        dM_dt = m.T_days * (Mf(xi, t + e_t) - Mf(xi, t - e_t)) / (2 * e_t)
        Hdot = m.T_days * (H(t + e_t) - H(t - e_t)) / (2 * e_t)
        dM_dxi = (Mf(xi + e_x, t) - Mf(xi - e_x, t)) / (2 * e_x)

        def q(x):
            dh = (m.h_tilde(x + e_x, t) - m.h_tilde(x - e_x, t)) / (2 * e_x)
            return -s.K(m.h_tilde(x, t)) * (dh / H(t) + 1.0)
        dq_dxi = (q(xi + e_x) - q(xi - e_x)) / (2 * e_x)
        R_fd = dM_dt - (xi * Hdot / H(t)) * dM_dxi + dq_dxi / H(t)
    err, scale = float((R - R_fd).abs().max()), float(R.abs().max())
    assert err < 1e-4 * max(scale, 1e-3), f'VI finite differences: max |dR| {err:.3e}, max |R| {scale:.3e}'


def test_vi_gradient_reaches_posterior():
    m, coef = make_vi([0.9, 1.1, 0.8, 1.2], randomise=True)
    q = GaussianPosterior(np.concatenate([coef.numpy(), [0.0]]), [0.05] * SP.n_coef + [0.3])
    u, _ = q.sample(2)
    t = 20 * torch.rand(50, 1)
    tot = 0.0
    for k in range(2):
        m.set_sample(u[k, :SP.n_coef], u[k, SP.n_coef])
        tot = tot + (m.theta_at(0.6, t) ** 2).sum() + (pde_residual(m, -torch.rand(30, 1), t[:30]) ** 2).sum()
    tot.backward()
    assert float(q.mu.grad[:SP.n_coef].abs().sum()) > 0.0, 'data reach the water-table posterior'
    assert float(q.mu.grad[SP.n_coef].abs()) > 0.0, 'physics reaches the Ks posterior'
    assert float(q.log_diag.grad.abs().sum()) > 0.0, 'gradients reach the posterior spread'


BASE = dict(theta_s=0.4193514611932402, theta_r=5.66e-09, alpha=0.822311364445854, n=2.7469217167092075,
            Ks=1.922838127196462e-06)
TWO = [dict(top_m=0.0, **BASE), dict(top_m=0.30, theta_s=BASE['theta_s'], theta_r=BASE['theta_r'],
                                    alpha=0.5 * BASE['alpha'], n=2.0, Ks=0.3 * BASE['Ks'])]


def make_layered(path_m, layers, randomise=False):
    t = np.linspace(0, 20, 481)
    if np.ndim(path_m) == 0:
        coef = np.full(SP.n_coef, np.log(np.expm1(float(path_m) - 0.05)))
    else:
        coef = ZbSplineNumpy.fit(SP, t, np.interp(t, np.linspace(0, 20, len(path_m)), path_m), 0.05).coef
    soil = LayeredSoilVG(layers, width_m=0.02, L=2.0)
    m = ProfilePINN(soil, SP, coef, 0.05, D_TOP, 20.0, 35.0, width=16, depth=2, head_form='scale')
    if randomise:
        torch.nn.init.normal_(m.N.net[-1].weight, std=0.05)
        torch.nn.init.normal_(m.N.net[-1].bias, std=0.05)
    return m


def test_layered_one_layer_equals_single():
    one = LayeredSoilVG([dict(top_m=0.0, **BASE)], L=2.0)
    h = -torch.rand(200, 1) * 2.0 + 0.05
    d = torch.rand(200, 1) * 2.0
    for f in ('theta', 'K', 'storage'):
        a, b = getattr(one, f)(h, d), getattr(SOIL, f)(h)
        assert float((a - b).abs().max()) < 1e-12, f'one layer: {f}'


def test_layered_matches_fem_blend():
    s = LayeredSoilVG(TWO, width_m=0.02, L=2.0)
    z = torch.linspace(0.0, 1.0, 101)
    w = 0.5 * (1.0 + torch.tanh((z - 0.30) / 0.02))
    ts, tr, a_t, n, m, kf = s.params(z)
    assert float((a_t / 2.0 - BASE['alpha'] * (1.0 + w * (0.5 - 1.0))).abs().max()) < 1e-12, 'alpha blend'
    assert float((n - (BASE['n'] + w * (2.0 - BASE['n']))).abs().max()) < 1e-12, 'n blend'
    assert float((kf - torch.exp(w * math.log(0.3))).abs().max()) < 1e-12, 'Ks blend (log-linear)'


def test_layered_rest_fd_and_sensors():
    m = make_layered(1.0, TWO)
    xi, tt = -torch.rand(400, 1), 20 * torch.rand(400, 1)
    assert float(pde_residual(m, xi, tt).abs().max()) < 1e-10, 'layered rest residual'
    t = 20 * torch.rand(50, 1)
    for d in (0.05, 0.6):
        ref = m.soil.theta(-(m.zb(t) - d) / 2.0, torch.full_like(t, d))
        assert float((m.theta_at(d, t) - ref).abs().max()) < 1e-12, f'layered hydrostatic theta at {d} m'
    assert float((m.theta_at(0.6, t) - SOIL.theta(-(m.zb(t) - 0.6) / 2.0)).abs().max()) > 1e-3, 'layer 2 differs'
    m = make_layered([0.9, 1.1, 0.8, 1.2], TWO, randomise=True)
    xi = -0.1 - 0.8 * torch.rand(20, 1)
    t = 1.0 + 18 * torch.rand(20, 1)
    R = pde_residual(m, xi, t).detach()
    e_t, e_x, s = 1e-4, 1e-4, m.soil
    with torch.no_grad():
        H = lambda tt: m.H_tilde(tt)
        dep = lambda x, tt: m.d_top - x * H(tt) * s.L
        Mf = lambda x, tt: s.storage(m.h_tilde(x, tt), dep(x, tt))
        dM_dt = m.T_days * (Mf(xi, t + e_t) - Mf(xi, t - e_t)) / (2 * e_t)
        Hdot = m.T_days * (H(t + e_t) - H(t - e_t)) / (2 * e_t)
        dM_dxi = (Mf(xi + e_x, t) - Mf(xi - e_x, t)) / (2 * e_x)

        def q(x):
            dh = (m.h_tilde(x + e_x, t) - m.h_tilde(x - e_x, t)) / (2 * e_x)
            return -s.K(m.h_tilde(x, t), dep(x, t)) * (dh / H(t) + 1.0)
        dq_dxi = (q(xi + e_x) - q(xi - e_x)) / (2 * e_x)
        R_fd = dM_dt - (xi * Hdot / H(t)) * dM_dxi + dq_dxi / H(t)
    err, scale = float((R - R_fd).abs().max()), float(R.abs().max())
    assert err < 1e-4 * max(scale, 1e-3), f'layered finite differences: max |dR| {err:.3e}, max |R| {scale:.3e}'


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
