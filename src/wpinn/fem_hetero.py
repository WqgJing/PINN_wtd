"""The wtd_pipeline FEM with soil that varies with depth (for a synthetic TRUTH with 'wrong physics' for the inversions).

The wtd_pipeline solver (read only, loaded by file) has one van Genuchten soil per member. Its hydraulic functions
broadcast per-node parameter arrays, so here the parameters are set node by node from each node's true depth at every
time step (the Landau mesh moves with the water table: depth = -xi * zb(t)):
  alpha(z), n(z) and a conductivity factor Ks(z) / Ks_ref (Ks_ref = the soil's Ks, which sets the solver's time scale).
theta_s and theta_r stay uniform, so theta = theta_r + (theta_s - theta_r) * Se holds at every node and time.

Profiles (dict 'soil_profile' of the synthetic config):
  {'kind': 'layer', 'depth_m': 0.30, 'width_m': 0.02, 'alpha_factor': 0.5, 'n': 2.0, 'Ks_factor': 0.3}
      the soil below depth_m (smooth tanh step of half-width width_m) has alpha x alpha_factor, n, Ks x Ks_factor
  {'kind': 'ks_exp', 'scale_m': 0.30}       Ks(z) = Ks_ref * exp(-z / scale_m); alpha and n unchanged
  {'kind': 'none'}                          uniform (for the regression check against the standard solver)
"""
import sys
import numpy as np
import torch

from .fem import load_fem


def profile_factors(depth_m, n_ref, profile):
    """(alpha factor, n, Ks factor), tensors shaped like depth_m [B, n_nodes]."""
    kind = profile.get('kind', 'none')
    one = torch.ones_like(depth_m)
    if kind == 'none':
        return one, one * n_ref, one
    if kind == 'layer':
        w = 0.5 * (1.0 + torch.tanh((depth_m - float(profile['depth_m'])) / float(profile['width_m'])))
        a_f = 1.0 + w * (float(profile['alpha_factor']) - 1.0)
        n = n_ref + w * (float(profile['n']) - n_ref)
        k_f = torch.exp(w * np.log(float(profile['Ks_factor'])))
        return a_f, n, k_f
    if kind == 'ks_exp':
        return one, one * n_ref, torch.exp(-torch.clamp(depth_m, min=0.0) / float(profile['scale_m']))
    raise ValueError(f'soil_profile kind {kind}')


def hetero_solver(root, soil, L, n_nodes, profile):
    """(solver, Sink): a wtd_pipeline solver whose soil follows `profile` with depth."""
    Solver, Sink = load_fem(root)
    solver = Solver(soil, L=L, n_nodes=n_nodes, device='cpu')
    VG, FEM = type(solver.vg), type(solver.fem)

    class HeteroVG(VG):
        def __init__(self, base):
            super().__init__(theta_s=base.theta_s, theta_r=base.theta_r, alpha_tilde=base.alpha_tilde, n=base.n,
                             l=base.l, device=base.device)
            self.alpha_ref, self.n_ref = base.alpha_tilde.clone(), float(base.n)
            self.ks_factor = None

        def set_depth(self, depth_m):
            a_f, n, k_f = profile_factors(depth_m, self.n_ref, profile)
            self.alpha_tilde, self.n, self.m, self.ks_factor = self.alpha_ref * a_f, n, 1.0 - 1.0 / n, k_f

        def K_tilde(self, h_tilde):
            kr = super().K_tilde(h_tilde)
            return kr if self.ks_factor is None else kr * self.ks_factor

    class HeteroFEM(FEM):
        def solve_timestep_picard(self, Se_old, q_surf_tilde, zb_tilde, zb_dot_tilde, dt_tilde, sink_tilde=None,
                                  **kw):
            self.vg.set_depth(-self.xi.unsqueeze(0) * (zb_tilde.reshape(-1, 1) * self.L))
            return super().solve_timestep_picard(Se_old, q_surf_tilde, zb_tilde, zb_dot_tilde, dt_tilde, sink_tilde,
                                                 **kw)

    solver.vg = HeteroVG(solver.vg)
    solver.fem = HeteroFEM(solver.vg, n_nodes, 'cpu', L=L, free_drainage=False)
    return solver, Sink


def run_fem_hetero(root, fem_cfg, soil, profile, t_days, zb, q, tp_mmd, n_nodes, root_beta=None):
    """As wpinn.fem.run_fem (one batched solve, hydrostatic start over zb(0), flux at the surface, h = 0 at the
    water table, Feddes sink), with soil varying with depth. theta is rebuilt from Se (uniform theta_s, theta_r)."""
    solver, Sink = hetero_solver(root, soil, fem_cfg['L'], n_nodes, profile)
    L = float(fem_cfg['L'])
    xi = solver.fem.xi
    zb_t = torch.as_tensor(zb, dtype=torch.float64)
    solver.vg.set_depth(-xi.unsqueeze(0) * zb_t[:, :1])                 # parameters at the start
    h0 = -(1.0 + xi).unsqueeze(0) * zb_t[:, :1]
    kw = {}
    if root_beta is not None:
        kw['root_params'] = sys.modules['root_water_uptake'].RootDistributionParameters(beta=float(root_beta))
    sink = Sink(tp_mmd, device='cpu', Tp_units='mm/day', **kw)
    res = solver.solve(t_days * 86400.0, zb_t, q, theta_initial=solver.vg.theta(h0 / L), sink_func=sink,
                       max_iter=fem_cfg['max_iter'], tol=fem_cfg['tol'], omega=fem_cfg['omega'], return_full=True)
    th_r, th_s = float(soil['theta_r']), float(soil['theta_s'])
    out = {'xi': res['xi'].cpu().numpy(), 'theta': (th_r + (th_s - th_r) * res['Se']).cpu().numpy(),
           'h': res['h'].cpu().numpy(), 'converged': res['converged'].cpu().numpy(),
           'n_iters': res['n_iters'].cpu().numpy()}
    print(f"[fem-hetero] {profile.get('kind')}: {zb.shape[0]} member(s), {t_days.size - 1} steps of "
          f"{24 * (t_days[1] - t_days[0]):.2f} h, {n_nodes} nodes: converged {100 * out['converged'].mean():.2f}% of "
          f"steps, max Picard iterations {int(out['n_iters'].max())}", flush=True)
    return out
