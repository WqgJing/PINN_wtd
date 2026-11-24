"""The wtd_pipeline FEM with a Dirichlet top: a column from the top sensor down to the water table.

The wtd_pipeline solver (read only, loaded by file through wpinn.fem.load_fem) has a flux at the surface node and h = 0
at the water table. Here the top node carries a prescribed Se (the top sensor's reading), so the column starts at the
sensor depth d_top and ends at the water table: column length zc(t) = zb(t) - d_top on the solver's Landau coordinate
(xi = 0 at the sensor, -1 at the water table). The Richards equation does not change under the shift (gravity is
uniform); only the root sink needs the true depth, d_top + depth in the column.
"""
import numpy as np
import torch

from .fem import load_fem


def top_dirichlet_solver(root, soil, L, n_nodes):
    """(solver, Sink): a wtd_pipeline ForwardSensitivitySolverPyTorch whose FEM has a Dirichlet top node, and the
    wtd_pipeline RootWaterUptakeTorch class. soil: dict theta_s, theta_r, alpha, n, Ks (scalars)."""
    Solver, Sink = load_fem(root)
    solver = Solver(soil, L=L, n_nodes=n_nodes, device='cpu')
    base = type(solver.fem)

    class TopDirichletFEM(base):
        def solve_timestep_picard(self, Se_old, se_top, zb_tilde, zb_dot_tilde, dt_tilde, sink_tilde=None,
                                  max_iter=50, tol=1e-6, omega=0.5):
            """The parent's Picard step with a second Dirichlet node at the top. The parent's solve() passes the
            surface-flux slot as the second argument; here that slot carries Se at the top node [batch]."""
            batch = Se_old.shape[0]
            top = self.n_nodes - 1
            se_top = se_top.reshape(batch)
            no_flux = torch.zeros(batch, dtype=Se_old.dtype, device=Se_old.device)
            Se_k = Se_old.clone()
            Se_k[:, 0], Se_k[:, top] = 1.0, se_top
            converged = torch.zeros(batch, dtype=torch.bool, device=Se_old.device)
            n_iters = torch.zeros(batch, dtype=torch.int32, device=Se_old.device)
            for it in range(max_iter):
                lo, di, up, b = self._assemble_picard_system(Se_k, Se_old, no_flux, zb_tilde, zb_dot_tilde, dt_tilde,
                                                             sink_tilde)
                lo, di, up, b = self._apply_dirichlet_bc(lo, di, up, b, node=0, value=1.0)
                lo, di, up, b = self._apply_dirichlet_bc(lo, di, up, b, node=top, value=se_top)
                try:
                    Se_new = self._tridiag_solve(lo, di, up, b)
                except RuntimeError:
                    return Se_k, converged, n_iters
                Se_new = torch.clamp(Se_new, min=1e-6, max=1.0 - 1e-6)
                Se_new[:, 0], Se_new[:, top] = 1.0, se_top
                delta = torch.abs(Se_new - Se_k).max(dim=1).values
                newly = (delta < tol) & (~converged)
                n_iters[newly] = it + 1
                converged = converged | (delta < tol)
                Se_k = torch.clamp(Se_k + omega * (Se_new - Se_k), min=1e-6, max=1.0 - 1e-6)
                Se_k[:, 0], Se_k[:, top] = 1.0, se_top
                if converged.all():
                    break
            n_iters[~converged] = max_iter
            return Se_k, converged, n_iters

    solver.fem = TopDirichletFEM(solver.vg, n_nodes, 'cpu', L=L, free_drainage=False)
    return solver, Sink


class OffsetSink:
    """Root sink at the true depth: the solver passes depth below the column top; the sensor sits d_top below the
    surface, where the root distribution (normalised over 0 - z_root_max from the surface) is defined."""

    def __init__(self, sink, d_top):
        self.sink, self.d_top = sink, float(d_top)

    def compute_sink(self, z_depth, time_idx, h):
        return self.sink.compute_sink(z_depth + self.d_top, time_idx, h)


def run_top(solver, Sink, fem_cfg, t_days, zc, se_top, tp_mmd, d_top):
    """One batched solve of the column [d_top, zb]. t_days [n_t] (solver steps); zc [B, n_t] column length (m);
    se_top [B, n_t] effective saturation at the top node; tp_mmd [n_t] transpiration. Start: hydrostatic over zc(0).
    Returns numpy xi [n_nodes], theta [B, n_t, n_nodes] (node 0 = water table, last = top), converged [B, n_t]."""
    L = float(fem_cfg['L'])
    Ks = float(solver.Ks)
    xi = solver.fem.xi
    zc_t = torch.as_tensor(zc, dtype=torch.float64)
    h0 = -(1.0 + xi).unsqueeze(0) * zc_t[:, :1]
    sink = OffsetSink(Sink(np.tile(np.asarray(tp_mmd, dtype=float), (zc_t.shape[0], 1)), device='cpu',
                           Tp_units='mm/day'), d_top)
    # the wrapper divides the flux slot by Ks; TopDirichletFEM reads that slot as Se at the top
    res = solver.solve(np.asarray(t_days) * 86400.0, zc_t, torch.as_tensor(se_top, dtype=torch.float64) * Ks,
                       theta_initial=solver.vg.theta(h0 / L), sink_func=sink, max_iter=int(fem_cfg['max_iter']),
                       tol=float(fem_cfg['tol']), omega=float(fem_cfg['omega']), return_full=True)
    return {'xi': res['xi'].numpy(), 'theta': res['theta'].numpy(), 'converged': res['converged'].numpy()}


def theta_at(xi, theta, zc, depth_rel, theta_s):
    """theta at depth_rel below the column top, for every member and time: theta [B, n_t, n_nodes] on uniform xi
    (-1 at the water table, 0 at the top), zc [B, n_t]. A point at or below the water table reads theta_s."""
    n = xi.size
    if not np.allclose(np.diff(xi), xi[1] - xi[0]):
        raise ValueError('xi is not uniform')
    x = -depth_rel / np.maximum(zc, 1e-9)                      # [B, n_t], in [-1, 0] above the water table
    pos = np.clip((x - xi[0]) / (xi[1] - xi[0]), 0.0, n - 1 - 1e-9)
    i0 = np.floor(pos).astype(int)
    w = pos - i0
    b_idx, t_idx = np.indices(zc.shape)
    out = (1.0 - w) * theta[b_idx, t_idx, i0] + w * theta[b_idx, t_idx, i0 + 1]
    return np.where(depth_rel >= zc, theta_s, out)
