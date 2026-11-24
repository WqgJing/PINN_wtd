"""The wtd_pipeline FEM (read only) as the truth solver, and point sensors from its profiles.

load_fem and run_fem follow ../pinn/zb_inverse/scripts/make_twin_fem.py (copied 2026-11-23); run_fem here returns the
full profiles and no layer mean.
"""
import importlib.util
import os
import sys
import numpy as np
import torch


def load_fem(root):
    """ForwardSensitivitySolverPyTorch and RootWaterUptakeTorch, loaded by file from the wtd_pipeline checkout."""
    mods = {}
    for name in ('fem_pytorch', 'root_water_uptake'):
        spec = importlib.util.spec_from_file_location(name, os.path.join(root, 'fem', name + '.py'))
        mods[name] = importlib.util.module_from_spec(spec)
        sys.modules[name] = mods[name]
        spec.loader.exec_module(mods[name])
    return mods['fem_pytorch'].ForwardSensitivitySolverPyTorch, mods['root_water_uptake'].RootWaterUptakeTorch


def run_fem(Solver, Sink, fem_cfg, soil_params, t_days, zb, q, tp_mmd, n_nodes):
    """One batched solve. zb, q [m/s, + up], tp_mmd [mm/d]: arrays [B, n_t] on t_days. Each member starts from the
    hydrostatic profile over its zb(0); the Feddes root sink takes its transpiration from tp_mmd (0: no sink).
    Returns numpy arrays: xi [n_nodes], theta and h [B, n_t, n_nodes] (last node = surface), converged, n_iters."""
    solver = Solver(soil_params, L=fem_cfg['L'], n_nodes=n_nodes, device='cpu')
    xi = solver.fem.xi
    h0 = -(1.0 + xi).unsqueeze(0) * torch.as_tensor(zb[:, :1], dtype=xi.dtype)
    sink = Sink(tp_mmd, device='cpu', Tp_units='mm/day')
    res = solver.solve(t_days * 86400.0, zb, q, theta_initial=solver.vg.theta(h0 / fem_cfg['L']), sink_func=sink,
                       max_iter=fem_cfg['max_iter'], tol=fem_cfg['tol'], omega=fem_cfg['omega'], return_full=True)
    out = {k: res[k].cpu().numpy() for k in ('xi', 'theta', 'h', 'converged', 'n_iters')}
    print(f"[fem] {zb.shape[0]} member(s), {t_days.size - 1} steps of {24 * (t_days[1] - t_days[0]):.2f} h, "
          f"{n_nodes} nodes: converged {100 * out['converged'].mean():.2f}% of steps, max Picard iterations "
          f"{int(out['n_iters'].max())}, surface head min {out['h'][:, :, -1].min():.2f} m", flush=True)
    return out


def point_sensors(xi, theta, zb, depths_m, theta_s):
    """theta at point depths [m below the surface] from nodal profiles theta [n_t, n_nodes] on xi (z / zb, -1 at the
    water table, 0 at the surface). A sensor at or below the water table reads theta_s (saturated).
    Returns [n_t, n_depths]."""
    xi = np.asarray(xi, dtype=float)
    order = np.argsort(xi)
    xs = xi[order]
    out = np.empty((theta.shape[0], len(depths_m)))
    for j, d in enumerate(depths_m):
        for k in range(theta.shape[0]):
            out[k, j] = theta_s if d >= zb[k] else np.interp(-d / zb[k], xs, theta[k][order])
    return out
