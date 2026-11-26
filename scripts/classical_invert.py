#!/usr/bin/env python
"""Classical inversion of the water-table path with the FEM (tasks/todo.md, Step 5 PLAN): the top sensor is a Dirichlet
boundary, the deeper sensor(s) are fitted by Levenberg-Marquardt over the same spline coefficients as the PINN.

Data: exactly those of a PINN run (build() of scripts/train_profile.py on that run's config.yaml): same noisy readings,
same start path, same truth (scoring only). Forward model: wpinn.fem_top, the wtd_pipeline FEM on the column from the
top sensor to the water table, root sink at true depth, hydrostatic start over the path's first value.
Top series: 'smooth' (centred mean over --smooth_hours of the noisy reading), 'raw' (noisy reading), 'true' (noise
free; check only).

Usage: python -u scripts/classical_invert.py --run runs/place_2_5-40 --top smooth --out runs/fem_2_5-40_smooth
       python -u scripts/classical_invert.py --run runs/place_2_5-40 --check --out runs/fem_2_5-40_check
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import train_profile as tpf                                          # noqa: E402  (build, load_config, score)
from wpinn.fem_top import top_dirichlet_solver, run_top, theta_at   # noqa: E402
from wpinn.spline import ZbSplineNumpy                               # noqa: E402


def centred_mean(y, m):
    """Mean over the m (odd) nearest points, fewer at the ends."""
    k = int(m) // 2
    return np.array([y[max(0, i - k):i + k + 1].mean() for i in range(y.size)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='PINN run whose config.yaml defines the data')
    ap.add_argument('--set', action='append', default=[], help='overrides of that config, as in train_profile.py')
    ap.add_argument('--pinn', default=None, help='PINN run drawn for comparison in result.png (default: --run)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--top', default='smooth', choices=['smooth', 'raw', 'true'])
    ap.add_argument('--smooth_hours', type=int, default=3)
    ap.add_argument('--check', action='store_true', help='forward runs on the true path only (no fit)')
    ap.add_argument('--dt_hours', type=float, default=0.25)
    ap.add_argument('--tol', type=float, default=1e-7)
    ap.add_argument('--max_iter_picard', type=int, default=100)
    ap.add_argument('--delta', type=float, default=0.01, help='central-difference step, spline-coefficient units')
    ap.add_argument('--max_iter', type=int, default=60)
    ap.add_argument('--rel_tol', type=float, default=1e-7, help='stop when a step lowers the cost by less than this '
                                                                'fraction')
    ap.add_argument('--zc_min', type=float, default=0.05, help='least column length below the top sensor, m')
    ap.add_argument('--est_top', action='store_true', help='fit a constant offset and a linear drift of the top '
                                                           'sensor with the water table (two extra unknowns)')
    ap.add_argument('--delta_top', type=float, default=0.002, help='central-difference step of those two, m3/m3')
    ap.add_argument('--max_step', type=float, default=1.0, help='trust region: largest change of any spline '
                    'coefficient per iteration (the step is scaled down to it); <= 0: no limit (original)')
    ap.add_argument('--damp_floor', type=float, default=1e-3, help='floor of the Marquardt scaling diag(J\'J) as a '
                    'fraction of its mean, so that coefficients the sensors barely see are still damped; '
                    '0 = plain (original)')
    args = ap.parse_args()
    tic0 = time.time()
    os.makedirs(args.out, exist_ok=True)

    cfg = tpf.load_config(os.path.join(args.run, 'config.yaml'), args.set)
    with open(os.path.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump({'data_config': cfg, 'args': vars(args)}, f, sort_keys=False)
    B = tpf.build(cfg, torch.device('cpu'))
    rec = dict(np.load(cfg['data']['npz'], allow_pickle=False))
    with open(os.path.join(os.path.dirname(cfg['data']['npz']), 'config.yaml')) as f:
        syn = yaml.safe_load(f)
    fem_cfg = {'L': syn['fem']['L'], 'n_nodes': syn['fem']['n_nodes'], 'omega': syn['fem'].get('omega', 0.5),
               'tol': args.tol, 'max_iter': args.max_iter_picard}
    s = B['soil_params']                                       # the soil the method is given (soil_factor applied)
    soil = {'theta_s': float(s[0]), 'theta_r': float(s[1]), 'alpha': float(s[2]), 'n': float(s[3]), 'Ks': float(s[4])}
    t0 = float(cfg['window']['start_day'])
    tw, depths, th_obs, th_true = B['tw'], B['depths'], B['th_obs'], B['th_true']
    d_top, deep = depths[0], depths[1:]
    top_obs = th_obs[:, 0].copy()
    if np.isnan(top_obs).any():                     # standard practice: straight-line fill of a gap in the boundary
        ok = np.isfinite(top_obs)
        top_obs = np.interp(tw, tw[ok], top_obs[ok])
    sig = float(cfg['sigmas']['theta'])
    sp, zb_min = B['spline'], float(cfg['zb_spline']['zb_min'])

    dt = args.dt_hours / 24.0
    n_s = int(round(tw[-1] / dt)) + 1
    t_s = np.arange(n_s) * dt
    k_obs = np.round(tw / dt).astype(int)
    if not np.allclose(t_s[k_obs], tw, atol=1e-9):
        raise SystemExit('observation times are not on the solver grid')
    tp_s = np.interp(t_s + t0, rec['t_nodes'], rec['tp_nodes_mmd'])
    solver, Sink = top_dirichlet_solver(syn['fem']['root'], soil, fem_cfg['L'], fem_cfg['n_nodes'])
    th_s, th_r = soil['theta_s'], soil['theta_r']

    def top_series(kind):
        y = {'true': th_true[:, 0], 'raw': top_obs, 'smooth': centred_mean(top_obs, args.smooth_hours)}[kind]
        return y, np.clip((np.interp(t_s, tw, y) - th_r) / (th_s - th_r), 1e-6, 1.0 - 1e-6)

    info = {'solves': 0, 'batched_solves': 0, 'not_converged_max': 0.0, 'solve_s': 0.0}

    def forward(C, se_top):
        """theta at the deeper sensors [B, n_deep * n_obs] (sensor-major) for coefficient rows C [B, n_coef]; se_top
        [n_s] (shared) or [B, n_s] (one boundary series per row)."""
        C = np.atleast_2d(C)
        zb = np.stack([ZbSplineNumpy(sp, c, zb_min).zb(t_s) for c in C])
        zc = np.maximum(zb - d_top, args.zc_min)
        se_rows = np.tile(se_top, (C.shape[0], 1)) if np.ndim(se_top) == 1 else se_top
        tic = time.time()
        try:
            A = run_top(solver, Sink, fem_cfg, t_s, zc, se_rows, tp_s, d_top)
        except Exception as e:                                   # a failed solve is a rejected step, not a crash
            print(f"[fem] solver failed: {type(e).__name__}: {e}", flush=True)
            return np.full((C.shape[0], len(deep) * tw.size), np.nan)
        info['solve_s'] += time.time() - tic
        info['solves'] += C.shape[0]
        info['batched_solves'] += 1
        info['not_converged_max'] = max(info['not_converged_max'], 1.0 - float(A['converged'].mean()))
        th = [theta_at(A['xi'], A['theta'][:, k_obs], zc[:, k_obs], d - d_top, th_s) for d in deep]
        return np.concatenate(th, axis=1)

    y = th_obs[:, 1:].T.reshape(-1)
    ok = np.isfinite(y)                                         # missing interior readings are left out of the cost
    y_true = th_true[:, 1:].T.reshape(-1)
    rms = lambda v: float(np.sqrt(np.nanmean(np.square(v))))
    print(f"[cls] run {args.run}: sensors {depths} m (top {d_top} m is the boundary); window {t0}-"
          f"{t0 + tw[-1]} d; {sp.n_coef} coefficients; solver dt {args.dt_hours} h, {n_s - 1} steps; soil given "
          f"{ {k: round(v, 6) for k, v in soil.items()} }; mean top reading {np.nanmean(th_obs[:, 0]):.4f}; top hours missing "
          f"{int(np.isnan(th_obs[:, 0]).sum())}", flush=True)

    if args.check:
        out = {}
        for kind in ('true', 'smooth', 'raw'):
            y_top, se = top_series(kind)
            tic = time.time()
            th = forward(B['truth'].coef, se)[0]
            out[kind] = {'top_rms_vs_truth': rms(y_top - th_true[:, 0]), 'deep_rms_vs_truth': rms(th - y_true),
                         'deep_rms_vs_obs': rms(th - y), 'cost_at_true_path': float(np.mean(((th - y) / sig) ** 2)),
                         'deep_maxabs_vs_truth': float(np.abs(th - y_true).max()), 'solve_s': time.time() - tic}
            print(f"[check] top {kind:6s}: top series vs truth rms {out[kind]['top_rms_vs_truth']:.4f}; deep sensor "
                  f"vs truth rms {out[kind]['deep_rms_vs_truth']:.4f} (largest {out[kind]['deep_maxabs_vs_truth']:.4f})"
                  f", vs readings {out[kind]['deep_rms_vs_obs']:.4f}; cost at the true path "
                  f"{out[kind]['cost_at_true_path']:.3f}; {out[kind]['solve_s']:.0f} s", flush=True)
        out['not_converged_max'] = info['not_converged_max']
        with open(os.path.join(args.out, 'check.json'), 'w') as f:
            json.dump(out, f, indent=1)
        print(f"[check] done in {time.time() - tic0:.0f} s", flush=True)
        return

    y_top, se_top = top_series(args.top)
    ramp_s = t_s / t_s[-1]                                      # drift of the top sensor: 0 at the start, b at the end

    def boundary(P):
        """Top boundary series [B, n_s] for parameter rows P: with --est_top the last two columns are the top
        sensor's offset a and drift b, removed from its reading before it becomes the boundary."""
        P = np.atleast_2d(P)
        if not args.est_top:
            return se_top
        y_s = np.interp(t_s, tw, y_top)[None, :] - P[:, -2:-1] - P[:, -1:] * ramp_s[None, :]
        return np.clip((y_s - th_r) / (th_s - th_r), 1e-6, 1.0 - 1e-6)

    def fwd(P):
        P = np.atleast_2d(P)
        return forward(P[:, :sp.n_coef], boundary(P))

    def cost_of(th_):
        return float(np.mean(((th_[ok] - y[ok]) / sig) ** 2))

    zb_true = B['zb_true']
    zb_err = lambda cc: rms(ZbSplineNumpy(sp, cc[:sp.n_coef], zb_min).zb(tw) - zb_true)
    tic_fit = time.time()
    c = np.asarray(B['coef0'], dtype=float).copy()
    deltas = np.full(c.size, args.delta)
    if args.est_top:                                            # start with no offset and no drift
        c = np.concatenate([c, [0.0, 0.0]])
        deltas = np.concatenate([deltas, [args.delta_top, args.delta_top]])
    n_c = c.size
    th = fwd(c)[0]
    cost = cost_of(th)
    lam, hist = 1e-2, []
    print(f"[cls] start: cost {cost:.4f}, water table vs truth {100 * zb_err(c):.2f} cm (scoring only); unknowns {n_c}; "
          f"interior readings used {int(ok.sum())} of {ok.size}", flush=True)
    for it in range(args.max_iter):
        E = np.diag(deltas)
        res = fwd(np.concatenate([c + E, c - E]))
        dif = res[:n_c] - res[n_c:]
        J = (np.where(np.isfinite(dif), dif, 0.0).T / (2 * deltas[None, :]) / sig)[ok]
        r = ((th - y) / sig)[ok]
        A, g = J.T @ J, J.T @ r
        D = np.diag(A).copy()                                   # Marquardt scaling
        if args.damp_floor > 0:                                 # floored: no undamped direction
            D = np.maximum(D, args.damp_floor * D.mean())
        accepted, rel = False, 0.0
        for _ in range(10):
            step = -np.linalg.solve(A + lam * np.diag(D) + 1e-12 * np.eye(n_c), g)
            s_max = float(np.abs(step[:sp.n_coef]).max())
            if args.max_step > 0 and s_max > args.max_step:            # trust region on the coefficients
                step = step * (args.max_step / s_max)
            th_new = fwd(c + step)[0]
            cost_new = cost_of(th_new)
            if np.isfinite(cost_new) and cost_new < cost:
                rel = (cost - cost_new) / cost
                c, th, cost = c + step, th_new, cost_new
                lam = max(lam / 3.0, 1e-8)
                accepted = True
                break
            lam *= 4.0
        hist.append({'iter': it, 'cost': cost, 'lambda': lam, 'zb_rmse_m': zb_err(c), 'wall_s': time.time() - tic_fit})
        top_txt = f"; top offset {c[-2]:+.4f}, drift {c[-1]:+.4f}" if args.est_top else ''
        print(f"[cls {it:2d}] cost {cost:.5f} lambda {lam:.1e} |step| {np.abs(step).max():.2e} water table vs truth "
              f"{100 * zb_err(c):.2f} cm  {time.time() - tic_fit:.0f} s{top_txt}", flush=True)
        if not accepted or rel < args.rel_tol or np.abs(step).max() < 1e-6:
            break
    wall_fit = time.time() - tic_fit

    m = ZbSplineNumpy(sp, c[:sp.n_coef], zb_min)
    zb = m.zb(tw)
    inner = (tw >= 1.0) & (tw <= tw[-1] - 1.0)
    th_deep = th.reshape(len(deep), tw.size).T
    summary = {
        'top': args.top, 'smooth_hours': args.smooth_hours if args.top == 'smooth' else None,
        'dt_hours': args.dt_hours, 'rel_tol': args.rel_tol, 'max_step': args.max_step, 'damp_floor': args.damp_floor,
        'depths_m': depths, 'iterations': len(hist),
        'solves': info['solves'],
        'batched_solves': info['batched_solves'], 'not_converged_max': info['not_converged_max'],
        'wall_fit_s': wall_fit, 'solve_s': info['solve_s'], 'wall_total_s': time.time() - tic0, 'cost': cost,
        'zb_vs_truth': tpf.score(zb, zb_true), 'zb_vs_truth_inner': tpf.score(zb[inner], zb_true[inner]),
        'start_vs_truth': tpf.score(ZbSplineNumpy(sp, B['coef0'], zb_min).zb(tw), zb_true),
        'deep_vs_obs_rmse': [rms(th_deep[:, j] - th_obs[:, 1 + j]) for j in range(len(deep))],
        'deep_vs_truth_rmse': [rms(th_deep[:, j] - th_true[:, 1 + j]) for j in range(len(deep))],
        'top_vs_truth_rmse': rms(y_top - th_true[:, 0]), 'est_top': bool(args.est_top),
        'top_offset_est': float(c[-2]) if args.est_top else None,
        'top_drift_est': float(c[-1]) if args.est_top else None,
        'contamination': B.get('contamination'), 'qc_removed': B.get('qc'),
        'interior_readings_used': int(ok.sum()), 'history': hist}
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    np.savez(os.path.join(args.out, 'predictions.npz'), t_days=tw, t_offset=t0, zb_est=zb, zb_true=zb_true,
             zb_start=ZbSplineNumpy(sp, B['coef0'], zb_min).zb(tw), coef=c, depths_m=np.asarray(depths),
             theta_deep_est=th_deep, theta_obs=th_obs, theta_true=th_true, top_used=y_top)
    plot(os.path.join(args.out, 'result.png'), args.pinn or args.run, tw + t0, zb_true, zb, depths, th_obs, y_top, th_deep)
    z = summary['zb_vs_truth']
    print(f"[cls] water table vs truth: RMSE {100 * z['rmse_m']:.2f} cm, bias {100 * z['bias_m']:+.2f}, largest "
          f"{100 * z['maxabs_m']:.2f}, r {z['r']:.4f}; inner {100 * summary['zb_vs_truth_inner']['rmse_m']:.2f} cm; "
          f"{len(hist)} iterations, {info['solves']} solves ({info['batched_solves']} batched), fit {wall_fit / 60:.1f} "
          f"min; deep sensor vs readings {summary['deep_vs_obs_rmse']}", flush=True)


def plot(path, run, ta, zb_true, zb, depths, th_obs, y_top, th_deep):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#6250d6']
    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax[0].plot(ta, zb_true, color='k', lw=2.0, label='truth')
    pinn = os.path.join(run, 'predictions.npz')
    if os.path.exists(pinn):
        ax[0].plot(ta, np.load(pinn)['zb_est'], color=colors[1], lw=1.6, label='PINN')
    ax[0].plot(ta, zb, color=colors[0], lw=1.6, label='FEM inversion')
    ax[0].invert_yaxis()
    ax[0].set_ylabel('water-table depth (m)')
    ax[0].legend(frameon=False, fontsize=9)
    ax[1].plot(ta, th_obs[:, 0], color=colors[0], lw=0.6, alpha=0.6)
    ax[1].plot(ta, y_top, color=colors[0], lw=1.6, label=f'{100 * depths[0]:.0f} cm (boundary)')
    for j in range(th_deep.shape[1]):
        ax[1].plot(ta, th_obs[:, 1 + j], color=colors[(2 + j) % 5], lw=0.6, alpha=0.6)
        ax[1].plot(ta, th_deep[:, j], color=colors[(2 + j) % 5], lw=1.6, label=f'{100 * depths[1 + j]:.0f} cm (fitted)')
    ax[1].set_ylabel('soil moisture (m$^3$/m$^3$)')
    ax[1].set_xlabel('time (days from the start of the record)')
    ax[1].legend(frameon=False, fontsize=9, ncol=len(depths))
    for a in ax:
        a.grid(True, color='#e1e0d9', lw=0.6)
        a.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == '__main__':
    main()
