#!/usr/bin/env python
"""Synthetic soil moisture at several depths: the wtd_pipeline FEM with prescribed forcing, water table and root uptake.

  forcing      site record (AmeriFlux BASE, half-hourly): ET = LE / (rho_w lambda), clipped at 0, gaps filled with
               the mean diurnal cycle; rain (missing = 0). Hourly nodes (centred means), linear in between.
                 q_surf = (1 - f_T) ET - P   [m/s, + up]     soil evaporation minus infiltration (no runoff)
                 T_p    = f_T ET             [mm/d]          the Feddes root sink of the FEM
               f_T: daily T_p / (E_soil + T_p) of the GLDAS pixel (the tower measures the total only).
  water table  prescribed: the daily-knot spline fit of the site's well record (smooth, shallow).
  truth        FEM on hourly steps, hydrostatic start over zb(0), full profile kept. Sensors: theta at points; a
               sensor at or below the water table reads theta_s.
  check        the same solve with check_dt_hours steps; the sensor difference is reported.

Usage: python -u scripts/make_synthetic.py --config configs/syn_nc2_spring23.yaml
"""
import argparse
import csv
import json
import os
import shutil
import sys
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from wpinn import sitedata                                    # noqa: E402
from wpinn.fem import load_fem, run_fem, point_sensors       # noqa: E402
from wpinn.spline import CubicBSpline1D, ZbSplineNumpy       # noqa: E402


def hourly_times(n_days, dt_hours=1.0):
    n = int(round(n_days * 24.0 / dt_hours))
    return np.arange(n + 1) * dt_hours / 24.0


def gldas_fraction(path, dates):
    """T_p / (E_soil + T_p) of the GLDAS daily rows for the given dates."""
    with open(path) as f:
        rows = {r['date']: r for r in csv.DictReader(f)}
    e = np.array([float(rows[str(d)]['E_soil_mm']) for d in dates])
    tp = np.array([float(rows[str(d)]['Tp_mm']) for d in dates])
    if np.any(e + tp <= 0.0):
        raise ValueError('GLDAS E_soil + T_p = 0 on a day of the window: no partition')
    return tp / (e + tp)


def site_forcing(cfg):
    """Half-hourly records -> hourly forcing nodes and the hourly well record (as make_real.site_series)."""
    sc, fc, wc = cfg['site'], cfg['forcing'], cfg['water_table']
    n_days = int(cfg['window']['n_days'])
    t0 = np.datetime64(cfg['window']['start'], 'm')
    w = fc['node_hours'] / 24.0
    half = int(fc.get('fill_half_window_days', 7))
    pad = np.timedelta64(half + 1, 'D')
    t_rec, rec = sitedata.read_amf_halfhourly(sc['amf_csv'], sc['columns'], t0 - pad,
                                              t0 + np.timedelta64(n_days, 'D') + pad)
    t_c = ((t_rec - t0) / np.timedelta64(1, 'm') + 15.0) / 1440.0
    p_missing = ~np.isfinite(rec['p'])
    p = np.where(p_missing, 0.0, rec['p']) / 1000.0 / 1800.0
    et, le_missing = sitedata.fill_diurnal(np.clip(rec['le'] * sitedata.LE_TO_ET, 0.0, None), half_window_days=half)
    t_nodes = np.arange(int(round(n_days / w)) + 1) * w
    et_n = sitedata.centred_means(et, t_c, t_nodes, w)
    p_n = sitedata.centred_means(p, t_c, t_nodes, w)
    dates = (t0 + np.round(t_nodes * 1440.0).astype('timedelta64[m]')).astype('datetime64[D]')
    t_h = hourly_times(n_days, 1.0)
    zb, n_zb = sitedata.fill_short_gaps(
        t_h, sitedata.centred_means(-rec['wtd'], t_c, t_h, 1.0 / 24.0, allow_missing=True),
        wc.get('max_gap_hours', 3.0) / 24.0)
    inside = (t_c >= 0) & (t_c < n_days)
    info = {'le_filled_frac': float(le_missing[inside].mean()), 'p_missing_records': int(p_missing[inside].sum()),
            'zb_hours_filled': n_zb, 'et_sum_mm': float(et[inside].sum() * 1800e3),
            'p_sum_mm': float(p[inside].sum() * 1800e3)}
    return {'t0': t0, 'n_days': n_days, 't_nodes': t_nodes, 'dates': dates, 'et': et_n, 'p': p_n, 't_h': t_h,
            'zb_record': zb, 'info': info}


def daily(t_nodes, y, n_days):
    """Daily totals [mm/d] of a series given at hourly nodes in m/s (trapezoid within each day)."""
    out = np.empty(n_days)
    for d in range(n_days):
        k = (t_nodes >= d - 1e-9) & (t_nodes <= d + 1 + 1e-9)
        out[d] = np.trapezoid(y[k], t_nodes[k]) * 86400e3
    return out


def plot(path, t_h, n_days, rain_d, et_d, tp_d, zb, depths, th, plot_depths, start):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#6250d6', '#e34948']
    td = np.arange(n_days) + 0.5
    fig, ax = plt.subplots(3, 1, figsize=(9, 8.5), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1.4]})
    ax[0].plot(td, rain_d, color=colors[0], lw=1.6, label='rain')
    ax[0].plot(td, et_d, color=colors[1], lw=1.6, label='evapotranspiration')
    ax[0].plot(td, tp_d, color=colors[2], lw=1.6, label='transpiration (root uptake demand)')
    ax[0].set_ylabel('water flux (mm/d)')
    ax[0].legend(frameon=False, fontsize=9)
    ax[1].plot(t_h, zb, color='k', lw=1.6)
    ax[1].invert_yaxis()
    ax[1].set_ylabel('water-table depth (m)')
    for j, d in enumerate(plot_depths):
        ax[2].plot(t_h, th[:, depths.index(d)], color=colors[j % len(colors)], lw=1.4, label=f'{100 * d:.0f} cm')
    ax[2].set_ylabel('soil moisture (m$^3$/m$^3$)')
    ax[2].set_xlabel(f'time (days from {start})')
    ax[2].legend(frameon=False, fontsize=9, ncol=len(plot_depths))
    for a in ax:
        a.grid(True, color='#e1e0d9', lw=0.6)
        a.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--no_check', action='store_true', help='skip the solve with check_dt_hours steps')
    ap.add_argument('--out', default=None, help='output directory instead of output.dir of the config')
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.out:
        cfg['output']['dir'] = args.out
    fc, wc, sc = cfg['fem'], cfg['water_table'], cfg['sensors']
    soil = {k: float(cfg['soil'][k]) for k in ('theta_s', 'theta_r', 'alpha', 'n', 'Ks')}
    s = site_forcing(cfg)
    n_days, t_h, t_n = s['n_days'], s['t_h'], s['t_nodes']
    print(f"[syn] {cfg['site']['name']} {s['t0']} + {n_days} d", flush=True)
    for k, v in s['info'].items():
        print(f"[syn] {k}: {v}", flush=True)

    offset = float(wc.get('offset_m', 0.0))                 # the record lowered by a constant, shape unchanged
    sp = CubicBSpline1D(0.0, n_days, wc['knot_days'])
    path = ZbSplineNumpy.fit(sp, t_h, s['zb_record'] + offset, wc['zb_min'])
    zb = path.zb(t_h)
    if offset:
        print(f"[syn] water-table record lowered by {offset:.3f} m", flush=True)
    depths_req = [float(d) for d in cfg['sensors']['depths_m']]
    if cfg['sensors'].get('require_above_water_table', False) and max(depths_req) >= zb.min():
        raise SystemExit(f"deepest sensor {max(depths_req):.2f} m is not above the water table (min {zb.min():.3f} m)")
    print(f"[syn] water table (spline, {sp.n_coef} coefficients) {zb.min():.3f} .. {zb.max():.3f} m; "
          f"rms minus the record {np.sqrt(np.mean((zb - s['zb_record']) ** 2)):.4f} m", flush=True)

    ft = gldas_fraction(cfg['site']['gldas_csv'], s['dates'])
    q_n = (1.0 - ft) * s['et'] - s['p']                     # m/s, + up
    tp_n = ft * s['et'] * 86400e3                            # mm/d
    Solver, Sink = load_fem(fc['root'])
    on = lambda t, y: np.interp(t, t_n, y)
    root_beta = (cfg.get('roots') or {}).get('beta')                 # None: the solver's default (0.96)
    if root_beta is not None:
        print(f"[syn] root distribution beta {float(root_beta)} (truth)", flush=True)
    profile = cfg.get('soil_profile')                                # None: uniform soil (the standard solver)
    if profile is not None:
        from wpinn.fem_hetero import run_fem_hetero
        print(f"[syn] soil varying with depth (truth): {profile}", flush=True)
        run = lambda t, z, q, tp: run_fem_hetero(fc['root'], fc, soil, profile, t, z, q, tp, fc['n_nodes'],
                                                 root_beta=root_beta)
    else:
        run = lambda t, z, q, tp: run_fem(Solver, Sink, fc, soil, t, z, q, tp, fc['n_nodes'], root_beta=root_beta)
    A = run(t_h, zb[None], on(t_h, q_n)[None], on(t_h, tp_n)[None])
    depths = [float(d) for d in sc['depths_m']]
    th = point_sensors(A['xi'], A['theta'][0], zb, depths, soil['theta_s'])
    if args.no_check:
        C, th_c = {'converged': np.full(1, np.nan)}, np.full_like(th, np.nan)
    else:
        t_c = hourly_times(n_days, fc['check_dt_hours'])
        C = run(t_c, path.zb(t_c)[None], on(t_c, q_n)[None], on(t_c, tp_n)[None])
        k_c = np.round(t_h / (fc['check_dt_hours'] / 24.0)).astype(int)
        th_c = point_sensors(C['xi'], C['theta'][0][k_c], zb, depths, soil['theta_s'])
    below = np.array([[d >= z for d in depths] for z in zb])

    out = cfg['output']['dir']
    os.makedirs(out, exist_ok=True)
    shutil.copy(args.config, os.path.join(out, 'config.yaml'))
    np.savez(os.path.join(out, 'synthetic.npz'),
             t_days=t_h, start=str(s['t0']), zb=zb, zb_coef=path.coef, zb_min=path.zb_min, knot_days=sp.dt,
             zb_record=s['zb_record'], zb_offset_m=offset, t_nodes=t_n, q_nodes=q_n, tp_nodes_mmd=tp_n, et_nodes=s['et'],
             p_nodes=s['p'], ft_nodes=ft, xi=A['xi'], theta=A['theta'][0], h=A['h'][0],
             sensor_depths_m=np.array(depths), theta_sensors=th, theta_sensors_check=th_c,
             sensor_below_water_table=below, converged=A['converged'][0],
             soil=np.array([soil[k] for k in ('theta_s', 'theta_r', 'alpha', 'n', 'Ks')]))
    rain_d, et_d, tp_d = daily(t_n, s['p'], n_days), daily(t_n, s['et'], n_days), daily(t_n, tp_n / 86400e3, n_days)
    summary = {'site': cfg['site']['name'], 'start': str(s['t0']), 'n_days': n_days, **s['info'],
               'zb_range_m': [float(zb.min()), float(zb.max())], 'zb_offset_m': offset, 'n_coef': sp.n_coef,
               'root_beta': None if root_beta is None else float(root_beta), 'soil_profile': profile,
               'deepest_sensor_min_gap_m': float(zb.min() - max(depths_req)),
               'rain_mm': float(rain_d.sum()), 'et_mm': float(et_d.sum()), 'tp_mm': float(tp_d.sum()),
               'ft_mean': float(ft.mean()), 'fem_converged_frac': float(A['converged'].mean()),
               'check_converged_frac': float(C['converged'].mean()),
               'surface_head_min_m': float(A['h'][0][:, -1].min()), 'sensors': {}}
    for j, d in enumerate(depths):
        summary['sensors'][f'{100 * d:.0f} cm'] = {
            'theta_range': [float(th[:, j].min()), float(th[:, j].max())],
            'hours_below_water_table_frac': float(below[:, j].mean()),
            'step_check_max_abs': float(np.abs(th[:, j] - th_c[:, j]).max()),
            'step_check_rms': float(np.sqrt(np.mean((th[:, j] - th_c[:, j]) ** 2)))}
        v = summary['sensors'][f'{100 * d:.0f} cm']
        print(f"[syn] sensor {100 * d:3.0f} cm: theta {v['theta_range'][0]:.4f} .. {v['theta_range'][1]:.4f}; below "
              f"the water table {100 * v['hours_below_water_table_frac']:.1f}% of hours; 1 h minus "
              f"{fc['check_dt_hours']} h step: rms {v['step_check_rms']:.2e}, max {v['step_check_max_abs']:.2e}",
              flush=True)
    with open(os.path.join(out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    plot(os.path.join(out, 'synthetic.png'), t_h, n_days, rain_d, et_d, tp_d, zb, depths, th,
         [float(d) for d in sc['plot_depths_m']], str(s['t0'])[:10])
    print(f"[syn] rain {summary['rain_mm']:.0f} mm, ET {summary['et_mm']:.0f} mm, T_p {summary['tp_mm']:.0f} mm; "
          f"saved {out}/synthetic.npz, summary.json, synthetic.png", flush=True)


if __name__ == '__main__':
    main()
