"""Real-data runs: water table against the well (RMSE, bias, correlation of the hourly series and of their changes,
start agreement) and the fit of each sensor (RMSE and mean residual per depth), with a figure: water table (well and
runs) and, per sensor, the readings and each run's fit. Lines only.

Usage (via sbatch): hpc/run_script_cpu.slurm scripts/analyze_site_runs.py --runs runs/hb2_1m,runs/hb2_2m,runs/hb2_fixed \
        --out output/sites/hb2_runs
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = ['#eb6834', '#2a78d6', '#1baf7a', '#8a4fd1']


def daily(t, z):
    d = np.floor(t + 1e-9).astype(int)
    return np.array([z[d == k].mean() for k in np.unique(d)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--out', required=True, help='path without extension')
    args = ap.parse_args()
    runs = args.runs.split(',')
    P = {r: dict(np.load(os.path.join(r, 'predictions.npz'))) for r in runs}
    p0 = P[runs[0]]
    t = p0['t_days'] + float(p0['t_offset'])
    depths = [float(d) for d in p0['depths_m']]
    obs = p0['theta_obs']
    report = {}
    for r, p in P.items():
        e = p['zb_est'] - p['zb_true']
        dz, dzt = np.diff(p['zb_est']), np.diff(p['zb_true'])
        res = p['theta_est'] - p['theta_obs']
        report[r] = {'zb_rmse_m': float(np.sqrt(np.mean(e ** 2))), 'zb_bias_m': float(e.mean()),
                     'zb_r': float(np.corrcoef(p['zb_est'], p['zb_true'])[0, 1]),
                     'zb_change_r': float(np.corrcoef(dz, dzt)[0, 1]),
                     'zb_rmse_after_bias_m': float(np.sqrt(np.mean((e - e.mean()) ** 2))),
                     'zb_range_est_m': float(np.ptp(p['zb_est'])), 'zb_range_well_m': float(np.ptp(p['zb_true'])),
                     'zb_slope_on_well': float(np.polyfit(p['zb_true'], p['zb_est'], 1)[0]),
                     'sensor_rmse': {f'{100 * d:.0f} cm': float(np.sqrt(np.nanmean(res[:, j] ** 2)))
                                     for j, d in enumerate(depths)},
                     'sensor_mean_residual': {f'{100 * d:.0f} cm': float(np.nanmean(res[:, j]))
                                              for j, d in enumerate(depths)}}
        rp = report[r]
        print(f"[site-runs] {r}: water table RMSE {100 * rp['zb_rmse_m']:.1f} cm, bias {100 * rp['zb_bias_m']:+.1f} cm, "
              f"r {rp['zb_r']:.3f}, r of changes {rp['zb_change_r']:.3f}, RMSE after removing the bias "
              f"{100 * rp['zb_rmse_after_bias_m']:.1f} cm; range {100 * rp['zb_range_est_m']:.1f} cm vs the well's "
              f"{100 * rp['zb_range_well_m']:.1f} cm, slope on the well {rp['zb_slope_on_well']:.2f}", flush=True)
        print(f"[site-runs]   sensor RMSE: " + ', '.join(f"{k} {v:.3f}" for k, v in rp['sensor_rmse'].items()), flush=True)
        print(f"[site-runs]   mean residual (model - reading): " +
              ', '.join(f"{k} {v:+.3f}" for k, v in rp['sensor_mean_residual'].items()), flush=True)
    inv = [r for r in runs if 'fixed' not in r]
    if len(inv) == 2:
        a, b = (daily(P[r]['t_days'], P[r]['zb_est']) for r in inv)
        report['start_agreement_share_days_5cm'] = float(np.mean(np.abs(a - b) <= 0.05))
        print(f"[site-runs] starts agree within 5 cm on {100 * report['start_agreement_share_days_5cm']:.0f} % of days",
              flush=True)
    with open(args.out + '.json', 'w') as f:
        json.dump(report, f, indent=1)

    n = len(depths)
    fig, axes = plt.subplots(n + 1, 1, figsize=(12, 2.1 * (n + 1)), sharex=True)
    axes[0].plot(t, p0['zb_true'], color='k', lw=2.0, label='well')
    for r, col in zip(runs, COLORS):
        axes[0].plot(t, P[r]['zb_est'], color=col, lw=1.4, label=os.path.basename(r))
    axes[0].invert_yaxis()
    axes[0].set_ylabel('water-table\ndepth (m)', fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, ncol=4)
    for j, d in enumerate(depths):
        ax = axes[j + 1]
        ax.plot(t, obs[:, j], color='k', lw=1.2, label='reading')
        for r, col in zip(runs, COLORS):
            ax.plot(t, P[r]['theta_est'][:, j], color=col, lw=1.0, label=os.path.basename(r))
        ax.set_ylabel(f'{100 * d:.0f} cm\n(m$^3$/m$^3$)', fontsize=9)
        if j == 0:
            ax.legend(frameon=False, fontsize=8, ncol=4)
    axes[-1].set_xlabel('time (days from the start of the record)')
    for ax in axes:
        ax.grid(True, color='#e1e0d9', lw=0.6)
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out + '.png', dpi=110)
    plt.close(fig)
    print(f"[site-runs] wrote {args.out}.json / .png", flush=True)


if __name__ == '__main__':
    main()
