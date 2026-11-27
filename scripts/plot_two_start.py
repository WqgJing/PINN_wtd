"""Multi-start identifiability check: for each case, the final water-table paths from several starts against the truth.
The first run of a case is the reference; per day, another run agrees with it if their daily means are within --tol.
Prints, per run, how far it moved from its own start, its error against the truth, its final sensor misfit and the share
of days it agrees with the reference. Dashed lines are the starting paths, solid lines the results. Lines only.

Usage (via sbatch): hpc/run_script_cpu.slurm scripts/plot_two_start.py --out output/compare/two_start.png \
        --case "shallow 0.7-1.1 m=runs/place_4_5-20-40-60,runs/ts_shallow_b1_4,runs/ts_shallow_b2_4" --case ...
"""
import argparse
import csv
import json
import os

import sys

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from wpinn.spline import CubicBSpline1D, ZbSplineNumpy   # noqa: E402

COLORS = {'hydrostatic': '#eb6834', 'hydrostatic+0.5': '#2a78d6', 'constant:1.1': '#1baf7a', 'constant:1.5': '#c2185b',
          'constant:1.0': '#2a78d6', 'constant:2.0': '#1baf7a',
          'constant:3.0': '#8a4fd1', 'constant:5.0': '#5d6d7e', 'constant:10.0': '#a0522d'}


def run_config(run):
    with open(os.path.join(run, 'config.yaml')) as f:
        return yaml.safe_load(f)


def start_path(cfg, p):
    """The run's starting water-table path, rebuilt from coef_start with the run's own spline settings."""
    w, zs = cfg['window'], cfg['zb_spline']
    sp = CubicBSpline1D(0.0, float(w['end_day']) - float(w['start_day']), float(zs['knot_days']))
    return ZbSplineNumpy(sp, p['coef_start'], float(zs['zb_min'])).zb(p['t_days'])


def start_label(init):
    if init == 'hydrostatic':
        return 'at-rest reading'
    if init.startswith('hydrostatic+'):
        return f"at-rest reading + {init.split('+', 1)[1]} m"
    if init.startswith('constant:'):
        return f"{init.split(':', 1)[1]} m"
    return init


def final_misfit(run):
    with open(os.path.join(run, 'log.csv')) as f:
        rows = list(csv.DictReader(f))
    return float(rows[-1]['data'])


def daily(t, z):
    d = np.floor(t + 1e-9).astype(int)
    return np.array([z[d == k].mean() for k in np.unique(d)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--case', action='append', required=True, help='label=runRef,run2,...')
    ap.add_argument('--tol', type=float, default=0.10, help='m')
    args = ap.parse_args()
    fig, axes = plt.subplots(len(args.case), 1, figsize=(9, 3.6 * len(args.case)), sharex=True, squeeze=False)
    report = {}
    for ax, item in zip(axes[:, 0], args.case):
        label, runs = item.split('=', 1)
        runs = runs.split(',')
        rep, ref_daily = {}, None
        p0 = dict(np.load(os.path.join(runs[0], 'predictions.npz')))
        ta = p0['t_days'] + float(p0['t_offset'])
        ax.plot(ta, p0['zb_true'], color='k', lw=2.0, label='truth')
        for run in runs:
            cfg = run_config(run)
            init = str(cfg['init'])
            p = dict(np.load(os.path.join(run, 'predictions.npz')))
            z0, z = start_path(cfg, p), p['zb_est']
            dz = daily(p['t_days'], z)
            if ref_daily is None:
                ref_daily = dz
            r = {'init': init, 'moved_from_own_start_m': float(np.mean(z - z0)),
                 'rmse_vs_truth_m': float(np.sqrt(np.mean((z - p['zb_true']) ** 2))),
                 'final_data_misfit': final_misfit(run),
                 'share_days_agree_with_ref': float(np.mean(np.abs(dz - ref_daily) <= args.tol))}
            rep[run] = r
            print(f"[multi-start] {label} | {run} ({init}): moved {100 * r['moved_from_own_start_m']:+.1f} cm, rmse "
                  f"{100 * r['rmse_vs_truth_m']:.1f} cm, misfit {r['final_data_misfit']:.3f}, agrees with ref on "
                  f"{100 * r['share_days_agree_with_ref']:.0f} % of days", flush=True)
            col = COLORS.get(init, '#7f7f7f')
            ax.plot(ta, z, color=col, lw=1.6, label=f'result from start {start_label(init)}')
            ax.plot(ta, z0, color=col, lw=1.0, ls='--', label=f'start {start_label(init)}')
        report[label] = rep
        ax.invert_yaxis()
        ax.set_ylabel(f'{label}\nwater-table depth (m)', fontsize=9)
        ax.grid(True, color='#e1e0d9', lw=0.6)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(frameon=False, fontsize=7, ncol=2)
    axes[-1, 0].set_xlabel('time (days from the start of the record)')
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    plt.close(fig)
    with open(os.path.splitext(args.out)[0] + '.json', 'w') as f:
        json.dump(report, f, indent=1)
    print(f"[multi-start] wrote {args.out}", flush=True)


if __name__ == '__main__':
    main()
