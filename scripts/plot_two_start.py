"""Two-start identifiability check: for each case, the final water-table paths from start A (at-rest reading of the
deepest sensor) and starts B1 / B2, against the truth; per day, A and B count as agreeing if their daily means are within
--tol. Prints the share of agreeing days and how far each run moved from its own start. Lines only.

Usage (via sbatch): hpc/run_script_cpu.slurm scripts/plot_two_start.py --out output/compare/two_start.png \
        --case shallow=runs/place_4_5-20-40-60,runs/ts_shallow_b1_4,runs/ts_shallow_b2_4 --case ...
"""
import argparse
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


def start_path(run, p):
    """The run's starting water-table path, rebuilt from coef_start with the run's own spline settings."""
    with open(os.path.join(run, 'config.yaml')) as f:
        cfg = yaml.safe_load(f)
    w, zs = cfg['window'], cfg['zb_spline']
    sp = CubicBSpline1D(0.0, float(w['end_day']) - float(w['start_day']), float(zs['knot_days']))
    return ZbSplineNumpy(sp, p['coef_start'], float(zs['zb_min'])).zb(p['t_days'])


def daily(t, z):
    d = np.floor(t + 1e-9).astype(int)
    return np.array([z[d == k].mean() for k in np.unique(d)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--case', action='append', required=True, help='label=runA,runB1,runB2')
    ap.add_argument('--tol', type=float, default=0.10, help='m')
    args = ap.parse_args()
    fig, axes = plt.subplots(len(args.case), 1, figsize=(9, 3.6 * len(args.case)), sharex=True, squeeze=False)
    report = {}
    for ax, item in zip(axes[:, 0], args.case):
        label, runs = item.split('=', 1)
        ra, rb1, rb2 = runs.split(',')
        pa, pb1, pb2 = (dict(np.load(os.path.join(r, 'predictions.npz'))) for r in (ra, rb1, rb2))
        for r, p in ((ra, pa), (rb1, pb1), (rb2, pb2)):
            p['zb_init'] = start_path(r, p)
        t = pa['t_days']
        ta = t + float(pa['t_offset'])
        za, zb1, zb2 = pa['zb_est'], pb1['zb_est'], pb2['zb_est']
        da, db1, db2 = daily(t, za), daily(t, zb1), daily(t, zb2)
        rep = {}
        for name, p, db in (('B1', pb1, db1), ('B2', pb2, db2)):
            rep[name] = {'share_days_agree': float(np.mean(np.abs(db - da) <= args.tol)),
                         'mean_abs_diff_m': float(np.mean(np.abs(db - da))),
                         'moved_from_own_start_m': float(np.mean(p['zb_est'] - p['zb_init'])),
                         'rmse_vs_truth_m': float(np.sqrt(np.mean((p['zb_est'] - p['zb_true']) ** 2)))}
        rep['A'] = {'moved_from_own_start_m': float(np.mean(pa['zb_est'] - pa['zb_init'])),
                    'rmse_vs_truth_m': float(np.sqrt(np.mean((za - pa['zb_true']) ** 2)))}
        report[label] = rep
        print(f"[two-start] {label}: A rmse {100 * rep['A']['rmse_vs_truth_m']:.1f} cm, moved "
              f"{100 * rep['A']['moved_from_own_start_m']:+.1f} cm | B1 agree {100 * rep['B1']['share_days_agree']:.0f} % "
              f"of days (mean |B1-A| {100 * rep['B1']['mean_abs_diff_m']:.1f} cm), moved "
              f"{100 * rep['B1']['moved_from_own_start_m']:+.1f} cm | B2 agree {100 * rep['B2']['share_days_agree']:.0f} % "
              f"(mean |B2-A| {100 * rep['B2']['mean_abs_diff_m']:.1f} cm), moved "
              f"{100 * rep['B2']['moved_from_own_start_m']:+.1f} cm", flush=True)
        ax.plot(ta, pa['zb_true'], color='k', lw=2.0, label='truth')
        ax.plot(ta, za, color='#eb6834', lw=1.6, label='start A: at-rest reading')
        ax.plot(ta, zb1, color='#2a78d6', lw=1.6, label='start B1: A + 0.5 m')
        ax.plot(ta, zb2, color='#1baf7a', lw=1.6, label='start B2: 1.1 m')
        ax.plot(ta, pb1['zb_init'], color='#2a78d6', lw=0.8, ls='--', label='B1 start')
        ax.plot(ta, pb2['zb_init'], color='#1baf7a', lw=0.8, ls='--', label='B2 start')
        ax.invert_yaxis()
        ax.set_ylabel(f'{label}\nwater-table depth (m)', fontsize=9)
        ax.grid(True, color='#e1e0d9', lw=0.6)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)
    axes[-1, 0].set_xlabel('time (days from the start of the record)')
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    plt.close(fig)
    with open(os.path.splitext(args.out)[0] + '.json', 'w') as f:
        json.dump(report, f, indent=1)
    print(f"[two-start] wrote {args.out}", flush=True)


if __name__ == '__main__':
    main()
