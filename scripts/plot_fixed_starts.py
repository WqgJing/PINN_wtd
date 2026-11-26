"""Shallow case after the fixes: water-table results from each start (solid) and the starts (dashed) against the truth.
Columns PINN, FEM (finished runs only); rows: full depth range, and only the runs that ended near the truth. Lines only.

Usage (via sbatch): hpc/run_script_cpu.slurm scripts/plot_fixed_starts.py --out output/compare/fixed_starts_shallow.png
"""
import argparse
import os
import sys

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from wpinn.spline import CubicBSpline1D, ZbSplineNumpy   # noqa: E402

STARTS = [('at-rest reading', '#eb6834', 'f2_A', 'fem2_A'), ('1.5 m', '#c2185b', 'f2_1p5', 'fem2_1p5'),
          ('3 m', '#8a4fd1', 'f2_3m', 'fem2_3m'), ('5 m', '#5d6d7e', 'f2_5m', 'fem2_5m'),
          ('10 m', '#a0522d', 'f2_10m', 'fem2_10m')]


def pinn_paths(run):
    p = np.load(os.path.join(run, 'predictions.npz'))
    with open(os.path.join(run, 'config.yaml')) as f:
        cfg = yaml.safe_load(f)
    w, zs = cfg['window'], cfg['zb_spline']
    sp = CubicBSpline1D(0.0, float(w['end_day']) - float(w['start_day']), float(zs['knot_days']))
    z0 = ZbSplineNumpy(sp, p['coef_start'], float(zs['zb_min'])).zb(p['t_days'])
    return p['t_days'] + float(p['t_offset']), p['zb_true'], z0, p['zb_est']


def fem_paths(run):
    p = np.load(os.path.join(run, 'predictions.npz'))
    return p['t_days'] + float(p['t_offset']), p['zb_true'], p['zb_start'], p['zb_est']


def panel(ax, loader, which, ylim, ylabel, only_in_range=False):
    truth_drawn = False
    for label, col, pinn, fem in STARTS:
        run = os.path.join('runs', pinn if which == 'pinn' else fem)
        if not os.path.exists(os.path.join(run, 'predictions.npz')):
            continue
        ta, zt, z0, z = loader(run)
        if only_in_range and not (ylim[0] <= z.min() and z.max() <= ylim[1]):
            continue
        if not truth_drawn:
            ax.plot(ta, zt, color='k', lw=2.0, label='truth')
            truth_drawn = True
        ax.plot(ta, z, color=col, lw=1.6, label=f'result from {label}')
        ax.plot(ta, z0, color=col, lw=1.0, ls='--', label=f'start {label}')
    ax.set_ylim(ylim[1], ylim[0])
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, color='#e1e0d9', lw=0.6)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc='lower right')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    panel(axes[0, 0], pinn_paths, 'pinn', (0.0, 11.5), 'PINN after the fixes\nwater-table depth (m)')
    panel(axes[0, 1], fem_paths, 'fem', (0.0, 11.5), 'FEM after the fixes (finished runs)\nwater-table depth (m)')
    panel(axes[1, 0], pinn_paths, 'pinn', (0.6, 1.6), 'PINN, runs that reached the truth\nwater-table depth (m)',
          only_in_range=True)
    panel(axes[1, 1], fem_paths, 'fem', (0.6, 1.6), 'FEM, runs that reached the truth\nwater-table depth (m)',
          only_in_range=True)
    for ax in axes[-1]:
        ax.set_xlabel('time (days from the start of the record)')
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    plt.close(fig)
    print(f"[fixed-starts] wrote {args.out}", flush=True)


if __name__ == '__main__':
    main()
