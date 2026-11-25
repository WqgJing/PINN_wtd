"""Water-table paths of several runs against the truth (one panel), optionally a second panel with the readings of one
sensor as used by a run (to show the sampling). Lines only; markers only where readings are sparse.

Usage (via sbatch): hpc/run_script_cpu.slurm scripts/plot_paths.py --out <png> --run label=dir [--run ...] \
        [--ylim 0.6,1.2] [--readings dir --sensor 3]
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = ['#eb6834', '#2a78d6', '#1baf7a', '#6250d6', '#eda100', '#898781']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--run', action='append', required=True, help='label=run directory (predictions.npz)')
    ap.add_argument('--ylim', default=None, help='min,max water-table depth (m) for the axis')
    ap.add_argument('--readings', default=None, help='run directory whose readings are drawn in a second panel')
    ap.add_argument('--sensor', type=int, default=3, help='sensor index for --readings')
    args = ap.parse_args()
    two = args.readings is not None
    fig, axes = plt.subplots(2 if two else 1, 1, figsize=(9, 7 if two else 4.2), sharex=True, squeeze=False)
    ax = axes[0, 0]
    truth_drawn = False
    for c, item in zip(COLORS, args.run):
        label, path = item.split('=', 1)
        p = np.load(os.path.join(path, 'predictions.npz'))
        ta = p['t_days'] + float(p['t_offset'])
        if not truth_drawn:
            ax.plot(ta, p['zb_true'], color='k', lw=2.0, label='truth')
            truth_drawn = True
        ax.plot(ta, p['zb_est'], color=c, lw=1.6, label=label)
    ax.invert_yaxis()
    if args.ylim:
        lo, hi = (float(v) for v in args.ylim.split(','))
        ax.set_ylim(hi, lo)
    ax.set_ylabel('water-table depth (m)')
    ax.legend(frameon=False, fontsize=9)
    if two:
        p = np.load(os.path.join(args.readings, 'predictions.npz'))
        ta = p['t_days'] + float(p['t_offset'])
        y = p['theta_obs'][:, args.sensor]
        d = p['depths_m'][args.sensor]
        a2 = axes[1, 0]
        a2.plot(ta, p['theta_true'][:, args.sensor], color='k', lw=1.2, label=f'truth, {100 * d:.0f} cm')
        ok = np.isfinite(y)
        a2.plot(ta[ok], y[ok], color=COLORS[0], lw=0.0, marker='o', ms=3, label=f'readings used, {100 * d:.0f} cm')
        a2.set_ylabel('soil moisture (m$^3$/m$^3$)')
        a2.legend(frameon=False, fontsize=9)
    axes[-1, 0].set_xlabel('time (days from the start of the record)')
    for a in axes.ravel():
        a.grid(True, color='#e1e0d9', lw=0.6)
        a.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.savefig(args.out, dpi=130)
    plt.close(fig)
    print(f"[paths] wrote {args.out}", flush=True)


if __name__ == '__main__':
    main()
