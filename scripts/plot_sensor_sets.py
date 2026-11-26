"""Water-table time series of the fixed PINN and FEM for every 3- and 2-sensor set (shallow record), from the constant
starts 1 m and 2 m, against the truth. One panel per sensor set (named in its axis label). Lines only.

Usage (via sbatch): hpc/run_script_cpu.slurm scripts/plot_sensor_sets.py --out output/compare/sensor_sets_timeseries.png
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SETS = [(3, '5-20-40'), (3, '5-20-60'), (3, '5-40-60'), (3, '20-40-60'),
        (2, '5-20'), (2, '5-40'), (2, '5-60'), (2, '20-40'), (2, '20-60'), (2, '40-60')]
LINES = [('s', '1m', '#eb6834', 'PINN, start 1 m'), ('s', '2m', '#f2a65a', 'PINN, start 2 m'),
         ('fs', '1m', '#2a78d6', 'FEM, start 1 m'), ('fs', '2m', '#86b6ec', 'FEM, start 2 m')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    fig, axes = plt.subplots(5, 2, figsize=(13, 15), sharex=True)
    for k, (ax, (n, st)) in enumerate(zip(axes.ravel(), SETS)):
        truth_drawn = False
        for pre, s0, col, name in LINES:
            f = os.path.join('runs', f'{pre}{n}_{st}_{s0}', 'predictions.npz')
            if not os.path.exists(f):
                continue
            p = np.load(f)
            ta = p['t_days'] + float(p['t_offset'])
            if not truth_drawn:
                ax.plot(ta, p['zb_true'], color='k', lw=2.0, label='truth')
                truth_drawn = True
            ax.plot(ta, p['zb_est'], color=col, lw=1.4, label=name)
        ax.set_ylim(1.2, 0.5)
        ax.set_ylabel(f'{n} sensors, {st} cm\nwater-table depth (m)', fontsize=9)
        ax.grid(True, color='#e1e0d9', lw=0.6)
        ax.spines[['top', 'right']].set_visible(False)
        if k == 0:
            ax.legend(frameon=False, fontsize=8, ncol=2, loc='lower left')
    for ax in axes[-1]:
        ax.set_xlabel('time (days from the start of the record)')
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=120)
    plt.close(fig)
    print(f"[sensor-sets] wrote {args.out}", flush=True)


if __name__ == '__main__':
    main()
