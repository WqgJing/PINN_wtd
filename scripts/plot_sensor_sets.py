"""Water-table time series of the fixed PINN and FEM for each sensor set, from the constant starts 1 m and 2 m, against
the truth. One panel per sensor set (named in its axis label). Lines only. Also prints, per set, each run's RMSE and
whether the two PINN starts agree (daily means within --tol on >= 90 % of days).

Usage (via sbatch): hpc/run_script_cpu.slurm scripts/plot_sensor_sets.py --out output/compare/sensor_sets_timeseries.png
        [--pinn_prefix s --fem_prefix fs --with4 --ylim 1.2 0.5]
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SETS4 = [(4, '5-20-40-60')]
SETS = [(3, '5-20-40'), (3, '5-20-60'), (3, '5-40-60'), (3, '20-40-60'),
        (2, '5-20'), (2, '5-40'), (2, '5-60'), (2, '20-40'), (2, '20-60'), (2, '40-60')]


def daily(t, z):
    d = np.floor(t + 1e-9).astype(int)
    return np.array([z[d == k].mean() for k in np.unique(d)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--pinn_prefix', default='s')
    ap.add_argument('--fem_prefix', default='fs')
    ap.add_argument('--with4', action='store_true', help='add the 4-sensor set as the first panel')
    ap.add_argument('--ylim', type=float, nargs=2, default=[1.2, 0.5], help='deep, shallow (m)')
    ap.add_argument('--tol', type=float, default=0.05, help='m, start agreement')
    args = ap.parse_args()
    sets = (SETS4 if args.with4 else []) + SETS
    lines = [(args.pinn_prefix, '1m', '#eb6834', 'PINN, start 1 m'), (args.pinn_prefix, '2m', '#f2a65a', 'PINN, start 2 m'),
             (args.fem_prefix, '1m', '#2a78d6', 'FEM, start 1 m'), (args.fem_prefix, '2m', '#86b6ec', 'FEM, start 2 m')]
    nrow = (len(sets) + 1) // 2
    fig, axes = plt.subplots(nrow, 2, figsize=(13, 3.0 * nrow), sharex=True, squeeze=False)
    for ax in axes.ravel()[len(sets):]:
        ax.set_visible(False)
    for k, (ax, (n, st)) in enumerate(zip(axes.ravel(), sets)):
        truth_drawn, pinn_daily, msg = False, [], []
        for pre, s0, col, name in lines:
            f = os.path.join('runs', f'{pre}{n}_{st}_{s0}', 'predictions.npz')
            if not os.path.exists(f):
                continue
            p = np.load(f)
            ta = p['t_days'] + float(p['t_offset'])
            if not truth_drawn:
                ax.plot(ta, p['zb_true'], color='k', lw=2.0, label='truth')
                truth_drawn = True
            ax.plot(ta, p['zb_est'], color=col, lw=1.4, label=name)
            msg.append(f"{name} {100 * np.sqrt(np.mean((p['zb_est'] - p['zb_true']) ** 2)):.1f} cm")
            if pre == args.pinn_prefix:
                pinn_daily.append(daily(p['t_days'], p['zb_est']))
        agree = (f"; PINN starts agree on {100 * np.mean(np.abs(pinn_daily[0] - pinn_daily[1]) <= args.tol):.0f} % "
                 f"of days") if len(pinn_daily) == 2 else ''
        print(f"[sensor-sets] {n} sensors {st}: " + ', '.join(msg) + agree, flush=True)
        ax.set_ylim(*args.ylim)
        ax.set_ylabel(f'{n} sensors, {st} cm\nwater-table depth (m)', fontsize=9)
        ax.grid(True, color='#e1e0d9', lw=0.6)
        ax.spines[['top', 'right']].set_visible(False)
        if k == 0:
            ax.legend(frameon=False, fontsize=8, ncol=2, loc='lower left')
    for j in range(2):
        last = max(i for i in range(len(sets)) if i % 2 == j)
        axes.ravel()[last].set_xlabel('time (days from the start of the record)')
        axes.ravel()[last].xaxis.set_tick_params(labelbottom=True)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=120)
    plt.close(fig)
    print(f"[sensor-sets] wrote {args.out}", flush=True)


if __name__ == '__main__':
    main()
