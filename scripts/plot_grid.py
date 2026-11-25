"""Grid of water-table time series (truth, PINN, FEM inversion) for representative conditions of the 4-sensor tests,
noise realisation 0. The condition is named in each panel's axis label. Lines only.

Usage (via sbatch): hpc/run_script_cpu.slurm scripts/plot_grid.py --out output/compare/grid_timeseries.png
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PANELS = [  # label, PINN run, FEM run, y-limits (shallow, deep) in m
    ('clean data', 'runs/place_4_5-20-40-60', 'runs/fem_4_clean', (0.55, 1.2)),
    ('top offset +0.03', 'runs/off_p0_4', 'runs/fem_off_p0_4', (0.55, 1.45)),
    ('top drift to +0.03', 'runs/drift_p0_4', 'runs/fem_drift_p0_4', (0.55, 1.35)),
    ('top clock +6 h', 'runs/t3_p6_s0_4', 'runs/fem_t3_p6_s0_4', (0.55, 1.2)),
    ('random errors, all sensors', 'runs/c2_all_r0_4', 'runs/fem_c2_all_r0_4', (0.4, 1.4)),
    ('daily means', 'runs/t4_dm_s0_4', 'runs/fem_t4_dm_s0_4', (0.55, 1.4)),
    ('wrong physics: two layers', 'runs/wp_layer_s0_4', 'runs/fem_wp_layer_s0_4', (0.55, 1.2)),
    ('water table 4.2-4.8 m', 'runs/wt_deep395_s0_4', 'runs/fem_wt_deep395_s0_4', (1.5, 8.0)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    fig, axes = plt.subplots(4, 2, figsize=(12, 12), sharex=True)
    for k, (ax, (label, pinn, fem, ylim)) in enumerate(zip(axes.ravel(), PANELS)):
        p = np.load(os.path.join(pinn, 'predictions.npz'))
        ta = p['t_days'] + float(p['t_offset'])
        ax.plot(ta, p['zb_true'], color='k', lw=2.0, label='truth')
        ax.plot(ta, p['zb_est'], color='#eb6834', lw=1.6, label='PINN')
        f = os.path.join(fem, 'predictions.npz')
        if os.path.exists(f):
            ax.plot(ta, np.load(f)['zb_est'], color='#2a78d6', lw=1.6, label='FEM inversion')
        ax.set_ylim(ylim[1], ylim[0])
        ax.set_ylabel(f'{label}\nwater-table depth (m)', fontsize=9)
        ax.grid(True, color='#e1e0d9', lw=0.6)
        ax.spines[['top', 'right']].set_visible(False)
        if k == 0:
            ax.legend(frameon=False, fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel('time (days from the start of the record)')
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=120)
    plt.close(fig)
    print(f"[grid] wrote {args.out}", flush=True)


if __name__ == '__main__':
    main()
