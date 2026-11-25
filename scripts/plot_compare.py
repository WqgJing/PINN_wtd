"""PINN against the classical FEM inversion on the same data: the water-table paths (top) and the water-table error
against wall-clock time (bottom; PINN from log.csv every eval_every steps, FEM from its summary history per step).

Usage (via sbatch): hpc/run_script_cpu.slurm scripts/plot_compare.py --pinn runs/place_2_5-40 \
                        --fem runs/fem_2_5-40_smooth_tol1e-3 --out output/compare/compare_5-40.png
"""
import argparse
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pinn', required=True)
    ap.add_argument('--fem', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    p = np.load(os.path.join(args.pinn, 'predictions.npz'))
    f = np.load(os.path.join(args.fem, 'predictions.npz'))
    if not np.allclose(p['zb_true'], f['zb_true']):
        raise SystemExit('the two runs do not share the truth')
    with open(os.path.join(args.pinn, 'log.csv'), newline='') as fh:
        rows = list(csv.DictReader(fh))
    with open(os.path.join(args.fem, 'summary.json')) as fh:
        fs = json.load(fh)
    ta = p['t_days'] + float(p['t_offset'])
    start = float(np.sqrt(np.mean((f['zb_start'] - f['zb_true']) ** 2)))

    fig, ax = plt.subplots(2, 1, figsize=(9, 7))
    ax[0].plot(ta, p['zb_true'], color='k', lw=2.0, label='truth')
    ax[0].plot(ta, p['zb_est'], color='#eb6834', lw=1.6, label='PINN')
    ax[0].plot(ta, f['zb_est'], color='#2a78d6', lw=1.6, label='FEM inversion')
    ax[0].invert_yaxis()
    ax[0].set_ylabel('water-table depth (m)')
    ax[0].set_xlabel('time (days from the start of the record)')
    ax[0].legend(frameon=False, fontsize=9)

    ax[1].plot([float(r['wall_s']) / 60 for r in rows], [100 * float(r['zb_rmse_m']) for r in rows],
               color='#eb6834', lw=1.6, label='PINN (1 GPU)')
    ax[1].plot([0.0] + [h['wall_s'] / 60 for h in fs['history']], [100 * start] + [100 * h['zb_rmse_m'] for h in
               fs['history']], color='#2a78d6', lw=1.6, marker='o', ms=4, label='FEM inversion (8 CPU cores)')
    ax[1].set_xlabel('wall-clock time (min)')
    ax[1].set_ylabel('water-table RMSE vs truth (cm)')
    ax[1].legend(frameon=False, fontsize=9)
    for a in ax:
        a.grid(True, color='#e1e0d9', lw=0.6)
        a.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    fig.savefig(args.out, dpi=130)
    plt.close(fig)
    print(f"[compare] wrote {args.out}", flush=True)


if __name__ == '__main__':
    main()
