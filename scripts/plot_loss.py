"""Training curves of one run: the loss terms (top) and the water-table error against the truth (bottom).

Usage (via sbatch): hpc/run_script_cpu.slurm scripts/plot_loss.py --run runs/<name> [--out <png>]
Reads <run>/log.csv (written every eval_every steps by train_profile.py). val_* columns are not plotted: the data
term there uses the training sensor points, so it is not a held-out check.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_log(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    return {k: [float(r[k]) for r in rows] for k in rows[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    log = read_log(os.path.join(args.run, 'log.csv'))
    out = args.out or os.path.join(args.run, 'loss_curve.png')
    step = log['step']
    colors = ['#2a78d6', '#eb6834', '#1baf7a', '#898781']

    fig, ax = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
    ax[0].plot(step, log['data'], color=colors[0], lw=1.6, label='data, mean of (sensor misfit / noise sd)$^2$')
    ax[0].plot(step, log['pde'], color=colors[1], lw=1.6, label='PDE residual')
    ax[0].plot(step, log['wb'], color=colors[2], lw=1.6, label='water balance')
    ax[0].plot([step[0], step[-1]], [1.0, 1.0], color=colors[3], lw=1.0, ls='--', label='data at noise level')
    ax[0].set_yscale('log')
    ax[0].set_ylabel('scaled loss (-)')
    ax[0].legend(frameon=False, fontsize=9)

    ax[1].plot(step, [100 * v for v in log['zb_rmse_m']], color='k', lw=1.6)
    ax[1].set_ylabel('water-table RMSE vs truth (cm)')
    ax[1].set_xlabel('training step')
    for a in ax:
        a.grid(True, color='#e1e0d9', lw=0.6)
        a.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[plot_loss] wrote {out} ({len(step)} logged steps, last step {int(step[-1])})", flush=True)


if __name__ == '__main__':
    main()
