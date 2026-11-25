"""Test #2 summary: empirical cumulative distribution of the water-table RMSE over the 8 realisations, per version
(rows: TOP, INTERIOR, ALL) and method (PINN, FEM-std, FEM-est). A fit without a summary (diverged, stopped) is drawn at
the right edge (cap). Lines only.

Usage (via sbatch): hpc/run_script_cpu.slurm scripts/plot_c2.py --out output/compare/c2_cdf.png
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS = {'top': 'top sensor contaminated', 'int': 'interior sensors contaminated', 'all': 'all sensors contaminated'}


def rmse(path, cap):
    f = os.path.join(path, 'summary.json')
    if not os.path.exists(f):
        return cap
    with open(f) as fh:
        return min(100.0 * json.load(fh)['zb_vs_truth']['rmse_m'], cap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--cap', type=float, default=300.0, help='cm; diverged or larger fits are drawn here')
    args = ap.parse_args()
    fig, ax = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
    for a, v in zip(ax, ['top', 'int', 'all']):
        for name, pat, col in [('PINN', 'runs/c2_{v}_r{r}_4', '#eb6834'),
                               ('FEM, top sensor trusted', 'runs/fem_c2_{v}_r{r}_4', '#2a78d6'),
                               ('FEM, top offset and drift fitted', 'runs/fest_c2_{v}_r{r}_4', '#1baf7a')]:
            e = np.sort([rmse(pat.format(v=v, r=r), args.cap) for r in range(8)])
            a.step(np.concatenate([[e[0]], e]), np.arange(0, 9) / 8.0, where='post', color=col, lw=1.6, label=name)
        a.set_xscale('log')
        a.set_ylabel(f'fraction of realisations\n({LABELS[v]})')
        a.grid(True, color='#e1e0d9', lw=0.6)
        a.spines[['top', 'right']].set_visible(False)
    ax[0].legend(frameon=False, fontsize=9)
    ax[-1].set_xlabel('water-table RMSE vs truth (cm)')
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    plt.close(fig)
    print(f"[c2] wrote {args.out}", flush=True)


if __name__ == '__main__':
    main()
