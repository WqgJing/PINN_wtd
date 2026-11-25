"""Overview of the 4-sensor tests: water-table RMSE of the PINN and the standard FEM inversion per condition (median
over noise realisations as a marker, min-max as a line; a run without a summary - diverged, timed out or stopped - and
any error above the cap are drawn at the cap). Lines and markers only.

Usage (via sbatch): hpc/run_script_cpu.slurm scripts/plot_overview.py --out output/compare/overview_4sensors.png
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

S3 = [0, 1, 2]
R8 = range(8)
CONDITIONS = [  # label, PINN runs, FEM runs
    ('clean data', ['runs/place_4_5-20-40-60'], ['runs/fem_4_clean']),
    ('top offset +0.03', [f'runs/off_p{s}_4' for s in S3], [f'runs/fem_off_p{s}_4' for s in S3]),
    ('top offset -0.03', [f'runs/off_m{s}_4' for s in S3], [f'runs/fem_off_m{s}_4' for s in S3]),
    ('top drift to +0.03', [f'runs/drift_p{s}_4' for s in S3], [f'runs/fem_drift_p{s}_4' for s in S3]),
    ('top drift to -0.03', [f'runs/drift_m{s}_4' for s in S3], [f'runs/fem_drift_m{s}_4' for s in S3]),
    ('top clock -12 h', [f'runs/t3_m12_s{s}_4' for s in S3], [f'runs/fem_t3_m12_s{s}_4' for s in S3]),
    ('top clock -6 h', [f'runs/t3_m6_s{s}_4' for s in S3], [f'runs/fem_t3_m6_s{s}_4' for s in S3]),
    ('top clock -3 h', [f'runs/t3_m3_s{s}_4' for s in S3], [f'runs/fem_t3_m3_s{s}_4' for s in S3]),
    ('top clock +3 h', [f'runs/t3_p3_s{s}_4' for s in S3], [f'runs/fem_t3_p3_s{s}_4' for s in S3]),
    ('top clock +6 h', [f'runs/t3_p6_s{s}_4' for s in S3], [f'runs/fem_t3_p6_s{s}_4' for s in S3]),
    ('top clock +12 h', [f'runs/t3_p12_s{s}_4' for s in S3], [f'runs/fem_t3_p12_s{s}_4' for s in S3]),
    ('random errors, top sensor', [f'runs/c2_top_r{r}_4' for r in R8], [f'runs/fem_c2_top_r{r}_4' for r in R8]),
    ('random errors, interior sensors', [f'runs/c2_int_r{r}_4' for r in R8], [f'runs/fem_c2_int_r{r}_4' for r in R8]),
    ('random errors, all sensors', [f'runs/c2_all_r{r}_4' for r in R8], [f'runs/fem_c2_all_r{r}_4' for r in R8]),
    ('daily means', [f'runs/t4_dm_s{s}_4' for s in S3], [f'runs/fem_t4_dm_s{s}_4' for s in S3]),
    ('one reading per day', [f'runs/t4_d1_s{s}_4' for s in S3], [f'runs/fem_t4_d1_s{s}_4' for s in S3]),
    ('one reading per week', [f'runs/t4_w1_s{s}_4' for s in S3], [f'runs/fem_t4_w1_s{s}_4' for s in S3]),
    ('wrong physics: two layers', [f'runs/wp_layer_s{s}_4' for s in S3], [f'runs/fem_wp_layer_s{s}_4' for s in S3]),
    ('wrong physics: Ks decays', [f'runs/wp_ksexp_s{s}_4' for s in S3], [f'runs/fem_wp_ksexp_s{s}_4' for s in S3]),
    ('wrong physics: deep roots', [f'runs/wp_rdeep_s{s}_4' for s in S3], [f'runs/fem_wp_rdeep_s{s}_4' for s in S3]),
    ('wrong physics: shallow roots', [f'runs/wp_rshal_s{s}_4' for s in S3], [f'runs/fem_wp_rshal_s{s}_4' for s in S3]),
    ('water table 2.2-2.9 m', ['runs/wt_edge195_s0_4'], ['runs/fem_wt_edge195_s0_4']),
    ('water table 4.2-4.8 m', ['runs/wt_deep395_s0_4'], ['runs/fem_wt_deep395_s0_4']),
]


def rmse_cm(path, cap):
    f = os.path.join(path, 'summary.json')
    if not os.path.exists(f):
        return cap
    with open(f) as fh:
        return min(100.0 * json.load(fh)['zb_vs_truth']['rmse_m'], cap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--cap', type=float, default=500.0)
    args = ap.parse_args()
    fig, ax = plt.subplots(figsize=(9, 10))
    labels = []
    for i, (label, pinn, fem) in enumerate(CONDITIONS):
        y = len(CONDITIONS) - 1 - i
        labels.append((y, label))
        for runs, col, dy, name in [(pinn, '#eb6834', 0.15, 'PINN'), (fem, '#2a78d6', -0.15, 'FEM inversion')]:
            e = np.array([rmse_cm(r, args.cap) for r in runs])
            ax.plot([e.min(), e.max()], [y + dy, y + dy], color=col, lw=1.6)
            ax.plot([np.median(e)], [y + dy], color=col, marker='o', ms=6, lw=0,
                    label=name if i == 0 else None)
    ax.set_yticks([y for y, _ in labels])
    ax.set_yticklabels([lab for _, lab in labels], fontsize=9)
    ax.set_xscale('log')
    ax.set_xlim(1.0, args.cap * 1.2)
    ax.set_xlabel('water-table RMSE vs truth (cm)')
    ax.legend(frameon=False, fontsize=9, loc='lower right')
    ax.grid(True, axis='x', color='#e1e0d9', lw=0.6)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    plt.close(fig)
    print(f"[overview] wrote {args.out}", flush=True)


if __name__ == '__main__':
    main()
