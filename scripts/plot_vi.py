"""Posterior of the variational PINN runs: (1) water table: truth, posterior median and 5 / 95 % lines per run;
(2) Ks: prior and posterior densities of log10 Ks per run, truth as a vertical line; (3) contraction of each spline
coefficient (1 - posterior / prior variance) per run. Lines only. Also prints the start-agreement rule (R3).

Usage (via sbatch): hpc/run_script_cpu.slurm scripts/plot_vi.py --runs runs/vi_ks_1m,runs/vi_ks_2m --out output/vi/vi_ks
"""
import argparse
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = ['#eb6834', '#2a78d6', '#1baf7a', '#8a4fd1']


def daily(t, z):
    d = np.floor(t + 1e-9).astype(int)
    return np.array([z[d == k].mean() for k in np.unique(d)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    runs = args.runs.split(',')
    P = {r: dict(np.load(os.path.join(r, 'posterior.npz'))) for r in runs}
    fig, ax = plt.subplots(3, 1, figsize=(11, 11), gridspec_kw={'height_ratios': [1.4, 1, 1]})
    p0 = P[runs[0]]
    ta = p0['t_days'] + float(p0['t_offset'])
    ax[0].plot(ta, p0['zb_true'], color='k', lw=2.0, label='truth')
    m = p0['mu'].size - 1
    for r, col in zip(runs, COLORS):
        p, name = P[r], os.path.basename(r)
        ax[0].plot(ta, p['zb_q50'], color=col, lw=1.5, label=f'{name} median')
        ax[0].plot(ta, p['zb_q05'], color=col, lw=0.8, ls='--', label=f'{name} 5 / 95 %')
        ax[0].plot(ta, p['zb_q95'], color=col, lw=0.8, ls='--')
        x = np.linspace(-7.5, -3.5, 400)                          # log10 Ks
        zx = (x * math.log(10.0)) - math.log(float(p['ks_ref']))    # z = ln(Ks / Ks_ref)
        sd_post, sd_prior = math.sqrt(p['cov_post'][m, m]), math.sqrt(p['cov_prior'][m, m])
        jac = math.log(10.0)
        post = jac * np.exp(-0.5 * ((zx - p['mu'][m]) / sd_post) ** 2) / (sd_post * math.sqrt(2 * math.pi))
        prior = jac * np.exp(-0.5 * (zx / sd_prior) ** 2) / (sd_prior * math.sqrt(2 * math.pi))
        ax[1].plot(x, post, color=col, lw=1.5, label=f'{name} posterior')
        if r == runs[0]:
            ax[1].plot(x, prior, color='#7f7f7f', lw=1.2, ls='--', label='prior')
            ax[1].axvline(math.log10(float(p['ks_true'])), color='k', lw=1.2, label='true Ks')
        contr = 1.0 - np.diag(p['cov_post'])[:m] / np.diag(p['cov_prior'])[:m]
        ax[2].plot(np.arange(m), contr, color=col, lw=1.5, marker='o', ms=3, label=name)
    ax[0].invert_yaxis()
    ax[0].set_ylabel('water-table depth (m)')
    ax[0].set_xlabel('time (days from the start of the record)')
    ax[1].set_xlabel('log10 Ks (Ks in m/s)')
    ax[1].set_ylabel('probability density (1 / log10 unit)')
    ax[2].set_xlabel('water-table spline coefficient (knot index, 1 per day)')
    ax[2].set_ylabel('contraction (1 - posterior / prior variance)')
    for a in ax:
        a.legend(frameon=False, fontsize=8, ncol=2)
        a.grid(True, color='#e1e0d9', lw=0.6)
        a.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out + '.png', dpi=120)
    plt.close(fig)
    rep = {r: json.load(open(os.path.join(r, 'summary.json'))) for r in runs}
    if len(runs) == 2:
        a, b = (daily(P[r]['t_days'], P[r]['zb_q50']) for r in runs)
        ka, kb = (rep[r]['ks_post'][1] for r in runs)
        r3 = bool(np.mean(np.abs(a - b) <= 0.05) >= 0.9 and max(ka, kb) / min(ka, kb) <= 1.5)
        print(f"[vi-plot] R3 start agreement: medians within 5 cm on {100 * np.mean(np.abs(a - b) <= 0.05):.0f} % of "
              f"days, Ks medians ratio {max(ka, kb) / min(ka, kb):.2f} -> {r3}", flush=True)
    for r in runs:
        s = rep[r]
        print(f"[vi-plot] {r}: median path {100 * s['zb_rmse_median_path_m']:.2f} cm, band covers "
              f"{100 * s['zb_band90_cover_truth']:.0f} % (width {100 * s['zb_band90_mean_width_m']:.1f} cm), Ks "
              f"{s['ks_post']} (truth {s['ks_true']:.3e}, in 90 %: {s['ks_true_in_90']}), contraction log Ks "
              f"{s['contraction_logks']:.3f}, level {s['contraction_level']:.3f}, corr(level, log Ks) "
              f"{s['corr_level_logks']:+.3f}; rules {s['rules']}", flush=True)
    print(f"[vi-plot] wrote {args.out}.png", flush=True)


if __name__ == '__main__':
    main()
