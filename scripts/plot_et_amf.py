"""Evapotranspiration at an AmeriFlux site by itself: (top) half-hourly ET (mm/h) from every LE column, (bottom) the
share of each day's half-hours present, per column and for their union. Lines only.

Usage (via sbatch): hpc/run_script_cpu.slurm scripts/plot_et_amf.py --csv <BASE_HH csv> --start 2024-03-01 \
        --end 2025-05-01 --out output/sites/US-Ho1_ET_2024-2025.png
"""
import argparse
import csv
import os
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import plot_site_amf as psa

COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#8a4fd1']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    with open(args.csv, newline='') as fh:
        rd = csv.reader(fh)
        next(rd), next(rd)
        head = next(rd)
    cols = [c for c in head if re.fullmatch(r'LE(_\d+_\d+_\d+)?', c)]
    t, v = psa.read(args.csv, cols)
    t0, t1 = np.datetime64(args.start), np.datetime64(args.end)
    m = (t >= t0) & (t <= t1)
    t = t[m]
    et = {c: v[c][m] / psa.LAMBDA * 3600.0 for c in cols}              # mm/h
    union = np.zeros(t.size, dtype=bool)
    fig, ax = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    day = t.astype('datetime64[D]')
    u, inv = np.unique(day, return_inverse=True)
    for c, col in zip(cols, COLORS):
        ok = np.isfinite(et[c])
        union |= ok
        ax[0].plot(t, et[c], color=col, lw=0.6, label=c)
        frac = np.bincount(inv, weights=ok.astype(float)) / np.bincount(inv)
        ax[1].plot(u, frac, color=col, lw=1.0, label=f'{c}: {100 * ok.mean():.0f} % of half-hours')
        print(f"[et] {c}: {100 * ok.mean():.1f} % of half-hours present; days with >= 80 %: "
              f"{int((frac >= 0.8).sum())} of {u.size}", flush=True)
    fu = np.bincount(inv, weights=union.astype(float)) / np.bincount(inv)
    ax[1].plot(u, fu, color='k', lw=1.4, label=f'any column: {100 * union.mean():.0f} % of half-hours')
    print(f"[et] union: {100 * union.mean():.1f} % of half-hours; days with >= 80 %: {int((fu >= 0.8).sum())} of {u.size}",
          flush=True)
    ax[0].set_ylabel('evapotranspiration (mm/h)')
    ax[1].set_ylabel("share of the day's half-hours present")
    ax[1].set_xlabel('date')
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    for a in ax:
        a.legend(frameon=False, fontsize=8, ncol=2)
        a.grid(True, color='#e1e0d9', lw=0.6)
        a.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=120)
    plt.close(fig)
    print(f"[et] wrote {args.out}", flush=True)


if __name__ == '__main__':
    main()
