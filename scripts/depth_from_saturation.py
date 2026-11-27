"""Estimate a soil-moisture sensor's depth from its saturation crossing: when the water table rises to the sensor, its
reading reaches the saturated plateau. For each sensor: median reading per water-table depth bin (unfrozen hours) and
the shallowest-to-deepest depth at which the median falls 5 % (of the plateau-to-dry range) below the plateau. Figure:
reading against water-table depth (points) and the bin medians (lines).

Usage (via sbatch): hpc/run_script_cpu.slurm scripts/depth_from_saturation.py --csv <BASE_HH csv> \
        --swc SWC_1_1_1,SWC_1_2_1 --wtd WTD --out output/sites/US-ONA_depth_from_saturation
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import plot_site_amf as psa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--swc', required=True)
    ap.add_argument('--wtd', default='WTD')
    ap.add_argument('--bin_m', type=float, default=0.02)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    cols = args.swc.split(',')
    t, v = psa.read(args.csv, cols + [args.wtd])
    w = v[args.wtd]
    print(f"[depth] {args.wtd}: {np.nanmin(w):.2f} .. {np.nanmax(w):.2f} (raw, valid {100 * np.isfinite(w).mean():.0f} %)",
          flush=True)
    zb = -w if np.nanmedian(w) < 0 else w                         # depth below the surface, positive down
    fig, ax = plt.subplots(figsize=(9, 6))
    rep = {}
    for c, col in zip(cols, ['#2a78d6', '#eb6834', '#1baf7a', '#8a4fd1']):
        th = np.where((v[c] > 0) & (v[c] <= 100), v[c], np.nan)
        th = th / 100.0 if np.nanmax(th) > 1.0 else th
        ok = np.isfinite(th) & np.isfinite(zb)
        x, y = zb[ok], th[ok]
        edges = np.arange(np.floor(x.min() / args.bin_m) * args.bin_m, x.max() + args.bin_m, args.bin_m)
        idx = np.digitize(x, edges)
        cen, med = [], []
        for k in np.unique(idx):
            s = idx == k
            if s.sum() >= 20:
                cen.append(edges[k - 1] + args.bin_m / 2)
                med.append(np.median(y[s]))
        cen, med = np.array(cen), np.array(med)
        plateau = np.percentile(med[cen <= np.percentile(cen, 10)], 50) if cen.size else np.nan
        dry = np.percentile(med, 5)
        thr = plateau - 0.05 * (plateau - dry)
        below = np.where(med < thr)[0]
        depth = float(cen[below[0]]) if below.size else float('nan')
        rep[c] = {'plateau': float(plateau), 'dry': float(dry), 'est_depth_m': depth, 'n': int(ok.sum()),
                  'reading_range': [float(np.nanmin(y)), float(np.nanmax(y))]}
        print(f"[depth] {c}: plateau {plateau:.3f}, dry {dry:.3f}; leaves the plateau at a water table of ~{depth:.2f} m "
              f"(n {ok.sum()})", flush=True)
        ax.plot(y[::7], x[::7], ls='none', marker='.', ms=1.5, color=col, alpha=0.3)
        ax.plot(med, cen, color=col, lw=2.0, label=f'{c} (median per {100 * args.bin_m:.0f} cm bin)')
    ax.invert_yaxis()
    ax.set_xlabel('soil moisture (m$^3$/m$^3$)')
    ax.set_ylabel('water-table depth (m)')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, color='#e1e0d9', lw=0.6)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out + '.png', dpi=120)
    plt.close(fig)
    with open(args.out + '.json', 'w') as f:
        json.dump(rep, f, indent=1)
    print(f"[depth] wrote {args.out}.png", flush=True)


if __name__ == '__main__':
    main()
