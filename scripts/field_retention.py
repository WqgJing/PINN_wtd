"""Field retention curves at an AmeriFlux site: each soil-moisture sensor's reading against its height above the
water table (well), at near-rest times, and a van Genuchten fit per sensor and per group of sensors.

Near rest: no rain in the previous --dry_hours, the sensor's reading changed by < --max_change over the previous 6 h,
the sensor above the water table by > --min_height_m. At rest the suction equals the height above the water table, so
these pairs trace the retention curve theta(h). Readings near the surface are drier than rest (evaporation, roots), so
the deep sensors are the more reliable. Times inside --exclude (the test window) are left out.

Usage (via sbatch): hpc/run_script_cpu.slurm scripts/field_retention.py --csv <BASE_HH csv> \
        --heights data/ameriflux/BASE_MeasurementHeight_20260908.csv --site US-HB2 --profile 3 \
        --exclude 2024-06-15 2024-07-20 --out output/sites/US-HB2_field_retention
"""
import argparse
import csv
import json
import os
import re

import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import plot_site_amf as psa

SAND = {'theta_r': 0.045, 'theta_s': 0.43, 'alpha': 14.5, 'n': 2.68}


def vg(H, tr, ts, a, n):
    return tr + (ts - tr) * (1.0 + (a * H) ** n) ** (-(1.0 - 1.0 / n))


def fit(H, th, theta_s_hint):
    """Least squares in (theta_r, theta_s, log alpha, log(n - 1))."""
    def res(p):
        return vg(H, p[0], p[1], np.exp(p[2]), 1.0 + np.exp(p[3])) - th
    lo, hi = [0.0, max(th.max(), 0.05), np.log(0.3), np.log(0.05)], [min(th.min(), 0.15), 0.6, np.log(40.0), np.log(6.0)]
    x0 = [min(0.02, th.min() * 0.5), min(max(theta_s_hint, th.max() + 0.01), 0.59), np.log(5.0), np.log(1.0)]
    x0 = np.clip(x0, np.array(lo) + 1e-6, np.array(hi) - 1e-6)
    r = least_squares(res, x0, bounds=(lo, hi))
    p = r.x
    return {'theta_r': float(p[0]), 'theta_s': float(p[1]), 'alpha': float(np.exp(p[2])), 'n': float(1.0 + np.exp(p[3])),
            'rmse': float(np.sqrt(np.mean(r.fun ** 2))), 'n_points': int(H.size)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--heights', required=True)
    ap.add_argument('--site', required=True)
    ap.add_argument('--profile', default='3')
    ap.add_argument('--exclude', nargs=2, default=None, help='first and last date to leave out')
    ap.add_argument('--dry_hours', type=float, default=48.0)
    ap.add_argument('--max_change', type=float, default=0.005)
    ap.add_argument('--min_height_m', type=float, default=0.02)
    ap.add_argument('--theta_s', type=float, default=0.445, help='saturated value hint (from the submerged sensor)')
    ap.add_argument('--groups', default='5-30:0.05,0.1,0.2,0.3;40-60:0.4,0.5,0.6;60-75:0.6,0.75;75:0.75')
    ap.add_argument('--out', required=True, help='path without extension')
    args = ap.parse_args()
    hts = psa.heights(args.heights, args.site)
    with open(args.csv, newline='') as fh:
        rd = csv.reader(fh)
        next(rd), next(rd)
        head = next(rd)
    swc = sorted([c for c in head if re.fullmatch(rf'SWC_{args.profile}_\d+_1', c)], key=lambda c: -hts[c])
    t, v = psa.read(args.csv, swc + ['WTD_1_1_1', 'P_RAIN'])
    for c in swc:
        v[c] = np.where((v[c] > 0.0) & (v[c] <= 100.0), v[c] / 100.0, np.nan)
    zb = -v['WTD_1_1_1']
    rain = np.nan_to_num(v['P_RAIN'])
    k = int(round(args.dry_hours * 2))
    rain_prev = np.convolve(rain, np.ones(k), mode='full')[:rain.size]          # sum over the previous k half-hours
    keep_t = (rain_prev == 0.0) & np.isfinite(zb)
    if args.exclude:
        e0, e1 = np.datetime64(args.exclude[0]), np.datetime64(args.exclude[1])
        keep_t &= ~((t >= e0) & (t <= e1))
    hourly = (t.astype('datetime64[m]').astype(int) % 60) == 0
    pairs = {}
    for c in swc:
        d = -hts[c]
        th = v[c]
        ch = np.full(th.size, np.nan)
        ch[12:] = np.abs(th[12:] - th[:-12])
        H = zb - d
        ok = keep_t & hourly & np.isfinite(th) & (ch < args.max_change) & (H > args.min_height_m)
        pairs[round(d, 3)] = (H[ok], th[ok])
    fits = {}
    for d, (H, th) in pairs.items():
        if H.size >= 30:
            fits[f'{100 * d:.0f} cm'] = fit(H, th, args.theta_s)
    groups = {}
    for g in args.groups.split(';'):
        name, ds = g.split(':')
        Hs = np.concatenate([pairs[round(float(x), 3)][0] for x in ds.split(',')])
        ths = np.concatenate([pairs[round(float(x), 3)][1] for x in ds.split(',')])
        if Hs.size >= 30:
            groups[name] = fit(Hs, ths, args.theta_s)
    sand_rmse = {f'{100 * d:.0f} cm': float(np.sqrt(np.mean((vg(H, SAND['theta_r'], args.theta_s, SAND['alpha'],
                                                                 SAND['n']) - th) ** 2))) if H.size else None
                 for d, (H, th) in pairs.items()}
    rep = {'site': args.site, 'excluded': args.exclude, 'per_sensor': fits, 'groups': groups,
           'sand_class_rmse_with_theta_s': sand_rmse,
           'points': {f'{100 * d:.0f} cm': int(H.size) for d, (H, th) in pairs.items()},
           'height_range_m': {f'{100 * d:.0f} cm': [float(H.min()), float(H.max())] if H.size else None
                              for d, (H, th) in pairs.items()}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out + '.json', 'w') as f:
        json.dump(rep, f, indent=1)
    for name, p in fits.items():
        print(f"[retention] sensor {name}: {p['n_points']} points, height {rep['height_range_m'][name][0]:.2f} - "
              f"{rep['height_range_m'][name][1]:.2f} m; theta_r {p['theta_r']:.3f} theta_s {p['theta_s']:.3f} alpha "
              f"{p['alpha']:.2f} n {p['n']:.2f}; rmse {p['rmse']:.4f} (sand class {sand_rmse[name]:.4f})", flush=True)
    for name, p in groups.items():
        print(f"[retention] group {name}: {p['n_points']} points; theta_r {p['theta_r']:.3f} theta_s {p['theta_s']:.3f} "
              f"alpha {p['alpha']:.2f} n {p['n']:.2f}; rmse {p['rmse']:.4f}", flush=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.get_cmap('viridis')
    ds = sorted(pairs)
    Hg = np.linspace(0.01, 1.4, 300)
    for i, d in enumerate(ds):
        H, th = pairs[d]
        col = cmap(i / max(len(ds) - 1, 1))
        if H.size:
            ax.plot(th, H, ls='none', marker='.', ms=2, color=col, alpha=0.5)
        name = f'{100 * d:.0f} cm'
        if name in fits:
            p = fits[name]
            ax.plot(vg(Hg, p['theta_r'], p['theta_s'], p['alpha'], p['n']), Hg, color=col, lw=1.6, label=f'{name}')
    ax.plot(vg(Hg, SAND['theta_r'], args.theta_s, SAND['alpha'], SAND['n']), Hg, color='k', lw=1.6, ls='--',
            label='sand class (used before)')
    ax.set_xlabel('soil moisture (m$^3$/m$^3$)')
    ax.set_ylabel('height above the water table = suction at rest (m)')
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.grid(True, color='#e1e0d9', lw=0.6)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out + '.png', dpi=120)
    plt.close(fig)
    print(f"[retention] wrote {args.out}.json / .png", flush=True)


if __name__ == '__main__':
    main()
