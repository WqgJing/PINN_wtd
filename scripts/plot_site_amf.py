"""Observations at an AmeriFlux site from its BASE half-hourly file: daily rain, daily evapotranspiration from the measured
latent heat flux, soil moisture of one profile at every depth (depths from the AmeriFlux measurement-height table), and
the water-table depth. Lines only. Standard library, numpy, matplotlib.

Usage (via sbatch): hpc/run_script_cpu.slurm scripts/plot_site_amf.py --csv <AMF_..._BASE_HH_...csv> \
        --heights data/ameriflux/BASE_MeasurementHeight_20260908.csv --site US-HB2 --profile 3 \
        --out output/sites/US-HB2_observations.png [--start 2022-09-27 --end 2024-12-31]
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

LAMBDA = 2.45e6          # J/kg, latent heat of vaporisation
ALIASES = {'LE': ['LE', 'LE_1_1_1', 'LE_PI_F', 'LE_PI_F_1_1_1'],            # generic name -> column names used by
           'P_RAIN': ['P_RAIN', 'P_RAIN_1_1_1', 'P', 'P_1_1_1', 'P_2_1_1']}   # sites, first present wins


def heights(path, site):
    out = {}
    with open(path, newline='') as fh:
        for r in csv.DictReader(fh):
            if r['Site_ID'] == site and r['Height'] not in ('', None):
                try:
                    out[r['Variable']] = float(r['Height'])
                except ValueError:
                    pass
    return out


def read(path, cols):
    with open(path, newline='') as fh:
        rd = csv.reader(fh)
        next(rd), next(rd)
        head = next(rd)
        idx = {}
        for c in cols:                                       # the requested name keys the result, even if read
            for name in ALIASES.get(c, [c]):                 # from an alias
                if name in head:
                    idx[c] = head.index(name)
                    break
        ts, vals = [], {c: [] for c in idx}
        for row in rd:
            ts.append(row[0])
            for c, i in idx.items():
                try:
                    v = float(row[i])
                except ValueError:
                    v = np.nan
                vals[c].append(np.nan if v == -9999 else v)
    t = np.array([f'{s[:4]}-{s[4:6]}-{s[6:8]}T{s[8:10]}:{s[10:12]}' for s in ts], dtype='datetime64[m]')
    return t, {c: np.array(v) for c, v in vals.items()}


def daily(t, x, how):
    d = t.astype('datetime64[D]')
    u, inv = np.unique(d, return_inverse=True)
    ok = np.isfinite(x)
    n = np.bincount(inv, weights=ok.astype(float))
    s = np.bincount(inv, weights=np.where(ok, x, 0.0))
    full = np.bincount(inv)
    if how == 'sum':
        val = np.where(n >= 0.8 * full, s, np.nan)
    else:
        val = np.where(n >= 0.8 * full, s / np.maximum(n, 1), np.nan)
    return u, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--heights', required=True)
    ap.add_argument('--site', required=True)
    ap.add_argument('--profile', default='3')
    ap.add_argument('--out', required=True)
    ap.add_argument('--start', default=None)
    ap.add_argument('--end', default=None)
    ap.add_argument('--max_depth_m', type=float, default=None, help='plot only sensors at most this deep')
    ap.add_argument('--le', default='LE', help='latent-heat column (default: LE or its first alias)')
    ap.add_argument('--ts', default=None, help='soil-temperature column: adds a panel (0 C marks frost)')
    ap.add_argument('--halfhourly', action='store_true', help='rain and ET as half-hourly rates (mm/h), not daily')
    args = ap.parse_args()
    hts = heights(args.heights, args.site)
    with open(args.csv, newline='') as fh:
        rd = csv.reader(fh)
        next(rd), next(rd)
        head = next(rd)
    swc = sorted([c for c in head if re.fullmatch(rf'SWC_{args.profile}_\d+_1', c)], key=lambda c: -hts.get(c, 0.0))
    if args.max_depth_m is not None:
        swc = [c for c in swc if -hts.get(c, 0.0) <= args.max_depth_m + 1e-9]
    rain = 'P_RAIN'
    wtd = [c for c in head if re.fullmatch(r'WTD(_\d+_\d+_\d+)?', c)]
    t, v = read(args.csv, swc + [rain, args.le] + wtd + ([args.ts] if args.ts else []))
    v['LE'] = v[args.le]
    for c in swc:                                            # SWC in %: other flags (e.g. 9999999) and zeros dropped
        v[c] = np.where((v[c] > 0.0) & (v[c] <= 100.0), v[c], np.nan)
    t0 = np.datetime64(args.start) if args.start else t[0]
    t1 = np.datetime64(args.end) if args.end else t[-1]
    nrow = 4 if args.ts else 3
    fig, axes = plt.subplots(nrow, 1, figsize=(13, 11 + 2.5 * (nrow - 3)), sharex=True,
                             gridspec_kw={'height_ratios': [1, 1.6, 1.2] + [0.8] * (nrow - 3)})
    if args.halfhourly:
        m = (t >= t0) & (t <= t1)
        axes[0].plot(t[m], 2.0 * v[rain][m], color='#2a78d6', lw=1.0, label='rain')
        axes[0].plot(t[m], v['LE'][m] / LAMBDA * 3600.0, color='#eb6834', lw=1.0,
                     label='evapotranspiration (eddy covariance)')
        axes[0].set_ylabel('water flux (mm/h)')
    else:
        td, pr = daily(t, v[rain], 'sum')
        _, le = daily(t, v['LE'], 'mean')
        m = (td >= t0.astype('datetime64[D]')) & (td <= t1.astype('datetime64[D]'))
        axes[0].plot(td[m], pr[m], color='#2a78d6', lw=1.0, label='rain')
        axes[0].plot(td[m], le[m] / LAMBDA * 86400.0, color='#eb6834', lw=1.0,
                     label='evapotranspiration (eddy covariance)')
        axes[0].set_ylabel('water flux (mm/d)')
    axes[0].legend(frameon=False, fontsize=9)
    m = (t >= t0) & (t <= t1)
    cmap = plt.get_cmap('viridis')
    for k, c in enumerate(swc):
        col = cmap(k / max(len(swc) - 1, 1))
        axes[1].plot(t[m], v[c][m] / 100.0, color=col, lw=0.9, label=f'{-100 * hts.get(c, np.nan):.0f} cm')
    axes[1].set_ylabel('soil moisture (m$^3$/m$^3$)')
    axes[1].legend(frameon=False, fontsize=8, ncol=len(swc))
    for c, col in zip(wtd, ('#1baf7a', '#8a4fd1')):
        if np.isfinite(v[c][m]).any():
            axes[2].plot(t[m], -v[c][m], color=col, lw=1.0, label=c)
    axes[2].invert_yaxis()
    axes[2].set_ylabel('water-table depth (m)')
    axes[2].legend(frameon=False, fontsize=9)
    if args.ts:
        x = v[args.ts][m]
        axes[3].plot(t[m], x, color='#8a4fd1', lw=1.0, label=args.ts)
        axes[3].axhline(0.0, color='k', lw=0.8, label='0 C')
        axes[3].set_ylabel('soil temperature (C)')
        axes[3].legend(frameon=False, fontsize=9)
        print(f"[site] {args.ts}: {np.nanmin(x):.1f} .. {np.nanmax(x):.1f} C, <= 0 C on {100 * np.nanmean(x <= 0.0):.0f} % "
              f"of valid half-hours, valid {100 * np.isfinite(x).mean():.0f} %", flush=True)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d' if args.halfhourly else '%Y-%m'))
    axes[-1].set_xlabel('date')
    for ax in axes:
        ax.grid(True, color='#e1e0d9', lw=0.6)
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=120)
    plt.close(fig)
    for c in swc:
        x = v[c][m] / 100.0
        print(f"[site] {c} at {-100 * hts.get(c, np.nan):.0f} cm: {np.nanmin(x):.3f} .. {np.nanmax(x):.3f}, "
              f"valid {100 * np.isfinite(x).mean():.0f} %", flush=True)
    for c in wtd:
        x = -v[c][m]
        if np.isfinite(x).any():
            print(f"[site] {c}: {np.nanmin(x):.2f} .. {np.nanmax(x):.2f} m deep, valid {100 * np.isfinite(x).mean():.0f} %",
                  flush=True)
    print(f"[site] rain {np.nansum(v[rain][m]):.0f} mm, ET {np.nansum(v['LE'][m]) * 1800 / LAMBDA:.0f} mm (gaps not filled), "
          f"LE valid {100 * np.isfinite(v['LE'][m]).mean():.0f} %; period {t0} - {t1}", flush=True)
    print(f"[site] wrote {args.out}", flush=True)


if __name__ == '__main__':
    main()
