"""Observations at an ICOS ecosystem station: daily rain (P_F) and ET (LE_F_MDS, gap-filled) from the FLUXNET_HH file of
the station's L2 archive; one soil-moisture profile at every depth (depths from VARINFO_METEOSENS); every water-table
well; the shallowest soil temperature. Lines only. Standard library, numpy, matplotlib.

Usage (via sbatch): hpc/run_script_cpu.slurm scripts/plot_site_icos.py --archive data/icos/FR-Bil_ARCHIVE.zip \
        --meteosens data/icos/FR-Bil.zip --profile 1 --out output/sites/FR-Bil_observations.png [--start ... --end ...]
"""
import argparse
import csv
import io
import os
import re
import zipfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

LAMBDA = 2.45e6


def member(z, pattern):
    return [n for n in z.namelist() if re.search(pattern, n)][0]


def read_cols(z, name, want):
    with z.open(name) as fh:
        rd = csv.reader(io.TextIOWrapper(fh, encoding='utf-8', errors='replace'))
        head = next(rd)
        idx = {c: head.index(c) for c in want if c in head}
        ts, vals = [], {c: [] for c in idx}
        bad = 0
        for row in rd:
            if not row or not re.fullmatch(r'\d{12}', row[0]):       # blank or broken line
                bad += 1
                continue
            ts.append(row[0])
            for c, i in idx.items():
                try:
                    v = float(row[i])
                except (ValueError, IndexError):
                    v = np.nan
                vals[c].append(np.nan if v <= -9990 else v)
    if bad:
        print(f'[icos-site] {name}: skipped {bad} lines without a timestamp', flush=True)
    t = np.array([f'{s[:4]}-{s[4:6]}-{s[6:8]}T{s[8:10]}:{s[10:12]}' for s in ts], dtype='datetime64[m]')
    return t, {c: np.array(v) for c, v in vals.items()}, head


def heights(z):
    name = member(z, r'VARINFO_METEOSENS_(INTERIM_)?L2\.csv$')
    g = {}
    with z.open(name) as fh:
        for r in csv.DictReader(io.TextIOWrapper(fh, encoding='utf-8')):
            g.setdefault(r['GROUP_ID'], {})[r['VARIABLE']] = r['DATAVALUE']
    out = {}
    for d in g.values():
        if 'VAR_INFO_VARNAME' in d and 'VAR_INFO_HEIGHT' in d:
            out[d['VAR_INFO_VARNAME']] = float(d['VAR_INFO_HEIGHT'])      # last entry per variable kept
    return out


def daily(t, x, how):
    d = t.astype('datetime64[D]')
    u, inv = np.unique(d, return_inverse=True)
    ok = np.isfinite(x)
    n = np.bincount(inv, weights=ok.astype(float))
    s = np.bincount(inv, weights=np.where(ok, x, 0.0))
    full = np.bincount(inv)
    return u, np.where(n >= 0.8 * full, s if how == 'sum' else s / np.maximum(n, 1), np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--archive', required=True)
    ap.add_argument('--meteosens', required=True)
    ap.add_argument('--profile', default='1')
    ap.add_argument('--out', required=True)
    ap.add_argument('--start', default=None)
    ap.add_argument('--end', default=None)
    ap.add_argument('--max_depth_m', type=float, default=None, help='plot only sensors at most this deep')
    ap.add_argument('--halfhourly', action='store_true', help='rain and ET as half-hourly rates (mm/h), not daily')
    args = ap.parse_args()
    za, zm = zipfile.ZipFile(args.archive), zipfile.ZipFile(args.meteosens)
    hts = heights(za)
    msname = [n for n in zm.namelist() if n.lower().endswith('.csv')][0]
    with zm.open(msname) as fh:
        head = next(csv.reader(io.TextIOWrapper(fh, encoding='utf-8')))
    swc = sorted([c for c in head if re.fullmatch(rf'SWC_{args.profile}_\d+_1', c)], key=lambda c: -hts.get(c, 0.0))
    if args.max_depth_m is not None:
        swc = [c for c in swc if -hts.get(c, -99.0) <= args.max_depth_m]
    wtd = [c for c in head if re.fullmatch(r'WTD_\d+_\d+_\d+', c)]
    ts = [c for c in head if re.fullmatch(rf'TS_{args.profile}_1_1', c)] or [c for c in head if re.fullmatch(r'TS_\d+_1_1', c)][:1]
    tm, vm, _ = read_cols(zm, msname, swc + wtd + ts)
    tf, vf, _ = read_cols(za, member(za, r'ICOSETC_[A-Z]{2}-\w{3}_FLUXNET_HH_(INTERIM_)?L2\.csv$'), ['P_F', 'LE_F_MDS'])
    t0 = np.datetime64(args.start) if args.start else tm[0]
    t1 = np.datetime64(args.end) if args.end else tm[-1]
    fig, ax = plt.subplots(4, 1, figsize=(13, 13), sharex=True, gridspec_kw={'height_ratios': [1, 1.6, 1.2, 0.7]})
    if args.halfhourly:                                                # P_F: mm per half-hour; LE: W/m2
        m = (tf >= t0) & (tf <= t1)
        ax[0].plot(tf[m], 2.0 * vf['P_F'][m], color='#2a78d6', lw=1.0, label='rain')
        ax[0].plot(tf[m], vf['LE_F_MDS'][m] / LAMBDA * 3600.0, color='#eb6834', lw=1.0,
                   label='evapotranspiration (gap-filled)')
        ax[0].set_ylabel('water flux (mm/h)')
    else:
        du, pr = daily(tf, vf['P_F'], 'sum')
        _, le = daily(tf, vf['LE_F_MDS'], 'mean')
        m = (du >= t0.astype('datetime64[D]')) & (du <= t1.astype('datetime64[D]'))
        ax[0].plot(du[m], pr[m], color='#2a78d6', lw=1.0, label='rain')
        ax[0].plot(du[m], le[m] / LAMBDA * 86400.0, color='#eb6834', lw=1.0, label='evapotranspiration (gap-filled)')
        ax[0].set_ylabel('water flux (mm/d)')
    m = (tm >= t0) & (tm <= t1)
    cmap = plt.get_cmap('viridis')
    for k, c in enumerate(swc):
        x = vm[c][m]
        x = x / 100.0 if np.nanmax(x) > 1.5 else x
        ax[1].plot(tm[m], x, color=cmap(k / max(len(swc) - 1, 1)), lw=0.8, label=f'{-100 * hts.get(c, np.nan):.0f} cm')
        print(f"[icos-site] {c} at {-100 * hts.get(c, np.nan):.1f} cm: {np.nanmin(x):.3f} .. {np.nanmax(x):.3f}, "
              f"present {100 * np.isfinite(x).mean():.0f} %", flush=True)
    ax[1].set_ylabel('soil moisture (m$^3$/m$^3$)')
    for c, col in zip(wtd, ['#1baf7a', '#8a4fd1', '#c2185b', '#5d6d7e', '#a0522d']):
        x = -vm[c][m]
        ax[2].plot(tm[m], x, color=col, lw=0.9, label=f'{c} (sensor {-hts.get(c, np.nan):.2f} m)')
        print(f"[icos-site] {c}: {np.nanmin(x):.2f} .. {np.nanmax(x):.2f} m deep, present {100 * np.isfinite(x).mean():.0f} %",
              flush=True)
    ax[2].invert_yaxis()
    ax[2].set_ylabel('water-table depth (m)')
    for c in ts:
        ax[3].plot(tm[m], vm[c][m], color='#8a4fd1', lw=0.8, label=c)
    ax[3].axhline(0.0, color='k', lw=0.8, label='0 C')
    ax[3].set_ylabel('soil temperature (C)')
    ax[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m' if t1 - t0 > np.timedelta64(120, 'D') else '%m-%d'))
    ax[3].set_xlabel('date')
    for a in ax:
        a.legend(frameon=False, fontsize=8, ncol=4)
        a.grid(True, color='#e1e0d9', lw=0.6)
        a.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=110)
    plt.close(fig)
    print(f"[icos-site] ET present {100 * np.isfinite(vf['LE_F_MDS']).mean():.0f} %; wrote {args.out}", flush=True)


if __name__ == '__main__':
    main()
