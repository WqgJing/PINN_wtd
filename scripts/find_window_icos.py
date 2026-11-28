"""Find windows of complete data at an ICOS station for a real-data PINN test (the ICOS twin of find_window.py): every
soil-moisture sensor of one profile down to --max_depth_m and the water table >= --min_valid of their half-hours, the
gap-filled latent heat flux >= --min_le, and the water table always deeper than --min_wt_m (so those sensors stay above
it) and shallower than --max_wt_m (so the well is not bottomed out). Ranks the windows by the water table's range and
lists the top. Also reports, per window, the share of soil-moisture readings <= 0.01 (dry-end calibration floor).

Usage (via sbatch): hpc/run_script_cpu.slurm scripts/find_window_icos.py --archive data/icos/FR-Bil_ARCHIVE.zip \
        --meteosens data/icos/FR-Bil.zip --profile 1 --max_depth_m 1.1 --wtd WTD_3_1_1 --days 21 --min_wt_m 1.2
"""
import argparse
import csv
import io
import re
import zipfile

import numpy as np

import plot_site_icos as psi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--archive', required=True)
    ap.add_argument('--meteosens', required=True)
    ap.add_argument('--profile', default='1')
    ap.add_argument('--max_depth_m', type=float, default=1.1)
    ap.add_argument('--wtd', default='WTD_1_1_1')
    ap.add_argument('--days', type=int, default=21)
    ap.add_argument('--min_valid', type=float, default=0.95)
    ap.add_argument('--min_le', type=float, default=0.9)
    ap.add_argument('--min_wt_m', type=float, default=1.2)
    ap.add_argument('--max_wt_m', type=float, default=1.65)
    ap.add_argument('--top', type=int, default=10)
    args = ap.parse_args()
    za, zm = zipfile.ZipFile(args.archive), zipfile.ZipFile(args.meteosens)
    hts = psi.heights(za)
    msname = [n for n in zm.namelist() if n.lower().endswith('.csv')][0]
    with zm.open(msname) as fh:
        head = next(csv.reader(io.TextIOWrapper(fh, encoding='utf-8')))
    swc = [c for c in head if re.fullmatch(rf'SWC_{args.profile}_\d+_1', c) and -hts.get(c, 99.0) <= args.max_depth_m]
    swc.sort(key=lambda c: -hts[c])
    print(f"[window] sensors: {', '.join(f'{c} ({-100 * hts[c]:.0f} cm)' for c in swc)}; well {args.wtd}", flush=True)
    tm, vm, _ = psi.read_cols(zm, msname, swc + [args.wtd])
    tf, vf, _ = psi.read_cols(za, psi.member(za, r'ICOSETC_[A-Z]{2}-\w{3}_FLUXNET_HH_(INTERIM_)?L2\.csv$'),
                              ['P_F', 'LE_F_MDS'])
    le = np.interp(tm.astype(float), tf.astype(float), np.where(np.isfinite(vf['LE_F_MDS']), 1.0, 0.0)) > 0.5
    rain = np.interp(tm.astype(float), tf.astype(float), np.nan_to_num(vf['P_F']))
    for c in swc:
        vm[c] = np.where(vm[c] > 1.5, vm[c] / 100.0, vm[c])
    day = tm.astype('datetime64[D]')
    days = np.unique(day)
    rows = []
    for d0 in days[::2]:
        m = (day >= d0) & (day < d0 + np.timedelta64(args.days, 'D'))
        if m.sum() < 48 * args.days * 0.99:
            continue
        ok_s = min(np.isfinite(vm[c][m]).mean() for c in swc)
        ok_w = np.isfinite(vm[args.wtd][m]).mean()
        ok_le = le[m].mean()
        if ok_s < args.min_valid or ok_w < args.min_valid or ok_le < args.min_le:
            continue
        wt = -vm[args.wtd][m]
        if np.nanmin(wt) < args.min_wt_m or np.nanmax(wt) > args.max_wt_m:
            continue
        floor = max(np.mean(vm[c][m][np.isfinite(vm[c][m])] <= 0.01) for c in swc)
        rd = np.bincount(np.unique(day[m], return_inverse=True)[1], weights=rain[m])
        rows.append((float(np.nanmax(wt) - np.nanmin(wt)), str(d0), float(np.nanmin(wt)), float(np.nanmax(wt)),
                     float(rd.sum()), int((rd > 5).sum()), ok_s, ok_w, ok_le, floor))
    rows.sort(reverse=True)
    print(f"[window] {len(rows)} windows of {args.days} d pass; top {args.top} by water-table range:", flush=True)
    for rg, d0, lo, hi, rs, nr, s, w, l, fl in rows[:args.top]:
        print(f"[window] start {d0}: water table {lo:.2f} - {hi:.2f} m (range {rg:.2f}), rain {rs:.0f} mm on {nr} days "
              f"> 5 mm, valid: soil {100 * s:.0f} %, water table {100 * w:.0f} %, ET {100 * l:.0f} %; "
              f"readings <= 0.01: {100 * fl:.0f} %", flush=True)


if __name__ == '__main__':
    main()
