"""Find windows of complete data at an AmeriFlux site for a real-data PINN test: every listed soil-moisture sensor and
the water table >= --min_valid of their half-hours, the latent heat flux >= --min_le, and the water table always deeper
than --min_wt_m (so that those sensors stay above it). Ranks the windows by the water table's range and lists the top.

Usage (via sbatch): hpc/run_script_cpu.slurm scripts/find_window.py --csv <BASE_HH csv> \
        --swc SWC_3_1_1,SWC_3_2_1,SWC_3_3_1,SWC_3_4_1,SWC_3_5_1 --days 21 --min_wt_m 0.45
"""
import argparse

import numpy as np

import plot_site_amf as psa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--swc', required=True, help='comma-separated soil-moisture columns that must be complete')
    ap.add_argument('--wtd', default='WTD_1_1_1')
    ap.add_argument('--le', default='LE', help='latent-heat column (default: LE or its first alias)')
    ap.add_argument('--days', type=int, default=21)
    ap.add_argument('--min_valid', type=float, default=0.95)
    ap.add_argument('--min_le', type=float, default=0.7)
    ap.add_argument('--min_wt_m', type=float, default=0.45)
    ap.add_argument('--top', type=int, default=8)
    args = ap.parse_args()
    swc = args.swc.split(',')
    rain = 'P_RAIN'
    t, v = psa.read(args.csv, swc + [args.wtd, args.le, rain])
    v['LE'] = v[args.le]
    for c in swc:
        v[c] = np.where((v[c] > 0.0) & (v[c] <= 100.0), v[c], np.nan)
    day = t.astype('datetime64[D]')
    days = np.unique(day)
    rows = []
    for d0 in days[::2]:
        m = (day >= d0) & (day < d0 + np.timedelta64(args.days, 'D'))
        if m.sum() < 48 * args.days * 0.99:
            continue
        ok_s = min(np.isfinite(v[c][m]).mean() for c in swc)
        ok_w = np.isfinite(v[args.wtd][m]).mean()
        ok_le = np.isfinite(v['LE'][m]).mean()
        if ok_s < args.min_valid or ok_w < args.min_valid or ok_le < args.min_le:
            continue
        wt = -v[args.wtd][m]
        if np.nanmin(wt) < args.min_wt_m:
            continue
        r = np.nan_to_num(v[rain][m])
        rd = np.bincount(np.unique(day[m], return_inverse=True)[1], weights=r)
        rows.append((float(np.nanmax(wt) - np.nanmin(wt)), str(d0), float(np.nanmin(wt)), float(np.nanmax(wt)),
                     float(r.sum()), int((rd > 5).sum()), ok_s, ok_w, ok_le))
    rows.sort(reverse=True)
    print(f"[window] {len(rows)} windows of {args.days} d pass; top {args.top} by water-table range:", flush=True)
    for rg, d0, lo, hi, rs, nr, s, w, le in rows[:args.top]:
        print(f"[window] start {d0}: water table {lo:.2f} - {hi:.2f} m (range {rg:.2f}), rain {rs:.0f} mm on {nr} days "
              f"> 5 mm, valid: soil {100 * s:.0f} %, water table {100 * w:.0f} %, LE {100 * le:.0f} %", flush=True)


if __name__ == '__main__':
    main()
