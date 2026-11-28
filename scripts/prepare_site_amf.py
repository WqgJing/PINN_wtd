"""A record for train_profile.py from an AmeriFlux site (the same npz keys as make_synthetic.py), half-hourly:
  t_days, sensor_depths_m, theta_obs [n_t, n_depths] (one soil-moisture profile, NaN where missing), zb (observed
  water-table depth, m, positive down; scoring only), soil [theta_s, theta_r, alpha, n, Ks], t_nodes and tp_nodes_mmd
  (root-uptake demand: the measured evapotranspiration, LE / lambda, clipped at 0, gaps filled with the mean of the
  same half-hour of the day over +-3 days).
Soil: a texture class (Carsel & Parrish 1988) with theta_s from the deepest sensor when the water table stood at least
--sat_margin_m above it (median over the site's whole record) and theta_r = min(class value, driest reading in the
record).

Usage (via sbatch): hpc/run_script_cpu.slurm scripts/prepare_site_amf.py --csv <BASE_HH csv> \
        --heights data/ameriflux/BASE_MeasurementHeight_20260908.csv --site US-HB2 --profile 3 \
        --start 2024-05-23 --end 2024-07-13 --out output/sites/US-HB2_2024-06-22/record.npz
"""
import argparse
import json
import os
import re

import numpy as np

import plot_site_amf as psa

CLASSES = {'sand': {'theta_r': 0.045, 'theta_s': 0.43, 'alpha': 14.5, 'n': 2.68, 'Ks': 8.25e-5}}   # C&P 1988


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--heights', required=True)
    ap.add_argument('--site', required=True)
    ap.add_argument('--profile', default='3')
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    ap.add_argument('--texture', default='sand', choices=sorted(CLASSES))
    ap.add_argument('--sat_margin_m', type=float, default=0.10)
    ap.add_argument('--soil', default=None, help='theta_s,theta_r,alpha,n,Ks given directly (e.g. from field_retention.py); '
                    'replaces the texture class and the theta_s / theta_r estimates')
    ap.add_argument('--wtd', default='WTD_1_1_1', help='water-table column (scoring only)')
    ap.add_argument('--le', default='LE', help='latent-heat column (default: LE or its first alias)')
    ap.add_argument('--wtd_despike_m', type=float, default=None, help='drop well readings farther than this from the '
                    'running 5-hour median (isolated glitches) before interpolating; off by default')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    hts = psa.heights(args.heights, args.site)
    import csv
    with open(args.csv, newline='') as fh:
        rd = csv.reader(fh)
        next(rd), next(rd)
        head = next(rd)
    swc = sorted([c for c in head if re.fullmatch(rf'SWC_{args.profile}_\d+_1', c)], key=lambda c: -hts[c])
    depths = [round(-hts[c], 4) for c in swc]
    t, v = psa.read(args.csv, swc + [args.wtd, args.le])
    v['WTD_1_1_1'], v['LE'] = v[args.wtd], v[args.le]
    for c in swc:
        v[c] = np.where((v[c] > 0.0) & (v[c] <= 100.0), v[c] / 100.0, np.nan)
    zb_all = -v['WTD_1_1_1']

    # soil: class values, theta_s from the deepest sensor when the water table stood above it, theta_r from the data
    soil = dict(CLASSES[args.texture])
    deep = swc[-1]
    sub = np.isfinite(zb_all) & (zb_all <= depths[-1] - args.sat_margin_m) & np.isfinite(v[deep])
    theta_s = float(np.median(v[deep][sub])) if sub.sum() > 48 else soil['theta_s']
    driest = float(np.nanmin(np.stack([v[c] for c in swc])))
    soil['theta_s'] = theta_s
    soil['theta_r'] = min(soil['theta_r'], max(driest, 0.0))
    if args.soil:
        soil = dict(zip(['theta_s', 'theta_r', 'alpha', 'n', 'Ks'], (float(x) for x in args.soil.split(','))))
        sub = np.zeros(1, dtype=bool)                              # theta_s given, not estimated here

    t0, t1 = np.datetime64(args.start), np.datetime64(args.end)
    m = (t >= t0) & (t <= t1)
    tm = t[m]
    t_days = (tm - t0).astype('timedelta64[m]').astype(float) / 1440.0
    th = np.stack([v[c][m] for c in swc], axis=1)
    zb = zb_all[m]
    n_spikes = 0
    if args.wtd_despike_m is not None:
        pad = np.pad(zb, 5, mode='edge')
        med = np.nanmedian(np.lib.stride_tricks.sliding_window_view(pad, 11), axis=1)
        spike = np.isfinite(zb) & (np.abs(zb - med) > args.wtd_despike_m)
        n_spikes = int(spike.sum())
        zb = np.where(spike, np.nan, zb)
    ok = np.isfinite(zb)
    zb = np.interp(t_days, t_days[ok], zb[ok])                     # scoring only; window is complete

    et = v['LE'][m] / 2.45e6 * 86400.0                               # mm/d
    et = np.clip(et, 0.0, None)
    tod = ((tm - tm.astype('datetime64[D]')).astype('timedelta64[m]').astype(int) // 30)
    filled = et.copy()
    for i in np.where(~np.isfinite(et))[0]:
        same = (tod == tod[i]) & (np.abs(t_days - t_days[i]) <= 3.0) & np.isfinite(et)
        filled[i] = et[same].mean() if same.any() else 0.0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, t_days=t_days, sensor_depths_m=np.array(depths), theta_obs=th, zb=zb,
             soil=np.array([soil['theta_s'], soil['theta_r'], soil['alpha'], soil['n'], soil['Ks']]),
             t_nodes=t_days, tp_nodes_mmd=filled, start=str(t0))
    info = {'site': args.site, 'profile': args.profile, 'start': str(t0), 'end': str(t1), 'depths_m': depths,
            'soil': soil, 'soil_given': bool(args.soil),
            'theta_s_from': 'given' if args.soil else (f'{deep} median over {int(sub.sum())} half-hours with the water '
                                                       f'table >= {args.sat_margin_m} m above it'),
            'driest_reading': driest,
            'valid_frac': {c: float(np.isfinite(th[:, j]).mean()) for j, c in enumerate(swc)},
            'et_filled_frac': float((~np.isfinite(et)).mean()), 'et_mean_mmd': float(filled.mean()),
            'zb_range_m': [float(zb.min()), float(zb.max())], 'well_spikes_removed': n_spikes}
    with open(os.path.splitext(args.out)[0] + '.json', 'w') as fh:
        json.dump(info, fh, indent=1)
    print(f"[prep] {json.dumps(info)}", flush=True)
    print(f"[prep] wrote {args.out}", flush=True)


if __name__ == '__main__':
    main()
