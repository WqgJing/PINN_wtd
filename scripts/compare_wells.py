"""A site run's water table against two wells: RMSE, bias and slope against each, over the times the shallower well is
not bottomed out (< --cap_m) and over the whole window. Prints only.

Usage (via sbatch): hpc/run_script_cpu.slurm scripts/compare_wells.py --csv <BASE_HH csv> --start 2020-05-06 \
        --runs runs/ho1j_1m,runs/ho1j_A --wells WTD_1_1_1,WTD_2_1_1 --cap_m 0.80
"""
import argparse
import os

import numpy as np

import plot_site_amf as psa


def stats(est, well):
    e = est - well
    return (100 * np.sqrt(np.mean(e ** 2)), 100 * e.mean(), np.polyfit(well, est, 1)[0] if np.ptp(well) > 0 else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--start', required=True, help='record start (day 0 of the run)')
    ap.add_argument('--runs', required=True)
    ap.add_argument('--wells', default='WTD_1_1_1,WTD_2_1_1')
    ap.add_argument('--cap_m', type=float, default=0.80)
    args = ap.parse_args()
    wells = args.wells.split(',')
    t, v = psa.read(args.csv, wells)
    td = (t - np.datetime64(args.start)).astype('timedelta64[m]').astype(float) / 1440.0
    for r in args.runs.split(','):
        p = np.load(os.path.join(r, 'predictions.npz'))
        tr = p['t_days'] + float(p['t_offset'])
        est = p['zb_est']
        w = {c: np.interp(tr, td, -v[c]) for c in wells}
        ok = w[wells[0]] < args.cap_m
        line = f"[wells] {r}: hours with {wells[0]} < {args.cap_m} m: {100 * ok.mean():.0f} %"
        for c in wells:
            a, b, s = stats(est[ok], w[c][ok])
            line += f" | vs {c} there: RMSE {a:.1f} cm bias {b:+.1f} slope {s:.2f}"
        a, b, s = stats(est, w[wells[1]])
        line += f" | vs {wells[1]} whole window: RMSE {a:.1f} bias {b:+.1f}"
        print(line, flush=True)
    d = w[wells[0]][ok] - w[wells[1]][ok]
    print(f"[wells] {wells[0]} minus {wells[1]} where uncapped: mean {100 * d.mean():+.1f} cm, range "
          f"{100 * d.min():+.1f} .. {100 * d.max():+.1f} cm", flush=True)


if __name__ == '__main__':
    main()
