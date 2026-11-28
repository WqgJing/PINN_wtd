"""Screen ICOS ETC L2 Meteosens files (zip with one half-hourly CSV) for a real-data test: per station, every water-table
column (share present, 5 / 50 / 95 % of its values), every soil-moisture profile (number of depth positions, share
present), and frost (share of the shallowest soil-temperature readings <= 0 C). Missing values -9999.

Usage (via sbatch): hpc/run_script_cpu.slurm scripts/screen_icos.py --dir data/icos --out output/sites/icos_screen.json
"""
import argparse
import csv
import glob
import io
import json
import os
import re
import zipfile

import numpy as np


def screen(path):
    with zipfile.ZipFile(path) as z:
        name = [n for n in z.namelist() if n.lower().endswith('.csv')][0]
        with z.open(name) as fh:
            rd = csv.reader(io.TextIOWrapper(fh, encoding='utf-8', errors='replace'))
            head = next(rd)
            wcols = [i for i, c in enumerate(head) if re.fullmatch(r'WTD_\d+_\d+_\d+', c)]
            scols = [i for i, c in enumerate(head) if re.fullmatch(r'SWC_\d+_\d+_\d+', c)]
            tcols = [i for i, c in enumerate(head) if re.fullmatch(r'TS_\d+_1_\d+', c)]
            keep = wcols + scols + tcols
            ts0 = head.index('TIMESTAMP_START') if 'TIMESTAMP_START' in head else 0
            vals = {i: [] for i in keep}
            first = last = None
            n = 0
            for row in rd:
                n += 1
                if first is None:
                    first = row[ts0]
                last = row[ts0]
                for i in keep:
                    try:
                        v = float(row[i])
                    except (ValueError, IndexError):
                        v = np.nan
                    vals[i].append(np.nan if v <= -9990 else v)
    out = {'file': os.path.basename(path), 'rows': n, 'start': first, 'end': last, 'water_table': {}, 'profiles': {},
           'frost_share': None}
    for i in wcols:
        x = np.array(vals[i])
        ok = np.isfinite(x)
        if ok.any():
            q = np.percentile(x[ok], [5, 50, 95])
            out['water_table'][head[i]] = {'present': float(ok.mean()), 'p05': float(q[0]), 'p50': float(q[1]),
                                           'p95': float(q[2])}
    prof = {}
    for i in scols:
        _, p, h, r = head[i].split('_')
        prof.setdefault(p, {})[h] = float(np.isfinite(np.array(vals[i])).mean())
    out['profiles'] = {p: {'depth_positions': len(d), 'present_by_position': d} for p, d in sorted(prof.items())}
    if tcols:
        x = np.concatenate([np.array(vals[i]) for i in tcols])
        ok = np.isfinite(x)
        out['frost_share'] = float(np.mean(x[ok] <= 0.0)) if ok.any() else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    rep = {}
    for z in sorted(glob.glob(os.path.join(args.dir, '*.zip'))):
        st = os.path.basename(z).split('.')[0]
        try:
            r = screen(z)
        except Exception as e:                                  # noqa: BLE001
            print(f"[icos] {st}: cannot read: {e}", flush=True)
            continue
        rep[st] = r
        wt = '; '.join(f"{k} {100 * v['present']:.0f} % [{v['p05']:.2f} {v['p50']:.2f} {v['p95']:.2f}]"
                       for k, v in r['water_table'].items())
        pr = ', '.join(f"P{p}:{d['depth_positions']}" for p, d in r['profiles'].items())
        fr = f"{100 * r['frost_share']:.0f} %" if r['frost_share'] is not None else 'n/a'
        print(f"[icos] {st} {r['start'][:8]}-{r['end'][:8]} | water table {wt} | SWC profiles {pr} | frost {fr}",
              flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(rep, f, indent=1)
    print(f"[icos] wrote {args.out}", flush=True)


if __name__ == '__main__':
    main()
