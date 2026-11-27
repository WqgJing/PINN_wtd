"""Scan AmeriFlux BASE half-hourly files for sites usable in a real-data test: soil-moisture profiles (sensors at one
horizontal position, several depths, depths from the site's BIF metadata) and a water-table record. For each site and
profile prints the sensor depths, the water-table range, and the longest run of days on which the water table and at
least --min_depths of that profile's sensors all have >= 80 % of their half-hours. Standard library and numpy only
(the BIF .xlsx is read as the zip of XML it is).

Usage (via sbatch): hpc/run_script_cpu.slurm scripts/scan_sites.py --root <uploaded_data> --out output/sites/scan.json
"""
import argparse
import csv
import glob
import json
import os
import re
import zipfile
import xml.etree.ElementTree as ET

import numpy as np

NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def xlsx_rows(path):
    """Rows (lists of strings) of the first worksheet of an .xlsx file."""
    with zipfile.ZipFile(path) as z:
        shared = []
        if 'xl/sharedStrings.xml' in z.namelist():
            for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si', NS):
                shared.append(''.join(t.text or '' for t in si.iter('{%s}t' % NS['m'])))
        sheet = sorted(n for n in z.namelist() if n.startswith('xl/worksheets/sheet'))[0]
        rows = []
        for r in ET.fromstring(z.read(sheet)).iter('{%s}row' % NS['m']):
            cells = {}
            for c in r.findall('m:c', NS):
                col = re.match(r'[A-Z]+', c.get('r')).group(0)
                v = c.find('m:v', NS)
                if v is None:
                    is_ = c.find('m:is', NS)
                    val = ''.join(t.text or '' for t in is_.iter('{%s}t' % NS['m'])) if is_ is not None else ''
                else:
                    val = shared[int(v.text)] if c.get('t') == 's' else v.text
                cells[col] = val
            idx = lambda s: sum((ord(ch) - 64) * 26 ** i for i, ch in enumerate(reversed(s)))
            if cells:
                n = max(idx(k) for k in cells)
                row = [''] * n
                for k, val in cells.items():
                    row[idx(k) - 1] = val
                rows.append(row)
        return rows


def bif_heights(path):
    """{variable name: height in m (negative = below the surface)} from a BIF file (last entry per variable)."""
    try:
        rows = xlsx_rows(path)
    except Exception as e:                                          # noqa: BLE001
        print(f"[scan] cannot read {path}: {e}", flush=True)
        return {}
    head = [h.upper() for h in rows[0]]
    col = {h: i for i, h in enumerate(head)}
    groups = {}
    for r in rows[1:]:
        r = r + [''] * (len(head) - len(r))
        if r[col['VARIABLE_GROUP']] != 'GRP_VAR_INFO':
            continue
        groups.setdefault(r[col['GROUP_ID']], {})[r[col['VARIABLE']]] = r[col['DATAVALUE']]
    out = {}
    for d in groups.values():
        name, h = d.get('VAR_INFO_VARNAME'), d.get('VAR_INFO_HEIGHT')
        try:
            out[str(name)] = float(h)
        except (TypeError, ValueError):
            pass
    return out


def read_columns(path, wanted):
    """Day index (days since the first record) and {column: float array} for the wanted columns (-9999 -> nan)."""
    with open(path, newline='') as fh:
        rd = csv.reader(fh)
        next(rd), next(rd)
        head = next(rd)
        idx = {c: head.index(c) for c in wanted}
        ts, vals = [], {c: [] for c in wanted}
        for row in rd:
            ts.append(row[0])
            for c, i in idx.items():
                try:
                    v = float(row[i])
                except (ValueError, IndexError):
                    v = np.nan
                vals[c].append(np.nan if v == -9999 else v)
    days = np.array([f'{s[:4]}-{s[4:6]}-{s[6:8]}' for s in ts], dtype='datetime64[D]')
    return days, {c: np.array(v) for c, v in vals.items()}


def daily_ok(days, x, thr=0.8):
    u, inv = np.unique(days, return_inverse=True)
    frac = np.bincount(inv, weights=np.isfinite(x).astype(float)) / np.bincount(inv)
    return u, frac >= thr


def longest_run(ok):
    best, cur, end = 0, 0, -1
    for i, v in enumerate(ok):
        cur = cur + 1 if v else 0
        if cur > best:
            best, end = cur, i
    return best, end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--min_depths', type=int, default=3)
    args = ap.parse_args()
    report = {}
    for f in sorted(glob.glob(os.path.join(args.root, 'AMF_*_BASE-BADM_*', 'AMF_*_BASE_HH_*.csv'))):
        site = os.path.basename(f).split('_')[1]
        with open(f, newline='') as fh:
            rd = csv.reader(fh)
            next(rd), next(rd)
            head = next(rd)
        wcols = [c for c in head if re.fullmatch(r'WTD(_\d+_\d+_\d+)?', c)]
        scols = [c for c in head if re.fullmatch(r'SWC_\d+_\d+_\d+', c)]
        profiles = {}
        for c in scols:
            profiles.setdefault(c.split('_')[1], []).append(c)
        profiles = {p: cs for p, cs in profiles.items() if len(cs) >= args.min_depths}
        if not wcols or not profiles:
            continue
        xl = glob.glob(os.path.join(os.path.dirname(f), '*BIF*.xlsx'))
        heights = bif_heights(xl[0]) if xl else {}
        days, v = read_columns(f, wcols + [c for cs in profiles.values() for c in cs])
        rep = {'file': f, 'years': [str(days[0])[:4], str(days[-1])[:4]], 'water_table': {}, 'profiles': {}}
        for w in wcols:
            x = v[w]
            if np.isfinite(x).any():
                rep['water_table'][w] = {'valid_frac': float(np.isfinite(x).mean()), 'min_m': float(np.nanmin(x)),
                                         'max_m': float(np.nanmax(x)), 'height_m': heights.get(w)}
        if not rep['water_table']:
            continue
        wbest = max(rep['water_table'], key=lambda k: rep['water_table'][k]['valid_frac'])
        u, wok = daily_ok(days, v[wbest])
        for p, cols in sorted(profiles.items()):
            sok = np.stack([daily_ok(days, v[c])[1] for c in cols], axis=1)
            ok = (sok.sum(axis=1) >= args.min_depths) & wok
            n, end = longest_run(ok)
            rep['profiles'][p] = {
                'sensors': {c: {'height_m': heights.get(c), 'valid_frac': float(np.isfinite(v[c]).mean())}
                            for c in cols},
                'days_all_ok': int(ok.sum()), 'longest_run_days': int(n),
                'longest_run': [str(u[end - n + 1]), str(u[end])] if n else None}
        report[site] = rep
        wt = rep['water_table'][wbest]
        print(f"[scan] {site} {rep['years']} water table {wbest} {wt['min_m']:.2f} .. {wt['max_m']:.2f} "
              f"(valid {100 * wt['valid_frac']:.0f} %, height {wt['height_m']})", flush=True)
        for p, pr in rep['profiles'].items():
            hs = ', '.join(f"{c}:{s['height_m']}" for c, s in pr['sensors'].items())
            print(f"[scan]   profile {p}: {hs} | days all ok {pr['days_all_ok']}, longest run "
                  f"{pr['longest_run_days']} d {pr['longest_run']}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(report, fh, indent=1)
    print(f"[scan] wrote {args.out}", flush=True)


if __name__ == '__main__':
    main()
