"""PINN vs FEM on matched cases (sensor set x constant start): accuracy, wall time and time to solution, scored with the
rules fixed in the notes before the runs:
  reproduces       water-table RMSE <= --tol_m AND fit at noise level (PINN data misfit <= 1.0; FEM cost <= 1.2)
  time to solution first logged time after which the RMSE stays <= --tol_m to the end (only for runs that reproduce)
  per sensor count "PINN faster" if the PINN's time to solution is lower in a majority of that count's cases (a success
  against a failure counts for the successful method; both failing counts for neither) AND the median FEM/PINN ratio over
  cases where both succeed is >= 1.2 (or there is no such case); "FEM faster" symmetric; else "no clear difference".
Writes <out>.json and <out>.png (time to solution, failures at the right edge as crosses; wall time to each
method's own stop; final RMSE).

Usage (via sbatch): hpc/run_script_cpu.slurm scripts/compare_speed.py --out output/compare/speed_1m2m
"""
import argparse
import csv
import json
import os
import re

import numpy as np
import matplotlib
import matplotlib.ticker
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SETS = {4: ['5-20-40-60'], 3: ['5-20-40', '5-20-60', '5-40-60', '20-40-60'],
        2: ['5-20', '5-40', '5-60', '20-40', '20-60', '40-60']}
STARTS = ['1m', '2m']


def load_json(path):
    with open(path) as f:
        return json.load(f)


def settle_time(times, errs, tol):
    """First time after which every later error is <= tol (None if the last one is above)."""
    t_set = None
    for t, e in zip(times, errs):
        if e <= tol:
            if t_set is None:
                t_set = t
        else:
            t_set = None
    return t_set


def gpu_of(tag):
    race = os.path.join('logs', 'race', f'{tag}.txt')
    if not os.path.exists(race):
        return None
    m = re.search(r'started (\d+)', open(race).read())
    if not m:
        return None
    log = os.path.join('logs', f'gpu_{m.group(1)}.out')
    if not os.path.exists(log):
        return None
    for line in open(log):
        if 'V100' in line or 'H200' in line or 'A100' in line:
            return line.split(',')[0].strip()
    return None


def pinn_case(tag, tol):
    run = os.path.join('runs', tag)
    if not os.path.exists(os.path.join(run, 'summary.json')):
        return {'run': run, 'finished': False}
    s = load_json(os.path.join(run, 'summary.json'))
    with open(os.path.join(run, 'log.csv')) as f:
        rows = list(csv.DictReader(f))
    misfit = float(rows[-1]['data'])
    rmse = float(s['zb_vs_truth']['rmse_m'])
    ok = rmse <= tol and misfit <= 1.0
    t = [float(r['wall_s']) for r in rows]
    e = [float(r['zb_rmse_m']) for r in rows]
    return {'run': run, 'finished': True, 'rmse_m': rmse, 'fit': misfit, 'reproduces': ok,
            'wall_s': float(s['wall_s']), 'tts_s': settle_time(t, e, tol) if ok else None, 'hardware': gpu_of(tag)}


def fem_case(tag, tol):
    run = os.path.join('runs', tag)
    if not os.path.exists(os.path.join(run, 'summary.json')):
        return {'run': run, 'finished': False}
    s = load_json(os.path.join(run, 'summary.json'))
    rmse, cost = float(s['zb_vs_truth']['rmse_m']), float(s['cost'])
    ok = rmse <= tol and cost <= 1.2
    h = s['history']
    t = [float(x['wall_s']) for x in h]
    e = [float(x['zb_rmse_m']) for x in h]
    return {'run': run, 'finished': True, 'rmse_m': rmse, 'fit': cost, 'reproduces': ok, 'iterations': len(h),
            'wall_s': float(s['wall_fit_s']), 'tts_s': settle_time(t, e, tol) if ok else None,
            'hardware': '8 CPU cores'}


def verdict(cases):
    p_win = f_win = 0
    ratios = []
    for c in cases:
        p, f = c['pinn'], c['fem']
        ps, fs = p.get('reproduces', False), f.get('reproduces', False)
        if ps and fs:
            ratios.append(f['tts_s'] / p['tts_s'])
            if p['tts_s'] < f['tts_s']:
                p_win += 1
            elif f['tts_s'] < p['tts_s']:
                f_win += 1
        elif ps:
            p_win += 1
        elif fs:
            f_win += 1
    n = len(cases)
    med = float(np.median(ratios)) if ratios else None
    if p_win > n / 2 and (med is None or med >= 1.2):
        v = 'PINN faster'
    elif f_win > n / 2 and (med is None or med <= 1 / 1.2):
        v = 'FEM faster'
    else:
        v = 'no clear difference'
    return {'cases': n, 'pinn_wins': p_win, 'fem_wins': f_win, 'median_fem_over_pinn': med, 'verdict': v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, help='output path without extension')
    ap.add_argument('--tol_m', type=float, default=0.05)
    ap.add_argument('--cap_min', type=float, default=1000.0, help='where failures are drawn, minutes')
    args = ap.parse_args()
    cases = []
    for n, sets in SETS.items():
        for st in sets:
            for s0 in STARTS:
                cases.append({'sensors': n, 'set': st, 'start': s0,
                              'pinn': pinn_case(f's{n}_{st}_{s0}', args.tol_m),
                              'fem': fem_case(f'fs{n}_{st}_{s0}', args.tol_m)})
    report = {'cases': cases, 'verdict_by_sensor_count': {}}
    for n in SETS:
        sub = [c for c in cases if c['sensors'] == n and c['pinn']['finished'] and c['fem']['finished']]
        report['verdict_by_sensor_count'][str(n)] = verdict(sub) if sub else None
    alls = [c for c in cases if c['pinn']['finished'] and c['fem']['finished']]
    report['verdict_all'] = verdict(alls) if alls else None
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out + '.json', 'w') as f:
        json.dump(report, f, indent=1)

    fmt = lambda v, k=1.0: '-' if v is None else f'{v * k:.1f}'
    print('| sensors | set | start | PINN rmse cm | PINN fit | PINN tts min | PINN wall min | FEM rmse cm | FEM fit | '
          'FEM tts min | FEM wall min | FEM iter |', flush=True)
    print('|---|---|---|---|---|---|---|---|---|---|---|---|', flush=True)
    for c in cases:
        p, f = c['pinn'], c['fem']
        print(f"| {c['sensors']} | {c['set']} | {c['start']} | {fmt(p.get('rmse_m'), 100)} | {fmt(p.get('fit'), 1)} | "
              f"{fmt(p.get('tts_s'), 1 / 60)} | {fmt(p.get('wall_s'), 1 / 60)} | {fmt(f.get('rmse_m'), 100)} | "
              f"{fmt(f.get('fit'), 1)} | {fmt(f.get('tts_s'), 1 / 60)} | {fmt(f.get('wall_s'), 1 / 60)} | "
              f"{f.get('iterations', '-')} |", flush=True)
    print(f"[speed] verdict by sensor count: {report['verdict_by_sensor_count']}", flush=True)
    print(f"[speed] verdict over all cases: {report['verdict_all']}", flush=True)
    print(f"[speed] PINN hardware: {sorted({str(c['pinn'].get('hardware')) for c in cases})}", flush=True)

    labels = [f"{c['sensors']} sensors {c['set']} cm, start {c['start']}" for c in cases]
    y = np.arange(len(cases))[::-1]
    fig, (a1, a3, a2) = plt.subplots(1, 3, figsize=(16, 0.42 * len(cases) + 1.5), sharey=True)
    for key, col, dy, name in (('pinn', '#eb6834', 0.15, 'PINN'), ('fem', '#2a78d6', -0.15, 'FEM')):
        tt = [c[key].get('tts_s') for c in cases]
        ok = [i for i, v in enumerate(tt) if v is not None]
        bad = [i for i, v in enumerate(tt) if v is None and c_fin(cases[i][key])]
        a1.plot([tt[i] / 60 for i in ok], [y[i] + dy for i in ok], color=col, marker='o', lw=0, ms=6, label=name)
        if bad:
            a1.plot([args.cap_min] * len(bad), [y[i] + dy for i in bad], color=col, marker='x', lw=0, ms=7,
                    label=f'{name}, did not reproduce')
        wl = [c[key].get('wall_s') for c in cases]
        fw = [i for i, v in enumerate(wl) if v is not None]
        a3.plot([wl[i] / 60 for i in fw], [y[i] + dy for i in fw], color=col, marker='o', lw=0, ms=6, label=name)
        rm = [c[key].get('rmse_m') for c in cases]
        fin = [i for i, v in enumerate(rm) if v is not None]
        a2.plot([100 * rm[i] for i in fin], [y[i] + dy for i in fin], color=col, marker='o', lw=0, ms=6, label=name)
    a1.set_xscale('log')
    a1.set_xlabel('time to solution (min)')
    a3.set_xscale('log')
    a3.set_xlabel('wall time to the method\'s own stop (min)')
    a2.set_xscale('log')
    a2.set_xlabel('water-table RMSE vs truth (cm)')
    for a, ticks in ((a1, [5, 10, 20, 50, 100, 200, 500, 1000]), (a3, [5, 10, 20, 50, 100, 200, 500, 1000]),
                     (a2, [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])):
        a.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
        a.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
        a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a1.set_yticks(y)
    a1.set_yticklabels(labels, fontsize=8)
    for a in (a1, a3, a2):
        a.grid(True, axis='x', color='#e1e0d9', lw=0.6)
        a.spines[['top', 'right']].set_visible(False)
    a1.legend(frameon=False, fontsize=8, loc='upper right')
    fig.tight_layout()
    fig.savefig(args.out + '.png', dpi=130)
    plt.close(fig)
    print(f"[speed] wrote {args.out}.json / .png", flush=True)


def c_fin(d):
    return d.get('finished', False)


if __name__ == '__main__':
    main()
