"""Observations at the USGS Panola Mountain Research Watershed (Georgia): daily rain and Priestley-Taylor PET, hourly
soil moisture at 15, 40 and 70 cm (soil moisture pit), and hourly water-table depth of the transect-3 wells (the nearest
to the pit: t3_65m ~80 m, t3_35m ~90 m, t3_8m ~105 m). Lines only.

Data: insitu_validation/data/panola_mountain (USGS ScienceBase, 2017 - 2020).
Usage (via sbatch): hpc/run_script_cpu.slurm scripts/plot_site_panola.py --data <panola_mountain dir> \
        --out output/sites/panola_observations.png [--start 2017-04-15 --end 2019-10-24]
"""
import argparse
import csv
import os
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

WELLS = [('t3_65m', '#2a78d6'), ('t3_35m', '#1baf7a'), ('t3_8m', '#8a4fd1')]


def num(s):
    try:
        return float(s)
    except ValueError:
        return np.nan


def read_met(path):
    t, rain, pet, sm = [], [], [], {15: [], 40: [], 70: []}
    with open(path, newline='') as fh:
        for r in csv.DictReader(fh):
            t.append(datetime.strptime(r['datetime'], '%m/%d/%Y %H:%M'))
            rain.append(num(r['ppt_cm']) * 10.0)                     # cm -> mm per hour
            pet.append(num(r['pt_pet_mm']))
            for d in sm:
                sm[d].append(num(r[f'vwc_{d}cm']))
    return np.array(t, dtype='datetime64[m]'), np.array(rain), np.array(pet), {d: np.array(v) for d, v in sm.items()}


def read_wells(path, ids):
    """Hourly means of the depth to the water table (m below the surface, positive down) for the given wells."""
    acc = {i: {} for i in ids}
    with open(path, newline='') as fh:
        rd = csv.reader(fh, delimiter='\t')
        next(rd)
        for r in rd:
            if r[0] not in acc:
                continue
            v = num(r[2])
            if np.isnan(v):
                continue
            h = r[1][:13]                                            # 'YYYY-mm-dd HH'
            s = acc[r[0]].setdefault(h, [0.0, 0])
            s[0] += -v                                               # m_bls: negative below the surface
            s[1] += 1
    out = {}
    for i, d in acc.items():
        keys = sorted(d)
        out[i] = (np.array([np.datetime64(k.replace(' ', 'T') + ':00') for k in keys], dtype='datetime64[m]'),
                  np.array([d[k][0] / d[k][1] for k in keys]))
    return out


def daily_sum(t, x):
    d = t.astype('datetime64[D]')
    u, inv = np.unique(d, return_inverse=True)
    return u, np.bincount(inv, weights=np.nan_to_num(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--start', default='2017-04-15')
    ap.add_argument('--end', default='2019-10-24')
    args = ap.parse_args()
    t0, t1 = np.datetime64(args.start), np.datetime64(args.end)
    t, rain, pet, sm = read_met(os.path.join(args.data, 'met_sm_pet_2017to2019.csv'))
    wells = read_wells(os.path.join(args.data, 'waterlevel_2017_2020.txt'), [w for w, _ in WELLS])
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1.4, 1.4]})
    td, rd = daily_sum(t, rain)
    _, pd_ = daily_sum(t, pet)
    m = (td >= t0) & (td <= t1)
    axes[0].plot(td[m], rd[m], color='#2a78d6', lw=1.0, label='rain')
    axes[0].plot(td[m], pd_[m], color='#eb6834', lw=1.0, label='potential ET (Priestley-Taylor)')
    axes[0].set_ylabel('water flux (mm/d)')
    axes[0].legend(frameon=False, fontsize=9)
    m = (t >= t0) & (t <= t1)
    for d, col in ((15, '#2a78d6'), (40, '#1baf7a'), (70, '#eb6834')):
        axes[1].plot(t[m], sm[d][m], color=col, lw=1.0, label=f'{d} cm')
    axes[1].set_ylabel('soil moisture (m$^3$/m$^3$)')
    axes[1].legend(frameon=False, fontsize=9, ncol=3)
    for w, col in WELLS:
        tw, zw = wells[w]
        k = (tw >= t0) & (tw <= t1)
        axes[2].plot(tw[k], zw[k], color=col, lw=1.0, label=f'well {w}')
    axes[2].invert_yaxis()
    axes[2].set_ylabel('water-table depth (m)')
    axes[2].legend(frameon=False, fontsize=9, ncol=3)
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[2].set_xlabel('date')
    for ax in axes:
        ax.grid(True, color='#e1e0d9', lw=0.6)
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=120)
    plt.close(fig)
    for d in (15, 40, 70):
        x = sm[d][m]
        print(f"[panola] soil moisture {d} cm: {np.nanmin(x):.3f} .. {np.nanmax(x):.3f}, valid {100 * np.isfinite(x).mean():.0f} %",
              flush=True)
    for w, _ in WELLS:
        tw, zw = wells[w]
        k = (tw >= t0) & (tw <= t1)
        print(f"[panola] well {w}: {zw[k].min():.2f} .. {zw[k].max():.2f} m, hours {k.sum()}", flush=True)
    print(f"[panola] wrote {args.out}", flush=True)


if __name__ == '__main__':
    main()
