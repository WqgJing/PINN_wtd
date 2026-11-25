"""Exploratory (user's question, 25 Nov): what information do NEIGHBOURING SENSORS carry about the water table, using the
sensor records alone (no rain, no ET)? Same weather and soil, water table at different depths; signatures between the
5 / 20 / 40 / 60 cm sensors, noise-free and with noise 0.005:
  - variability with depth (std of 6 h means) and its ratio to the top sensor;
  - lag and correlation of 6 h increments between neighbours (positive lag: the deeper sensor follows);
  - linear predictability both ways (Granger-type): share of a sensor's increment variance explained by the
    neighbour's past beyond its own past; 'down' = upper -> lower, 'up' = lower -> upper;
  - diurnal cycle: amplitude (and its ratio to the top) and phase lag at each depth (1-cycle-per-day fit, hourly);
  - dry spells (top sensor falling for 48 h): share of hours with UPWARD total-head gradient between 40 and 60 cm
    (pressure heads from the known retention curve), i.e. water moving up toward the roots / surface;
  - time near saturation of the deepest sensor, and its at-rest water-table reading.

Usage (via sbatch): hpc/run_script_cpu.slurm scripts/sensor_signatures.py --out output/signatures \
        --rec very_shallow=output/syn_nc2_spring23 --rec shallow=output/syn_nc2_spring23_deep_dt16 ...
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

DEPTHS = [0.05, 0.20, 0.40, 0.60]


def h_from_theta(theta, ts, tr, alpha, n):
    m = 1.0 - 1.0 / n
    se = np.clip((theta - tr) / (ts - tr), 1e-6, 1.0 - 1e-6)
    return -(se ** (-1.0 / m) - 1.0) ** (1.0 / n) / alpha


def block_mean(x, k):
    n = (x.shape[0] // k) * k
    return x[:n].reshape(-1, k, *x.shape[1:]).mean(axis=1)


def lagged_xcorr(a, b, max_lag):
    """corr(a[t], b[t + lag]) for lag in [-max_lag, max_lag]; positive lag: b follows a."""
    out = []
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x, y = a[:len(a) - lag], b[lag:]
        else:
            x, y = a[-lag:], b[:len(b) + lag]
        out.append(np.corrcoef(x, y)[0, 1])
    return np.arange(-max_lag, max_lag + 1), np.array(out)


def granger_gain(target, other, p):
    """Share of the target's residual variance (given its own p past values) removed by adding other's p past values."""
    n = len(target)
    rows = range(p, n)
    own = np.array([target[i - p:i][::-1] for i in rows])
    oth = np.array([other[i - p:i][::-1] for i in rows])
    y = target[p:]
    xa = np.column_stack([np.ones(len(y)), own])
    xb = np.column_stack([xa, oth])
    sse = lambda x: float(np.sum((y - x @ np.linalg.lstsq(x, y, rcond=None)[0]) ** 2))
    a, b = sse(xa), sse(xb)
    return (a - b) / a


def diurnal(x_h, t_days):
    """Amplitude and phase (hours of the peak after midnight) of the 1-cycle-per-day component of x minus its 24 h
    running mean."""
    k = 24
    pad = np.pad(x_h, (k // 2, k // 2), mode='edge')
    trend = np.convolve(pad, np.ones(k) / k, mode='same')[k // 2:-k // 2]
    r = x_h - trend
    w = 2 * np.pi * t_days
    A = np.column_stack([np.cos(w), np.sin(w)])
    c, s = np.linalg.lstsq(A, r, rcond=None)[0]
    amp = float(np.hypot(c, s))
    phase_h = float((np.arctan2(s, c) / (2 * np.pi) * 24.0) % 24.0)
    return amp, phase_h


def signatures(th, t, soil):
    ts, tr, alpha, n = soil[0], soil[1], soil[2], soil[3]
    out = {}
    x6 = block_mean(th, 6)
    out['mean'] = [float(v) for v in x6.mean(axis=0)]
    out['std'] = [float(v) for v in x6.std(axis=0)]
    out['std_ratio_to_top'] = [float(v / out['std'][0]) for v in out['std']]
    d6 = np.diff(x6, axis=0)
    lag, cc, gd, gu = [], [], [], []
    for j in range(3):
        lags, c = lagged_xcorr(d6[:, j], d6[:, j + 1], 8)
        i = int(np.argmax(c))
        lag.append(float(6 * lags[i]))
        cc.append(float(c[i]))
        gd.append(granger_gain(d6[:, j + 1], d6[:, j], 4))
        gu.append(granger_gain(d6[:, j], d6[:, j + 1], 4))
    out['lag_h'], out['xcorr'], out['granger_down'], out['granger_up'] = lag, cc, gd, gu
    amps, phases = [], []
    for j in range(th.shape[1]):
        a, p = diurnal(th[:, j], t)
        amps.append(a)
        phases.append(p)
    out['diurnal_amp'] = amps
    out['diurnal_amp_ratio_to_top'] = [float(a / amps[0]) for a in amps]
    out['diurnal_phase_h'] = phases
    k = 24
    sm = np.convolve(np.pad(th[:, 0], (k, k), mode='edge'), np.ones(k) / k, mode='same')[k:-k]
    dsm = np.gradient(sm)
    dry = np.zeros(len(t), dtype=bool)
    for i in range(48, len(t)):
        dry[i] = np.all(dsm[i - 48:i + 1] < 0)
    h40 = h_from_theta(th[:, 2], ts, tr, alpha, n)
    h60 = h_from_theta(th[:, 3], ts, tr, alpha, n)
    up = (h60 - 0.60) > (h40 - 0.40)                     # total head higher at 60 cm: water moves up
    out['dry_hours'] = int(dry.sum())
    out['dry_upward_share_40_60'] = float(up[dry].mean()) if dry.any() else None
    out['all_upward_share_40_60'] = float(up.mean())
    out['deep_near_saturation_share'] = float(np.mean(th[:, 3] >= ts - 0.005))
    out['deep_at_rest_wt_m_median'] = float(np.median(0.60 - h_from_theta(th[:, 3], ts, tr, alpha, n)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rec', action='append', required=True, help='label=record directory')
    ap.add_argument('--out', required=True)
    ap.add_argument('--noise', type=float, default=0.005)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    res = {}
    for item in args.rec:
        label, path = item.split('=', 1)
        r = np.load(os.path.join(path, 'synthetic.npz'))
        dep = [float(d) for d in r['sensor_depths_m']]
        cols = [dep.index(d) for d in DEPTHS]
        th = r['theta_sensors'][:, cols]
        t = r['t_days']
        soil = [float(v) for v in r['soil']]
        noisy = th + np.random.default_rng(0).normal(0.0, args.noise, th.shape)
        res[label] = {'zb_range_m': [float(r['zb'].min()), float(r['zb'].max())],
                      'noise_free': signatures(th, t, soil), 'noisy': signatures(noisy, t, soil)}
        for kind in ('noise_free', 'noisy'):
            s = res[label][kind]
            print(f"[sig] {label:13s} {kind:10s} zb {res[label]['zb_range_m'][0]:.2f}-{res[label]['zb_range_m'][1]:.2f} m |"
                  f" std {np.round(s['std'], 4)} ratio {np.round(s['std_ratio_to_top'], 2)} | lag_h {s['lag_h']} xcorr "
                  f"{np.round(s['xcorr'], 2)} | granger down {np.round(s['granger_down'], 3)} up "
                  f"{np.round(s['granger_up'], 3)} | diurnal amp ratio {np.round(s['diurnal_amp_ratio_to_top'], 3)} "
                  f"phase_h {np.round(s['diurnal_phase_h'], 1)} | dry h {s['dry_hours']}, upward 40-60 in dry "
                  f"{s['dry_upward_share_40_60']}, all {s['all_upward_share_40_60']:.2f} | 60 cm near sat "
                  f"{s['deep_near_saturation_share']:.2f}, at-rest WT {s['deep_at_rest_wt_m_median']:.2f} m", flush=True)
    with open(os.path.join(args.out, 'signatures.json'), 'w') as f:
        json.dump(res, f, indent=1)
    colors = ['#2a78d6', '#1baf7a', '#eda100', '#eb6834', '#6250d6']
    for kind in ('noise_free', 'noisy'):
        fig, ax = plt.subplots(2, 2, figsize=(10, 7))
        for c, (label, v) in zip(colors, res.items()):
            s = v[kind]
            lab = f"{label} ({v['zb_range_m'][0]:.1f}-{v['zb_range_m'][1]:.1f} m)"
            ax[0, 0].plot([100 * d for d in DEPTHS], s['std_ratio_to_top'], color=c, lw=1.6, marker='o', label=lab)
            ax[0, 1].plot([1, 2, 3], s['granger_up'], color=c, lw=1.6, marker='o', label=lab)
            ax[1, 0].plot([1, 2, 3], s['lag_h'], color=c, lw=1.6, marker='o', label=lab)
            ax[1, 1].plot([100 * d for d in DEPTHS], s['diurnal_amp_ratio_to_top'], color=c, lw=1.6, marker='o',
                          label=lab)
        ax[0, 0].set_xlabel('sensor depth (cm)')
        ax[0, 0].set_ylabel('variability / variability at 5 cm (-)')
        ax[0, 1].set_xlabel('sensor pair (1: 5-20, 2: 20-40, 3: 40-60 cm)')
        ax[0, 1].set_ylabel('upward predictability (share of variance)')
        ax[1, 0].set_xlabel('sensor pair (1: 5-20, 2: 20-40, 3: 40-60 cm)')
        ax[1, 0].set_ylabel('lag of the deeper sensor (h)')
        ax[1, 1].set_xlabel('sensor depth (cm)')
        ax[1, 1].set_ylabel('daily-cycle amplitude / amplitude at 5 cm (-)')
        ax[0, 0].legend(frameon=False, fontsize=8)
        for a in ax.ravel():
            a.grid(True, color='#e1e0d9', lw=0.6)
            a.spines[['top', 'right']].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f'signatures_{kind}.png'), dpi=130)
        plt.close(fig)
    print(f"[sig] wrote {args.out}/signatures.json and signatures_*.png", flush=True)


if __name__ == '__main__':
    main()
