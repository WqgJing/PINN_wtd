#!/usr/bin/env python
"""Invert the water-table path from soil moisture at several depths with a PINN (no surface flux, no initial profile).

Data: a synthetic record of scripts/make_synthetic.py; a window of it; sensors at the given depths with noise added.
Known: the soil (van Genuchten-Mualem of the record) and the transpiration (Feddes root sink of the FEM).
Trained together: the network weights and the water-table spline coefficients (mode invert), or the weights only with
the path held at the truth (mode fixed: the sanity run, can the network and the losses reproduce the record at all).
Loss: data at every sensor (the top one is the column's upper boundary) + Richards residual + block water balance,
each scaled by its sigma (noise; rate of change of the data; noise). The truth path is used to SCORE only.

Usage: python -u scripts/train_profile.py --config configs/wt4_d35_55.yaml [--set train.seed=1 ...] --out runs/<tag>
"""
import argparse
import copy
import csv
import json
import math
import os
import sys
import time
import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from wpinn.forcing import ForcingMemoryFeatures      # noqa: E402
from wpinn.model import ProfilePINN                  # noqa: E402
from wpinn.physics import pde_residual, block_defect  # noqa: E402
from wpinn.roots import RootSink                     # noqa: E402
from wpinn.scales import signal_rate_scale           # noqa: E402
from wpinn.soil import SoilVG                        # noqa: E402
from wpinn.spline import CubicBSpline1D, ZbSplineNumpy  # noqa: E402

torch.set_default_dtype(torch.float64)


def load_config(path, overrides):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, val = item.split('=', 1)
        node = cfg
        for p in key.split('.')[:-1]:
            node = node.setdefault(p, {})
        node[key.split('.')[-1]] = yaml.safe_load(val)
    return cfg


def score(est, ref):
    d = np.asarray(est) - np.asarray(ref)
    return {'rmse_m': float(np.sqrt(np.mean(d ** 2))), 'bias_m': float(d.mean()), 'maxabs_m': float(np.abs(d).max()),
            'r': float(np.corrcoef(est, ref)[0, 1])}


def build(cfg, dev):
    dc, wc, sc = cfg['data'], cfg['window'], cfg['sensors']
    rec = dict(np.load(dc['npz'], allow_pickle=False))
    t_rec = rec['t_days']
    all_depths = [float(d) for d in rec['sensor_depths_m']]
    cols = [all_depths.index(float(d)) for d in sc['depths_m']]
    # noise drawn for EVERY depth of the record, then the chosen sensors taken: a sensor at a given depth has the same
    # noise whichever other sensors are used (runs before 2026-11-24 drew it for the chosen sensors only)
    th_true_all = rec[dc['truth_key']]
    rng = np.random.default_rng(int(sc['noise_seed']))
    noise_all = rng.normal(0.0, float(sc['noise_sigma']), th_true_all.shape)
    th_true_rec = th_true_all[:, cols]                               # noise-free truth at the chosen sensors
    th_obs_rec = (th_true_all + noise_all)[:, cols]
    bias = sc.get('bias')                                            # optional constant offset per chosen sensor
    if bias is not None:                                             # (a calibration error; the truth stays unbiased)
        if len(bias) != len(cols):
            raise ValueError(f'sensors.bias has {len(bias)} values for {len(cols)} sensors')
        th_obs_rec = th_obs_rec + np.asarray(bias, dtype=float)[None, :]
    drift = sc.get('drift')                                          # optional linear drift per chosen sensor: 0 at
    if drift is not None:                                            # the window start, the value at its end
        if len(drift) != len(cols):
            raise ValueError(f'sensors.drift has {len(drift)} values for {len(cols)} sensors')
        ws, we = float(wc['start_day']), float(wc['end_day'])
        ramp = np.clip((t_rec - ws) / (we - ws), 0.0, 1.0)
        th_obs_rec = th_obs_rec + ramp[:, None] * np.asarray(drift, dtype=float)[None, :]
    # optional gaps: [[sensor index, first day, last day], ...] in record days; missing readings are left out of the
    # data loss. Inputs that need a full series (memory features, start path) get a straight-line fill instead.
    avail_rec = np.ones(th_obs_rec.shape, dtype=bool)
    for j, a, b in (sc.get('gaps') or []):
        avail_rec[(t_rec >= float(a) - 1e-9) & (t_rec <= float(b) + 1e-9), int(j)] = False
    th_fill_rec = th_obs_rec.copy()
    for j in range(th_obs_rec.shape[1]):
        if not avail_rec[:, j].all():
            th_fill_rec[:, j] = np.interp(t_rec, t_rec[avail_rec[:, j]], th_obs_rec[avail_rec[:, j], j])
    t0, t1 = float(wc['start_day']), float(wc['end_day'])
    k = (t_rec >= t0 - 1e-9) & (t_rec <= t1 + 1e-9)
    tw = t_rec[k] - t0
    s = np.array(rec['soil'], dtype=float)
    names = ['theta_s', 'theta_r', 'alpha', 'n', 'Ks']
    for key, f in (cfg.get('soil_factor') or {}).items():           # a wrong soil guess: the record's soil times f
        s[names.index(key)] *= float(f)
    soil = SoilVG(theta_s=s[0], theta_r=s[1], alpha=s[2], n=s[3], Ks=s[4], L=float(cfg['scaling']['L']))
    sp = CubicBSpline1D(0.0, t1 - t0, float(cfg['zb_spline']['knot_days']))
    zb_min = float(cfg['zb_spline']['zb_min'])
    zb_true = rec['zb'][k]
    truth = ZbSplineNumpy.fit(sp, tw, zb_true, zb_min)
    depths = [float(d) for d in sc['depths_m']]
    d_top, d_deep = depths[0], depths[-1]
    # hydrostatic reading of the deepest sensor: its height above the water table at rest, from the retention curve
    zb_hyd = d_deep - soil.h_from_theta(th_fill_rec[k, -1])
    init = str(cfg['init'])
    if init == 'truth':
        coef0 = truth.coef
    elif init == 'hydrostatic':
        coef0 = ZbSplineNumpy.fit(sp, tw, np.maximum(zb_hyd, d_deep + 0.02), zb_min).coef
    elif init.startswith('constant:'):
        coef0 = ZbSplineNumpy.fit(sp, tw, np.full_like(tw, float(init.split(':')[1])), zb_min).coef
    else:
        raise ValueError(f'init {init}')
    sink = RootSink(rec['t_nodes'], rec['tp_nodes_mmd']) if cfg['physics'].get('root_sink', True) else None
    mem = None
    if cfg['net'].get('memory_taus_days'):
        top = th_fill_rec[:, 0]
        mem = ForcingMemoryFeatures([(t_rec, top - top.mean()), (rec['t_nodes'], rec['tp_nodes_mmd'])],
                                    cfg['net']['memory_taus_days'])
    model = ProfilePINN(soil, sp, coef0, zb_min, d_top, t1 - t0, t0, sink=sink, memories=mem,
                        width=int(cfg['net']['width']), depth=int(cfg['net']['depth']),
                        fourier_periods_days=cfg['net']['fourier_periods_days'],
                        path_taus_days=cfg['net']['path_taus_days']).to(dev)
    sig_th = float(cfg['sigmas']['theta'])
    avail = avail_rec[k]                                             # (n_t, n_sensors) in the window
    rates = [signal_rate_scale(tw[avail[:, j]], th_obs_rec[k, j][avail[:, j]], soil, sig_th)
             for j in range(len(depths))]
    sig_pde = float(cfg['sigmas'].get('pde_factor', 1.0)) * float(np.sqrt(np.mean(np.square(rates))))
    keep = avail.T.reshape(-1)                                       # sensor-major, as the arrays below
    obs = {'t': torch.tensor(np.tile(tw, len(depths))[keep], device=dev).reshape(-1, 1),
           'depth': torch.tensor(np.repeat(depths, tw.size)[keep], device=dev).reshape(-1, 1),
           'theta': torch.tensor(th_obs_rec[k].T.reshape(-1)[keep], device=dev).reshape(-1, 1)}
    return {'model': model, 'soil': soil, 'spline': sp, 'truth': truth, 'tw': tw, 'zb_true': zb_true, 'zb_hyd': zb_hyd,
            'th_obs': np.where(avail, th_obs_rec[k], np.nan), 'th_true': th_true_rec[k], 'depths': depths, 'obs': obs,
            'sig_th': sig_th, 'sig_pde': sig_pde, 'rates': rates, 'coef0': np.asarray(coef0),
            'soil_params': [float(v) for v in s], 'obs_avail': avail}


def collocation(n, Tw, dev, gen):
    u = torch.rand(n, 1, device=dev, generator=gen)
    xi = torch.where(torch.arange(n, device=dev).reshape(-1, 1) % 2 == 0, -u, -u ** 2)   # half dense near the top
    t = Tw * torch.rand(n, 1, device=dev, generator=gen)
    return xi, t


def blocks(n, nt, nx, Tw, dev, gen):
    r = lambda *s: torch.rand(*s, device=dev, generator=gen)
    dur = torch.exp(math.log(1.0 / 24.0) + (math.log(min(2.0, Tw)) - math.log(1.0 / 24.0)) * r(n, 1))
    dxi = torch.exp(math.log(0.05) + (math.log(1.0) - math.log(0.05)) * r(n, 1))
    t0 = (Tw - dur) * r(n, 1)
    xb = -(1.0 - dxi) * r(n, 1)                                                         # upper face, in [-1 + dxi, 0]
    t = t0 + dur * (torch.arange(nt, device=dev)[None, :] + r(n, nt)) / nt              # (n, nt)
    xi = xb[:, :, None] - dxi[:, :, None] * (torch.arange(nx, device=dev)[None, None, :] + r(n, nt, nx)) / nx
    return xi, t, dur, dxi


def losses(B, cfg, gen, dev, held_out=False):
    m, o, tc = B['model'], B['obs'], cfg['train']
    th = m.theta_at(o['depth'], o['t'])
    out = {'data': (((th - o['theta']) / B['sig_th']) ** 2).mean()}
    xi, t = collocation(int(tc['n_col']), m.Tw, dev, gen)
    R = pde_residual(m, xi, t)
    out['pde'] = ((R / B['sig_pde']) ** 2).mean()
    if int(tc.get('n_box', 0)):
        bx = blocks(int(tc['n_box']), int(tc['box_n_t']), int(tc['box_n_xi']), m.Tw, dev, gen)
        out['wb'] = ((block_defect(m, *bx) / B['sig_th']) ** 2).mean()
    if held_out:
        out = {kk: v.detach() for kk, v in out.items()}
    return out


def evaluate(B, cfg, dev, gen):
    m = B['model']
    tw = torch.tensor(B['tw'], device=dev).reshape(-1, 1)
    with torch.no_grad():
        zb = m.zb(tw).cpu().numpy().reshape(-1)
        th = np.stack([m.theta_at(d, tw).cpu().numpy().reshape(-1) for d in B['depths']], axis=1)
    lo = losses(B, cfg, gen, dev, held_out=True)
    return zb, th, {kk: float(v) for kk, v in lo.items()}


def plot(path, tw, t0, zb_true, zb_est, zb_hyd, depths, th_obs, th_est):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#6250d6']
    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ta = tw + t0
    ax[0].plot(ta, zb_true, color='k', lw=2.0, label='truth')
    ax[0].plot(ta, zb_hyd, color='#898781', lw=1.0, label='hydrostatic reading of the deepest sensor')
    ax[0].plot(ta, zb_est, color=colors[1], lw=1.6, label='PINN')
    ax[0].invert_yaxis()
    ax[0].set_ylabel('water-table depth (m)')
    ax[0].legend(frameon=False, fontsize=9)
    for j, d in enumerate(depths):
        ax[1].plot(ta, th_obs[:, j], color=colors[j % 5], lw=0.6, alpha=0.6)
        ax[1].plot(ta, th_est[:, j], color=colors[j % 5], lw=1.6, label=f'{100 * d:.0f} cm')
    ax[1].set_ylabel('soil moisture (m$^3$/m$^3$)')
    ax[1].set_xlabel('time (days from the start of the record)')
    ax[1].legend(frameon=False, fontsize=9, ncol=len(depths))
    for a in ax:
        a.grid(True, color='#e1e0d9', lw=0.6)
        a.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--set', action='append', default=[])
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    tc = cfg['train']
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(int(tc['seed']))
    np.random.seed(int(tc['seed']))
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(tc['seed']))
    gen_val = torch.Generator(device=dev)
    gen_val.manual_seed(10_000 + int(tc['seed']))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    B = build(cfg, dev)
    m = B['model']
    mode = str(cfg['mode'])
    print(f"[train] device {dev}; mode {mode}; init {cfg['init']}; sensors {B['depths']} m; window "
          f"{cfg['window']['start_day']} - {cfg['window']['end_day']} d; {B['spline'].n_coef} path coefficients; "
          f"sigma_theta {B['sig_th']}, sigma_pde {B['sig_pde']:.4g} (per sensor {[round(r, 4) for r in B['rates']]}); "
          f"T = {m.T_days:.3f} d", flush=True)
    print(f"[train] start path vs truth (scoring only): {score(m.zb(torch.tensor(B['tw'], device=dev).reshape(-1, 1)).detach().cpu().numpy().reshape(-1), B['zb_true'])}", flush=True)
    print(f"[train] hydrostatic reading vs truth (scoring only): {score(B['zb_hyd'], B['zb_true'])}", flush=True)

    net_params = [p for n_, p in m.named_parameters() if n_ != 'coef']
    lr, lr_end, steps = float(tc['lr']), float(tc['lr_final']), int(tc['steps'])
    groups = [{'params': net_params, 'lr': lr}]
    if mode == 'invert':
        groups.append({'params': [m.coef], 'lr': lr * float(tc['coef_lr_factor'])})
    opt = torch.optim.Adam(groups)
    warm = int(tc.get('warmup_steps', 0)) if mode == 'invert' else steps
    m.coef.requires_grad_(False)
    log_path = os.path.join(args.out, 'log.csv')
    fields = ['step', 'wall_s', 'data', 'pde', 'wb', 'val_data', 'val_pde', 'val_wb', 'zb_rmse_m', 'zb_bias_m']
    tic = time.time()
    with open(log_path, 'w', newline='') as flog:
        w = csv.DictWriter(flog, fieldnames=fields)
        w.writeheader()
        for step in range(steps + 1):
            if mode == 'invert' and step == warm:
                m.coef.requires_grad_(True)
                print(f"[train] step {step}: water-table coefficients released", flush=True)
            frac = step / max(steps, 1)
            for g_i, g in enumerate(opt.param_groups):
                base = lr * (1.0 if g_i == 0 else float(tc['coef_lr_factor']))
                g['lr'] = base * (lr_end / lr) ** frac
            lo = losses(B, cfg, gen, dev)
            total = sum(lo.values())
            opt.zero_grad(set_to_none=True)
            total.backward()
            opt.step()
            if step % int(tc['eval_every']) == 0 or step == steps:
                zb, _, val = evaluate(B, cfg, dev, gen_val)
                sc_ = score(zb, B['zb_true'])
                row = {'step': step, 'wall_s': round(time.time() - tic, 1), 'data': float(lo['data']),
                       'pde': float(lo['pde']), 'wb': float(lo.get('wb', float('nan'))), 'val_data': val['data'],
                       'val_pde': val['pde'], 'val_wb': val.get('wb', float('nan')), 'zb_rmse_m': sc_['rmse_m'],
                       'zb_bias_m': sc_['bias_m']}
                w.writerow(row)
                flog.flush()
                print(f"[train] {step:6d} {row['wall_s']:7.0f}s data {row['data']:.3f} pde {row['pde']:.3f} wb "
                      f"{row['wb']:.3f} | val data {row['val_data']:.3f} pde {row['val_pde']:.3f} wb {row['val_wb']:.3f}"
                      f" | zb vs truth {100 * sc_['rmse_m']:.2f} cm (bias {100 * sc_['bias_m']:+.2f})", flush=True)

    zb, th, val = evaluate(B, cfg, dev, gen_val)
    tw = B['tw']
    inner = (tw >= 1.0) & (tw <= tw[-1] - 1.0)
    summary = {'mode': mode, 'init': cfg['init'], 'seed': int(tc['seed']), 'depths_m': B['depths'],
               'sigma_theta': B['sig_th'], 'sigma_pde': B['sig_pde'], 'steps': steps, 'wall_s': time.time() - tic,
               'held_out': val, 'zb_vs_truth': score(zb, B['zb_true']),
               'zb_vs_truth_inner': score(zb[inner], B['zb_true'][inner]),
               'hydrostatic_vs_truth': score(B['zb_hyd'], B['zb_true']),
               'start_vs_truth': score(ZbSplineNumpy(B['spline'], B['coef0'], m.zb_min).zb(tw), B['zb_true']),
               'theta_vs_obs_rmse': [float(np.sqrt(np.nanmean((th[:, j] - B['th_obs'][:, j]) ** 2)))
                                     for j in range(len(B['depths']))],
               'theta_vs_truth_rmse': [float(np.sqrt(np.mean((th[:, j] - B['th_true'][:, j]) ** 2)))
                                       for j in range(len(B['depths']))]}
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    np.savez(os.path.join(args.out, 'predictions.npz'), t_days=tw, t_offset=float(cfg['window']['start_day']),
             zb_est=zb, zb_true=B['zb_true'], zb_hyd=B['zb_hyd'], coef=m.coef.detach().cpu().numpy(),
             coef_start=B['coef0'], coef_truth=B['truth'].coef, depths_m=np.array(B['depths']), theta_est=th,
             theta_obs=B['th_obs'], theta_true=B['th_true'])
    torch.save(m.state_dict(), os.path.join(args.out, 'model.pt'))
    plot(os.path.join(args.out, 'result.png'), tw, float(cfg['window']['start_day']), B['zb_true'], zb, B['zb_hyd'],
         B['depths'], B['th_obs'], th)
    z = summary['zb_vs_truth']
    print(f"[train] final: water table vs truth {100 * z['rmse_m']:.2f} cm (bias {100 * z['bias_m']:+.2f}, max "
          f"{100 * z['maxabs_m']:.2f}, r {z['r']:.4f}); inner {100 * summary['zb_vs_truth_inner']['rmse_m']:.2f} cm; "
          f"hydrostatic reading {100 * summary['hydrostatic_vs_truth']['rmse_m']:.2f} cm; theta vs obs "
          f"{[round(v, 4) for v in summary['theta_vs_obs_rmse']]}; held-out {val}", flush=True)


if __name__ == '__main__':
    main()
