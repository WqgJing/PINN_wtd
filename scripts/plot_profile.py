#!/usr/bin/env python
"""Soil-moisture profile of a trained run as depth-time maps: the network, the FEM of the synthetic record, and their
difference. Depths from the top sensor down; below the water table the soil is saturated (theta_s).

The FEM profile stored in the record is the one of its 1 h solve (the sensors' truth is the 1/16 h solve; the 1 h one
is off by up to ~0.03 at wetting fronts), so the difference map shows that solver error too.

Usage: python -u scripts/plot_profile.py --run runs/wt4_sanity [--dz 0.005]
"""
import argparse
import os
import sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from train_profile import build, load_config     # noqa: E402

torch.set_default_dtype(torch.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--dz', type=float, default=0.005)
    args = ap.parse_args()
    cfg = load_config(os.path.join(args.run, 'config.yaml'), [])
    dev = torch.device('cpu')
    B = build(cfg, dev)
    m = B['model']
    m.load_state_dict(torch.load(os.path.join(args.run, 'model.pt'), map_location=dev, weights_only=True))
    m.eval()
    tw = B['tw']
    t = torch.tensor(tw).reshape(-1, 1)
    with torch.no_grad():
        zb = m.zb(t).numpy().reshape(-1)
    d_top = B['depths'][0]
    depths = np.arange(d_top, float(np.ceil(zb.max() * 20) / 20) + 1e-9, args.dz)
    th_s = m.soil.theta_s
    net = np.full((depths.size, tw.size), th_s)
    with torch.no_grad():
        for i, d in enumerate(depths):
            above = d < zb
            if above.any():
                net[i, above] = m.theta_at(float(d), t[above]).numpy().reshape(-1)
    rec = dict(np.load(cfg['data']['npz'], allow_pickle=False))
    k = (rec['t_days'] >= float(cfg['window']['start_day']) - 1e-9) & (rec['t_days'] <= float(cfg['window']['end_day']) + 1e-9)
    xi, th_fem, zb_fem = rec['xi'], rec['theta'][k], rec['zb'][k]
    order = np.argsort(xi)
    fem = np.full_like(net, th_s)
    for j in range(tw.size):
        above = depths < zb_fem[j]
        fem[above, j] = np.interp(-depths[above] / zb_fem[j], xi[order], th_fem[j][order])
    diff = net - fem
    np.savez(os.path.join(args.run, 'profile_maps.npz'), t_days=tw, t_offset=float(cfg['window']['start_day']),
             depths_m=depths, theta_net=net, theta_fem_1h=fem, zb_net=zb, zb_record=zb_fem)
    print(f"[profile] {args.run}: {depths.size} depths x {tw.size} hours; network minus FEM (1 h): rms "
          f"{np.sqrt(np.mean(diff ** 2)):.2e}, max {np.abs(diff).max():.2e}; theta range {net.min():.3f} .. "
          f"{net.max():.3f}", flush=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ta = tw + float(cfg['window']['start_day'])
    lo, hi = float(min(net.min(), fem.min())), th_s
    lim = float(np.abs(diff).max())
    fig, ax = plt.subplots(3, 1, figsize=(9, 9.5), sharex=True)
    panels = [(net, 'Blues', lo, hi, 'soil moisture, network (m$^3$/m$^3$)'),
              (fem, 'Blues', lo, hi, 'soil moisture, FEM 1 h (m$^3$/m$^3$)'),
              (diff, 'RdBu_r', -lim, lim, 'network minus FEM (m$^3$/m$^3$)')]
    for a, (z, cmap, vmin, vmax, lab) in zip(ax, panels):
        pc = a.pcolormesh(ta, depths, z, cmap=cmap, vmin=vmin, vmax=vmax, shading='nearest', rasterized=True)
        a.plot(ta, zb, color='k', lw=1.4)
        a.set_ylim(depths[-1], depths[0])
        a.set_ylabel('depth (m)')
        fig.colorbar(pc, ax=a, label=lab, pad=0.01)
    ax[-1].set_xlabel('time (days from the start of the record)')
    fig.tight_layout()
    out = os.path.join(args.run, 'profile_heatmap.png')
    fig.savefig(out, dpi=110)
    print(f"[profile] saved {out}", flush=True)


if __name__ == '__main__':
    main()
