"""Which loss term pushes the water table, and through which route? For a trained run (its config.yaml and model.pt),
on one fixed draw of collocation points and blocks:
  - the gradient of each loss term (data, pde, wb) along a uniform shift of all water-table coefficients
    (> 0: a deeper water table raises that term, so descent pushes the water table UP; < 0: pushes it DOWN),
    in full and with the path features (network inputs) detached, i.e. only through h = 0 at the water table,
    the column height and the sensor positions;
  - each term after shifting all coefficients by delta with the network held fixed.

Usage (via sbatch): hpc/run_script_cpu.slurm scripts/diag_wt_gradient.py --runs runs/f2_5m,runs/f2_3m --out output/diag/wt_gradient.json
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_profile as tpf   # noqa: E402


def terms(B, cfg, dev, seed):
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    return tpf.losses(B, cfg, gen, dev)


def shift_gradients(B, cfg, dev, seed):
    m = B['model']
    out = {}
    for k, v in terms(B, cfg, dev, seed).items():
        g, = torch.autograd.grad(v, m.coef, retain_graph=False, allow_unused=True)
        out[k] = 0.0 if g is None else float(g.sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True, help='comma-separated run folders')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=123)
    args = ap.parse_args()
    dev = torch.device('cpu')
    report = {}
    for run in args.runs.split(','):
        cfg = tpf.load_config(os.path.join(run, 'config.yaml'), [])
        B = tpf.build(cfg, dev)
        m = B['model']
        m.load_state_dict(torch.load(os.path.join(run, 'model.pt'), map_location=dev))
        m.coef.requires_grad_(True)
        rep = {'full': shift_gradients(B, cfg, dev, args.seed)}
        orig = m.path.forward
        m.path.forward = lambda t, c, _o=orig: _o(t, c.detach())          # path features as fixed inputs
        rep['features_detached'] = shift_gradients(B, cfg, dev, args.seed)
        m.path.forward = orig
        c0 = m.coef.detach().clone()
        scan = {}
        for d in (-0.5, -0.2, -0.05, 0.0, 0.05, 0.2, 0.5):
            with torch.no_grad():
                m.coef.copy_(c0 + d)
            lo = terms(B, cfg, dev, args.seed)
            tw = torch.tensor(B['tw'], device=dev).reshape(-1, 1)
            scan[str(d)] = {k: float(v) for k, v in lo.items()}
            scan[str(d)]['mean_zb_m'] = float(m.zb(tw).mean())
        with torch.no_grad():
            m.coef.copy_(c0)
        rep['shift_scan'] = scan
        report[run] = rep
        print(f"[diag] {run}: d/dshift full {rep['full']} | features detached {rep['features_detached']}", flush=True)
        for d, v in scan.items():
            print(f"[diag]   shift {d:>5s}: mean zb {v['mean_zb_m']:.3f} m, data {v['data']:.3f}, pde {v['pde']:.4f}, "
                  f"wb {v.get('wb', float('nan')):.4f}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(report, f, indent=1)
    print(f"[diag] wrote {args.out}", flush=True)


if __name__ == '__main__':
    main()
