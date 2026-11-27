"""Water-table error of PINN runs split by the true depth (inner window, first and last day left out), to see where the
error comes from. Usage (via sbatch): hpc/run_script_cpu.slurm scripts/depth_error_bins.py --runs 'runs/m[234]_*'
"""
import argparse
import glob
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True, help='glob of run folders')
    ap.add_argument('--shallow_m', type=float, default=2.0)
    ap.add_argument('--deep_m', type=float, default=2.2)
    args = ap.parse_args()
    print(f"{'run':24s} {'all':>6s} {'inner':>6s} {'<=' + str(args.shallow_m) + 'm':>7s} {'>' + str(args.deep_m) + 'm':>6s}  (cm)")
    for r in sorted(glob.glob(args.runs)):
        p = np.load(os.path.join(r, 'predictions.npz'))
        t, z, zt = p['t_days'], p['zb_est'], p['zb_true']
        e = z - zt
        inner = (t >= 1.0) & (t <= t[-1] - 1.0)
        sh, dp = inner & (zt <= args.shallow_m), inner & (zt > args.deep_m)
        f = lambda m: 100 * np.sqrt(np.mean(e[m] ** 2)) if m.any() else float('nan')
        print(f"{os.path.basename(r):24s} {f(np.ones_like(inner)):6.1f} {f(inner):6.1f} {f(sh):7.1f} {f(dp):6.1f}  "
              f"share of inner hours deeper than {args.deep_m} m: {100 * dp.sum() / inner.sum():.0f} %", flush=True)


if __name__ == '__main__':
    main()
