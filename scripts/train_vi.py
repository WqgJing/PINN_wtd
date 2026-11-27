"""Variational PINN: joint Gaussian posterior over the water-table spline coefficients and log Ks, trained by the
ELBO with the PINN as the conditional profile (src/wpinn/vi.py). Uses build() and losses() of train_profile.py.

Loss per step = mean over K samples u_k ~ q of [ N_eff/2 (L_data + L_pde + L_wb)(u_k) - log p(u_k) + log q(u_k) ],
N_eff = number of readings in the window. Ks_ref = the soil Ks of the config (record Ks x soil_factor.Ks): the prior
median of Ks and the reference of z = log(Ks / Ks_ref).

Usage (via sbatch): hpc/run_script_gpu.slurm scripts/train_vi.py --config <run config.yaml> [--set k=v ...] --out runs/<x>
    config keys used besides train_profile's: vi.samples (8), vi.n_col (1024), vi.n_box (16), vi.level_m (3.0),
    vi.level_sd (2.5), vi.step_sd (0.5), vi.z_sd (1.1513), vi.sd_init_c (0.05), vi.sd_init_z (0.3), vi.n_draw (4000)
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_profile as tpf                                            # noqa: E402
from wpinn.vi import VarKsSoil, VIProfilePINN, GaussianPosterior, WaterTableKsPrior   # noqa: E402
from wpinn.spline import ZbSplineNumpy                                  # noqa: E402

torch.set_default_dtype(torch.float64)


def schedule(step, steps, lr, lr_end, hold=0.0):
    frac = step / max(steps, 1)
    f = max(frac - hold, 0.0) / (1.0 - hold)
    return lr * (lr_end / lr) ** f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--set', action='append', default=[])
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    cfg = tpf.load_config(args.config, args.set)
    tc, vc = cfg['train'], cfg.setdefault('vi', {})
    K = int(vc.get('samples', 8))
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(int(tc['seed']))
    np.random.seed(int(tc['seed']))
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(tc['seed']))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    B = tpf.build(cfg, dev)
    m0 = B['model']
    s = B['soil_params']                                                # theta_s, theta_r, alpha, n, Ks_ref
    soil = VarKsSoil(theta_s=s[0], theta_r=s[1], alpha=s[2], n=s[3], Ks=s[4], L=float(cfg['scaling']['L']))
    model = VIProfilePINN(soil, B['spline'], B['coef0'], m0.zb_min, m0.d_top, m0.Tw, m0.t_off, sink=m0.sink,
                          memories=m0.memories, width=int(cfg['net']['width']), depth=int(cfg['net']['depth']),
                          fourier_periods_days=cfg['net']['fourier_periods_days'],
                          path_taus_days=cfg['net']['path_taus_days'],
                          head_form=str(cfg['net'].get('head_form', 'shift'))).to(dev)
    B['model'], B['soil'] = model, soil
    m = B['spline'].n_coef
    ks_ref = float(s[4])
    ks_true = float(np.load(cfg['data']['npz'])['soil'][4])            # scoring only

    post = GaussianPosterior(np.concatenate([B['coef0'], [0.0]]),
                             [float(vc.get('sd_init_c', 0.05))] * m + [float(vc.get('sd_init_z', 0.3))]).to(dev)
    level_m = float(vc.get('level_m', 3.0))
    prior = WaterTableKsPrior(m, level_mean=math.log(math.expm1(level_m - model.zb_min)),
                              level_sd=float(vc.get('level_sd', 2.5)), step_sd=float(vc.get('step_sd', 0.5)),
                              z_sd=float(vc.get('z_sd', math.log(10.0) / 2.0))).to(dev)
    n_eff = float(B['obs']['theta'].shape[0])
    cfg_k = copy.deepcopy(cfg)
    cfg_k['train']['n_col'] = int(vc.get('n_col', 1024))
    cfg_k['train']['n_box'] = int(vc.get('n_box', 16))

    lr, lr_end, steps = float(tc['lr']), float(tc['lr_final']), int(tc['steps'])
    q_factor, hold = float(tc['coef_lr_factor']), float(tc.get('coef_hold_frac', 0.0))
    opt = torch.optim.Adam([{'params': list(model.N.parameters()), 'lr': lr},
                            {'params': list(post.parameters()), 'lr': lr * q_factor}])
    warm = int(tc.get('warmup_steps', 0))
    for p in post.parameters():
        p.requires_grad_(False)
    print(f"[vi] device {dev}; {m} coefficients + log Ks; K {K}; N_eff {n_eff:.0f}; Ks_ref {ks_ref:.3e} (truth "
          f"{ks_true:.3e}, scoring only); start path {cfg['init']}", flush=True)
    tw = torch.tensor(B['tw'], device=dev).reshape(-1, 1)
    zb_true = B['zb_true']
    fields = ['step', 'wall_s', 'neg_elbo', 'fit', 'neg_logp', 'logq', 'zb_rmse_mean_path_m', 'ks_median',
              'z_sd', 'level_sd']
    tic = time.time()
    with open(os.path.join(args.out, 'log.csv'), 'w', newline='') as flog:
        w = csv.DictWriter(flog, fieldnames=fields)
        w.writeheader()
        for step in range(steps + 1):
            if step == warm:
                for p in post.parameters():
                    p.requires_grad_(True)
                print(f"[vi] step {step}: posterior released", flush=True)
            opt.param_groups[0]['lr'] = schedule(step, steps, lr, lr_end)
            opt.param_groups[1]['lr'] = schedule(step, steps, lr * q_factor, lr_end * q_factor, hold)
            u, logq = post.sample(K, generator=gen)
            fits = []
            for k in range(K):
                model.set_sample(u[k, :m], u[k, m])
                lo = tpf.losses(B, cfg_k, gen, dev)
                fits.append(0.5 * n_eff * sum(lo.values()))
            fit = torch.stack(fits).mean()
            logp = prior.log_prob(u).mean()
            lq = logq.mean()
            loss = fit - logp + lq
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % int(tc['eval_every']) == 0 or step == steps:
                with torch.no_grad():
                    model.set_sample(post.mu[:m], post.mu[m])
                    zb_mu = model.zb(tw).cpu().numpy().reshape(-1)
                    cov = post.covariance()
                    a = torch.full((m,), 1.0 / m, device=dev, dtype=cov.dtype)
                    row = {'step': step, 'wall_s': round(time.time() - tic, 1), 'neg_elbo': float(loss),
                           'fit': float(fit), 'neg_logp': float(-logp), 'logq': float(lq),
                           'zb_rmse_mean_path_m': float(np.sqrt(np.mean((zb_mu - zb_true) ** 2))),
                           'ks_median': float(ks_ref * torch.exp(post.mu[m])), 'z_sd': float(torch.sqrt(cov[m, m])),
                           'level_sd': float(torch.sqrt(a @ cov[:m, :m] @ a))}
                w.writerow(row)
                flog.flush()
                print(f"[vi] {step:6d} {row['wall_s']:7.0f}s -elbo {row['neg_elbo']:.1f} fit {row['fit']:.1f} "
                      f"-logp {row['neg_logp']:.1f} logq {row['logq']:.1f} | mean path vs truth "
                      f"{100 * row['zb_rmse_mean_path_m']:.2f} cm | Ks median {row['ks_median']:.3e} (truth "
                      f"{ks_true:.3e}), sd(log Ks) {row['z_sd']:.3f}, sd(level c) {row['level_sd']:.3f}", flush=True)

    # posterior summary
    n_draw = int(vc.get('n_draw', 4000))
    with torch.no_grad():
        ud, _ = post.sample(n_draw, generator=gen)
        ud = ud.cpu().numpy()
        cov_post = post.covariance().cpu().numpy()
        cov_prior = prior.covariance().cpu().numpy()
        mu = post.mu.detach().cpu().numpy()
    Bmat = B['spline'].basis_np(B['tw'])
    zb_draws = model.zb_min + np.logaddexp(0.0, ud[:, :m] @ Bmat.T)     # (n_draw, n_t)
    q05, q50, q95 = np.percentile(zb_draws, [5, 50, 95], axis=0)
    ks_draws = ks_ref * np.exp(ud[:, m])
    ks05, ks50, ks95 = np.percentile(ks_draws, [5, 50, 95])
    a = np.full(m, 1.0 / m)
    var_lv_post, var_lv_prior = a @ cov_post[:m, :m] @ a, a @ cov_prior[:m, :m] @ a
    contraction = 1.0 - np.diag(cov_post) / np.diag(cov_prior)
    corr_level_z = float((a @ cov_post[:m, m]) / math.sqrt(var_lv_post * cov_post[m, m]))
    cover = float(np.mean((zb_true >= q05) & (zb_true <= q95)))
    rmse_med = float(np.sqrt(np.mean((q50 - zb_true) ** 2)))
    summary = {'init': cfg['init'], 'steps': steps, 'K': K, 'n_eff': n_eff, 'wall_s': time.time() - tic,
               'ks_true': ks_true, 'ks_ref_prior_median': ks_ref, 'ks_post': [float(ks05), float(ks50), float(ks95)],
               'ks_true_in_90': bool(ks05 <= ks_true <= ks95),
               'zb_rmse_median_path_m': rmse_med, 'zb_band90_cover_truth': cover,
               'zb_band90_mean_width_m': float(np.mean(q95 - q05)),
               'contraction_logks': float(contraction[m]),
               'contraction_level': float(1.0 - var_lv_post / var_lv_prior),
               'contraction_coef': [float(v) for v in contraction[:m]],
               'corr_level_logks': corr_level_z,
               'rules': {'R1_rmse_le_5cm': rmse_med <= 0.05,
                         'R2_cover_ge_80_and_ks_in_90': (cover >= 0.8) and bool(ks05 <= ks_true <= ks95)}}
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    np.savez(os.path.join(args.out, 'posterior.npz'), t_days=B['tw'], t_offset=float(cfg['window']['start_day']),
             zb_true=zb_true, zb_q05=q05, zb_q50=q50, zb_q95=q95, mu=mu, cov_post=cov_post, cov_prior=cov_prior,
             ks_draws=ks_draws, ks_true=ks_true, ks_ref=ks_ref, coef_start=B['coef0'])
    torch.save({'model': model.state_dict(), 'post': post.state_dict()}, os.path.join(args.out, 'model.pt'))
    print(f"[vi] posterior: median path vs truth {100 * rmse_med:.2f} cm; 90 % band covers the truth on "
          f"{100 * cover:.0f} % of hours (mean width {100 * summary['zb_band90_mean_width_m']:.1f} cm); Ks 90 % "
          f"[{ks05:.2e}, {ks95:.2e}] median {ks50:.2e} (truth {ks_true:.2e}, in: {summary['ks_true_in_90']}); "
          f"contraction log Ks {summary['contraction_logks']:.3f}, level {summary['contraction_level']:.3f}; "
          f"corr(level, log Ks) {corr_level_z:+.3f}", flush=True)
    print(f"[vi] rules: {summary['rules']}", flush=True)


if __name__ == '__main__':
    main()
