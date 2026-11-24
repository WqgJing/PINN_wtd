"""Residual scale from the data. signal_rate_scale copied unchanged from ../pinn/zb_inverse/src/zbinv/train.py on 2026-11-23."""
import numpy as np


def signal_rate_scale(t_obs, theta_obs, soil, sigma_theta):
    """sigma_pde 'auto': RMS of d Se / d t~ of the noise-free signal, estimated from day means.

    Differencing noisy hourly data measures the noise, sqrt(2) sigma / (theta* dt~): sigma_theta 0.005 gives
    1.35 vs 0.084 for clean data, i.e. a PDE term 256x weaker than intended (1000x at 0.01), so the network
    fits the noise by bending the physics. Day means cut the noise by sqrt(n per day); the remaining known
    noise variance of each difference, sigma^2 (1/n_i + 1/n_j) / (theta* dt~)^2, is subtracted.
    Incomplete days (fewer than half the usual count) are dropped. Daily data: n = 1 per day."""
    t_obs = np.asarray(t_obs, dtype=float)
    theta_obs = np.asarray(theta_obs, dtype=float)
    day = np.floor(t_obs + 1e-9).astype(int)
    keys, counts = np.unique(day, return_counts=True)
    keys = keys[counts >= 0.5 * counts.max()]
    tm = np.array([t_obs[day == k].mean() for k in keys])
    se = np.array([(theta_obs[day == k].mean() - soil.theta_r) / soil.theta_star for k in keys])
    nk = np.array([(day == k).sum() for k in keys], dtype=float)
    dtt = np.diff(tm) / (soil.T / 86400.0)
    rate2 = (np.diff(se) / dtt) ** 2
    noise2 = (float(sigma_theta) / soil.theta_star) ** 2 * (1.0 / nk[1:] + 1.0 / nk[:-1]) / dtt ** 2
    var = float(np.mean(rate2 - noise2))
    if var < 0.25 * float(np.mean(rate2)):
        print(f"[train] WARNING sigma_pde auto: noise removes {100 * (1 - var / np.mean(rate2)):.0f}% of the "
              f"day-mean rate variance; signal barely resolved, using 0.5 x raw", flush=True)
        var = 0.25 * float(np.mean(rate2))
    return float(np.sqrt(var))
