"""Small helpers. torch_interp copied from ../pinn/zb_inverse/src/zbinv/data.py on 2026-11-23."""
import torch


def torch_interp(x, xp, fp):
    """Piecewise-linear interpolation, clamped at the ends. x: any shape; xp, fp: 1-D sorted."""
    xf = x.reshape(-1)
    xc = torch.clamp(xf, xp[0], xp[-1])
    idx = torch.clamp(torch.searchsorted(xp, xc.detach()), 1, xp.numel() - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    y0, y1 = fp[idx - 1], fp[idx]
    w = (xc - x0) / (x1 - x0)
    return (y0 + w * (y1 - y0)).reshape(x.shape)
