"""Losses and checkpoint loading."""

import math

import torch
import torch.nn.functional as F


# Clamp range for log σ² (u space). The lower bound keeps σ_dBZ >= 0.64: without it the optimiser
# drives σ to 1e-4 dBZ and the probabilistic metrics degenerate (NLL 1e10, ECE 0.78, CRPS = MAE).
# The upper bound stops σ -> inf from being a cheap way to lower the NLL.
LOGVAR_MIN = -10.0
LOGVAR_MAX = 2.0


def gaussian_nll(mu, logvar, target, reduction='mean'):
    logvar = logvar.clamp(min=LOGVAR_MIN, max=LOGVAR_MAX)
    inv_var = torch.exp(-logvar)
    nll = 0.5 * (logvar + (target - mu).pow(2) * inv_var + math.log(2 * math.pi))

    if reduction == 'mean':
        return nll.mean()
    if reduction == 'sum':
        return nll.sum()
    return nll


def mse_loss(mu, target):
    return F.mse_loss(mu, target)


def mse_nll_loss(mu, logvar, target, lambda_nll=1.0, return_parts=False,
                 intensity_alpha=0.0, heavy_threshold=0.4215):
    a = float(intensity_alpha or 0.0)
    if a > 0.0:
        w = 1.0 + a * (target >= float(heavy_threshold)).to(target.dtype)
        mse = (w * (mu - target) ** 2).sum() / w.sum().clamp(min=1e-8)
    else:
        mse = F.mse_loss(mu, target)
    nll = gaussian_nll(mu.detach(), logvar, target)
    total = mse + float(lambda_nll) * nll
    if return_parts:
        return total, {'mse': mse, 'nll': nll}
    return total


def compute_prediction_loss(model, mu, target, logvar=None):
    if getattr(model, 'heteroscedastic', False) and logvar is not None:
        return gaussian_nll(mu, logvar, target)
    return mse_loss(mu, target)


def load_state_dict_compat(model, state_dict, logger=None, context=''):
    def _emit(msg, level='warning'):
        if logger is not None and hasattr(logger, level):
            getattr(logger, level)(msg)
        else:
            print(msg)

    try:
        model.load_state_dict(state_dict, strict=True)
        return []
    except RuntimeError as strict_err:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        tag = f"[{context}] " if context else ""
        _emit(f"{tag}严格加载失败，改用宽松加载。")
        _emit(f"{tag}  缺失键 {len(missing)} 个: {missing[:8]}"
              f"{' ...' if len(missing) > 8 else ''}")
        _emit(f"{tag}  多余键 {len(unexpected)} 个: {unexpected[:8]}"
              f"{' ...' if len(unexpected) > 8 else ''}")

        # Only newly added logvar keys are tolerated; anything else missing is a structural mismatch.
        backbone_missing = [k for k in missing if 'logvar' not in k]
        if backbone_missing:
            raise RuntimeError(
                f"{tag}骨干权重缺失 {len(backbone_missing)} 个键 "
                f"（例: {backbone_missing[:5]}）。这是权重结构与模型配置不匹配，"
                f"不允许以随机权重继续。原始错误: {strict_err}"
            ) from strict_err

        _emit(f"{tag}仅 logvar 头为新初始化（旧权重 + 新代码），其余权重已恢复。", level='info')
        return missing
