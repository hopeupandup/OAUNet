"""Probabilistic scores."""

import numpy as np
from scipy import stats


def negative_log_likelihood(pred_mean, pred_var, target):
    pred_var = np.maximum(pred_var, 1e-8)
    nll = 0.5 * (np.log(2 * np.pi * pred_var) + (target - pred_mean) ** 2 / pred_var)
    return float(np.mean(nll))


def crps_gaussian(pred_mean, pred_std, target):
    pred_std = np.maximum(pred_std, 1e-8)
    z = (target - pred_mean) / pred_std

    phi_z = stats.norm.pdf(z)
    Phi_z = stats.norm.cdf(z)

    crps = pred_std * (z * (2 * Phi_z - 1) + 2 * phi_z - 1.0 / np.sqrt(np.pi))
    return float(np.mean(crps))


def expected_calibration_error(pred_mean, pred_var, target, n_bins=10, batch_size=100):
    levels = (np.arange(n_bins, dtype=np.float64) + 0.5) / n_bins
    q = stats.norm.ppf(0.5 + levels / 2.0)

    n = pred_mean.shape[0]
    hits = np.zeros(n_bins, dtype=np.int64)
    total_count = 0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        b_mean = np.asarray(pred_mean[start:end], dtype=np.float32).flatten()
        # Standard deviation is taken per batch, so memory stays proportional to batch_size.
        b_std = np.sqrt(np.maximum(
            np.asarray(pred_var[start:end], dtype=np.float32).flatten(), 1e-8))
        b_target = np.asarray(target[start:end], dtype=np.float32).flatten()

        # |standardised residual| should follow |N(0,1)| when the predictive distribution is calibrated.
        z_abs = np.abs((b_target - b_mean) / b_std)
        total_count += z_abs.size

        for i in range(n_bins):
            hits[i] += int(np.count_nonzero(z_abs <= q[i]))

    ece = 0.0
    bin_details = []
    for i in range(n_bins):
        empirical = float(hits[i] / max(total_count, 1))
        bin_details.append({
            'nominal': float(levels[i]),
            'empirical': empirical,
            'count': int(total_count),
        })
        ece += abs(float(levels[i]) - empirical)

    ece /= n_bins
    return float(ece), bin_details


def evaluate_probabilistic(pred_mean, pred_var, target, batch_size=100):
    if np.max(pred_var) < 1e-6:
        return {
            'nll': None,
            'crps': None,
            'ece': None,
            'note': '非贝叶斯模型，无不确定性估计，跳过概率指标'
        }

    n = pred_mean.shape[0]
    nll_total = 0.0
    crps_total = 0.0
    total_elements = 0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_mean = np.asarray(pred_mean[start:end], dtype=np.float32)
        batch_var = np.asarray(pred_var[start:end], dtype=np.float32)
        batch_std = np.sqrt(np.maximum(batch_var, 1e-8))
        batch_target = np.asarray(target[start:end], dtype=np.float32)

        batch_elements = batch_mean.size
        nll_total += negative_log_likelihood(batch_mean, batch_var, batch_target) * batch_elements
        crps_total += crps_gaussian(batch_mean, batch_std, batch_target) * batch_elements
        total_elements += batch_elements

    results = {
        'nll': float(nll_total / max(total_elements, 1)),
        'crps': float(crps_total / max(total_elements, 1)),
    }

    ece, bin_details = expected_calibration_error(pred_mean, pred_var, target)
    results['ece'] = ece
    results['ece_bin_details'] = bin_details

    return results
