"""Online variance recalibration (OVR)."""

import math
from collections import deque

import numpy as np
import torch

# Φ⁻¹(0.75) = 0.6745, the |N(0,1)| median: the reference value when q = 0.5.
_GAUSSIAN_MEDIAN_ABS = 0.6744897501960817


class OnlineVarianceRecalibrator:
    def __init__(self, window_size=200, min_window=30,
                 s_min=0.1, s_max=10.0, quantile=0.5):
        self.window_size = int(window_size)
        self.min_window = int(min_window)
        self.s_min = float(s_min)
        self.s_max = float(s_max)
        self.quantile = float(quantile)
        # P(|Z| <= t) = q  =>  t = Φ⁻¹((1+q)/2)
        from scipy import stats
        self._ref = float(stats.norm.ppf((1.0 + self.quantile) / 2.0))

        self._m_window = deque(maxlen=self.window_size)
        self.s = 1.0
        self.n_seen = 0
        self.n_applied = 0
        self.history_s = []


    def update(self, mean_pred, target, variance):
        self.n_seen += 1

        with torch.no_grad():
            v_max = float(variance.max().item()) if variance.numel() else 0.0
            if v_max <= 1e-12:
                return variance

            sigma = torch.sqrt(variance.clamp_min(1e-12))
            z = (target - mean_pred).abs() / sigma
            if self.quantile == 0.5:
                m = float(torch.median(z.flatten()).item())
            else:
                m = float(torch.quantile(z.flatten().float(), self.quantile).item())

            self._m_window.append(m)

            if len(self._m_window) >= max(self.min_window, 2):
                ref = self._ref if self._ref > 1e-9 else _GAUSSIAN_MEDIAN_ABS
                s = float(np.mean(self._m_window)) / ref
                self.s = float(min(max(s, self.s_min), self.s_max))
                self.n_applied += 1

            scaled = variance * (self.s ** 2)

        self.history_s.append((self.n_seen, self.s))
        return scaled


    def get_stats(self):
        return {
            'enabled': True,
            's': self.s,
            'n_seen': self.n_seen,
            'n_applied': self.n_applied,
            'window_size': self.window_size,
            'min_window': self.min_window,
            's_min': self.s_min,
            's_max': self.s_max,
            'quantile': self.quantile,
            'ref_quantile': self._ref,
            'mean_m_window': float(np.mean(self._m_window)) if self._m_window else None,
            'window_len': len(self._m_window),
        }

    def reset(self):
        self._m_window.clear()
        self.s = 1.0
        self.n_seen = 0
        self.n_applied = 0
        self.history_s = []


def build_recalibrator(config):
    cfg = (config.get('recalibration') or {}) if isinstance(config, dict) else {}
    if not cfg.get('enable', False):
        return None
    return OnlineVarianceRecalibrator(
        window_size=cfg.get('window_size', 200),
        min_window=cfg.get('min_window', 30),
        s_min=cfg.get('s_min', 0.1),
        s_max=cfg.get('s_max', 10.0),
        quantile=cfg.get('quantile', 0.5),
    )


def apply_recalibration(recalibrator, mean_pred, target, variance):
    if recalibrator is None:
        return variance
    return recalibrator.update(mean_pred, target, variance)
