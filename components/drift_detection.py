"""ADWIN-style drift detection with a configurable bound."""

import math
import numpy as np


class ADWINDriftDetector:
    def __init__(self, delta=0.01, min_window=10, max_window=500,
                 epsilon_mode='hoeffding', sudden_mode='legacy',
                 sudden_factor=2.0):
        self.delta = delta
        self.min_window = min_window
        self.max_window = max_window

        if epsilon_mode not in ('hoeffding', 'variance'):
            raise ValueError(f"未知 epsilon_mode: {epsilon_mode!r}")
        self.epsilon_mode = epsilon_mode
        if sudden_mode not in ('legacy', 'se'):
            raise ValueError(f"未知 sudden_mode: {sudden_mode!r}")
        # 'legacy' tests the jump against 2*std = 0.218, above this stream's largest jump 0.103.
        self.sudden_mode = sudden_mode
        self.sudden_factor = float(sudden_factor)

        self.window = []
        self.drift_count = 0
        self.last_drift_step = -1
        self.step = 0

        self.last_split = None
        self.out_of_range_count = 0

    def add_element(self, value):
        # The cutoff is derived on [0,1]; out-of-range input breaks the bound, so it is clipped and counted.
        self.step += 1
        value = float(value)
        if value < 0.0 or value > 1.0:
            self.out_of_range_count += 1
            value = min(1.0, max(0.0, value))
        self.window.append(value)
        if len(self.window) > self.max_window:
            self.window.pop(0)

    def detected_change(self):
        if len(self.window) < self.min_window * 2:
            return False

        n = len(self.window)

        for i in range(self.min_window, n - self.min_window + 1):
            w1 = self.window[:i]
            w2 = self.window[i:]

            mean1 = np.mean(w1)
            mean2 = np.mean(w2)
            mean_diff = abs(mean1 - mean2)

            n1, n2 = len(w1), len(w2)
            m = (n1 * n2) / (n1 + n2)

            if self.epsilon_mode == 'variance':
                var_w = float(np.var(self.window))
                ln_term = math.log(2.0 / self.delta)
                epsilon = (math.sqrt((2.0 / m) * var_w * ln_term)
                           + (2.0 / (3.0 * m)) * ln_term)
            else:
                epsilon = math.sqrt((1.0 / (2.0 * m)) * math.log(4.0 / self.delta))

            if mean_diff > epsilon:
                # Split evidence is frozen before the window is truncated, or is_sudden loses the 'before' part.
                self.last_split = {
                    'w1': list(w1),
                    'w2': list(w2),
                    'mean1': float(mean1),
                    'mean2': float(mean2),
                    'epsilon': float(epsilon),
                    'step': self.step,
                }
                self.window = self.window[i:]
                self.drift_count += 1
                self.last_drift_step = self.step
                return True

        return False

    def is_sudden(self, short_window=5, factor=None):
        split = self.last_split
        if split is None:
            return False

        w1 = np.asarray(split['w1'], dtype=float)
        w2 = np.asarray(split['w2'], dtype=float)
        if len(w1) < 2 or len(w2) < 2:
            return False

        factor = self.sudden_factor if factor is None else factor
        jump = abs(float(np.mean(w2)) - float(np.mean(w1)))

        if self.sudden_mode == 'se':
            var_w = float(np.var(np.concatenate([w1, w2])))
            se = math.sqrt(max(var_w, 1e-12) * (1.0 / len(w1) + 1.0 / len(w2)))
            reference = se
        else:
            reference = max(float(np.std(w1)), float(np.std(w2)))

        trend = max(self._internal_trend(w1), self._internal_trend(w2))

        return jump > factor * reference and jump > factor * trend

    @staticmethod
    def _internal_trend(arr):
        n = len(arr)
        if n < 2:
            return 0.0
        half = n // 2
        return abs(float(np.mean(arr[half:])) - float(np.mean(arr[:half])))

    def get_stats(self):
        stats = {
            'step': self.step,
            'window_size': len(self.window),
            'drift_count': self.drift_count,
            'last_drift_step': self.last_drift_step,
            'current_mean': np.mean(self.window) if self.window else 0.0,
            'current_std': np.std(self.window) if len(self.window) > 1 else 0.0,
            'out_of_range_count': self.out_of_range_count,
        }
        if self.last_split is not None:
            stats['last_split'] = {
                'mean_before': self.last_split['mean1'],
                'mean_after': self.last_split['mean2'],
                'jump': abs(self.last_split['mean2'] - self.last_split['mean1']),
                'n_before': len(self.last_split['w1']),
                'n_after': len(self.last_split['w2']),
                'epsilon': self.last_split['epsilon'],
            }
        return stats

    def reset(self):
        self.window = []
        self.drift_count = 0
        self.last_drift_step = -1
        self.step = 0
        self.last_split = None
        self.out_of_range_count = 0
