"""Uncertainty-gated selective updating."""

import numpy as np


class RunningStats:
    def __init__(self, var_init=1.0):
        self.mean = 0.0
        self.var = var_init
        self.count = 0
        self._buffer = []

    def update(self, value, warmup_steps, ema_alpha=None):
        self.count += 1
        self._buffer.append(float(value))

        if self.count <= warmup_steps:
            self.mean = float(np.mean(self._buffer))
            if len(self._buffer) > 1:
                self.var = float(np.var(self._buffer))
            return

        alpha = ema_alpha if ema_alpha is not None else 2.0 / (self.count + 1)
        self.mean = (1 - alpha) * self.mean + alpha * value
        diff = value - self.mean
        self.var = (1 - alpha) * self.var + alpha * (diff ** 2)

    def threshold(self, beta):
        return self.mean + beta * np.sqrt(max(self.var, 1e-8))

    def std(self):
        return float(np.sqrt(max(self.var, 1e-8)))

    def reset(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self._buffer = []


class UncertaintyActiveLearner:
    def __init__(self, beta_U=1.5, beta_E=1.5, warmup_steps=10, ema_alpha=None,
                 gate_mode='difficulty', uniform_rate=0.094, seed=42):
        self.beta_U = beta_U
        self.beta_E = beta_E
        self.warmup_steps = warmup_steps
        self.ema_alpha = ema_alpha


        # difficulty: update when U/E exceeds EMA mean + β·std. uniform: the same budget drawn at random,
        # which separates the cost of selection bias from the cost of updating less often.
        if gate_mode not in ('difficulty', 'uniform'):
            raise ValueError(f"未知 gate_mode: {gate_mode!r}")
        self.gate_mode = gate_mode
        self.uniform_rate = float(uniform_rate)
        # A private RandomState, so the gate does not consume the global stream used by MC sampling.
        self._rng = np.random.RandomState(seed)

        self.u_stats = RunningStats()
        self.e_stats = RunningStats()

        self.update_count = 0
        self.uncertainty_trigger_count = 0
        self.error_trigger_count = 0
        self.uniform_trigger_count = 0
        self.last_trigger = None

    @property
    def count(self):
        return self.u_stats.count

    def update_statistics(self, uncertainty, error=None):
        self.u_stats.update(uncertainty, self.warmup_steps, self.ema_alpha)
        if error is not None:
            self.e_stats.update(error, self.warmup_steps, self.ema_alpha)

    def should_update(self, uncertainty, error=None):
        if self.u_stats.count < self.warmup_steps:
            self.update_count += 1
            self.last_trigger = 'warmup'
            return True

        if self.gate_mode == 'uniform':
            if self._rng.rand() < self.uniform_rate:
                self.update_count += 1
                self.uniform_trigger_count += 1
                self.last_trigger = 'uniform'
                return True
            self.last_trigger = None
            return False

        if uncertainty > self.u_stats.threshold(self.beta_U):
            self.update_count += 1
            self.uncertainty_trigger_count += 1
            self.last_trigger = 'uncertainty'
            return True

        if error is not None and self.e_stats.count >= self.warmup_steps:
            if error > self.e_stats.threshold(self.beta_E):
                self.update_count += 1
                self.error_trigger_count += 1
                self.last_trigger = 'error'
                return True

        self.last_trigger = None
        return False

    def get_threshold(self):
        return self.u_stats.threshold(self.beta_U)

    def get_error_threshold(self):
        return self.e_stats.threshold(self.beta_E)

    def get_stats(self):
        return {
            'count': self.count,
            'update_count': self.update_count,
            'update_ratio': self.update_count / max(1, self.count),
            'running_mean': self.u_stats.mean,
            'running_std': self.u_stats.std(),
            'threshold': self.get_threshold(),
            'beta_U': self.beta_U,
            'gate_mode': self.gate_mode,
            'uniform_rate': self.uniform_rate,
            'uncertainty_trigger_count': self.uncertainty_trigger_count,
            'error_trigger_count': self.error_trigger_count,
            'uniform_trigger_count': self.uniform_trigger_count,
            'error_threshold': self.get_error_threshold(),
            'error_running_mean': self.e_stats.mean,
            'error_running_std': self.e_stats.std(),
            'beta_E': self.beta_E,
            'last_trigger': self.last_trigger,
        }

    def reset(self):
        self.u_stats.reset()
        self.e_stats.reset()
        self.update_count = 0
        self.uncertainty_trigger_count = 0
        self.error_trigger_count = 0
        self.uniform_trigger_count = 0
        self.last_trigger = None
