"""Adaptation loop: gate, update, detect, and report predictive variance."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

from models.trajgru import TrajGRU
from components.online_vi import OnlineVI
from components.active_learning import UncertaintyActiveLearner
from components.drift_detection import ADWINDriftDetector
from components.meta_learning import MAMLLearner
from components.variance_recalibration import build_recalibrator, apply_recalibration
from utils.losses import load_state_dict_compat


class OAUNet:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        self.enable_vi = config.get('enable_online_vi', True)
        self.enable_al = config.get('enable_active_learning', True)
        self.enable_dd = config.get('enable_drift_detection', True)
        self.enable_ml = config.get('enable_meta_learning', True)

        self.model = self._load_model()

        if self.enable_vi:
            vi_cfg = config.get('online_vi', {})
            self.online_vi = OnlineVI(
                self.model,
                prior_sigma=vi_cfg.get('prior_sigma', 1.0),
                learning_rate=vi_cfg.get('learning_rate', 1e-3),
                beta_kl=vi_cfg.get('beta_kl', 1e-4),
                kl_scaling=vi_cfg.get('kl_scaling', 'per_param'),

                update_mode=vi_cfg.get('update_mode', 'single'),
                buffer_size=vi_cfg.get('buffer_size', 64),
                mini_batch=vi_cfg.get('mini_batch', 8),
                inner_steps=vi_cfg.get('inner_steps', 5),
                buffer_lr=vi_cfg.get('buffer_lr', None),

                prior_center=vi_cfg.get('prior_center', 'zero'),

                adapt_scope=vi_cfg.get('adapt_scope', 'full'),

                intensity_alpha=vi_cfg.get('intensity_alpha', 0.0),
                heavy_threshold=vi_cfg.get('heavy_threshold', 0.4215),
            )
            self.model = self.online_vi.model
        else:
            self.online_vi = None

        if self.enable_al:
            al_cfg = config.get('active_learning', {})
            self.active_learner = UncertaintyActiveLearner(
                beta_U=al_cfg.get('beta_U', 1.5),
                beta_E=al_cfg.get('beta_E', 1.5),
                warmup_steps=al_cfg.get('warmup_steps', 10),
                gate_mode=al_cfg.get('gate_mode', 'difficulty'),
                uniform_rate=al_cfg.get('uniform_rate', 0.094)
            )
        else:
            self.active_learner = None

        if self.enable_dd:
            dd_cfg = config.get('drift_detection', {})
            self.drift_detector = ADWINDriftDetector(
                delta=dd_cfg.get('delta', 0.01),
                min_window=dd_cfg.get('min_window', 10),
                max_window=dd_cfg.get('max_window', 500),

                epsilon_mode=dd_cfg.get('epsilon_mode', 'hoeffding'),
                sudden_mode=dd_cfg.get('sudden_mode', 'legacy'),
                sudden_factor=dd_cfg.get('sudden_factor', 2.0)
            )
        else:
            self.drift_detector = None

        self.drift_error_scale = config.get('drift_detection', {}).get('error_scale', None)

        if self.enable_ml:
            ml_cfg = config.get('meta_learning', {})
            self.meta_learner = MAMLLearner(
                self.model,
                inner_lr=ml_cfg.get('inner_lr', 0.01),
                outer_lr=ml_cfg.get('outer_lr', 0.001),
                inner_steps=ml_cfg.get('inner_steps', 5)
            )
        else:
            self.meta_learner = None

        self.mc_samples = config.get('mc_samples', 10)

        self.adapt_buffer = []
        self.adapt_buffer_size = config.get('meta_learning', {}).get('adapt_buffer_size', 5)

        self.recalibrator = build_recalibrator(config)
        self.enable_recal = self.recalibrator is not None

        self.update_count = 0
        self.total_count = 0
        self.drift_count = 0
        self.vi_update_count = 0
        self.maml_update_count = 0
        self.history = []
        self._history_max_len = 5000

    def _load_model(self):
        model_cfg = self.config.get('model', {})
        model = TrajGRU(
            input_channels=model_cfg.get('input_channels', 1),
            hidden_channels=model_cfg.get('hidden_channels', [32, 64]),
            num_layers=model_cfg.get('num_layers', 2),
            output_frames=model_cfg.get('output_frames', 10),
            num_flows=model_cfg.get('num_flows', [9, 9]),
            use_checkpoint=model_cfg.get('use_checkpoint', False),
            heteroscedastic=model_cfg.get('heteroscedastic', False)
        )

        checkpoint_path = self.config.get('checkpoint_path')
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            load_state_dict_compat(model, checkpoint['model_state_dict'],
                                   context='framework._load_model')

        model.to(self.device)
        return model

    def meta_train(self, train_inputs, train_outputs, train_labels, n_epochs=50,
                    checkpoint_dir=None):
        if not self.enable_ml:
            return

        meta_cfg = self.config.get('meta_learning', {})
        tasks = MAMLLearner.construct_tasks(
            train_inputs, train_outputs, train_labels,
            n_tasks=meta_cfg.get('n_tasks', 20),
            support_size=meta_cfg.get('support_size', 30),
            query_size=meta_cfg.get('query_size', 20)
        )
        self.meta_learner.meta_train(
            tasks, n_epochs=n_epochs, device=self.device,
            checkpoint_dir=checkpoint_dir
        )

    def online_step(self, x, y):
        self.total_count += 1
        x = x.to(self.device)
        y = y.to(self.device)

        mean_pred, epistemic, aleatoric = self._predict_with_uncertainty(x)

        if self.online_vi is not None:
            self.online_vi.observe(x, y)

        if self.enable_ml:
            self.adapt_buffer.append((x.detach().clone(), y.detach().clone()))
            if len(self.adapt_buffer) > self.adapt_buffer_size:
                self.adapt_buffer.pop(0)


        # The AL gate uses the scale-free relative error |err|/(|err|+|y|); the drift detector uses an
        # absolute-scale error |err|/(|err|+c) with a global constant c, which is what reacts to drift.
        c = self.drift_error_scale
        with torch.no_grad():
            abs_err = (mean_pred - y).abs().mean()
            y_scale = y.abs().mean() + 1e-6
        error = (abs_err / (abs_err + y_scale)).item()
        drift_signal = (abs_err / (abs_err + c)).item() if (c is not None and c > 0) else error

        # Gating uses epistemic variance only; the reported total adds aleatoric to cover data noise.
        scalar_uncertainty = epistemic.mean().item()
        scalar_aleatoric = aleatoric.mean().item()

        if self.enable_al:
            self.active_learner.update_statistics(scalar_uncertainty, error)

        should_update = False
        drift_detected = False
        update_type = 'none'

        if self.enable_al:
            should_update = self.active_learner.should_update(scalar_uncertainty, error)
        else:
            should_update = True

        if self.enable_dd:
            self.drift_detector.add_element(drift_signal)
            drift_detected = self.drift_detector.detected_change()

        if drift_detected:
            self.drift_count += 1

        if drift_detected or should_update:
            if drift_detected and self.enable_dd and self.drift_detector.is_sudden() and self.enable_ml:
                buf_x = torch.cat([bx for bx, _ in self.adapt_buffer], dim=0)
                buf_y = torch.cat([by for _, by in self.adapt_buffer], dim=0)
                adapted = self.meta_learner.rapid_adapt(
                    buf_x, buf_y,
                    k_steps=self.config.get('meta_learning', {}).get('adapt_steps', 5),
                    device=self.device
                )
                self._sync_model(adapted)
                self.adapt_buffer.clear()
                update_type = 'maml'
                self.maml_update_count += 1
            elif self.enable_vi:

                # drift_lr_mult defaults to 1.0: the legacy x3 path was never exercised before the fix.
                mult = float(self.config.get('online_vi', {}).get('drift_lr_mult', 1.0))
                scaled = False
                if drift_detected and self.enable_dd and mult != 1.0:
                    original_lr = self.online_vi.learning_rate
                    self.online_vi.set_learning_rate(original_lr * mult)
                    scaled = True

                self.online_vi.online_update(x, y)

                if scaled:
                    self.online_vi.reset_learning_rate()

                update_type = 'vi'
                self.vi_update_count += 1

            if update_type != 'none':
                self.update_count += 1

        self.history.append({
            'step': self.total_count,
            'error': error,
            'drift_signal': drift_signal,
            'uncertainty': scalar_uncertainty,
            'aleatoric': scalar_aleatoric,
            'updated': update_type != 'none',
            'update_type': update_type,
            'drift_detected': drift_detected
        })
        if len(self.history) > self._history_max_len:
            self.history = self.history[-self._history_max_len:]

        total_variance = epistemic + aleatoric
        total_variance = apply_recalibration(
            self.recalibrator, mean_pred, y, total_variance)
        return mean_pred.cpu(), total_variance.cpu()

    def _predict_with_uncertainty(self, x):
        if self.online_vi is not None:
            return self.online_vi.compute_uncertainty(x, n_samples=self.mc_samples)

        self.model.eval()
        with torch.no_grad():
            if getattr(self.model, 'heteroscedastic', False):
                pred, logvar = self.model(x, return_logvar=True)
                ale = torch.exp(logvar.clamp(-10.0, 2.0))
                return pred, torch.zeros_like(ale), ale
            pred = self.model(x)
            zero = torch.zeros_like(pred)
            return pred, zero, zero

    def _sync_model(self, source_model):
        self.model.load_state_dict(source_model.state_dict())
        if self.online_vi is not None:
            self.online_vi.model.load_state_dict(source_model.state_dict())
            self.online_vi.rebuild_optimizer()

    def get_statistics(self):
        stats = {
            'total_steps': self.total_count,
            'update_count': self.update_count,
            'update_ratio': self.update_count / max(1, self.total_count),
            'vi_update_count': self.vi_update_count,
            'maml_update_count': self.maml_update_count,
            'drift_count': self.drift_count,
            'history': self.history
        }

        if self.active_learner is not None:
            stats['active_learning'] = self.active_learner.get_stats()
        if self.drift_detector is not None:
            stats['drift_detection'] = self.drift_detector.get_stats()
        if self.recalibrator is not None:
            stats['recalibration'] = self.recalibrator.get_stats()

        return stats
