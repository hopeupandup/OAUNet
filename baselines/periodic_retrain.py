"""Baseline: retrain on a sliding buffer every N samples."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from models.trajgru import TrajGRU
from utils.losses import mse_nll_loss, load_state_dict_compat
from components.variance_recalibration import build_recalibrator, apply_recalibration


class PeriodicRetrain:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        model_cfg = config.get('model', {})
        self.model = TrajGRU(
            input_channels=model_cfg.get('input_channels', 1),
            hidden_channels=model_cfg.get('hidden_channels', [32, 64]),
            num_layers=model_cfg.get('num_layers', 2),
            output_frames=model_cfg.get('output_frames', 10),
            num_flows=model_cfg.get('num_flows', [9, 9]),
            use_checkpoint=model_cfg.get('use_checkpoint', False),
            heteroscedastic=model_cfg.get('heteroscedastic', False)
        )
        self.heteroscedastic = bool(getattr(self.model, 'heteroscedastic', False))

        checkpoint_path = config.get('checkpoint_path')
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            load_state_dict_compat(self.model, checkpoint['model_state_dict'],
                                   context='periodic_retrain')

        self.model.to(self.device)

        retrain_cfg = config.get('periodic_retrain', {})
        self.retrain_interval = retrain_cfg.get('interval', 100)
        self.retrain_epochs = retrain_cfg.get('epochs', 5)
        self.learning_rate = retrain_cfg.get('learning_rate', 1e-4)

        self.recalibrator = build_recalibrator(config)

        self.total_count = 0
        self.retrain_count = 0
        self.buffer_x = []
        self.buffer_y = []
        self.history = []

    def online_step(self, x, y):
        self.total_count += 1
        x = x.to(self.device)
        y = y.to(self.device)

        self.buffer_x.append(x.cpu())
        self.buffer_y.append(y.cpu())

        max_buffer = self.retrain_interval * 3
        if len(self.buffer_x) > max_buffer:
            self.buffer_x = self.buffer_x[-max_buffer:]
            self.buffer_y = self.buffer_y[-max_buffer:]

        self.model.eval()
        with torch.no_grad():
            if self.heteroscedastic:
                pred, logvar = self.model(x, return_logvar=True)
                var = torch.exp(logvar.clamp(-10.0, 2.0))
            else:
                pred = self.model(x)
                var = torch.zeros_like(pred)

            var = apply_recalibration(self.recalibrator, pred, y, var)

        error = F.mse_loss(pred.cpu(), y.cpu()).item()

        updated = False
        if self.total_count % self.retrain_interval == 0:
            self._retrain()
            self.retrain_count += 1
            updated = True

        self.history.append({
            'step': self.total_count,
            'error': error,
            'uncertainty': 0.0,
            'updated': updated,
            'update_type': 'retrain' if updated else 'none',
            'drift_detected': False
        })

        return pred.cpu(), var.cpu()

    def _retrain(self):
        if len(self.buffer_x) == 0:
            return

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        window = min(len(self.buffer_x), self.retrain_interval * 2)
        recent_x = torch.cat(self.buffer_x[-window:], dim=0)
        recent_y = torch.cat(self.buffer_y[-window:], dim=0)

        mini_batch = 4
        n = recent_x.shape[0]

        for _ in range(self.retrain_epochs):
            for start in range(0, n, mini_batch):
                end = min(start + mini_batch, n)
                batch_x = recent_x[start:end].to(self.device)
                batch_y = recent_y[start:end].to(self.device)

                # μ follows MSE and the σ head the split NLL, as everywhere else in the codebase.
                if self.heteroscedastic:
                    pred_r, logvar_r = self.model(batch_x, return_logvar=True)
                    loss = mse_nll_loss(pred_r, logvar_r, batch_y)
                    del pred_r, logvar_r
                else:
                    loss = F.mse_loss(self.model(batch_x), batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.model.eval()
        del recent_x, recent_y
        torch.cuda.empty_cache()

    def get_statistics(self):
        stats = {
            'total_steps': self.total_count,
            'update_count': self.retrain_count,
            'update_ratio': self.retrain_count / max(1, self.total_count),
            'history': self.history
        }
        if self.recalibrator is not None:
            stats['recalibration'] = self.recalibrator.get_stats()
        return stats
