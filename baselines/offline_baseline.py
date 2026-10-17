"""Offline baseline: the base model applied without any online adaptation."""

import torch
import torch.nn as nn
import numpy as np
from models.trajgru import TrajGRU
from utils.losses import load_state_dict_compat
from components.variance_recalibration import build_recalibrator, apply_recalibration


class OfflineBaseline:
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
                                   context='offline_baseline')

        self.model.to(self.device)
        self.model.eval()

        # OVR re-scales σ online from observed residuals and leaves μ frozen.
        self.recalibrator = build_recalibrator(config)

        self.total_count = 0
        self.history = []

    def online_step(self, x, y):
        self.total_count += 1
        x = x.to(self.device)
        y = y.to(self.device)

        with torch.no_grad():
            if self.heteroscedastic:
                pred, logvar = self.model(x, return_logvar=True)
                var = torch.exp(logvar.clamp(-10.0, 2.0))
            else:
                pred = self.model(x)
                var = torch.zeros_like(pred)

            var = apply_recalibration(self.recalibrator, pred, y, var)

        error = nn.functional.mse_loss(pred.detach().cpu(), y.cpu()).item()
        self.history.append({
            'step': self.total_count,
            'error': error,
            'uncertainty': 0.0,
            'updated': False,
            'update_type': 'none',
            'drift_detected': False
        })

        return pred.detach().cpu(), var.detach().cpu()

    def get_statistics(self):
        stats = {
            'total_steps': self.total_count,
            'update_count': 0,
            'update_ratio': 0.0,
            'history': self.history
        }
        if self.recalibrator is not None:
            stats['recalibration'] = self.recalibrator.get_stats()
        return stats
