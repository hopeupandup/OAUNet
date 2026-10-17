"""Online baseline: one gradient step on every incoming sample."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.trajgru import TrajGRU
from utils.losses import mse_nll_loss, load_state_dict_compat
from components.variance_recalibration import build_recalibrator, apply_recalibration


class OnlineGD:
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
                                   context='online_gd')

        self.model.to(self.device)

        ogd_cfg = config.get('online_gd', {})
        self.learning_rate = ogd_cfg.get('learning_rate', 1e-3)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.recalibrator = build_recalibrator(config)
        self.total_count = 0
        self.history = []

    def online_step(self, x, y):
        self.total_count += 1
        x = x.to(self.device)
        y = y.to(self.device)

        self.model.train()
        # μ follows MSE and the σ head the split NLL, so μ is not pulled by the 1/σ² weighting.
        if self.heteroscedastic:
            pred, logvar = self.model(x, return_logvar=True)
            loss = mse_nll_loss(pred, logvar, y)
            var = torch.exp(logvar.clamp(-10.0, 2.0)).detach()
        else:
            pred = self.model(x)
            loss = F.mse_loss(pred, y)
            var = None

        error = F.mse_loss(pred.detach(), y).item()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.history.append({
            'step': self.total_count,
            'error': error,
            'uncertainty': 0.0,
            'updated': True,
            'update_type': 'gd',
            'drift_detected': False
        })

        pred_cpu = pred.detach().cpu()
        if var is None:
            return pred_cpu, torch.zeros_like(pred_cpu)
        var = apply_recalibration(self.recalibrator, pred.detach(), y, var)
        return pred_cpu, var.cpu()

    def get_statistics(self):
        stats = {
            'total_steps': self.total_count,
            'update_count': self.total_count,
            'update_ratio': 1.0,
            'history': self.history
        }
        if self.recalibrator is not None:
            stats['recalibration'] = self.recalibrator.get_stats()
        return stats
