"""Online variational inference over the fitted weights."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import collections
import numpy as np

from utils.losses import mse_nll_loss


class BayesianConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, prior_sigma=1.0, bias=True,
                 prior_center='zero'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.prior_sigma = prior_sigma
        self.has_bias = bias
        if prior_center not in ('zero', 'anchor'):
            raise ValueError(f"未知 prior_center: {prior_center!r}（应为 'zero' | 'anchor'）")
        self.prior_center = prior_center

        self.weight_mu = nn.Parameter(torch.empty(out_channels, in_channels, *self.kernel_size))
        self.weight_log_sigma = nn.Parameter(torch.empty(out_channels, in_channels, *self.kernel_size))
        if bias:
            self.bias_mu = nn.Parameter(torch.empty(out_channels))
            self.bias_log_sigma = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_log_sigma', None)

        self.register_buffer('prior_w_mu',
                             torch.zeros(out_channels, in_channels, *self.kernel_size))
        if bias:
            self.register_buffer('prior_b_mu', torch.zeros(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight_mu)
        # σ init -3 (σ≈0.05) rather than -5: at -5 the MC variance degenerates into constant noise.
        nn.init.constant_(self.weight_log_sigma, -3.0)
        if self.has_bias:
            nn.init.zeros_(self.bias_mu)
            nn.init.constant_(self.bias_log_sigma, -3.0)

    def forward(self, x):
        if self.training:
            weight_sigma = torch.exp(self.weight_log_sigma)
            weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
            if self.has_bias:
                bias_sigma = torch.exp(self.bias_log_sigma)
                bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)
            else:
                bias = None
        else:
            weight = self.weight_mu
            bias = self.bias_mu if self.has_bias else None

        return F.conv2d(x, weight, bias, self.stride, self.padding)

    def kl_divergence(self):
        weight_sigma = torch.exp(self.weight_log_sigma)

        kl_weight = 0.5 * (
            (weight_sigma ** 2 + (self.weight_mu - self.prior_w_mu) ** 2) / self.prior_sigma ** 2
            - 1 - 2 * self.weight_log_sigma + 2 * math.log(self.prior_sigma)
        )
        kl_weight = kl_weight.sum() / self.weight_mu.numel()

        if self.has_bias:
            bias_sigma = torch.exp(self.bias_log_sigma)
            kl_bias = 0.5 * (
                (bias_sigma ** 2 + (self.bias_mu - self.prior_b_mu) ** 2) / self.prior_sigma ** 2
                - 1 - 2 * self.bias_log_sigma + 2 * math.log(self.prior_sigma)
            )
            kl_bias = kl_bias.sum() / self.bias_mu.numel()
        else:
            kl_bias = 0.0

        return kl_weight + kl_bias

    def kl_divergence_sum(self):
        weight_sigma = torch.exp(self.weight_log_sigma)
        kl_w = 0.5 * (
            (weight_sigma ** 2 + (self.weight_mu - self.prior_w_mu) ** 2) / self.prior_sigma ** 2
            - 1 - 2 * self.weight_log_sigma + 2 * math.log(self.prior_sigma)
        ).sum()
        n = self.weight_mu.numel()

        if self.has_bias:
            bias_sigma = torch.exp(self.bias_log_sigma)
            kl_b = 0.5 * (
                (bias_sigma ** 2 + (self.bias_mu - self.prior_b_mu) ** 2) / self.prior_sigma ** 2
                - 1 - 2 * self.bias_log_sigma + 2 * math.log(self.prior_sigma)
            ).sum()
            n += self.bias_mu.numel()
        else:
            kl_b = torch.zeros((), dtype=kl_w.dtype, device=kl_w.device)

        return kl_w + kl_b, n


def make_bayesian(model, prior_sigma=1.0, prior_center='zero'):
    for name, child in model.named_children():
        if isinstance(child, nn.Conv2d):
            bayesian = BayesianConv2d(
                child.in_channels, child.out_channels, child.kernel_size,
                child.stride, child.padding, prior_sigma=prior_sigma,
                bias=(child.bias is not None), prior_center=prior_center
            )
            bayesian = bayesian.to(child.weight.device)
            bayesian.weight_mu.data = child.weight.data.clone()
            # σ is scaled per layer as 0.1·|μ|; one constant σ cannot say which weights are uncertain.
            bayesian.weight_log_sigma.data = torch.log(
                child.weight.data.abs().mul(0.1).clamp(min=1e-3)
            )
            if prior_center == 'anchor':
                bayesian.prior_w_mu.copy_(child.weight.data)
            if child.bias is not None:
                bayesian.bias_mu.data = child.bias.data.clone()
                bayesian.bias_log_sigma.data = torch.log(
                    child.bias.data.abs().mul(0.1).clamp(min=1e-3)
                )
                if prior_center == 'anchor':
                    bayesian.prior_b_mu.copy_(child.bias.data)
            setattr(model, name, bayesian)
        else:
            make_bayesian(child, prior_sigma, prior_center)
    return model


def get_bayesian_layers(model):
    layers = []
    for module in model.modules():
        if isinstance(module, BayesianConv2d):
            layers.append(module)
    return layers


class OnlineVI:
    def __init__(self, model, prior_sigma=1.0, learning_rate=1e-3, beta_kl=1e-2,
                 kl_scaling='per_param', update_mode='single',
                 buffer_size=64, mini_batch=8, inner_steps=5, buffer_lr=None,
                 prior_center='zero', adapt_scope='full',
                 intensity_alpha=0.0, heavy_threshold=0.4215):
        self.heteroscedastic = bool(getattr(model, 'heteroscedastic', False))

        if prior_center not in ('zero', 'anchor'):
            raise ValueError(f"未知 prior_center: {prior_center!r}")
        self.prior_center = prior_center
        self.model = make_bayesian(model, prior_sigma, prior_center)
        self.prior_sigma = prior_sigma


        # Restricting the update scope is the structural defence against catastrophic forgetting.
        if adapt_scope not in ('full', 'decoder', 'head'):
            raise ValueError(f"未知 adapt_scope: {adapt_scope!r}")
        self.adapt_scope = adapt_scope
        self._apply_adapt_scope(adapt_scope)
        self.learning_rate = learning_rate
        self._base_lr = learning_rate
        self.beta_kl = beta_kl

        self.intensity_alpha = float(intensity_alpha or 0.0)
        self.heavy_threshold = float(heavy_threshold)
        if kl_scaling == 'legacy':
            kl_scaling = 'per_layer_mean'
        self.kl_scaling = kl_scaling


        # buffer: draw mini_batch of the last buffer_size observations, take inner_steps SGD steps.
        if update_mode not in ('single', 'buffer'):
            raise ValueError(f"未知 update_mode: {update_mode!r}")
        self.update_mode = update_mode
        self.buffer_size = int(buffer_size)
        self.mini_batch = int(mini_batch)
        self.inner_steps = int(inner_steps)
        self.buffer_lr = float(buffer_lr) if buffer_lr is not None else learning_rate
        if self.update_mode == 'buffer':
            self._base_lr = self.buffer_lr
            self.learning_rate = self.buffer_lr
        self._buffer = collections.deque(maxlen=max(self.buffer_size, self.mini_batch))

        self.optimizer = torch.optim.Adam(self.trainable_parameters(), lr=self._base_lr)
        self.update_step = 0


    def _apply_adapt_scope(self, scope):
        for name, p in self.model.named_parameters():
            if scope == 'full':
                p.requires_grad_(True)
            elif scope == 'decoder':
                p.requires_grad_(name.startswith('decoder.'))
            else:
                p.requires_grad_(('output_conv' in name) or ('logvar_conv' in name))
        try:
            self.model.stop_grad_encoder = (scope != 'full')
        except AttributeError:
            pass
        self.adapt_scope = scope

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def scope_stats(self):
        total = sum(p.numel() for p in self.model.parameters())
        train = sum(p.numel() for p in self.trainable_parameters())
        return {'adapt_scope': self.adapt_scope,
                'total_params': int(total),
                'trainable_params': int(train),
                'trainable_ratio': float(train) / max(total, 1)}

    def rebuild_optimizer(self):
        self._apply_adapt_scope(self.adapt_scope)
        self.optimizer = torch.optim.Adam(self.trainable_parameters(), lr=self._base_lr)


    def observe(self, x, y):
        if self.update_mode != 'buffer':
            return
        self._buffer.append((x.detach().to('cpu', copy=True),
                             y.detach().to('cpu', copy=True)))

    def buffer_size_current(self):
        return len(self._buffer)


    def _kl_term(self):
        layers = get_bayesian_layers(self.model)

        if self.kl_scaling == 'per_layer_mean':
            kl = sum(layer.kl_divergence() for layer in layers)
            return kl, kl

        kl_sum = None
        n_param = 0
        for layer in layers:
            s, n = layer.kl_divergence_sum()
            kl_sum = s if kl_sum is None else kl_sum + s
            n_param += n

        if kl_sum is None:
            zero = torch.zeros((), dtype=torch.float32,
                               device=next(self.model.parameters()).device)
            return zero, zero

        if self.kl_scaling == 'per_param':
            return kl_sum / max(n_param, 1), kl_sum
        if self.kl_scaling == 'sum':
            return kl_sum, kl_sum
        raise ValueError(f"未知 kl_scaling: {self.kl_scaling!r}")

    def compute_elbo(self, pred, target, logvar=None):
        if logvar is not None:
            recon_loss = mse_nll_loss(pred, logvar, target,
                                      intensity_alpha=self.intensity_alpha,
                                      heavy_threshold=self.heavy_threshold)
        elif self.intensity_alpha > 0.0:
            y = target
            w = 1.0 + self.intensity_alpha * (y >= self.heavy_threshold).to(y.dtype)
            recon_loss = (w * (pred - y) ** 2).sum() / w.sum().clamp(min=1e-8)
        else:
            recon_loss = F.mse_loss(pred, target)
        kl_loss, _ = self._kl_term()
        total = recon_loss + self.beta_kl * kl_loss
        return total, recon_loss, kl_loss

    def online_update(self, x, y):
        if self.update_mode == 'buffer':
            return self._buffered_update(x, y)

        self.model.train()
        if self.heteroscedastic:
            pred, logvar = self.model(x, return_logvar=True)
        else:
            pred, logvar = self.model(x), None
        loss, recon, kl = self.compute_elbo(pred, y, logvar=logvar)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.update_step += 1

        return {
            'loss': loss.item(),
            'recon_loss': recon.item(),
            'kl_loss': kl.item(),
            'kl_scaling': self.kl_scaling
        }

    def _buffered_update(self, x, y):
        if len(self._buffer) == 0:
            self.observe(x, y)

        self.model.train()
        dev = next(self.model.parameters()).device
        k = min(self.mini_batch, len(self._buffer))
        losses = []
        for _ in range(max(1, self.inner_steps)):
            idx = np.random.choice(len(self._buffer), size=k, replace=False)
            bx = torch.cat([self._buffer[i][0] for i in idx], dim=0).to(dev)
            by = torch.cat([self._buffer[i][1] for i in idx], dim=0).to(dev)

            if self.heteroscedastic:
                pred, logvar = self.model(bx, return_logvar=True)
            else:
                pred, logvar = self.model(bx), None
            loss, recon, kl = self.compute_elbo(pred, by, logvar=logvar)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.update_step += 1
            losses.append(float(loss.item()))
            del bx, by, pred, logvar, loss

        return {
            'loss': float(np.mean(losses)),
            'recon_loss': None,
            'kl_loss': float(kl.item()),
            'kl_scaling': self.kl_scaling,
            'update_mode': 'buffer',
            'grad_steps': len(losses),
            'mini_batch': k,
        }

    def compute_uncertainty(self, x, n_samples=10):
        self.model.train()
        predictions = []
        logvars = [] if self.heteroscedastic else None

        for _ in range(n_samples):
            with torch.no_grad():
                if self.heteroscedastic:
                    pred, logvar = self.model(x, return_logvar=True)
                    logvars.append(logvar)
                else:
                    pred = self.model(x)
                predictions.append(pred)

        self.model.eval()

        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        epistemic = predictions.var(dim=0)
        if logvars:
            aleatoric = torch.stack(logvars).exp().mean(dim=0)
        else:
            aleatoric = torch.zeros_like(epistemic)

        return mean_pred, epistemic, aleatoric

    def set_learning_rate(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.learning_rate = lr

    def reset_learning_rate(self):
        self.set_learning_rate(self._base_lr)
