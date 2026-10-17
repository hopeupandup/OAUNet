"""First-order MAML adaptation (evaluated and rejected; kept for completeness)."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from tqdm import tqdm

from utils.losses import mse_nll_loss, load_state_dict_compat


class MAMLLearner:
    def __init__(self, model, inner_lr=0.01, outer_lr=0.001,
                 inner_steps=5, first_order=True):
        self.model = model
        self.heteroscedastic = bool(getattr(model, 'heteroscedastic', False))
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        self.first_order = first_order
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=outer_lr)

    @staticmethod
    def construct_tasks(inputs, outputs, labels, n_tasks=20,
                        support_size=5, query_size=5):
        tasks = []
        sorted_indices = np.argsort(labels)
        n_samples = len(labels)
        range_size = max(1, n_samples // n_tasks)

        for i in range(n_tasks):
            start = i * range_size
            end = min((i + 1) * range_size, n_samples)
            if end - start < support_size + query_size:
                continue

            indices = sorted_indices[start:end]
            np.random.shuffle(indices)

            support_idx = indices[:support_size]
            query_idx = indices[support_size:support_size + query_size]

            task = {
                'support': (
                    torch.FloatTensor(inputs[support_idx]).unsqueeze(2),
                    torch.FloatTensor(outputs[support_idx]).unsqueeze(2)
                ),
                'query': (
                    torch.FloatTensor(inputs[query_idx]).unsqueeze(2),
                    torch.FloatTensor(outputs[query_idx]).unsqueeze(2)
                )
            }
            tasks.append(task)

        return tasks

    def meta_train(self, tasks, n_epochs=50, device='cpu',
                    checkpoint_dir=None):
        self.model.to(device)
        self.model.train()

        start_epoch = 0
        loss_history = []

        if checkpoint_dir:
            start_epoch, loss_history = self._load_meta_checkpoint(
                checkpoint_dir, device
            )

        epoch_iter = tqdm(
            range(start_epoch, n_epochs),
            desc="Meta-Training", ncols=80,
            initial=start_epoch, total=n_epochs
        )

        for epoch in epoch_iter:
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            original_state = {
                name: p.data.clone()
                for name, p in self.model.named_parameters()
            }

            self.meta_optimizer.zero_grad()
            epoch_loss = 0.0

            for task in tasks:
                support_x, support_y = task['support']
                query_x, query_y = task['query']
                support_x = support_x.to(device)
                support_y = support_y.to(device)
                query_x = query_x.to(device)
                query_y = query_y.to(device)

                for name, p in self.model.named_parameters():
                    p.data.copy_(original_state[name])

                self._inner_loop(support_x, support_y)

                # The outer objective uses the same split loss as the inner loop, or σ gets no outer gradient.
                if self.heteroscedastic:
                    pred, logvar = self.model(query_x, return_logvar=True)
                    task_loss = mse_nll_loss(pred, logvar, query_y)
                else:
                    pred, logvar = self.model(query_x), None
                    task_loss = F.mse_loss(pred, query_y)

                task_loss.backward()
                epoch_loss += task_loss.item()

                del pred, logvar, task_loss, support_x, support_y, query_x, query_y
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            for name, p in self.model.named_parameters():
                p.data.copy_(original_state[name])

            # Only trainable parameters are clipped, matching _inner_loop.
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            # The meta-optimiser stays over all parameters: Adam skips a None grad, and filtering would
            # misalign the positional state_dict of a resumed meta checkpoint.
            self.meta_optimizer.step()

            loss_history.append(epoch_loss)
            tqdm.write(
                f"  Epoch {epoch+1}/{n_epochs} | "
                f"Meta Loss: {epoch_loss:.6f}"
            )

            if checkpoint_dir:
                self._save_meta_checkpoint(
                    checkpoint_dir, epoch, loss_history
                )

        return self.model

    def _save_meta_checkpoint(self, checkpoint_dir, epoch, loss_history):
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(checkpoint_dir, 'meta_checkpoint.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.meta_optimizer.state_dict(),
            'loss_history': loss_history,
            'inner_lr': self.inner_lr,
            'outer_lr': self.outer_lr,
            'inner_steps': self.inner_steps,
        }, ckpt_path)
        tqdm.write(f"  [checkpoint] 已保存到 epoch {epoch + 1}")

    def _load_meta_checkpoint(self, checkpoint_dir, device):
        ckpt_path = os.path.join(checkpoint_dir, 'meta_checkpoint.pt')
        if not os.path.exists(ckpt_path):
            return 0, []

        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            load_state_dict_compat(self.model, ckpt['model_state_dict'],
                                   logger=None, context='meta_learning._load_meta_checkpoint')
            self.meta_optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt['epoch']
            loss_history = ckpt.get('loss_history', [])
            print(f"  从 epoch {start_epoch} 继续元训练（已完成 {start_epoch} 个 epoch）")
            return start_epoch, loss_history
        except Exception as e:
            print(f"  [警告] 元训练 checkpoint 加载失败: {e}，从头开始")
            return 0, []

    def _inner_loop(self, x, y):
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable:
            return

        for _ in range(self.inner_steps):
            if self.heteroscedastic:
                pred, logvar = self.model(x, return_logvar=True)
                loss = mse_nll_loss(pred, logvar, y)
            else:
                pred, logvar = self.model(x), None
                loss = F.mse_loss(pred, y)

            grads = torch.autograd.grad(
                loss,
                trainable,
                create_graph=not self.first_order,
                allow_unused=True,
            )

            for p, g in zip(trainable, grads):
                if g is not None:
                    p.data -= self.inner_lr * g.data

            del pred, logvar, loss, grads

    def rapid_adapt(self, x, y, k_steps=5, device='cpu'):
        adapted_model = copy.deepcopy(self.model)
        adapted_model.to(device)
        adapted_model.train()

        frozen_params = []
        for module in adapted_model.modules():
            if hasattr(module, 'weight_log_sigma'):
                module.weight_log_sigma.requires_grad_(False)
                frozen_params.append(module.weight_log_sigma)
            if hasattr(module, 'bias_log_sigma'):
                module.bias_log_sigma.requires_grad_(False)
                frozen_params.append(module.bias_log_sigma)

        trainable_params = [p for p in adapted_model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(trainable_params, lr=self.inner_lr)

        x = x.to(device)
        y = y.to(device)

        for _ in range(k_steps):
            if self.heteroscedastic:
                pred, logvar = adapted_model(x, return_logvar=True)
                loss = mse_nll_loss(pred, logvar, y)
            else:
                pred = adapted_model(x)
                loss = F.mse_loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return adapted_model
