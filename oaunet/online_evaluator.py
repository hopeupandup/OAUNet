"""Online evaluator in dBZ, with per-sample metrics and on-disk memmaps."""

import os
import time
import json
import numpy as np
import torch
from tqdm import tqdm

from utils.logger import Logger
from metrics.deterministic import (
    csi, hss, evaluate_deterministic, normalized_to_dbz, DEFAULT_DBZ_THRESHOLDS,
)

# dBZ = u * 95 - 10, so this is the linear scale factor from u to dBZ.
_DBZ_SCALE = 95.0


class OnlineEvaluator:
    CHECKPOINT_INTERVAL = 100

    def __init__(self, framework, test_loader, device='cpu',
                 thresholds=None, use_dbz=True, max_value=255.0,
                 resume_dir=None, logger=None):
        self.framework = framework
        self.test_loader = test_loader
        self.device = device
        self.thresholds = [float(t) for t in (thresholds or DEFAULT_DBZ_THRESHOLDS)]
        self.use_dbz = bool(use_dbz)
        self.max_value = max_value
        self.result_dir = None
        self.logger = logger

        # Per-sample arrays go to on-disk memmaps; holding them in RAM needs >4 GB at 2000 samples.
        self.preds_mm = None
        self.targets_mm = None
        self.uncs_mm = None
        self._mm_paths = {}
        self._n_written = 0
        self._has_unc = False
        self._sample_shape = None
        self._ephemeral_store = None
        self.step_errors = []
        self.step_times = []
        self.start_sample = 0

        self.per_sample_csi = {t: [] for t in self.thresholds}
        self.per_sample_hss = {t: [] for t in self.thresholds}

        if resume_dir and os.path.exists(resume_dir):
            self._load_checkpoint(resume_dir)


    def _log(self, message, level='info'):
        if self.logger:
            getattr(self.logger, level)(message)
        else:
            print(message)


    def _get_checkpoint_path(self, result_dir):
        return os.path.join(result_dir, 'checkpoint.npz')

    def _get_framework_checkpoint_path(self, result_dir):
        return os.path.join(result_dir, 'framework_state.pth')

    def _get_meta_path(self, result_dir):
        return os.path.join(result_dir, 'checkpoint_meta.json')

    def _get_pred_path(self, result_dir):
        return os.path.join(result_dir, 'predictions_partial.npz')


    def _mm_paths_for(self, store_dir):
        return {
            'preds': os.path.join(store_dir, 'all_preds.npy'),
            'targets': os.path.join(store_dir, 'all_targets.npy'),
            'uncs': os.path.join(store_dir, 'all_uncertainties.npy'),
        }

    def _open_memmaps(self, store_dir, total_samples, sample_shape, resume=False):
        os.makedirs(store_dir, exist_ok=True)
        self._mm_paths = self._mm_paths_for(store_dir)
        sample_shape = tuple(sample_shape)
        self._sample_shape = sample_shape
        shape = (int(total_samples), *sample_shape)

        def _open(p, mode):
            return np.lib.format.open_memmap(p, mode=mode, dtype=np.float32, shape=shape)

        if resume and all(os.path.exists(p) for p in self._mm_paths.values()):
            modes = ('r+', 'r+', 'r+')
        else:
            modes = ('w+', 'w+', 'w+')
        self.preds_mm = _open(self._mm_paths['preds'], modes[0])
        self.targets_mm = _open(self._mm_paths['targets'], modes[1])
        self.uncs_mm = _open(self._mm_paths['uncs'], modes[2])

    def _flush_memmaps(self):
        for mm in (self.preds_mm, self.targets_mm, self.uncs_mm):
            if mm is not None:
                try:
                    mm.flush()
                except Exception:
                    pass

    def _cleanup_memmaps(self, move_to=None):
        for key, p in (self._mm_paths or {}).items():
            if not p or not os.path.exists(p):
                continue
            try:
                if move_to:
                    os.makedirs(move_to, exist_ok=True)
                    os.replace(p, os.path.join(move_to, os.path.basename(p)))
                else:
                    os.remove(p)
            except Exception:
                pass

    def release_memmaps(self, delete=True, move_to=None):
        self._flush_memmaps()
        self.preds_mm = None
        self.targets_mm = None
        self.uncs_mm = None
        if delete or move_to:
            self._cleanup_memmaps(move_to=move_to)


    def _to_metric_space(self, arr_np):
        if self.use_dbz:
            return normalized_to_dbz(arr_np)
        return arr_np * self.max_value


    def save_checkpoint(self, result_dir, current_step):
        os.makedirs(result_dir, exist_ok=True)
        self._flush_memmaps()

        ckpt_path = self._get_checkpoint_path(result_dir)
        save_dict = {
            'step_errors': np.array(self.step_errors, dtype=np.float32),
            'step_times': np.array(self.step_times, dtype=np.float32),
            'n_written': np.array([self._n_written], dtype=np.int64),
            'has_unc': np.array([int(self._has_unc)], dtype=np.int8),
        }
        for t in self.thresholds:
            if self.per_sample_csi[t]:
                save_dict[f'per_sample_csi_{t}'] = np.array(self.per_sample_csi[t], dtype=np.float32)
                save_dict[f'per_sample_hss_{t}'] = np.array(self.per_sample_hss[t], dtype=np.float32)
        np.savez(ckpt_path, **save_dict)

        fw_path = self._get_framework_checkpoint_path(result_dir)
        try:
            fw_state = self.framework.get_statistics()
            if hasattr(self.framework, 'model'):
                fw_state['model_state_dict'] = self.framework.model.state_dict()
            if hasattr(self.framework, 'optimizer'):
                fw_state['optimizer_state_dict'] = self.framework.optimizer.state_dict()
            for key in ('total_count', 'update_count', 'drift_count',
                        'vi_update_count', 'maml_update_count'):
                fw_state[key] = getattr(self.framework, key, 0)
            if hasattr(self.framework, 'adapt_buffer'):
                fw_state['adapt_buffer'] = [
                    (bx.detach().cpu(), by.detach().cpu())
                    for bx, by in self.framework.adapt_buffer
                ]
            torch.save(fw_state, fw_path)
        except Exception as e:
            self._log(f"  [警告] 框架状态保存失败: {e}", 'warning')

        meta = {
            'current_step': current_step,
            'total_preds': self._n_written,
            'total_errors': len(self.step_errors),
            'total_uncertainties': self._n_written if self._has_unc else 0,
            'metric_space': 'dBZ' if self.use_dbz else 'pixel',
            'thresholds': self.thresholds,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(self._get_meta_path(result_dir), 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def _load_checkpoint(self, result_dir):
        meta_path = self._get_meta_path(result_dir)
        ckpt_path = self._get_checkpoint_path(result_dir)
        fw_path = self._get_framework_checkpoint_path(result_dir)

        if not os.path.exists(meta_path):
            return 0

        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        self.start_sample = int(meta.get('current_step', 0))

        mm_paths = self._mm_paths_for(result_dir)
        has_mm = all(os.path.exists(mm_paths[k]) for k in ('preds', 'targets'))
        restart = False

        if os.path.exists(ckpt_path):
            data = np.load(ckpt_path, allow_pickle=True)
            if 'step_errors' in data:
                self.step_errors = data['step_errors'].tolist()
            if 'step_times' in data:
                self.step_times = data['step_times'].tolist()

            for t in self.thresholds:
                k_csi = f'per_sample_csi_{t}'
                k_hss = f'per_sample_hss_{t}'
                if k_csi in data:
                    self.per_sample_csi[t] = data[k_csi].tolist()
                    self.per_sample_hss[t] = data[k_hss].tolist()
                    self._log(f"  已恢复逐样本指标 (阈值 {t} dBZ): {len(self.per_sample_csi[t])} 个")

            if 'n_written' in data:
                self._n_written = int(np.asarray(data['n_written']).reshape(-1)[0])
            else:
                self._n_written = len(self.step_errors)
            if 'has_unc' in data:
                self._has_unc = bool(int(np.asarray(data['has_unc']).reshape(-1)[0]))

            # Resume only if the checkpoint is in the memmap-era format AND the memmaps are consistent;
            # a half-written cache would otherwise write the remaining samples into the wrong slots.

            new_format = 'n_written' in data
            if 'all_preds' in data or 'all_uncertainties' in data:
                self._log("  [兼容] 检测到旧格式断点（含大数组，续跑会二次 OOM）", 'warning')
            if (not new_format) or (not has_mm):
                if new_format and not has_mm:
                    self._log("  磁盘缓存不完整", 'warning')
                self._log("  该实验本次从头评估", 'warning')
                restart = True

        if (not restart) and self.start_sample > 0 and \
                ((not has_mm) or self._n_written == 0):
            self._log("  断点与磁盘缓存不一致（缓存缺失/计数为 0），从头评估", 'warning')
            restart = True

        if restart:
            self.start_sample = 0
            self._n_written = 0
            self.step_errors = []
            self.step_times = []
            for t in self.thresholds:
                self.per_sample_csi[t] = []
                self.per_sample_hss[t] = []
            self._has_unc = False
            self._mm_paths = mm_paths
            try:
                backup_dir = os.path.join(result_dir, 'checkpoint_backup')
                os.makedirs(backup_dir, exist_ok=True)
                for _fn in ('checkpoint.npz', 'checkpoint_meta.json'):
                    _fp = os.path.join(result_dir, _fn)
                    if os.path.exists(_fp):
                        os.replace(_fp, os.path.join(backup_dir, _fn))
                self._cleanup_memmaps()
                self._log(f"  旧断点已移入 {os.path.join(result_dir, 'checkpoint_backup')}")
            except Exception as e:
                self._log(f"  [警告] 清理旧断点失败: {e}", 'warning')

        if has_mm and not restart and self._n_written > 0 and self.start_sample > 0:
            try:
                self.preds_mm = np.load(mm_paths['preds'], mmap_mode='r+')
                self.targets_mm = np.load(mm_paths['targets'], mmap_mode='r+')
                if self._has_unc and os.path.exists(mm_paths['uncs']):
                    self.uncs_mm = np.load(mm_paths['uncs'], mmap_mode='r+')
                self._mm_paths = mm_paths
                self._log(f"  已续开磁盘缓存：已写 {self._n_written} 个样本，"
                          f"形状 {tuple(self.preds_mm.shape)}")
            except Exception as e:
                self._log(f"  [警告] 磁盘缓存续开失败: {e}，本次从头评估", 'warning')
                self.preds_mm = self.targets_mm = self.uncs_mm = None
                self.start_sample = 0
                self._n_written = 0
                restart = True

        if (not restart) and os.path.exists(fw_path) and hasattr(self.framework, 'model'):
            try:
                fw_state = torch.load(fw_path, map_location='cpu')
                if 'model_state_dict' in fw_state:
                    self.framework.model.load_state_dict(fw_state['model_state_dict'])
                    self._log("  框架模型权重已恢复")
                if 'optimizer_state_dict' in fw_state and hasattr(self.framework, 'optimizer'):
                    self.framework.optimizer.load_state_dict(fw_state['optimizer_state_dict'])
                if 'history' in fw_state and hasattr(self.framework, 'history'):
                    self.framework.history = fw_state['history']
                for key in ('total_count', 'update_count', 'drift_count',
                            'vi_update_count', 'maml_update_count'):
                    if key in fw_state and hasattr(self.framework, key):
                        setattr(self.framework, key, fw_state[key])
                if 'adapt_buffer' in fw_state and hasattr(self.framework, 'adapt_buffer'):
                    try:
                        self.framework.adapt_buffer = [
                            (bx.to(self.device), by.to(self.device))
                            for bx, by in fw_state['adapt_buffer']
                        ]
                    except Exception as e:
                        self._log(f"  [警告] adapt_buffer 恢复失败: {e}，从空缓冲开始", 'warning')
            except Exception as e:
                self._log(f"  [警告] 框架状态恢复失败: {e}，从头开始", 'warning')

        self._log(f"  从第 {self.start_sample} 个样本继续（已完成 {self.start_sample} 个）")
        return self.start_sample


    def evaluate(self, result_dir=None):
        self.result_dir = result_dir
        total_samples = len(self.test_loader.dataset)
        space = 'dBZ' if self.use_dbz else 'pixel'
        self._log(f"开始在线评估... 共 {total_samples} 个样本，从第 {self.start_sample} 个开始")
        self._log(f"  评测空间: {space}，阈值: {self.thresholds}")
        start_time = time.time()
        sample_count = 0

        store_dir = result_dir
        if store_dir is None:
            import tempfile
            store_dir = tempfile.mkdtemp(prefix='oaunet_eval_')
            self._ephemeral_store = store_dir
            self._log(f"  [注意] 未指定 result_dir，逐样本结果暂存于 {store_dir}")

        for batch_idx, (batch_x, batch_y) in enumerate(
                tqdm(self.test_loader, desc="在线评估", ncols=80)):

            if self.preds_mm is None:
                self._open_memmaps(store_dir, total_samples,
                                   tuple(batch_y.shape[1:]), resume=False)
                self._log(f"  逐样本结果直接落盘（不再驻留内存）: "
                          f"{os.path.basename(self._mm_paths['preds'])} "
                          f"shape={tuple(self.preds_mm.shape)}")

            for i in range(batch_x.shape[0]):
                sample_count += 1

                if sample_count <= self.start_sample:
                    continue

                x = batch_x[i:i + 1]
                y = batch_y[i:i + 1]

                step_start = time.time()
                pred, uncertainty = self.framework.online_step(x, y)
                step_time = time.time() - step_start

                pred_m = self._to_metric_space(pred.detach().numpy())
                target_m = self._to_metric_space(y.detach().numpy())

                p_flat = pred_m.reshape(-1)
                t_flat = target_m.reshape(-1)
                for th in self.thresholds:
                    self.per_sample_csi[th].append(csi(p_flat, t_flat, th))
                    self.per_sample_hss[th].append(hss(p_flat, t_flat, th))

                error = float(np.mean((pred_m - target_m) ** 2))

                if (sample_count - self.start_sample) % 50 == 0 or sample_count == self.start_sample + 1:
                    elapsed = time.time() - start_time
                    processed = sample_count - self.start_sample
                    speed = processed / elapsed if elapsed > 0 else 0
                    remaining = total_samples - sample_count
                    eta = remaining / speed if speed > 0 else 0
                    stats = self.framework.get_statistics()
                    updates = stats.get('update_count', 0)
                    csi_hi = self.per_sample_csi[self.thresholds[-1]][-1]
                    csi_lo = self.per_sample_csi[self.thresholds[0]][-1]
                    self._log(f"  [{sample_count}/{total_samples}] "
                              f"速度={speed:.1f}样本/s ETA={eta:.0f}s "
                              f"误差={error:.2f}(dBZ^2) 更新={updates} "
                              f"CSI@{self.thresholds[0]:g}={csi_lo:.3f} "
                              f"CSI@{self.thresholds[-1]:g}={csi_hi:.3f} "
                              f"耗时={step_time * 1000:.0f}ms")

                idx = self._n_written
                self.preds_mm[idx] = pred_m[0]
                self.targets_mm[idx] = target_m[0]
                if uncertainty is not None:
                    unc = uncertainty.detach().numpy()
                    scale = _DBZ_SCALE if self.use_dbz else self.max_value
                    self.uncs_mm[idx] = (unc * (scale ** 2))[0]
                    self._has_unc = True
                self._n_written = idx + 1

                self.step_errors.append(error)
                self.step_times.append(step_time)

                if result_dir and (sample_count % self.CHECKPOINT_INTERVAL == 0):
                    self.save_checkpoint(result_dir, sample_count)
                    self._log(f"  [checkpoint] 已保存到第 {sample_count} 个样本")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        total_time = time.time() - start_time
        self._flush_memmaps()

        if self.preds_mm is None:
            self._log("  [警告] 测试集为空（0 样本），跳过指标计算", 'warning')
            return {
                'deterministic': self._compute_deterministic(0),
                'per_sample': self._compute_per_sample_metrics(),
                'online': self._compute_online_metrics(),
                'timing': {'total_time': total_time, 'avg_time_per_sample': 0.0,
                           'avg_time_per_frame': 0.0,
                           'metric_space': 'dBZ' if self.use_dbz else 'pixel'},
                'framework_stats': self.framework.get_statistics(),
            }

        n = min(self._n_written, int(self.preds_mm.shape[0]))
        n_frames = int(self.preds_mm.shape[1])

        results = {
            'deterministic': self._compute_deterministic(n),
            'per_sample': self._compute_per_sample_metrics(),
            'online': self._compute_online_metrics(),
            'timing': {
                'total_time': total_time,
                'avg_time_per_sample': total_time / max(1, len(self.step_errors)),
                'avg_time_per_frame': total_time / max(1, len(self.step_errors) * n_frames),
                'metric_space': 'dBZ' if self.use_dbz else 'pixel',
            }
        }

        if self._has_unc and self.uncs_mm is not None:
            results['probabilistic'] = self._compute_probabilistic(n)

        results['framework_stats'] = self.framework.get_statistics()

        self._log("\n在线评估完成:")
        self._log(f"  总样本数: {len(self.step_errors)}")
        self._log(f"  总时间: {total_time:.1f}s")
        self._log(f"  更新次数: {results['framework_stats'].get('update_count', 'N/A')}")
        update_ratio = results['framework_stats'].get('update_ratio', 0.0)
        self._log(f"  更新比例: {update_ratio * 100:.2f}%")
        det = results['deterministic']
        for th, v in zip(det['thresholds'], det['csi']):
            self._log(f"  CSI@{th:g} dBZ: {v:.4f}")

        return results


    def _compute_deterministic(self, n):
        from metrics.deterministic import (
            evaluate_deterministic, evaluate_deterministic_chunked,
        )
        if self.preds_mm is None or n <= 0:
            empty = np.zeros((0,), dtype=np.float32)
            return evaluate_deterministic(empty, empty,
                                          thresholds=self.thresholds, scale='dbz')
        return evaluate_deterministic_chunked(
            self.preds_mm[:n], self.targets_mm[:n], thresholds=self.thresholds)

    def _compute_per_sample_metrics(self):
        out = {
            'thresholds': self.thresholds,
            'unit': 'dBZ',
            'note': 'decay_rate 已按 higher_is_better 归一：= -(末窗-首窗)/首窗；正值表示退化，负值表示提升',
            'csi': {},
            'hss': {},
        }
        for t in self.thresholds:
            csi_series = np.array(self.per_sample_csi[t], dtype=np.float64)
            hss_series = np.array(self.per_sample_hss[t], dtype=np.float64)
            out['csi'][str(t)] = csi_series.tolist()
            out['hss'][str(t)] = hss_series.tolist()
            out[f'csi_mean_{t:g}'] = float(csi_series.mean()) if csi_series.size else 0.0
            out[f'csi_decay_{t:g}'] = _window_decay(csi_series, higher_is_better=True)
        return out

    def _compute_probabilistic(self, n):
        from metrics.probabilistic import evaluate_probabilistic
        if self.uncs_mm is None or self.preds_mm is None or n <= 0:
            return {
                'nll': None, 'crps': None, 'ece': None,
                'note': '无不确定性估计，跳过概率指标',
            }
        return evaluate_probabilistic(
            self.preds_mm[:n], self.uncs_mm[:n], self.targets_mm[:n])

    def _compute_online_metrics(self):
        from metrics.online_metrics import evaluate_online_metrics
        history = self.framework.get_statistics().get('history', [])
        return evaluate_online_metrics(history, self.step_errors)


    def save_results(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        n = min(self._n_written,
                int(self.preds_mm.shape[0]) if self.preds_mm is not None else 0)

        np.save(os.path.join(output_dir, 'step_errors.npy'),
                np.array(self.step_errors, dtype=np.float32))

        if n > 0 and self.preds_mm is not None:
            self._flush_memmaps()
            np.savez_compressed(
                os.path.join(output_dir, 'predictions.npz'),
                preds=self.preds_mm[:n], targets=self.targets_mm[:n],
                metric_space=np.array(['dBZ' if self.use_dbz else 'pixel'])
            )
            if self._has_unc and self.uncs_mm is not None:
                np.save(os.path.join(output_dir, 'uncertainties.npy'),
                        self.uncs_mm[:n])

        per_sample_path = os.path.join(output_dir, 'per_sample_csi.npz')
        np.savez(
            per_sample_path,
            **{f'csi_{t}': np.array(self.per_sample_csi[t], dtype=np.float32)
               for t in self.thresholds},
            **{f'hss_{t}': np.array(self.per_sample_hss[t], dtype=np.float32)
               for t in self.thresholds},
            thresholds=np.array(self.thresholds),
        )

        history = self.framework.get_statistics().get('history', [])
        if history:
            with open(os.path.join(output_dir, 'online_history.json'), 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)

        backup_dir = os.path.join(output_dir, 'checkpoint_backup')
        os.makedirs(backup_dir, exist_ok=True)
        for fname in ['checkpoint.npz', 'framework_state.pth',
                      'checkpoint_meta.json', 'predictions_partial.npz']:
            fpath = os.path.join(output_dir, fname)
            if os.path.exists(fpath):
                os.replace(fpath, os.path.join(backup_dir, fname))

        self._log(f"结果已保存到: {output_dir}")
        self._log(f"  逐样本 CSI 序列: {per_sample_path}")


def _window_decay(series, window_size=50, higher_is_better=False):
    arr = np.asarray(series, dtype=np.float64)
    if arr.size < window_size * 2:
        return 0.0
    first = arr[:window_size].mean()
    last = arr[-window_size:].mean()
    rate = (last - first) / max(abs(first), 1e-8)
    return float(-rate if higher_is_better else rate)
