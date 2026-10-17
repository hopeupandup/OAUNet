"""Runs the experiment grid and writes comparison.json."""

import os
import sys
import argparse
import yaml
import json
import time
import shutil
import random
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from data.dataset import RadarDataset, CompactRadarDataset, get_dataloader, load_split
from oaunet.framework import OAUNet
from oaunet.online_evaluator import OnlineEvaluator
from baselines.offline_baseline import OfflineBaseline
from baselines.periodic_retrain import PeriodicRetrain
from baselines.online_gd import OnlineGD
from metrics.deterministic import evaluate_deterministic, save_results, DEFAULT_DBZ_THRESHOLDS
from utils.logger import Logger
from utils.losses import load_state_dict_compat
from utils.run_mode import add_run_mode_args, apply_run_mode
from utils.paths import PROJECT_ROOT, RESULT_DIR, resolve_config_path


# Execution order: the most expensive arm goes last, so a partial run still fills whole columns.
ALL_EXPERIMENTS = ['offline', 'offline_ovr', 'offline_sigma', 'oaunet_uniform',
                   'oaunet_vi_only', 'oaunet_no_ml', 'oaunet_no_dd',
                   'oaunet', 'oaunet_no_al', 'online_gd',
                   'periodic_retrain']


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def warn_if_stale_checkpoint(config, log):
    if not config.get('model', {}).get('heteroscedastic', False):
        return
    ckpt_path = config.get('checkpoint_path')
    if not ckpt_path or not os.path.exists(ckpt_path):
        return
    try:
        state = torch.load(ckpt_path, map_location='cpu')
        sd = state.get('model_state_dict', state)
        has_logvar = any('logvar' in k for k in sd.keys())
    except Exception as e:
        log(f"  [警告] 无法检查 checkpoint 是否含 σ 头: {e}")
        return

    if not has_logvar:
        log("[警告] 权重不含 σ 头，但 config 已 heteroscedastic=True："
            "NLL / CRPS / ECE 不可信，请先重跑 train.py 生成含 σ 头的权重。")


def create_experiment_config(base_config, experiment_name, checkpoint_path):
    config = base_config.copy()
    config['checkpoint_path'] = checkpoint_path

    if experiment_name == 'oaunet':
        config['enable_online_vi'] = True
        config['enable_active_learning'] = True
        config['enable_drift_detection'] = True
        config['enable_meta_learning'] = True

    elif experiment_name == 'oaunet_no_al':
        config['enable_online_vi'] = True
        config['enable_active_learning'] = False
        config['enable_drift_detection'] = True
        config['enable_meta_learning'] = True

    elif experiment_name == 'oaunet_no_dd':
        config['enable_online_vi'] = True
        config['enable_active_learning'] = True
        config['enable_drift_detection'] = False
        config['enable_meta_learning'] = True

    elif experiment_name == 'oaunet_no_ml':
        config['enable_online_vi'] = True
        config['enable_active_learning'] = True
        config['enable_drift_detection'] = True
        config['enable_meta_learning'] = False

    elif experiment_name == 'oaunet_vi_only':
        config['enable_online_vi'] = True
        config['enable_active_learning'] = False
        config['enable_drift_detection'] = False
        config['enable_meta_learning'] = False

    elif experiment_name == 'oaunet_uniform':

        config['enable_online_vi'] = True
        config['enable_active_learning'] = True
        config['enable_drift_detection'] = True
        config['enable_meta_learning'] = True
        al = dict(config.get('active_learning') or {})
        al['gate_mode'] = 'uniform'
        config['active_learning'] = al

    elif experiment_name == 'offline_ovr':
        config['enable_online_vi'] = False
        config['enable_active_learning'] = False
        config['enable_drift_detection'] = False
        config['enable_meta_learning'] = False
        rec = dict(config.get('recalibration') or {})
        rec['enable'] = True
        config['recalibration'] = rec

    elif experiment_name == 'offline_sigma':

        # offline_sigma and offline_ovr differ only in OVR, so any calibration gain is attributable to it.
        config['enable_online_vi'] = False
        config['enable_active_learning'] = False
        config['enable_drift_detection'] = False
        config['enable_meta_learning'] = False
        rec = dict(config.get('recalibration') or {})
        rec['enable'] = False
        config['recalibration'] = rec

    return config


def get_experiment_status(config):
    results_dir = config['paths']['results_dir']
    checkpoints_dir = config['paths']['checkpoints_dir']
    status = {}
    for exp in ALL_EXPERIMENTS:
        exp_dir = os.path.join(results_dir, exp)
        summary_path = os.path.join(exp_dir, 'summary.json')
        checkpoint_path = os.path.join(exp_dir, 'checkpoint_meta.json')

        if os.path.exists(summary_path):
            status[exp] = 'completed'
        elif os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r') as f:
                meta = json.load(f)
            status[exp] = f'checkpoint (step {meta.get("current_step", "?")})'
        else:
            if exp in ('oaunet', 'oaunet_no_al', 'oaunet_no_dd', 'oaunet_uniform'):
                meta_ckpt = os.path.join(checkpoints_dir, 'meta_checkpoints', 'meta_checkpoint.pt')
                meta_done = os.path.join(checkpoints_dir, 'meta_trained.pt')
                if os.path.exists(meta_ckpt):
                    try:
                        ckpt = torch.load(meta_ckpt, map_location='cpu')
                        epoch = ckpt.get('epoch', '?')
                        status[exp] = f'meta-ckpt (epoch {epoch})'
                        continue
                    except Exception:
                        pass
                elif os.path.exists(meta_done):
                    status[exp] = 'meta-done (pending eval)'
                    continue
            status[exp] = 'pending'
    return status


def clear_meta_checkpoint(checkpoints_dir):
    meta_ckpt_dir = os.path.join(checkpoints_dir, 'meta_checkpoints')
    meta_done = os.path.join(checkpoints_dir, 'meta_trained.pt')
    if os.path.exists(meta_ckpt_dir):
        shutil.rmtree(meta_ckpt_dir)
    if os.path.exists(meta_done):
        os.remove(meta_done)


def interactive_menu(config):
    status = get_experiment_status(config)
    mode = config.get('run_mode', 'formal')
    lt = config.get('_limit_train')
    ls = config.get('_limit_test')
    scope = (f"训练 {lt} / 测试 {ls} 样本" if mode == 'verify'
             else "训练 2000 / 测试 2000 样本（全量）")

    print(f"\n{'='*60}")
    print("  OAUNet 实验管理器")
    print(f"{'='*60}")
    print(f"  运行模式: {mode}（{scope}）")
    print(f"  产出目录: {config['paths']['results_dir']}")
    print(f"\n{'实验名称':<20} {'状态':<30}")
    print("-" * 50)
    for exp in ALL_EXPERIMENTS:
        s = status[exp]
        if s == 'completed':
            tag = '[完成]'
        elif s.startswith('checkpoint') or s.startswith('meta-'):
            tag = '[断点]'
        else:
            tag = '[待跑]'
        print(f"  {tag} {exp:<18} {s}")

    print(f"\n{'选项':<5} {'说明'}")
    print("-" * 50)
    print(f"  [1] 运行所有未完成的实验（跳过已完成 + 从断点续跑）")
    print(f"  [2] 运行所有实验（强制重新开始）")
    print(f"  [3] 选择单个实验运行")
    print(f"  [4] 仅运行未完成的消融实验")
    print(f"  [5] 退出")

    while True:
        choice = input("\n请选择 [1-5]: ").strip()

        if choice == '1':
            to_run = []
            for exp in ALL_EXPERIMENTS:
                if status[exp] != 'completed':
                    to_run.append(exp)
            if not to_run:
                print("所有实验已完成！")
                return None, None, None
            print(f"\n将运行: {to_run}")
            return to_run, True, True

        elif choice == '2':
            print(f"\n将重新运行所有实验: {ALL_EXPERIMENTS}")
            clear_meta_checkpoint(config['paths']['checkpoints_dir'])
            return ALL_EXPERIMENTS, False, False

        elif choice == '3':
            print("\n可选实验:")
            for i, exp in enumerate(ALL_EXPERIMENTS, 1):
                print(f"  [{i}] {exp} ({status[exp]})")
            while True:
                try:
                    idx = int(input("选择实验编号: ").strip()) - 1
                    if 0 <= idx < len(ALL_EXPERIMENTS):
                        exp = ALL_EXPERIMENTS[idx]
                        if status[exp].startswith('checkpoint'):
                            r = input(f"  {exp} 有断点，是否续跑？[Y/n]: ").strip().lower()
                            resume = r != 'n'
                        elif status[exp] == 'completed':
                            r = input(f"  {exp} 已完成，是否强制重新运行？[y/N]: ").strip().lower()
                            if r != 'y':
                                continue
                            resume = False
                            if exp in ('oaunet', 'oaunet_no_al', 'oaunet_no_dd', 'oaunet_uniform'):
                                clear_meta_checkpoint(config['paths']['checkpoints_dir'])
                        else:
                            resume = False
                        return [exp], False, resume
                    print("无效编号，请重新选择")
                except ValueError:
                    print("请输入数字")

        elif choice == '4':
            ablation = ['oaunet_vi_only', 'oaunet_no_al', 'oaunet_no_dd',
                        'oaunet_no_ml', 'oaunet', 'oaunet_uniform']
            to_run = [e for e in ablation if status[e] != 'completed']
            if not to_run:
                print("所有消融实验已完成！")
                return None, None, None
            print(f"\n将运行: {to_run}")
            return to_run, True, True

        elif choice == '5':
            sys.exit(0)
        else:
            print("无效选择，请输入 1-5")


def _compute_per_frame_metrics(evaluator, thresholds, chunk=64):
    preds_mm = getattr(evaluator, 'preds_mm', None)
    targets_mm = getattr(evaluator, 'targets_mm', None)
    n = int(getattr(evaluator, '_n_written', 0))
    if preds_mm is None or targets_mm is None or n <= 0:
        return None
    n = min(n, int(preds_mm.shape[0]))
    n_frames = int(preds_mm.shape[1])
    thresholds = [float(t) for t in thresholds]

    m = len(thresholds)
    tp = np.zeros((m, n_frames), dtype=np.int64)
    fp = np.zeros((m, n_frames), dtype=np.int64)
    fn = np.zeros((m, n_frames), dtype=np.int64)
    tn = np.zeros((m, n_frames), dtype=np.int64)

    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        p = np.asarray(preds_mm[s:e], dtype=np.float32)
        t = np.asarray(targets_mm[s:e], dtype=np.float32)
        for k in range(n_frames):
            pk = p[:, k]
            tk = t[:, k]
            for j, th in enumerate(thresholds):
                pb = pk >= th
                tb = tk >= th
                tp[j, k] += int(np.count_nonzero(pb & tb))
                fp[j, k] += int(np.count_nonzero(pb & ~tb))
                fn[j, k] += int(np.count_nonzero(~pb & tb))
                tn[j, k] += int(np.count_nonzero(~pb & ~tb))

    per_frame = {'thresholds': thresholds,
                 'frames': list(range(1, n_frames + 1))}
    for j, th in enumerate(thresholds):
        csi_list, hss_list = [], []
        for k in range(n_frames):
            denom = int(tp[j, k] + fp[j, k] + fn[j, k])
            csi_list.append(float(tp[j, k] / denom) if denom else 0.0)
            num = 2.0 * (float(tp[j, k]) * float(tn[j, k])
                         - float(fp[j, k]) * float(fn[j, k]))
            den = float((tp[j, k] + fn[j, k]) * (fn[j, k] + tn[j, k])
                        + (tp[j, k] + fp[j, k]) * (fp[j, k] + tn[j, k]))
            hss_list.append(float(num / den) if den else 0.0)
        per_frame[f'csi_{th}'] = csi_list
        per_frame[f'hss_{th}'] = hss_list
    return per_frame


def run_single_experiment(experiment_name, config, test_loader, train_data=None,
                          resume=False, logger=None):
    log = logger.info if logger else print
    log(f"\n{'='*60}")
    log(f"实验: {experiment_name}")
    log(f"{'='*60}")

    result_dir = os.path.join(config['paths']['results_dir'], experiment_name)
    figure_dir = os.path.join(config['paths']['figures_dir'], experiment_name)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    if experiment_name == 'offline':
        framework = OfflineBaseline(config)
    elif experiment_name == 'periodic_retrain':
        framework = PeriodicRetrain(config)
    elif experiment_name == 'online_gd':
        framework = OnlineGD(config)
    else:
        framework = OAUNet(config)

        if config.get('enable_meta_learning', False) and train_data is not None:
            shared_meta_ckpt = os.path.join(
                config['paths']['checkpoints_dir'], 'meta_trained.pt'
            )
            meta_ckpt_dir = os.path.join(
                config['paths']['checkpoints_dir'], 'meta_checkpoints'
            )
            if os.path.exists(shared_meta_ckpt):
                log("加载共享元训练权重...")
                try:
                    fw_state = torch.load(shared_meta_ckpt, map_location='cpu')
                    if 'model_state_dict' in fw_state:
                        load_state_dict_compat(framework.model,
                                               fw_state['model_state_dict'],
                                               logger=logger,
                                               context='run_experiments.meta_trained')
                except Exception as e:
                    log(f"  [警告] 元训练权重加载失败: {e}")
                    log("  将重新执行元训练（若这是权重结构与 config 不匹配，请先确认无误再继续）")
                    framework.meta_train(
                        train_data['inputs'], train_data['outputs'], train_data['labels'],
                        n_epochs=config.get('meta_learning', {}).get('meta_epochs', 50),
                        checkpoint_dir=meta_ckpt_dir
                    )
                    torch.save({'model_state_dict': framework.model.state_dict()}, shared_meta_ckpt)
            else:
                meta_ckpt_path = os.path.join(meta_ckpt_dir, 'meta_checkpoint.pt')
                if os.path.exists(meta_ckpt_path):
                    log("发现元训练 checkpoint，从断点继续...")
                else:
                    log("执行元训练（首次运行，结果会缓存供后续消融实验复用）...")
                framework.meta_train(
                    train_data['inputs'], train_data['outputs'], train_data['labels'],
                    n_epochs=config.get('meta_learning', {}).get('meta_epochs', 50),
                    checkpoint_dir=meta_ckpt_dir
                )
                torch.save({'model_state_dict': framework.model.state_dict()}, shared_meta_ckpt)
                log(f"  元训练权重已保存到: {shared_meta_ckpt}")
                if os.path.exists(meta_ckpt_path):
                    os.remove(meta_ckpt_path)

    resume_dir = result_dir if resume else None
    eval_cfg = config.get('evaluation', {})
    evaluator = OnlineEvaluator(
        framework, test_loader,
        device=config.get('device', 'cpu'),
        thresholds=eval_cfg.get('thresholds', DEFAULT_DBZ_THRESHOLDS),
        use_dbz=eval_cfg.get('use_dbz', True),
        max_value=eval_cfg.get('max_value', 255.0),
        resume_dir=resume_dir,
        logger=logger
    )

    results = evaluator.evaluate(result_dir=result_dir)
    evaluator.save_results(result_dir)

    per_frame = _compute_per_frame_metrics(
        evaluator, eval_cfg.get('thresholds', DEFAULT_DBZ_THRESHOLDS)
    )

    try:
        evaluator.release_memmaps()
    except Exception as e:
        log(f"  [警告] 释放磁盘缓存失败: {e}")

    summary = {
        'experiment': experiment_name,
        'deterministic': results['deterministic'],
        'per_frame': per_frame,
        'online': results['online'],
        'timing': results['timing'],
        'framework_stats': {
            k: v for k, v in results['framework_stats'].items()
            if k != 'history'
        }
    }
    ps = results.get('per_sample')
    if ps:
        ths = ps.get('thresholds', [])
        summary['per_sample_summary'] = {
            'unit': ps.get('unit'),
            'note': ps.get('note'),
            'csi_mean': {str(t): ps.get(f'csi_mean_{t:g}') for t in ths},
            'csi_decay': {str(t): ps.get(f'csi_decay_{t:g}') for t in ths},
        }
    if 'probabilistic' in results:
        summary['probabilistic'] = results['probabilistic']

    summary_path = os.path.join(result_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    det = results['deterministic']
    th_list = det.get('thresholds', [])
    try:
        n_test = len(test_loader.dataset)
    except Exception:
        n_test = '?'
    log(f"\n{experiment_name} 结果（测试 {n_test} 样本）:")
    if th_list:
        unit = det.get('threshold_unit', 'dBZ')
        csi_list = det.get('csi', [])
        hss_list = det.get('hss', [])
        for i, th in enumerate(th_list):
            c = csi_list[i] if i < len(csi_list) else float('nan')
            h = hss_list[i] if i < len(hss_list) else float('nan')
            log(f"  CSI@{th:g} {unit}: {c:.4f}   HSS@{th:g}: {h:.4f}")
    log(f"  更新比例: {results['framework_stats'].get('update_ratio', 'N/A')}")
    pb = results.get('probabilistic')
    if isinstance(pb, dict) and pb:
        def _fmt(v):
            return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
        log(f"  概率指标: NLL={_fmt(pb.get('nll'))}  "
            f"CRPS={_fmt(pb.get('crps'))}  ECE={_fmt(pb.get('ece'))}")
    ps = results.get('per_sample')
    if isinstance(ps, dict) and ps.get('thresholds'):
        dec = {f"{t:g}dBZ": ps.get(f'csi_decay_{t:g}') for t in ps['thresholds']}
        log(f"  CSI 衰减率: {dec}")
    log(f"  产出目录: {result_dir}")

    return summary


def main():
    parser = argparse.ArgumentParser(description='OAUNet 实验')
    parser.add_argument('--config', type=str, default='configs/oaunet_temporal.yaml',
                        help='配置文件路径（本副本默认 configs/oaunet_temporal.yaml）')
    parser.add_argument('--experiment', type=str, default=None,
                        choices=['all', 'oaunet', 'baselines', 'ablation',
                                 'offline', 'offline_ovr', 'offline_sigma', 'periodic_retrain', 'online_gd',
                                 'oaunet_no_al', 'oaunet_no_dd', 'oaunet_no_ml',
                                 'oaunet_vi_only', 'oaunet_uniform'])
    parser.add_argument('--data_prefix', type=str, default='test')
    parser.add_argument('--split', type=str, default=None,
                        help='split 文件名（如 compact_split.json），指定后自动区分 train/test 索引')
    parser.add_argument('--compact', action='store_true',
                        help='使用紧凑数据集（从 data_compact/compact.npz 加载，遗留协议）')
    parser.add_argument('--temporal', action='store_true',
                        help='使用时间切分数据集（从 data_temporal/ 加载）')
    parser.add_argument('--temporal_name', type=str, default=None,
                        help='时间切分数据名（不含扩展名，如 temporal_strong）；'
                             '留空则自动解析 data_temporal/ 下的 npz')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--gate_mode', type=str, default=None,
                        choices=['difficulty', 'uniform'],
                        help='覆盖 active_learning.gate_mode（对照实验用）。'
                             'difficulty=现状（只挑最难 ~9.4%%）；'
                             'uniform=同预算随机抽，用于分离选择偏置与更新次数')
    parser.add_argument('--uniform_rate', type=float, default=None,
                        help='覆盖 active_learning.uniform_rate（仅 gate_mode=uniform 生效）。'
                             '用于扫「更新预算 vs 精度」的权衡曲线（C3）：'
                             '0.094≈283 次 / 0.03≈90 次 / 0.01≈30 次（3000 样本）')
    parser.add_argument('--epsilon_mode', type=str, default=None,
                        choices=['hoeffding', 'variance'],
                        help='覆盖 drift_detection.epsilon_mode（漂移阈值口径）。'
                             'hoeffding=现状（不使用方差，偏松）；'
                             'variance=Bernstein 方差自适应界（更紧）')
    parser.add_argument('--sudden_mode', type=str, default=None,
                        choices=['legacy', 'se'],
                        help='覆盖 drift_detection.sudden_mode（突变判据口径）。'
                             'legacy=现状（2×逐样本 std，本流结构上不可能通过）；'
                             'se=均值差标准误（可检出块级突变，让 MAML 有触发机会）')
    parser.add_argument('--sudden_factor', type=float, default=None,
                        help='覆盖 drift_detection.sudden_factor（se 口径的倍数，默认 2.0）')
    parser.add_argument('--recalibration', type=str, default=None,
                        choices=['on', 'off'],
                        help='覆盖 recalibration.enable（在线方差重校准 OVR）。'
                             'on=对所有档位启用；off=强制关闭。'
                             '注意：offline_ovr 档位自身会强制开启，不受此项影响；'
                             '若要与它做同 μ 对照，请直接用 offline_sigma 档位（OVR 关）。')
    parser.add_argument('--update_mode', type=str, default=None,
                        choices=['single', 'buffer'],
                        help='覆盖 online_vi.update_mode（在线更新规则）。'
                             'single=现状（单样本一步 Adam）；'
                             'buffer=缓冲小批量（对齐 periodic_retrain 的胜出配方）')
    parser.add_argument('--buffer_lr', type=float, default=None,
                        help='覆盖 online_vi.buffer_lr（buffer 模式的学习率，默认 1e-4）')
    parser.add_argument('--max_test', type=int, default=None,
                        help='限制测试样本数（用于便宜的先行试验 pilot）。'
                             '截取的是测试流的**前** N 个样本；正式结果不要用')
    parser.add_argument('--results_tag', type=str, default=None,
                        help='给产出目录追加后缀（如 --results_tag buf）⇒ '
                             'metrics/<split>_buf/。用于同名档位的 A/B 先行试验，'
                             '避免互相覆盖。正式结果不要用')
    parser.add_argument('--resume', action='store_true',
                        help='从断点续跑（加载 checkpoint 继续）')
    parser.add_argument('--skip_completed', action='store_true',
                        help='跳过已完成的实验')
    parser.add_argument('--force', action='store_true',
                        help='强制重新运行（忽略已有结果）')
    add_run_mode_args(parser)
    args = parser.parse_args()

    current_dir = Path(__file__).resolve().parent

    config = load_config(args.config)

    mode_info = apply_run_mode(config, run_mode=args.run_mode,
                               limit_train=args.limit_train,
                               limit_test=args.limit_test)

    # --limit_test takes the *first* N test samples, where drift has barely unfolded: ranking only.
    if args.max_test is not None:
        config['_limit_test'] = int(args.max_test)

    if args.device:
        config['device'] = args.device
    elif config.get('device') == 'auto':
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.gate_mode:
        config.setdefault('active_learning', {})['gate_mode'] = args.gate_mode
    if args.uniform_rate is not None:
        config.setdefault('active_learning', {})['uniform_rate'] = args.uniform_rate

    if args.epsilon_mode:
        config.setdefault('drift_detection', {})['epsilon_mode'] = args.epsilon_mode
    if args.sudden_mode:
        config.setdefault('drift_detection', {})['sudden_mode'] = args.sudden_mode
    if args.sudden_factor is not None:
        config.setdefault('drift_detection', {})['sudden_factor'] = args.sudden_factor

    if args.recalibration is not None:
        rec = dict(config.get('recalibration') or {})
        rec['enable'] = (args.recalibration == 'on')
        config['recalibration'] = rec

    if args.update_mode is not None:
        config.setdefault('online_vi', {})['update_mode'] = args.update_mode
    if args.buffer_lr is not None:
        config.setdefault('online_vi', {})['buffer_lr'] = args.buffer_lr

    for key in ('data_dir', 'checkpoints_dir', 'results_dir', 'figures_dir', 'logs_dir'):
        if key in config.get('paths', {}):
            config['paths'][key] = resolve_config_path(config['paths'][key])
    if config.get('checkpoint_path'):
        config['checkpoint_path'] = resolve_config_path(config['checkpoint_path'])

    seed = config.get('seed', config.get('training', {}).get('seed', 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logs_dir = config.get('paths', {}).get('logs_dir') or str(RESULT_DIR / 'logs')
    logger = Logger(log_dir=logs_dir, log_name='experiments')
    log = logger.info

    log(mode_info['banner'])
    log(f"设备: {config['device']}")

    warn_if_stale_checkpoint(config, log)

    compact_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_compact')
    npz_path = os.path.join(compact_dir, 'compact.npz')
    split_json_path = os.path.join(compact_dir, 'compact_split.json')

    if args.temporal:
        data_mode = 'temporal'
    elif args.compact:
        data_mode = 'compact'
    elif args.split:
        data_mode = 'split'
    else:
        data_mode = 'plain'

    if data_mode == 'temporal':
        from utils.paths import find_temporal_split
        from data.dataset import TemporalRadarDataset

        prefer = args.temporal_name or config.get('data', {}).get('temporal_name')
        split_path = find_temporal_split(prefer)
        if split_path is None:
            logger.error(f"未在 data_temporal/ 找到时间切分索引"
                         f"（指定名: {prefer or '自动'}）")
            logger.error("请先运行 python -m data.make_temporal 生成索引")
            return
        split_json_path = str(split_path)

        log(f"时间切分协议模式，索引: {split_json_path}")

        # One output directory per drift arm, so a stale summary.json cannot silently skip a newer run.
        drift_tag = Path(split_json_path).stem
        for _k in ('results_dir', 'figures_dir'):
            if _k in config.get('paths', {}):
                config['paths'][_k] = os.path.join(config['paths'][_k], drift_tag)
        os.makedirs(config['paths']['results_dir'], exist_ok=True)
        os.makedirs(config['paths']['figures_dir'], exist_ok=True)
        log(f"结果目录（按档位隔离）: {config['paths']['results_dir']}")
        log(f"图目录（按档位隔离）: {config['paths']['figures_dir']}")

        test_dataset = TemporalRadarDataset(split_json_path, split='test',
                                            limit=config.get('_limit_test'))
        test_loader = get_dataloader(test_dataset, batch_size=1, shuffle=False)

        with open(split_json_path, 'r', encoding='utf-8') as f:
            split_info = json.load(f)
        train_indices = split_info['train_indices']
        test_indices = split_info['test_indices']
        log(f"  漂移档位: {split_info.get('drift', '?')} "
            f"({split_info.get('drift_desc', '')})")
        log(f"  训练源索引段: {split_info.get('train_range')}  "
            f"测试源索引段: {split_info.get('test_range')}  "
            f"gap: {split_info.get('gap_range')}")
    elif data_mode == 'compact':
        if not os.path.exists(npz_path):
            logger.error(f"紧凑数据文件不存在: {npz_path}")
            logger.error("请先运行 python -m data.make_compact 生成数据")
            return

        log(f"紧凑数据集模式: {npz_path}")
        test_dataset = CompactRadarDataset(npz_path, split='test',
                                           limit=config.get('_limit_test'))
        test_loader = get_dataloader(test_dataset, batch_size=1, shuffle=False)

        with open(split_json_path, 'r') as f:
            split_info = json.load(f)
        train_indices = split_info['train_indices']
        test_indices = split_info['test_indices']
    else:
        data_dir = config['paths']['data_dir']
        test_indices = None
        train_indices = None
        if args.split:
            split_path = os.path.join(data_dir, args.split)
            if not os.path.exists(split_path):
                logger.error(f"Split 文件不存在: {split_path}")
                return
            train_indices = load_split(split_path, split='train')
            test_indices = load_split(split_path, split='test')
            inputs_path = os.path.join(data_dir, f'{args.data_prefix}_inputs.npy')
            outputs_path = os.path.join(data_dir, f'{args.data_prefix}_outputs.npy')
            log(f"Split 模式: 训练 {len(train_indices)} 样本, 测试 {len(test_indices)} 样本")
        else:
            inputs_path = os.path.join(data_dir, f'{args.data_prefix}_inputs.npy')
            outputs_path = os.path.join(data_dir, f'{args.data_prefix}_outputs.npy')

        if not os.path.exists(inputs_path):
            logger.error(f"测试数据不存在: {inputs_path}")
            logger.error("请先运行 preprocess.py 生成测试数据")
            return

        test_dataset = RadarDataset(inputs_path, outputs_path, indices=test_indices,
                                    limit=config.get('_limit_test'))
        test_loader = get_dataloader(test_dataset, batch_size=1, shuffle=False)

    if args.results_tag:
        for _k in ('results_dir', 'figures_dir'):
            if _k in config.get('paths', {}):
                config['paths'][_k] = f"{config['paths'][_k]}_{args.results_tag}"
        os.makedirs(config['paths']['results_dir'], exist_ok=True)
        os.makedirs(config['paths']['figures_dir'], exist_ok=True)
        log(f"产出目录标签: _{args.results_tag} → {config['paths']['results_dir']}")

    checkpoint_path = config['checkpoint_path']

    if args.experiment is None:
        experiments, skip_completed, resume = interactive_menu(config)
        if experiments is None:
            return
    elif args.experiment == 'all':
        experiments = ALL_EXPERIMENTS
    elif args.experiment == 'baselines':
        experiments = ['offline', 'offline_ovr', 'offline_sigma', 'online_gd', 'periodic_retrain']
    elif args.experiment == 'ablation':
        experiments = ['oaunet_vi_only', 'oaunet_no_al', 'oaunet_no_dd',
                       'oaunet_no_ml', 'oaunet', 'oaunet_uniform']
    else:
        experiments = [args.experiment]

    skip_completed = False
    if args.force:
        skip_completed = False
        resume = False
        clear_meta_checkpoint(config['paths']['checkpoints_dir'])
    else:
        if args.skip_completed:
            skip_completed = True
        elif len(experiments) > 1:
            skip_completed = True
        resume = True

    if skip_completed:
        results_dir = config['paths']['results_dir']
        original = experiments[:]
        experiments = []
        for exp in original:
            summary_path = os.path.join(results_dir, exp, 'summary.json')
            if os.path.exists(summary_path):
                log(f"  跳过已完成: {exp}")
            else:
                experiments.append(exp)

    if not experiments:
        log("\n所有实验已完成! 如需重新运行，加 --force 参数")
        return

    log(f"\n即将运行 {len(experiments)} 个实验: {experiments}")
    if resume:
        log("模式: 自动跳过已完成 + 断点续跑（--force 强制重跑）")

    train_data = None
    needs_train = any(e in ('oaunet', 'oaunet_no_al', 'oaunet_no_dd',
                            'oaunet_uniform') for e in experiments)

    if needs_train:
        _lt = config.get('_limit_train')
        if data_mode == 'temporal':
            from data.dataset import load_temporal_train_data

            _n = min(len(train_indices), _lt) if _lt else len(train_indices)
            log(f"加载训练数据（时间切分，{_n} 样本，按索引只读）...")
            train_data = load_temporal_train_data(split_json_path, limit=_lt)
        elif data_mode == 'compact':
            _ti = train_indices[:_lt] if _lt else train_indices
            log(f"加载训练数据（紧凑模式，{len(_ti)} 样本）...")
            data = np.load(npz_path)
            all_labels = data['labels'] if 'labels' in data else np.zeros(len(data['inputs']))
            train_data = {
                'inputs': data['inputs'][_ti],
                'outputs': data['outputs'][_ti],
                'labels': all_labels[_ti]
            }
            log(f"训练数据已加载: {train_data['inputs'].shape[0]} 样本")
        elif data_mode == 'split':
            _ti = train_indices[:_lt] if _lt else train_indices
            log(f"加载训练数据（split 模式，{len(_ti)} 样本）...")
            all_inputs = np.load(inputs_path)
            all_outputs = np.load(outputs_path)
            labels_path = os.path.join(data_dir, f'{args.data_prefix}_labels.npy')
            all_labels = np.load(labels_path) if os.path.exists(labels_path) else np.zeros(len(all_inputs))
            train_data = {
                'inputs': all_inputs[_ti],
                'outputs': all_outputs[_ti],
                'labels': all_labels[_ti]
            }
            log(f"训练数据已加载: {train_data['inputs'].shape[0]} 样本")
        else:
            train_inputs_path = os.path.join(data_dir, 'train_inputs.npy')
            train_outputs_path = os.path.join(data_dir, 'train_outputs.npy')
            train_labels_path = os.path.join(data_dir, 'train_labels.npy')
            if os.path.exists(train_inputs_path) and os.path.exists(train_labels_path):
                log("正在加载训练数据（用于元学习，约 5.8GB，请稍候）...")
                train_data = {
                    'inputs': np.load(train_inputs_path),
                    'outputs': np.load(train_outputs_path),
                    'labels': np.load(train_labels_path)
                }
                log(f"训练数据已加载: {train_data['inputs'].shape[0]} 样本")

    all_results = {}
    for exp_name in experiments:
        exp_config = create_experiment_config(config, exp_name, checkpoint_path)
        summary = run_single_experiment(
            exp_name, exp_config, test_loader, train_data,
            resume=resume, logger=logger
        )
        all_results[exp_name] = summary

    if len(all_results) > 1:
        comparison_path = os.path.join(config['paths']['results_dir'], 'comparison.json')
        with open(comparison_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        log(f"\n对比结果已保存: {comparison_path}")

    log(f"\n{'='*60}")
    log("所有实验完成!")
    log(f"运行模式: {config['run_mode']}")
    log(f"产出目录: {config['paths']['results_dir']}")
    if config['run_mode'] == 'verify':
        log("提示: 这是 100 样本的检验结果，仅供链路自检；"
            "正式结果请用 formal 模式（不加 --run_mode）重跑，并加 --force。")
    log(f"{'='*60}")


if __name__ == '__main__':
    main()
