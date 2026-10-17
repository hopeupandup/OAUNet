"""Trains the base model on the temporal split."""

import os
import argparse
import yaml
import numpy as np
import random
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

from data.dataset import RadarDataset, CompactRadarDataset, get_dataloader, load_split
from models.trajgru import TrajGRU, count_parameters
from utils.logger import Logger
from utils.losses import gaussian_nll, mse_nll_loss, load_state_dict_compat
from utils.run_mode import add_run_mode_args, apply_run_mode


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_model(config):
    hidden_channels = config['model'].get('hidden_channels', [64, 128])
    num_flows = config['model'].get('num_flows', [9, 9])

    if isinstance(hidden_channels, int):
        hidden_channels = [hidden_channels] * config['model']['num_layers']
    if isinstance(num_flows, int):
        num_flows = [num_flows] * config['model']['num_layers']

    model = TrajGRU(
        input_channels=config['model']['input_channels'],
        hidden_channels=hidden_channels,
        num_layers=config['model']['num_layers'],
        output_frames=config['model']['output_frames'],
        num_flows=num_flows,
        use_checkpoint=config['model'].get('use_checkpoint', False),
        heteroscedastic=config['model'].get('heteroscedastic', False)
    )
    return model


def get_optimizer(config, model):
    optimizer_name = config['training']['optimizer'].lower()
    lr = config['training']['learning_rate']

    # The σ head gets its own larger lr: μ and σ gradients differ by orders of magnitude.
    sigma_mult = float(config['training'].get('sigma_lr_mult', 3.0))
    sigma_params = [p for n, p in model.named_parameters() if 'logvar' in n]
    base_params = [p for n, p in model.named_parameters() if 'logvar' not in n]
    groups = []
    if base_params:
        groups.append({'params': base_params, 'lr': lr})
    if sigma_params:
        groups.append({'params': sigma_params, 'lr': lr * sigma_mult})
    params = groups if groups else model.parameters()

    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(params, lr=lr)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)
    else:
        raise ValueError(f"未知优化器类型: {optimizer_name}")

    return optimizer


def get_loss_fn(config):
    loss_name = config['training']['loss'].lower()

    if loss_name in ('gaussian_nll', 'nll', 'heteroscedastic', 'mse_nll'):
        return None
    if loss_name == 'mse':
        return nn.MSELoss()
    if loss_name == 'mae':
        return nn.L1Loss()
    raise ValueError(f"未知损失函数类型: {loss_name}")


def compute_train_loss(model, loss_fn, batch_x, batch_y, config=None):
    if getattr(model, 'heteroscedastic', False):
        loss_name = ((config or {}).get('training', {}) or {}).get('loss', '').lower()
        mu, logvar = model(batch_x, return_logvar=True)
        if loss_name == 'mse_nll':
            lam = float((config or {}).get('training', {}).get('lambda_nll', 1.0))
            loss, parts = mse_nll_loss(mu, logvar, batch_y,
                                       lambda_nll=lam, return_parts=True)
            return loss, logvar, {'mse': parts['mse'].detach(),
                                  'nll': parts['nll'].detach()}
        nll = gaussian_nll(mu, logvar, batch_y)
        return nll, logvar, {'nll': nll.detach()}
    return loss_fn(model(batch_x), batch_y), None, None


def train_epoch(model, dataloader, optimizer, loss_fn, device, config, epoch, logger):
    model.train()
    total_loss = 0.0
    log_interval = config['logging']['log_interval']

    for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        loss, _, _ = compute_train_loss(model, loss_fn, batch_x, batch_y, config)
        total_loss += loss.item()

        loss.backward()

        grad_clip = config['training']['grad_clip']
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            logger.train(f"Epoch [{epoch+1}/{config['training']['epochs']}], "
                         f"Batch [{batch_idx+1}/{len(dataloader)}], "
                         f"Loss: {avg_loss:.6f}")

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def save_checkpoint(model, optimizer, epoch, loss, save_path, logger, train_losses=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'train_losses': train_losses if train_losses is not None else []
    }
    torch.save(checkpoint, save_path)
    logger.info(f"模型检查点已保存: {save_path}")


def find_latest_checkpoint(checkpoints_dir):
    checkpoints_dir = Path(checkpoints_dir)
    files = list(checkpoints_dir.glob('model_epoch_*.pth'))
    if not files:
        final_path = checkpoints_dir / 'model_final.pth'
        if final_path.exists():
            return str(final_path)
        return None
    files.sort(key=lambda f: int(f.stem.split('_epoch_')[1]))
    return str(files[-1])


def main():
    parser = argparse.ArgumentParser(description='OAUNet 训练脚本')
    parser.add_argument('--config', type=str, default='configs/temporal.yaml',
                        help='配置文件路径（本副本默认 configs/temporal.yaml）')
    parser.add_argument('--data_prefix', type=str, default='train',
                        help='数据文件前缀: sample/train/test/compact')
    parser.add_argument('--split', type=str, default=None,
                        help='使用 split 文件区分 train/test（如 compact_split.json）')
    parser.add_argument('--compact', action='store_true',
                        help='使用紧凑数据集（从 data_compact/compact.npz 加载，遗留协议）')
    parser.add_argument('--temporal', action='store_true',
                        help='使用时间切分数据集（从 data_temporal/ 加载）')
    parser.add_argument('--temporal_name', type=str, default=None,
                        help='时间切分数据名（不含扩展名，如 temporal_strong）；'
                             '留空则自动解析 data_temporal/ 下的 npz')
    parser.add_argument('--device', type=str, default=None,
                        help='训练设备: cpu/cuda (默认自动检测)')
    parser.add_argument('--resume', action='store_true',
                        help='从最新 checkpoint 继续训练')
    parser.add_argument('--epochs', type=int, default=None,
                        help='覆盖 config 中的训练轮数（冒烟测试用，如 --epochs 1）')
    add_run_mode_args(parser)
    args = parser.parse_args()

    from utils.paths import resolve_config_path

    config_path = Path(__file__).resolve().parent / args.config
    if not config_path.exists():
        print(f"错误: 配置文件不存在: {config_path}")
        return
    config = load_config(str(config_path))

    mode_info = apply_run_mode(config, run_mode=args.run_mode,
                               limit_train=args.limit_train,
                               limit_test=args.limit_test)
    if args.epochs is not None:
        config.setdefault('training', {})['epochs'] = int(args.epochs)
        print(f"[覆盖] 训练轮数设为 {args.epochs}")

    for key in ('data_dir', 'checkpoints_dir', 'logs_dir'):
        if key in config.get('paths', {}):
            config['paths'][key] = resolve_config_path(config['paths'][key])

    seed = config.get('seed', config.get('training', {}).get('seed', 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logs_dir = Path(config['paths']['logs_dir'])
    logger = Logger(log_dir=str(logs_dir))

    logger.log_config(config)
    logger.info(mode_info['banner'])

    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"使用设备: {device}")

    data_dir = Path(config['paths']['data_dir'])

    if args.temporal:
        from utils.paths import find_temporal_split
        from data.dataset import TemporalRadarDataset

        prefer = args.temporal_name or config.get('data', {}).get('temporal_name')
        split_path = find_temporal_split(prefer)
        if split_path is None:
            logger.error(f"未在 data_temporal/ 找到时间切分索引"
                         f"（指定名: {prefer or '自动'}）")
            logger.error("请先运行 python -m data.make_temporal 生成索引")
            return

        logger.info(f"训练数据: 时间切分协议，索引 {split_path}")

        logger.info("\n创建数据集（源数据 mmap 只读，不复制）...")
        dataset = TemporalRadarDataset(str(split_path), split='train',
                                       limit=config.get('_limit_train'))
        dataloader = get_dataloader(
            dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=0
        )
        logger.info(f"数据集大小: {len(dataset)} 样本")
    elif args.compact:
        compact_dir = Path(__file__).resolve().parent / 'data_compact'
        npz_path = compact_dir / 'compact.npz'
        if not npz_path.exists():
            logger.error(f"紧凑数据文件不存在: {npz_path}")
            logger.error("请先运行 python -m data.make_compact 生成数据")
            return

        logger.info(f"训练数据: 紧凑数据集（遗留协议），文件 {npz_path}")

        logger.info("\n创建数据集...")
        dataset = CompactRadarDataset(str(npz_path), split='train')
        dataloader = get_dataloader(
            dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=0
        )
        logger.info(f"数据集大小: {len(dataset)} 样本")
    else:
        inputs_path = data_dir / f"{args.data_prefix}_inputs.npy"
        outputs_path = data_dir / f"{args.data_prefix}_outputs.npy"

        if not inputs_path.exists():
            logger.error(f"输入文件不存在: {inputs_path}")
            logger.error("请先运行 preprocess.py 生成 .npy 文件")
            return

        if not outputs_path.exists():
            logger.error(f"输出文件不存在: {outputs_path}")
            return

        indices = None
        if args.split:
            split_path = data_dir / args.split
            if not split_path.exists():
                logger.error(f"Split 文件不存在: {split_path}")
                return
            indices = load_split(str(split_path), split='train')
            logger.info(f"使用 split 文件: {split_path}, 训练样本数: {len(indices)}")

        logger.info(f"训练数据: {inputs_path} -> {outputs_path}")

        logger.info("\n创建数据集...")
        dataset = RadarDataset(str(inputs_path), str(outputs_path), indices=indices)
        dataloader = get_dataloader(
            dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=0
        )
        logger.info(f"数据集大小: {len(dataset)} 样本")

    logger.info("\n创建模型...")
    model = get_model(config)
    model.to(device)
    count_parameters(model)

    optimizer = get_optimizer(config, model)
    loss_fn = get_loss_fn(config)

    # gaussian_nll requires model.heteroscedastic=true, otherwise there is no log σ² to regress.
    loss_name = config['training']['loss'].lower()
    hetero = getattr(model, 'heteroscedastic', False)
    if loss_fn is None and not hetero:
        logger.error(f"config 指定 loss={loss_name}，但 model.heteroscedastic 未开启。"
                     "请在 model 段加 `heteroscedastic: true`（否则没有 log σ² 可言）。")
        return
    if loss_fn is not None and hetero:
        logger.warning(f"model.heteroscedastic=true 但 loss={loss_name}，"
                       "σ 头将得不到有效梯度（建议 loss: mse_nll）。")

    scheduler = None
    sched_name = str(config['training'].get('lr_scheduler', 'none')).lower()
    if sched_name == 'cosine':
        base_lr = config['training']['learning_rate']
        eta_min_ratio = float(config['training'].get('eta_min_ratio', 0.01))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, config['training']['epochs']),
            eta_min=base_lr * eta_min_ratio)
        logger.info(f"LR 调度: cosine，T_max={config['training']['epochs']}，"
                    f"eta_min={base_lr * eta_min_ratio:.2e}")

    save_interval = config['logging']['save_interval']
    checkpoints_dir = Path(config['paths']['checkpoints_dir'])

    start_epoch = 0
    train_losses = []
    if args.resume:
        latest_ckpt = find_latest_checkpoint(checkpoints_dir)
        if latest_ckpt:
            logger.info(f"从 checkpoint 恢复: {latest_ckpt}")
            checkpoint = torch.load(latest_ckpt, map_location=device)
            load_state_dict_compat(model, checkpoint['model_state_dict'],
                                   logger=logger, context='train.py --resume')
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            train_losses = checkpoint.get('train_losses', [])
            logger.info(f"  从 epoch {start_epoch} 继续训练")
            logger.info(f"  已恢复 {len(train_losses)} 个 epoch 的损失记录")
        else:
            logger.info("未找到 checkpoint，从头开始训练")

    logger.train(f"\n开始训练 (共 {config['training']['epochs']} 个 epoch, 从 epoch {start_epoch} 开始)...")

    avg_loss = 0.0
    best_loss = float('inf')
    best_epoch = -1
    patience = int(config['training'].get('early_stop_patience', 0))
    no_improve = 0

    for epoch in tqdm(range(start_epoch, config['training']['epochs']), desc="Training"):
        model.train()
        total_loss = 0.0
        mse_sum = 0.0
        nll_sum = 0.0
        sigma_u_sum = 0.0

        for batch_idx, (batch_x, batch_y) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            loss, logvar, aux = compute_train_loss(model, loss_fn, batch_x, batch_y, config)
            total_loss += loss.item()
            if aux is not None:
                if aux.get('mse') is not None:
                    mse_sum += aux['mse'].item()
                if aux.get('nll') is not None:
                    nll_sum += aux['nll'].item()
            if logvar is not None:
                sigma_u_sum += logvar.detach().clamp(-10.0, 2.0).exp().sqrt().mean().item()

            loss.backward()

            grad_clip = config['training']['grad_clip']
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

        avg_loss = total_loss / len(dataloader)
        train_losses.append(avg_loss)

        if getattr(model, 'heteroscedastic', False):
            avg_sigma_u = sigma_u_sum / len(dataloader)
            if loss_name == 'mse_nll':
                logger.train(f"Epoch [{epoch+1}/{config['training']['epochs']}] 完成, "
                             f"总Loss: {avg_loss:.6f} (MSE: {mse_sum / len(dataloader):.6f}, "
                             f"NLL: {nll_sum / len(dataloader):.6f}), "
                             f"平均σ: {avg_sigma_u:.4f}u ({avg_sigma_u * 95.0:.2f} dBZ)")
            else:
                logger.train(f"Epoch [{epoch+1}/{config['training']['epochs']}] 完成, "
                             f"NLL: {avg_loss:.6f}, 平均σ: {avg_sigma_u:.4f}u "
                             f"({avg_sigma_u * 95.0:.2f} dBZ)")
        else:
            logger.train(f"Epoch [{epoch+1}/{config['training']['epochs']}] 完成, Loss: {avg_loss:.6f}")

        # model_final is the lowest-loss epoch, not the last one, which may have diverged.
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            no_improve = 0
            save_checkpoint(model, optimizer, epoch, avg_loss,
                            str(checkpoints_dir / "model_final.pth"), logger, train_losses)
            logger.info(f"  [best] epoch {epoch+1} 损失创新低 {best_loss:.6f} → 已更新 model_final.pth")
        else:
            no_improve += 1

        if (epoch + 1) % save_interval == 0:
            save_path = checkpoints_dir / f"model_epoch_{epoch+1}.pth"
            save_checkpoint(model, optimizer, epoch, avg_loss, str(save_path), logger, train_losses)

        if scheduler is not None:
            scheduler.step()

        if patience > 0 and no_improve >= patience:
            logger.info(f"\n早停：连续 {patience} 轮训练损失无改善"
                        f"（最优 epoch {best_epoch+1}，Loss {best_loss:.6f}），停止训练。")
            break

    final_path = checkpoints_dir / "model_final.pth"
    if best_epoch < 0:
        save_checkpoint(model, optimizer, config['training']['epochs'], avg_loss,
                        str(final_path), logger, train_losses)

    logger.info(f"\n{'='*60}")
    logger.info("训练完成!")
    logger.info(f"运行模式: {config['run_mode']}")
    loss_desc = {'mse_nll': 'MSE(μ) + 分离式 NLL(σ)（基准忠实原文，σ 为我们的组件）',
                 'gaussian_nll': '整体高斯 NLL（旧行为）',
                 'nll': '整体高斯 NLL（旧行为）'}.get(loss_name, loss_name.upper())
    logger.info(f"损失口径: {loss_desc}")
    if best_epoch >= 0:
        logger.info(f"最优 epoch: {best_epoch + 1}（训练损失 {best_loss:.6f}）"
                    f" —— model_final.pth 存的是这一轮，不是末轮")
    logger.info(f"末轮 Loss: {avg_loss:.6f}")
    logger.info(f"权重: {final_path}")
    if config['run_mode'] == 'verify':
        logger.info("提示: 这是 100 样本的检验权重，仅供链路自检；"
                    "正式结果请用 formal 模式（不加 --run_mode）重训。")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()
