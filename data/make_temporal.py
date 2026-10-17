"""Builds the temporal split indices (indices only, no data copied)."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_ROOT / 'data_raw'
SOURCE_DIR_REL = 'data_raw'
OUTPUT_DIR = PROJECT_ROOT / 'data_temporal'

# dBZ thresholds, following the CIKM2017 protocol.
DBZ_THRESHOLDS = [5.0, 15.0, 30.0]

DRIFT_PRESETS = {
    'weak':   (2000, 4000),
    'mid':    (5000, 7000),
    'strong': (8000, 10000),
}
DRIFT_DESC = {
    'weak':   '测试紧邻训练段，漂移最小',
    'mid':    '中等时间间隔',
    'strong': '约 1.5 年间隔，漂移最强（推荐）',
}

# The train range stops at 1999, leaving a gap before the weak test split.
GAP_START = 2000


def normalize_to_dbz(u):
    return np.asarray(u, dtype=np.float32) * 95.0 - 10.0


def _iter_blocks(mmap_arr, indices, chunk):
    for s in range(0, len(indices), chunk):
        e = min(s + chunk, len(indices))
        yield np.asarray(mmap_arr[indices[s:e]], dtype=np.float32)


def _mean_std(mmap_arr, indices, chunk=100):
    n = 0
    s1 = 0.0
    s2 = 0.0
    for blk in _iter_blocks(mmap_arr, indices, chunk):
        flat = blk.reshape(-1).astype(np.float64)
        n += flat.size
        s1 += float(flat.sum())
        s2 += float(np.dot(flat, flat))
    mean = s1 / max(n, 1)
    var = max(s2 / max(n, 1) - mean * mean, 0.0)
    return mean, math.sqrt(var)


def _positive_ratio(mmap_arr, indices, thresh_dbz, chunk=100):
    u_thr = (thresh_dbz + 10.0) / 95.0
    hit = 0
    total = 0
    for blk in _iter_blocks(mmap_arr, indices, chunk):
        hit += int(np.count_nonzero(blk >= u_thr))
        total += blk.size
    return hit / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description='时间切分协议构建（只产索引）')
    parser.add_argument('--drift', type=str, default='strong',
                        choices=sorted(DRIFT_PRESETS.keys()),
                        help='漂移强度：weak/mid/strong（决定测试段位置）')
    parser.add_argument('--num_train', type=int, default=2000, help='训练样本数')
    parser.add_argument('--num_test', type=int, default=2000, help='测试样本数')
    parser.add_argument('--out_name', type=str, default=None,
                        help='输出文件名（默认按 drift 命名，如 temporal_strong）')
    parser.add_argument('--no_check', action='store_true',
                        help='跳过漂移自检（仅写 JSON，最快）')
    args = parser.parse_args()

    src_inputs = SOURCE_DIR / 'train_inputs.npy'
    src_outputs = SOURCE_DIR / 'train_outputs.npy'
    src_labels = SOURCE_DIR / 'train_labels.npy'
    for p in (src_inputs, src_outputs, src_labels):
        if not p.exists():
            print(f'错误: 源数据不存在: {p}')
            sys.exit(1)

    print('=' * 68)
    print('时间切分协议构建（只产索引，不复制数据）')
    print('=' * 68)


    print('\n[1/4] 打开源数据（只读 mmap，不载入内存）...')
    inputs = np.load(src_inputs, mmap_mode='r')
    outputs = np.load(src_outputs, mmap_mode='r')
    labels = np.load(src_labels)
    total_src = inputs.shape[0]
    print(f'  源序列: {total_src} 条, inputs {inputs.shape}, outputs {outputs.shape}')
    print(f'  源数据目录（只读，不会被修改）: {SOURCE_DIR}')


    print('\n[2/4] 构造时间切分索引...')
    if args.num_train > GAP_START:
        print(f'  警告: num_train={args.num_train} 超过 gap 起点 {GAP_START}，自动收敛')
    num_train = min(args.num_train, GAP_START)

    t_start, t_end = DRIFT_PRESETS[args.drift]
    if args.num_test > (t_end - t_start):
        print(f'  警告: num_test={args.num_test} 超过该档可用量 {t_end - t_start}，自动收敛')
    num_test = min(args.num_test, t_end - t_start)

    train_idx = np.arange(0, num_train, dtype=np.int64)
    test_idx = np.arange(t_end - num_test, t_end, dtype=np.int64)

    print(f'  漂移档位: {args.drift} —— {DRIFT_DESC[args.drift]}')
    print(f'  训练段源索引: [{train_idx[0]}, {train_idx[-1]}]  ({len(train_idx)} 条, 最早时段)')
    print(f'  gap  源索引: [{GAP_START}, {test_idx[0] - 1}]  ({test_idx[0] - GAP_START} 条, 不参与)')
    print(f'  测试段源索引: [{test_idx[0]}, {test_idx[-1]}]  ({len(test_idx)} 条, 最晚时段)')


    drift_check = {}
    if not args.no_check:
        print('\n[3/4] 漂移自检（分块 mmap 统计，内存友好）...')
        print('  ' + '-' * 62)
        print(f'  {"指标":<30}{"训练段":>14}{"测试段":>14}')
        print('  ' + '-' * 62)

        def _row(name, a, b):
            print(f'  {name:<30}{a:>14.4f}{b:>14.4f}')
            drift_check[name] = {'train': a, 'test': b}

        tr_in_mean, tr_in_std = _mean_std(inputs, train_idx)
        te_in_mean, te_in_std = _mean_std(inputs, test_idx)
        _row('输入均值', tr_in_mean, te_in_mean)
        _row('输入标准差', tr_in_std, te_in_std)

        for d in DBZ_THRESHOLDS:
            _row(f'目标 强回波({d:g} dBZ) 占比%',
                 100 * _positive_ratio(outputs, train_idx, d),
                 100 * _positive_ratio(outputs, test_idx, d))

        tr_lb = labels[train_idx].astype(np.float64)
        te_lb = labels[test_idx].astype(np.float64)
        _row('降水标签 均值(mm)', float(tr_lb.mean()), float(te_lb.mean()))
        _row('降水标签 中位数(mm)', float(np.median(tr_lb)), float(np.median(te_lb)))
        _row('降水标签 暴雨(>10mm) 占比%',
             100 * float((tr_lb > 10).mean()), 100 * float((te_lb > 10).mean()))
        print('  ' + '-' * 62)

        print('\n  评测阈值换算自检（dBZ = u*95 - 10）:')
        for d in DBZ_THRESHOLDS:
            u = (d + 10.0) / 95.0
            print(f'    {d:5.1f} dBZ  ->  u = {u:.4f}  (pixel = {u * 255:.1f})')


    print('\n[4/4] 写索引 JSON...')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_name = args.out_name or f'temporal_{args.drift}'
    split_info = {
        'protocol': 'temporal',
        'drift': args.drift,
        'drift_desc': DRIFT_DESC[args.drift],
        'source_dir': SOURCE_DIR_REL,
        'source_files': {
            'inputs': 'train_inputs.npy',
            'outputs': 'train_outputs.npy',
            'labels': 'train_labels.npy',
        },
        'source_total': int(total_src),
        'train_indices': train_idx.tolist(),
        'test_indices': test_idx.tolist(),
        'train_range': [int(train_idx[0]), int(train_idx[-1])],
        'test_range': [int(test_idx[0]), int(test_idx[-1])],
        'gap_range': [int(GAP_START), int(test_idx[0] - 1)],
        'train_n': int(len(train_idx)),
        'test_n': int(len(test_idx)),
        'label_thresholds_dbz': DBZ_THRESHOLDS,
        'drift_check': drift_check,
        'note': ('训练段与测试段均取自 CIKM2017 原始 train 文件（按探测时间排序）；'
                 '二者在文件中均为连续区间，Dataset 直接按索引 mmap 读取，不复制数据。'),
    }
    split_path = OUTPUT_DIR / f'{out_name}_split.json'
    with open(split_path, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)
    print(f'  {split_path}  ({split_path.stat().st_size} bytes)')

    print('\n完成（未复制任何数据，磁盘零增长）。')
    print(f'  训练 {len(train_idx)} 条  +  测试 {len(test_idx)} 条  =  {len(train_idx) + len(test_idx)} 条')
    print(f'  下一步: python train.py --config configs/temporal.yaml --temporal')
    print(f'           python run_experiments.py --config configs/oaunet_temporal.yaml --temporal')


if __name__ == '__main__':
    main()
