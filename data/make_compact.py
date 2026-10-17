"""Compact dataset construction (legacy protocol, not used in this paper)."""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_ROOT / 'data_raw'
OUTPUT_DIR = PROJECT_ROOT / 'data_compact'

# Stratification: (lower bound, upper bound, train count, test count); no-rain is dropped.
CUSTOM_STRATA = [
    {'name': '微量',     'low': 0,   'high': 0.5,  'train': 80,  'test': 20},
    {'name': '小雨',     'low': 0.5, 'high': 2,    'train': 160, 'test': 40},
    {'name': '中雨',     'low': 2,   'high': 5,    'train': 320, 'test': 80},
    {'name': '大雨',     'low': 5,   'high': 10,   'train': 320, 'test': 80},
    {'name': '暴雨',     'low': 10,  'high': 30,   'train': 480, 'test': 120},
    {'name': '大暴雨',   'low': 30,  'high': 50,   'train': 400, 'test': 100},
    {'name': '特大暴雨', 'low': 50,  'high': 200,  'train': 240, 'test': 60},
]


def get_stratum_indices(labels, low, high):
    return np.where((labels > low) & (labels <= high))[0]


def custom_stratified_split(labels, seed):
    rng = np.random.RandomState(seed)

    train_indices_all = []
    test_indices_all = []
    stats = []

    for s in CUSTOM_STRATA:
        indices = get_stratum_indices(labels, s['low'], s['high'])
        available = len(indices)
        need_train = s['train']
        need_test = s['test']
        need_total = need_train + need_test

        if available < need_total:
            print(f"  警告: {s['name']} 可用 {available} < 需要 {need_total}，全部使用")
            need_train = min(need_train, available)
            need_test = min(need_test, available - need_train)

        perm = rng.permutation(available)
        selected = indices[perm]
        train_idx = selected[:need_train]
        test_idx = selected[need_train:need_train + need_test]

        train_indices_all.append(train_idx)
        test_indices_all.append(test_idx)

        stats.append({
            'name': s['name'],
            'range_mm': f"{s['low']}-{s['high']}",
            'available': available,
            'train_n': len(train_idx),
            'test_n': len(test_idx),
        })

    train_indices = np.concatenate(train_indices_all)
    test_indices = np.concatenate(test_indices_all)
    train_indices.sort()
    test_indices.sort()

    return train_indices, test_indices, stats


def main():
    parser = argparse.ArgumentParser(description='数据裁剪脚本（自定义分层抽样）')
    parser.add_argument('--seed', type=int, default=123, help='随机种子')
    args = parser.parse_args()

    src_files = {
        'train_inputs': SOURCE_DIR / 'train_inputs.npy',
        'train_outputs': SOURCE_DIR / 'train_outputs.npy',
        'train_labels': SOURCE_DIR / 'train_labels.npy',
    }
    for name, path in src_files.items():
        if not path.exists():
            print(f"错误: {name} 不存在: {path}")
            sys.exit(1)

    print("加载源数据...")
    train_inputs = np.load(src_files['train_inputs'], mmap_mode='r')
    train_outputs = np.load(src_files['train_outputs'], mmap_mode='r')
    train_labels = np.load(src_files['train_labels'])
    print(f"  训练集: {train_inputs.shape[0]} 样本")
    print(f"  标签范围: [{train_labels.min():.2f}, {train_labels.max():.2f}]")

    print("\n自定义分层抽样...")
    train_indices, test_indices, stats = custom_stratified_split(train_labels, args.seed)

    print(f"\n{'降水量':<10} {'可用':>6} {'训练':>6} {'测试':>6} {'合计':>6}")
    print("-" * 40)
    for s in stats:
        total = s['train_n'] + s['test_n']
        print(f"{s['name']:<10} {s['available']:>6} {s['train_n']:>6} {s['test_n']:>6} {total:>6}")
    print("-" * 40)
    total_train = sum(s['train_n'] for s in stats)
    total_test = sum(s['test_n'] for s in stats)
    print(f"{'合计':<10} {'':>6} {total_train:>6} {total_test:>6} {total_train+total_test:>6}")

    print("\n合并数据...")
    compact_inputs = np.concatenate([
        train_inputs[train_indices],
        train_inputs[test_indices]
    ], axis=0)
    compact_outputs = np.concatenate([
        train_outputs[train_indices],
        train_outputs[test_indices]
    ], axis=0)
    compact_labels = np.concatenate([
        train_labels[train_indices],
        train_labels[test_indices]
    ], axis=0)

    total = len(train_indices) + len(test_indices)
    print(f"  合并后: {total} 样本")
    print(f"  inputs: {compact_inputs.shape}, dtype: {compact_inputs.dtype}")
    print(f"  outputs: {compact_outputs.shape}, dtype: {compact_outputs.dtype}")

    new_train_indices = list(range(len(train_indices)))
    new_test_indices = list(range(len(train_indices), total))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    npz_path = OUTPUT_DIR / 'compact.npz'
    print(f"\n保存到 {npz_path}...")
    np.savez(npz_path,
             inputs=compact_inputs.astype(np.float32),
             outputs=compact_outputs.astype(np.float32),
             labels=compact_labels.astype(np.float32))

    split_info = {
        'train_indices': new_train_indices,
        'test_indices': new_test_indices,
        'train_n': len(train_indices),
        'test_n': len(test_indices),
        'seed': args.seed,
        'stratification': [
            {
                'name': s['name'],
                'range_mm': s['range_mm'],
                'available': s['available'],
                'train_n': s['train_n'],
                'test_n': s['test_n']
            }
            for s in stats
        ]
    }
    split_path = OUTPUT_DIR / 'compact_split.json'
    with open(split_path, 'w') as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)

    print("\n输出文件:")
    for path in [npz_path, split_path]:
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  {path.name}: {size_mb:.1f} MB")

    print(f"\n完成! 共 {total} 样本 (训练 {len(train_indices)} + 测试 {len(test_indices)})")
    print(f"数据路径: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
