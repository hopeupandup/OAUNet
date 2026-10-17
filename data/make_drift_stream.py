"""Builds the drift stream: concatenates real blocks, writes an index JSON only."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# <repo>/data/make_drift_stream.py -> parent.parent is the repository root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_ROOT / 'data_raw'
SOURCE_DIR_REL = 'data_raw'
OUTPUT_DIR = PROJECT_ROOT / 'data_temporal'

DBZ_THRESHOLDS = [5.0, 15.0, 30.0]

# (source start, source end, segment name, measured baseline RMSE in dBZ). The order below is the
# order inside the stream; the last two blocks are swapped so the switch is a single clean jump.
BLOCKS = [
    (9500, 10000, 'S1_stable', 8.43),
    (5500, 6000, 'S1_stable', 9.27),
    (7000, 7500, 'S2_gradual', 9.26),
    (6500, 7000, 'S2_gradual', 9.69),
    (8000, 8500, 'S3_abrupt', 12.09),
    (4000, 4500, 'S3_abrupt', 12.11),
]

ABRUPT_AT = 2000


def main():
    ap = argparse.ArgumentParser(description='漂移协议测试流构建（只产索引）')
    ap.add_argument('--out_name', type=str, default='temporal_drift3000')
    ap.add_argument('--num_train', type=int, default=2000)
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()

    src_in = SOURCE_DIR / 'train_inputs.npy'
    src_out = SOURCE_DIR / 'train_outputs.npy'
    if not src_in.exists():
        print(f'错误: 源数据不存在: {src_in}')
        sys.exit(1)

    inputs = np.load(str(src_in), mmap_mode='r')
    outputs = np.load(str(src_out), mmap_mode='r')
    total_src = inputs.shape[0]

    train_idx = np.arange(0, args.num_train, dtype=np.int64)
    parts = [np.arange(a, b, dtype=np.int64) for a, b, _n, _e in BLOCKS]
    test_idx = np.concatenate(parts)

    # Leak check: the test range must not overlap the train range.
    overlap = np.intersect1d(train_idx, test_idx)
    if len(overlap) > 0:
        print(f'错误: 测试段与训练段重叠 {len(overlap)} 个索引，存在数据泄漏')
        sys.exit(1)

    print('=' * 70)
    print('漂移协议测试流构建（构造 regime switch + 渐变；只产索引）')
    print('=' * 70)
    print(f'  源序列: {total_src} 条  inputs {inputs.shape}')
    print(f'  训练段: [0,{args.num_train}) （{len(train_idx)} 条，与基线权重一致）')
    print(f'  测试段: {len(BLOCKS)} 块拼接，共 {len(test_idx)} 条')
    print()
    print('  %-10s%-8s%-16s%9s%11s%11s' % ('段', '流内', '源区间', 'in_mean', 'out_std', '基线RMSE'))
    print('  ' + '-' * 66)

    acc, block_meta = 0, []
    for a, b, seg_name, rmse_est in BLOCKS:
        n = b - a
        x = np.array(inputs[a:b], dtype=np.float32)
        np.clip(x, 0.0, 1.0, out=x)
        y = np.array(outputs[a:b], dtype=np.float32)
        np.clip(y, 0.0, 1.0, out=y)
        in_mean = float(x.mean())
        out_std = float(y.std())
        mark = '  <== 突变点' if acc == ABRUPT_AT else ''
        print('  %-10s%-8s%-16s%9.4f%11.4f%11.2f%s' % (
            seg_name, f'[{acc},{acc + n})', f'[{a},{b})', in_mean, out_std, rmse_est, mark))
        block_meta.append({
            'segment': seg_name, 'stream_range': [acc, acc + n],
            'source_range': [a, b], 'n': n,
            'in_mean': round(in_mean, 4), 'out_std': round(out_std, 4),
            'baseline_rmse_dbz_est': rmse_est,
        })
        acc += n
    print('  ' + '-' * 66)
    print(f'  合计 {acc} 条；突变点位于流内 step {ABRUPT_AT}')

    drift_check = {}
    tr_in = float(np.asarray(inputs[train_idx], dtype=np.float32).mean())
    te_in = float(np.asarray(inputs[test_idx], dtype=np.float32).mean())
    drift_check['输入均值'] = {'train': tr_in, 'test': te_in}
    print(f'\n  漂移自检: 输入均值 训练={tr_in:.4f}  测试={te_in:.4f}')

    if args.dry_run:
        print('\n[dry_run] 未写文件。')
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    split_info = {
        'protocol': 'temporal_spliced',
        'drift': 'gradual+abrupt',
        'drift_desc': '构造的 regime switch：按输入强度单调递减拼接 6 个真实数据块（含 1 个突变点）',
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
        'test_range': [int(test_idx.min()), int(test_idx.max())],
        'gap_range': [int(args.num_train), int(total_src) - 1],
        'train_n': int(len(train_idx)),
        'test_n': int(len(test_idx)),
        'label_thresholds_dbz': DBZ_THRESHOLDS,
        'drift_check': drift_check,
        'constructed': True,
        'construction_blocks': block_meta,
        'abrupt_step': ABRUPT_AT,
        'gradual_range': [0, ABRUPT_AT],
        'construction_note': (
            '⚠️ 构造协议：测试流由 6 个真实数据块拼接而成，各块按「输入强度均值（in_mean）」'
            '单调递减排列，用于制造声明式的协变量漂移（covariate shift）。'
            '块 5（源 [4000,4500)）接入处为突变点（流内 step 2000）。'
            '所有块均为 CIKM2017 真实观测，被设计的只是块的选取与拼接顺序；'
            '不得声称自然界发生了突变。'
        ),
        'note': ('训练段 [0,2000) 与 temporal_strong 完全一致，可直接复用同一套基线权重；'
                 'Dataset 按 test_indices 逐条 mmap 读取，不复制数据。'),
    }
    split_path = OUTPUT_DIR / f'{args.out_name}_split.json'
    with open(split_path, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)
    print(f'\n  已写: {split_path}  ({split_path.stat().st_size} bytes)')
    print(f'  训练 {len(train_idx)} 条 + 测试 {len(test_idx)} 条')
    print(f'  下一步: python train.py --config configs/temporal.yaml --temporal')
    print('=' * 70)


if __name__ == '__main__':
    main()
