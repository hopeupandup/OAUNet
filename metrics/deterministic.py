"""Deterministic scores in dBZ."""

import numpy as np
import json
import os
from typing import List, Dict, Optional


# dBZ = u * 95 - 10, with u = pixel / 255.
_PIXEL_TO_DBZ_SCALE = 95.0 / 255.0
_PIXEL_TO_DBZ_OFFSET = -10.0

DEFAULT_DBZ_THRESHOLDS = [5.0, 15.0, 30.0]


def normalized_to_dbz(u: np.ndarray) -> np.ndarray:
    return np.asarray(u, dtype=np.float32) * 95.0 - 10.0


def pixel_to_dbz(p: np.ndarray) -> np.ndarray:
    return np.asarray(p, dtype=np.float32) * _PIXEL_TO_DBZ_SCALE + _PIXEL_TO_DBZ_OFFSET


def dbz_to_normalized(d: np.ndarray) -> np.ndarray:
    return (np.asarray(d, dtype=np.float32) - _PIXEL_TO_DBZ_OFFSET) / 95.0


def dbz_to_threshold_u(thresh_dbz: float) -> float:
    return (float(thresh_dbz) + 10.0) / 95.0


def csi(pred: np.ndarray, target: np.ndarray, threshold: float = 5.0) -> float:
    pred_binary = pred >= threshold
    target_binary = target >= threshold

    tp = np.sum(pred_binary & target_binary)
    fp = np.sum(pred_binary & ~target_binary)
    fn = np.sum(~pred_binary & target_binary)

    denominator = tp + fp + fn
    if denominator == 0:
        return 0.0
    return float(tp / denominator)


def hss(pred: np.ndarray, target: np.ndarray, threshold: float = 5.0) -> float:
    pred_binary = pred >= threshold
    target_binary = target >= threshold

    tp = np.sum(pred_binary & target_binary)
    fp = np.sum(pred_binary & ~target_binary)
    fn = np.sum(~pred_binary & target_binary)
    tn = np.sum(~pred_binary & ~target_binary)

    numerator = 2 * (tp * tn - fp * fn)
    denominator = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)

    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    total = 0.0
    count = 0
    n = pred.shape[0]
    for i in range(n):
        diff = pred[i] - target[i]
        total += np.sum(diff * diff)
        count += diff.size
    return float(total / max(count, 1))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    total = 0.0
    count = 0
    n = pred.shape[0]
    for i in range(n):
        total += np.sum(np.abs(pred[i] - target[i]))
        count += pred[i].size
    return float(total / max(count, 1))


def evaluate_deterministic(pred: np.ndarray, target: np.ndarray,
                           thresholds: Optional[List[float]] = None,
                           scale: str = 'dbz',
                           max_value: float = 255.0) -> Dict[str, object]:
    if thresholds is None:
        thresholds = list(DEFAULT_DBZ_THRESHOLDS)

    if scale == 'dbz':
        p, t = pred, target
    elif scale == 'normalized':
        p, t = normalized_to_dbz(pred), normalized_to_dbz(target)
    elif scale == 'raw':
        p, t = pixel_to_dbz(pred), pixel_to_dbz(target)
    else:
        raise ValueError(f"未知 scale: {scale}（可选 dbz / normalized / raw）")

    results = {
        'thresholds': list(thresholds),
        'threshold_unit': 'dBZ',
        'scale': scale,
        'csi': [],
        'hss': [],
        'mse': mse(p, t),
        'mae': mae(p, t),
    }

    for thresh in thresholds:
        results['csi'].append(csi(p, t, thresh))
        results['hss'].append(hss(p, t, thresh))

    return results


def evaluate_deterministic_chunked(pred, target, thresholds=None, chunk=64) -> Dict[str, object]:
    if thresholds is None:
        thresholds = list(DEFAULT_DBZ_THRESHOLDS)
    thresholds = [float(t) for t in thresholds]

    n = int(pred.shape[0])
    m = len(thresholds)
    tp = np.zeros(m, dtype=np.int64)
    fp = np.zeros(m, dtype=np.int64)
    fn = np.zeros(m, dtype=np.int64)
    tn = np.zeros(m, dtype=np.int64)
    sq_sum = 0.0
    abs_sum = 0.0
    count = 0

    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        p = np.asarray(pred[s:e], dtype=np.float32)
        t = np.asarray(target[s:e], dtype=np.float32)
        diff = p - t
        sq_sum += float(np.sum(diff * diff))
        abs_sum += float(np.sum(np.abs(diff)))
        count += diff.size
        for j, th in enumerate(thresholds):
            pb = p >= th
            tb = t >= th
            tp[j] += int(np.count_nonzero(pb & tb))
            fp[j] += int(np.count_nonzero(pb & ~tb))
            fn[j] += int(np.count_nonzero(~pb & tb))
            tn[j] += int(np.count_nonzero(~pb & ~tb))

    csi_list, hss_list = [], []
    for j in range(m):
        denom = int(tp[j] + fp[j] + fn[j])
        csi_list.append(float(tp[j] / denom) if denom else 0.0)
        num = 2.0 * (float(tp[j]) * float(tn[j]) - float(fp[j]) * float(fn[j]))
        den = float((tp[j] + fn[j]) * (fn[j] + tn[j])
                    + (tp[j] + fp[j]) * (fp[j] + tn[j]))
        hss_list.append(float(num / den) if den else 0.0)

    return {
        'thresholds': thresholds,
        'threshold_unit': 'dBZ',
        'scale': 'dbz',
        'csi': csi_list,
        'hss': hss_list,
        'mse': float(sq_sum / max(count, 1)),
        'mae': float(abs_sum / max(count, 1)),
    }


def positive_ratio_by_threshold(arr, thresholds=None, scale='normalized',
                                chunk=32) -> Dict[float, float]:
    if thresholds is None:
        thresholds = list(DEFAULT_DBZ_THRESHOLDS)

    out = {}
    n = arr.shape[0]
    for thresh in thresholds:
        if scale == 'normalized':
            thr = dbz_to_threshold_u(thresh)
        elif scale == 'dbz':
            thr = thresh
        elif scale == 'raw':
            thr = thresh / _PIXEL_TO_DBZ_SCALE - _PIXEL_TO_DBZ_OFFSET
        else:
            raise ValueError(f"未知 scale: {scale}")

        hit = 0
        total = 0
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            block = np.asarray(arr[s:e], dtype=np.float32)
            hit += int(np.count_nonzero(block >= thr))
            total += block.size
        out[float(thresh)] = hit / max(total, 1)
    return out


def print_deterministic_results(results: Dict[str, object]):
    print("回归指标:")
    print(f"  MSE: {results.get('mse', 0):.6f}")
    print(f"  MAE: {results.get('mae', 0):.6f}")

    print("\n分类指标（各阈值）:")
    unit = results.get('threshold_unit', 'dBZ')
    print(f"{'阈值 (' + unit + ')':<16} {'CSI':<10} {'HSS':<10}")
    print('-' * 36)

    for thresh, csi_val, hss_val in zip(results['thresholds'], results['csi'], results['hss']):
        print(f"{thresh:<16} {csi_val:<10.4f} {hss_val:<10.4f}")


def save_results(results: Dict[str, object], save_path: str) -> None:
    dir_path = os.path.dirname(save_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    _, ext = os.path.splitext(save_path)

    if ext.lower() == '.json':
        save_data = {
            'regression': {
                'mse': float(results.get('mse', 0)),
                'mae': float(results.get('mae', 0)),
                'unit': results.get('threshold_unit', 'dBZ'),
            },
            'classification': {
                'thresholds': results['thresholds'],
                'threshold_unit': results.get('threshold_unit', 'dBZ'),
                'csi': [float(v) for v in results['csi']],
                'hss': [float(v) for v in results['hss']],
            }
        }
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=4)
        print(f"评估结果已保存到 JSON 文件: {save_path}")

    elif ext.lower() == '.csv':
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(f"MSE,{results.get('mse', 0):.6f}\n")
            f.write(f"MAE,{results.get('mae', 0):.6f}\n")
            f.write('\n')
            f.write('threshold_dBZ,CSI,HSS\n')
            for thresh, csi_val, hss_val in zip(results['thresholds'], results['csi'], results['hss']):
                f.write(f'{thresh},{csi_val:.6f},{hss_val:.6f}\n')
        print(f"评估结果已保存到 CSV 文件: {save_path}")

    else:
        save_results(results, save_path + '.json')


def test_deterministic_metrics():
    print(f"\n{'='*60}")
    print("测试确定性评估指标（dBZ 口径）")
    print(f"{'='*60}")

    u = np.array([0.0, 0.25, 0.5, 0.7725], dtype=np.float32)
    assert np.allclose(dbz_to_normalized(normalized_to_dbz(u)), u, atol=1e-5)
    assert np.allclose(pixel_to_dbz(u * 255.0), normalized_to_dbz(u), atol=1e-4)
    print("[OK] dBZ / 归一化 / 像素值 三者换算自洽")

    expect = {5.0: 0.1579, 15.0: 0.2632, 30.0: 0.4211}
    for d, eu in expect.items():
        assert abs(dbz_to_threshold_u(d) - eu) < 5e-4, d
    print(f"[OK] dBZ 阈值换算: { {d: round(dbz_to_threshold_u(d), 4) for d in expect} }")

    np.random.seed(42)
    field = np.full((2, 40, 40), -10.0, dtype=np.float32)
    field[:, 10:30, 10:30] = 25.0
    field[:, 15:20, 15:20] = 35.0
    for t in DEFAULT_DBZ_THRESHOLDS:
        assert abs(csi(field, field, t) - 1.0) < 1e-6, t
    print("[OK] 完全匹配时 CSI = 1.0（所有阈值）")

    csi_good = csi(field, field + np.random.randn(*field.shape).astype(np.float32) * 2.0, 15.0)
    csi_bad = csi(field, field + np.random.randn(*field.shape).astype(np.float32) * 15.0, 15.0)
    assert csi_good > csi_bad, (csi_good, csi_bad)
    print(f"[OK] 误差增大 CSI 下降: {csi_good:.4f} -> {csi_bad:.4f}")

    layered = np.full((2, 10, 10), -10.0, dtype=np.float32)
    layered[:, :5, :] = 35.0     # 30+ dBZ
    layered[:, 5:8, :] = 20.0    # 15-30 dBZ
    layered[:, 8:9, :] = 8.0     # 5-15 dBZ
    ratio = positive_ratio_by_threshold(layered, scale='dbz')
    print(f"     正例占比自检: { {k: round(v, 4) for k, v in ratio.items()} }")
    assert ratio[5.0] > ratio[15.0] > ratio[30.0], ratio
    print("[OK] 正例占比随阈值递减（口径生效，未退化）")

    print(f"\n{'='*60}")
    print("全部通过")
    print(f"{'='*60}")


if __name__ == '__main__':
    test_deterministic_metrics()
