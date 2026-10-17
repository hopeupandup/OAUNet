"""Dataset and dataloader over the temporal split."""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class RadarDataset(Dataset):
    def __init__(self, inputs_path, outputs_path, transform=None, indices=None,
                 limit=None):
        inputs = np.load(inputs_path)
        outputs = np.load(outputs_path)

        if indices is not None:
            inputs = inputs[indices]
            outputs = outputs[indices]

        if limit is not None:
            limit = int(limit)
            if limit <= 0:
                raise ValueError(f"limit 必须为正整数，收到 {limit}")
            inputs = inputs[:limit]
            outputs = outputs[:limit]

        self.inputs = inputs
        self.outputs = outputs

        assert len(self.inputs) == len(self.outputs), \
            f"输入和输出样本数量不匹配: {len(self.inputs)} vs {len(self.outputs)}"

        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx]
        y = self.outputs[idx]

        x = np.expand_dims(x, axis=1)
        y = np.expand_dims(y, axis=1)

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        if self.transform:
            x = self.transform(x)
            y = self.transform(y)

        return x, y


class CompactRadarDataset(Dataset):
    def __init__(self, npz_path, split='train', split_path=None, transform=None,
                 split_name=None, limit=None):
        from pathlib import Path

        data = np.load(npz_path)
        all_inputs = data['inputs']
        all_outputs = data['outputs']

        if split_path is None:
            if split_name is None:
                split_name = 'compact_split.json'
            split_path = str(Path(npz_path).parent / split_name)

        with open(split_path, 'r') as f:
            split_info = json.load(f)

        if split == 'train':
            indices = split_info['train_indices']
        elif split == 'test':
            indices = split_info['test_indices']
        else:
            raise ValueError(f"未知 split: {split}, 可选: train/test")

        if limit is not None:
            limit = int(limit)
            if limit <= 0:
                raise ValueError(f"limit 必须为正整数，收到 {limit}")
            indices = indices[:limit]

        self.inputs = all_inputs[indices]
        self.outputs = all_outputs[indices]
        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx]
        y = self.outputs[idx]

        x = np.expand_dims(x, axis=1)
        y = np.expand_dims(y, axis=1)

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        if self.transform:
            x = self.transform(x)
            y = self.transform(y)

        return x, y


class TemporalRadarDataset(Dataset):
    def __init__(self, split_path, split='train', source_dir=None,
                 clip=True, transform=None, verbose=True, limit=None):
        from pathlib import Path
        from utils.paths import PROJECT_ROOT

        with open(str(split_path), 'r', encoding='utf-8') as f:
            info = json.load(f)

        if source_dir is None:
            source_dir = info.get('source_dir', 'data_raw')
        src = Path(source_dir)
        if not src.is_absolute():
            src = PROJECT_ROOT / src
        self.source_dir = str(src)

        files = info.get('source_files', {})
        self._inputs = np.load(str(src / files.get('inputs', 'train_inputs.npy')),
                               mmap_mode='r')
        self._outputs = np.load(str(src / files.get('outputs', 'train_outputs.npy')),
                                mmap_mode='r')

        self.indices = list(info.get(f'{split}_indices', []))
        if not self.indices:
            raise ValueError(f"split json 中不存在 {split}_indices: {split_path}")

        self.limit = None
        if limit is not None:
            limit = int(limit)
            if limit <= 0:
                raise ValueError(f"limit 必须为正整数，收到 {limit}")
            self.limit = limit
            self.indices = self.indices[:limit]

        self.clip = clip
        self.transform = transform
        self.info = info

        if verbose:
            print(f"TemporalRadarDataset: split={split}, "
                  f"drift={info.get('drift')}, {len(self.indices)} samples")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        j = self.indices[idx]
        # memmap sample -> writable copy; negative reflectivity is clipped to 0.
        x = np.array(self._inputs[j], dtype=np.float32)
        y = np.array(self._outputs[j], dtype=np.float32)
        if self.clip:
            np.clip(x, 0.0, 1.0, out=x)
            np.clip(y, 0.0, 1.0, out=y)

        x = np.expand_dims(x, axis=1)   # (5, 1, H, W)
        y = np.expand_dims(y, axis=1)   # (10, 1, H, W)
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        if self.transform:
            x = self.transform(x)
            y = self.transform(y)

        return x, y


def load_temporal_train_data(split_path, source_dir=None, clip=True, verbose=True,
                             limit=None):
    from pathlib import Path
    from utils.paths import PROJECT_ROOT

    with open(str(split_path), 'r', encoding='utf-8') as f:
        info = json.load(f)

    if source_dir is None:
        source_dir = info.get('source_dir', 'data_raw')
    src = Path(source_dir)
    if not src.is_absolute():
        src = PROJECT_ROOT / src

    files = info.get('source_files', {})
    idx = np.asarray(info['train_indices'], dtype=np.int64)
    if limit is not None:
        limit = int(limit)
        if limit <= 0:
            raise ValueError(f"limit 必须为正整数，收到 {limit}")
        idx = idx[:limit]

    inputs = np.load(str(src / files.get('inputs', 'train_inputs.npy')), mmap_mode='r')
    outputs = np.load(str(src / files.get('outputs', 'train_outputs.npy')), mmap_mode='r')
    X = np.asarray(inputs[idx], dtype=np.float32)
    Y = np.asarray(outputs[idx], dtype=np.float32)
    if clip:
        np.clip(X, 0.0, 1.0, out=X)
        np.clip(Y, 0.0, 1.0, out=Y)

    labels_path = src / files.get('labels', 'train_labels.npy')
    if Path(labels_path).exists():
        labels = np.load(str(labels_path))[idx].astype(np.float32)
    else:
        labels = np.zeros(len(idx), dtype=np.float32)

    if verbose:
        print(f"Temporal split loaded into memory: {X.shape[0]} samples")

    return {'inputs': X, 'outputs': Y, 'labels': labels}


def load_split(split_path, split='train'):
    with open(split_path, 'r') as f:
        split_info = json.load(f)

    if split == 'train':
        return split_info['train_indices']
    elif split == 'test':
        return split_info['test_indices']
    else:
        raise ValueError(f"未知 split: {split}, 可选: train/test")


def get_dataloader(dataset, batch_size=2, shuffle=True, num_workers=0, pin_memory=False):
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    return dataloader


def test_dataset():
    import os
    from pathlib import Path
    from utils.paths import DATA_DIR

    data_dir = DATA_DIR

    if not data_dir.exists():
        print(f"错误: 数据目录不存在: {data_dir}")
        print(f"请确认已运行 preprocess.py 生成 .npy 文件")
        return

    inputs_files = [f for f in os.listdir(data_dir) if f.endswith('_inputs.npy')]
    if not inputs_files:
        print(f"错误: 在 {data_dir} 中找不到 *_inputs.npy 文件")
        return

    inputs_file = inputs_files[0]
    basename = inputs_file.replace('_inputs.npy', '')

    inputs_path = os.path.join(data_dir, f'{basename}_inputs.npy')
    outputs_path = os.path.join(data_dir, f'{basename}_outputs.npy')

    print(f"\n{'='*60}")
    print(f"测试 RadarDataset")
    print(f"{'='*60}")
    print(f"输入文件: {inputs_path}")
    print(f"输出文件: {outputs_path}")

    dataset = RadarDataset(inputs_path, outputs_path)
    print(f"\n测试单个样本:")
    x, y = dataset[0]
    print(f"  样本 0:")
    print(f"    输入 shape: {x.shape}, dtype: {x.dtype}")
    print(f"    输出 shape: {y.shape}, dtype: {y.dtype}")
    print(f"    输入范围: [{x.min():.4f}, {x.max():.4f}]")
    print(f"    输出范围: [{y.min():.4f}, {y.max():.4f}]")

    print(f"\n测试 DataLoader (batch_size=2):")
    dataloader = get_dataloader(dataset, batch_size=2, shuffle=False)

    for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
        print(f"  Batch {batch_idx}:")
        print(f"    输入 shape: {batch_x.shape}")
        print(f"    输出 shape: {batch_y.shape}")

        if batch_idx >= 1:
            print(f"    ...")
            break

    print(f"\n总 batch 数: {len(dataloader)}")
    print(f"{'='*60}")
    print(f"✅ RadarDataset 测试通过!")
    print(f"{'='*60}")


if __name__ == '__main__':
    test_dataset()
