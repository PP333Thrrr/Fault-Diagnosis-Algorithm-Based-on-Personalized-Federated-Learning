import json
import os
from pathlib import Path

import numpy as np


def sanitize_signal(signal):
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    return np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)


def sliding_window_slice(signal, window_size=1024, stride=1024):
    if window_size <= 0:
        raise ValueError('window_size must be positive.')
    if stride <= 0:
        raise ValueError('stride must be positive.')

    signal = sanitize_signal(signal)
    if signal.size < window_size:
        return np.empty((0, window_size), dtype=np.float32)

    windows = []
    max_start = signal.size - window_size
    for start in range(0, max_start + 1, stride):
        windows.append(signal[start:start + window_size])

    if not windows:
        return np.empty((0, window_size), dtype=np.float32)
    return np.stack(windows).astype(np.float32)


def normalize_time_series_samples(samples, method='zscore', eps=1e-6):
    samples = np.asarray(samples, dtype=np.float32)
    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)

    if method in (None, 'none'):
        return samples.astype(np.float32)

    squeeze_back = False
    if samples.ndim == 1:
        samples = samples[None, :]
        squeeze_back = True

    if method != 'zscore':
        raise ValueError(f'Unsupported normalization method: {method}')

    means = np.mean(samples, axis=1, keepdims=True)
    stds = np.std(samples, axis=1, keepdims=True)
    stds = np.where(stds < eps, 1.0, stds)
    normalized = (samples - means) / stds
    normalized = normalized.astype(np.float32)

    if squeeze_back:
        return normalized[0]
    return normalized


def remap_targets_to_contiguous(targets):
    targets = np.squeeze(np.asarray(targets))
    unique_targets = np.unique(targets)
    target_mapping = {}
    remapped_targets = np.zeros(len(targets), dtype=np.int64)

    for idx, label in enumerate(unique_targets):
        python_label = label.item() if isinstance(label, np.generic) else label
        target_mapping[python_label] = idx
        remapped_targets[targets == label] = idx

    return remapped_targets, target_mapping


def _allocate_split_counts(class_count, split_ratios):
    raw_counts = np.array(split_ratios, dtype=np.float64) * class_count
    counts = np.floor(raw_counts).astype(int)
    remainder = class_count - int(np.sum(counts))

    if remainder > 0:
        order = np.argsort(-(raw_counts - counts))
        for idx in order[:remainder]:
            counts[idx] += 1

    if class_count >= 3:
        for split_idx in range(len(counts)):
            if counts[split_idx] == 0:
                donor_candidates = np.where(counts > 1)[0]
                if donor_candidates.size == 0:
                    break
                donor_idx = donor_candidates[np.argmax(counts[donor_candidates])]
                counts[donor_idx] -= 1
                counts[split_idx] += 1

    return counts.tolist()


def summarize_class_distribution(labels, indices):
    if len(indices) == 0:
        return {}
    labels = np.asarray(labels)
    split_labels = labels[np.asarray(indices, dtype=np.int64)]
    unique_labels, counts = np.unique(split_labels, return_counts=True)
    return {
        str(int(label)): int(count)
        for label, count in zip(unique_labels, counts)
    }


def build_stratified_split_record(
    labels,
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    seed=0,
):
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError('train_ratio + val_ratio + test_ratio must equal 1.0')

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    rng = np.random.RandomState(seed)
    split_ratios = [train_ratio, val_ratio, test_ratio]
    split_names = ['train', 'val', 'test']
    split_indices = {name: [] for name in split_names}

    for label in np.unique(labels):
        class_indices = np.where(labels == label)[0]
        rng.shuffle(class_indices)
        class_counts = _allocate_split_counts(len(class_indices), split_ratios)

        from_index = 0
        for split_name, split_count in zip(split_names, class_counts):
            to_index = from_index + split_count
            split_indices[split_name].extend(class_indices[from_index:to_index].tolist())
            from_index = to_index

    for split_name in split_names:
        split_indices[split_name] = sorted(split_indices[split_name])

    split_record = {
        'seed': int(seed),
        'ratios': {
            'train': float(train_ratio),
            'val': float(val_ratio),
            'test': float(test_ratio),
        },
        'total_samples': int(len(labels)),
        'splits': {},
    }

    for split_name in split_names:
        split_record['splits'][split_name] = {
            'count': int(len(split_indices[split_name])),
            'indices': split_indices[split_name],
            'class_distribution': summarize_class_distribution(labels, split_indices[split_name]),
        }

    return split_record


def _jsonify_mapping(mapping):
    return {str(key): int(value) for key, value in mapping.items()}


def save_fault_dataset(output_dir, samples, labels, metadata, split_record):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    samples = np.asarray(samples, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    np.save(output_path / 'x.npy', samples)
    np.save(output_path / 'y.npy', labels)

    metadata = dict(metadata)
    if 'label_mapping' in metadata:
        metadata['label_mapping'] = _jsonify_mapping(metadata['label_mapping'])

    with open(output_path / 'preprocess_meta.json', 'w', encoding='utf-8') as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    with open(output_path / 'split_record.json', 'w', encoding='utf-8') as file:
        json.dump(split_record, file, ensure_ascii=False, indent=2)


def maybe_load_split_record(dataset_dir):
    split_record_path = Path(dataset_dir) / 'split_record.json'
    if not split_record_path.exists():
        return None
    with open(split_record_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def ensure_directory(path):
    os.makedirs(path, exist_ok=True)
