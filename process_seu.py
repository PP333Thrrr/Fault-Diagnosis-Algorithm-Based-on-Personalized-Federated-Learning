import argparse
import os

import numpy as np
import pandas as pd

from datautil.fault_preprocess import (
    build_stratified_split_record,
    normalize_time_series_samples,
    remap_targets_to_contiguous,
    save_fault_dataset,
    sliding_window_slice,
)


BEARING_FAULTS = {
    'health': 0,
    'ball': 1,
    'inner': 2,
    'outer': 3,
    'comb': 4,
}

GEAR_FAULTS = {
    'health': 5,
    'chipped': 6,
    'miss': 7,
    'root': 8,
    'surface': 9,
}


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess the SEU fault dataset.')
    parser.add_argument('--input-dir', type=str, required=True, help='Directory containing the SEU gearbox dataset.')
    parser.add_argument('--output-dir', type=str, default='data/seu', help='Directory to save x.npy/y.npy and metadata.')
    parser.add_argument('--window-size', type=int, default=1024, help='Sliding-window length.')
    parser.add_argument('--stride', type=int, default=1024, help='Sliding-window stride.')
    parser.add_argument('--normalize', type=str, default='zscore', choices=['zscore', 'none'], help='Normalization method.')
    parser.add_argument('--train-ratio', type=float, default=0.6, help='Global train split ratio.')
    parser.add_argument('--val-ratio', type=float, default=0.2, help='Global validation split ratio.')
    parser.add_argument('--test-ratio', type=float, default=0.2, help='Global test split ratio.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for split recording.')
    parser.add_argument('--include-gear', action='store_true', help='Include gearset files in addition to bearingset.')
    return parser.parse_args()


def read_numeric_series_from_csv(file_path):
    dataframe = pd.read_csv(file_path, header=None, sep=None, engine='python')
    numeric_values = pd.to_numeric(dataframe.stack(), errors='coerce').dropna()
    return numeric_values.to_numpy(dtype=np.float32)


def collect_windows_from_folder(folder_path, label_mapping, args, source_name):
    samples = []
    raw_labels = []
    source_records = []

    if not os.path.isdir(folder_path):
        return samples, raw_labels, source_records

    for file_name in sorted(os.listdir(folder_path)):
        if not file_name.endswith('.csv'):
            continue

        prefix = file_name.split('_')[0].lower()
        if prefix not in label_mapping:
            continue

        file_path = os.path.join(folder_path, file_name)
        signal = read_numeric_series_from_csv(file_path)
        windows = sliding_window_slice(signal, window_size=args.window_size, stride=args.stride)
        if windows.shape[0] == 0:
            print(f'跳过 {file_name}：信号长度不足 {args.window_size}。')
            continue

        windows = normalize_time_series_samples(windows, method=args.normalize)
        raw_label = label_mapping[prefix]
        samples.append(windows)
        raw_labels.extend([raw_label] * windows.shape[0])
        source_records.append({
            'source': source_name,
            'file_name': file_name,
            'raw_label': int(raw_label),
            'window_count': int(windows.shape[0]),
        })
        print(f'处理文件 {file_name}，生成 {windows.shape[0]} 个窗口样本，原始标签 {raw_label}')

    return samples, raw_labels, source_records


def preprocess_seu_dataset(args):
    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f'SEU input directory not found: {args.input_dir}')

    all_samples = []
    raw_labels = []
    source_records = []

    bearing_dir = os.path.join(args.input_dir, 'bearingset')
    bearing_samples, bearing_labels, bearing_records = collect_windows_from_folder(
        bearing_dir,
        BEARING_FAULTS,
        args,
        source_name='bearing',
    )
    all_samples.extend(bearing_samples)
    raw_labels.extend(bearing_labels)
    source_records.extend(bearing_records)

    if args.include_gear:
        gear_dir = os.path.join(args.input_dir, 'gearset')
        gear_samples, gear_labels, gear_records = collect_windows_from_folder(
            gear_dir,
            GEAR_FAULTS,
            args,
            source_name='gear',
        )
        all_samples.extend(gear_samples)
        raw_labels.extend(gear_labels)
        source_records.extend(gear_records)

    if not all_samples:
        raise RuntimeError('No valid SEU samples were generated. Please check the input directory.')

    samples = np.vstack(all_samples).astype(np.float32)
    remapped_labels, label_mapping = remap_targets_to_contiguous(raw_labels)
    split_record = build_stratified_split_record(
        remapped_labels,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    metadata = {
        'dataset': 'seu',
        'window_size': int(args.window_size),
        'stride': int(args.stride),
        'normalization': args.normalize,
        'include_gear': bool(args.include_gear),
        'num_samples': int(samples.shape[0]),
        'num_classes': int(len(np.unique(remapped_labels))),
        'label_mapping': label_mapping,
        'source_records': source_records,
    }

    save_fault_dataset(
        args.output_dir,
        samples,
        remapped_labels,
        metadata,
        split_record,
    )

    print('\n预处理完成。')
    print(f'样本形状: {samples.shape}')
    print(f'标签形状: {remapped_labels.shape}')
    print(f'类别数: {len(np.unique(remapped_labels))}')
    print(f'输出目录: {args.output_dir}')
    print(f'划分记录: {os.path.join(args.output_dir, "split_record.json")}')


if __name__ == '__main__':
    preprocess_seu_dataset(parse_args())
