import argparse
import os

import numpy as np
import scipy.io as sio

from datautil.fault_preprocess import (
    build_stratified_split_record,
    normalize_time_series_samples,
    remap_targets_to_contiguous,
    save_fault_dataset,
    sliding_window_slice,
)


FAULT_TYPES = {
    '97': 0,
    '98': 0,
    '99': 1,
    '100': 2,
    '105': 3,
    '106': 4,
    '107': 5,
    '108': 6,
    '118': 7,
    '119': 8,
    '120': 9,
    '121': 10,
    '130': 11,
    '131': 12,
    '132': 13,
    '133': 14,
}


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess the CWRU fault dataset.')
    parser.add_argument('--input-dir', type=str, required=True, help='Directory containing CWRU .mat files.')
    parser.add_argument('--output-dir', type=str, default='data/cwru', help='Directory to save x.npy/y.npy and metadata.')
    parser.add_argument('--window-size', type=int, default=1024, help='Sliding-window length.')
    parser.add_argument('--stride', type=int, default=1024, help='Sliding-window stride.')
    parser.add_argument('--normalize', type=str, default='zscore', choices=['zscore', 'none'], help='Normalization method.')
    parser.add_argument('--train-ratio', type=float, default=0.6, help='Global train split ratio.')
    parser.add_argument('--val-ratio', type=float, default=0.2, help='Global validation split ratio.')
    parser.add_argument('--test-ratio', type=float, default=0.2, help='Global test split ratio.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for split recording.')
    return parser.parse_args()


def extract_drive_end_signal(mat_dict, file_prefix):
    preferred_key = f'X{file_prefix}_DE_time'
    if preferred_key in mat_dict:
        return mat_dict[preferred_key].reshape(-1)

    for key, value in mat_dict.items():
        if key.endswith('DE_time'):
            return np.asarray(value).reshape(-1)
    return None


def preprocess_cwru_dataset(args):
    all_samples = []
    raw_labels = []
    source_records = []

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f'CWRU input directory not found: {args.input_dir}')

    for filename in sorted(os.listdir(args.input_dir)):
        if not filename.endswith('.mat'):
            continue

        file_prefix = os.path.splitext(filename)[0]
        if file_prefix not in FAULT_TYPES:
            continue

        file_path = os.path.join(args.input_dir, filename)
        mat_data = sio.loadmat(file_path)
        signal = extract_drive_end_signal(mat_data, file_prefix)
        if signal is None:
            print(f'跳过 {filename}：未找到驱动端振动信号。')
            continue

        windows = sliding_window_slice(signal, window_size=args.window_size, stride=args.stride)
        if windows.shape[0] == 0:
            print(f'跳过 {filename}：信号长度不足 {args.window_size}。')
            continue

        windows = normalize_time_series_samples(windows, method=args.normalize)
        raw_label = FAULT_TYPES[file_prefix]
        all_samples.append(windows)
        raw_labels.extend([raw_label] * windows.shape[0])
        source_records.append({
            'file_name': filename,
            'raw_label': int(raw_label),
            'window_count': int(windows.shape[0]),
        })
        print(f'处理文件 {filename}，生成 {windows.shape[0]} 个窗口样本，原始标签 {raw_label}')

    if not all_samples:
        raise RuntimeError('No valid CWRU samples were generated. Please check the input directory.')

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
        'dataset': 'cwru',
        'window_size': int(args.window_size),
        'stride': int(args.stride),
        'normalization': args.normalize,
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
    preprocess_cwru_dataset(parse_args())
