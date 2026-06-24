import numpy as np
import pywt
import pandas as pd
from scipy.signal import butter, filtfilt


def wavelet_denoise(signal, wavelet='db4', level=5):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    threshold = np.median(np.abs(coeffs[-1])) / 0.6745 * np.sqrt(2 * np.log(len(signal)))
    denoised_coeffs = [coeffs[0]]
    for c in coeffs[1:]:
        denoised = pywt.threshold(c, threshold, mode='soft')
        denoised_coeffs.append(denoised)
    denoised_signal = pywt.waverec(denoised_coeffs, wavelet)
    if len(denoised_signal) > len(signal):
        denoised_signal = denoised_signal[:len(signal)]
    return denoised_signal


def bandpass_filter(signal, fs=360, lowcut=0.5, highcut=40.0, order=4):
    nyq = fs / 2.0
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)
    return filtered


def remove_baseline_wander(signal, fs=360):
    return bandpass_filter(signal, fs, lowcut=0.5, highcut=5.0, order=2)


def segment_heartbeat(signal, r_peaks, labels, before=99, after=200):
    segments = []
    segment_labels = []
    total_len = before + after + 1

    for peak, label in zip(r_peaks, labels):
        start = peak - before
        end = peak + after + 1
        if start >= 0 and end <= len(signal):
            segment = signal[start:end]
            segments.append(segment)
            segment_labels.append(label)

    return np.array(segments), np.array(segment_labels)


def process_record(record_data, denoise=True):
    signal = record_data['signal']
    filtered_data = record_data['filtered_data']
    fs = record_data.get('fs', 360)

    if denoise:
        signal = wavelet_denoise(signal)
        signal = bandpass_filter(signal, fs)

    r_peaks = np.array([d['r_peak'] for d in filtered_data])
    labels = np.array([d['label'] for d in filtered_data])

    segments, seg_labels = segment_heartbeat(signal, r_peaks, labels)

    return segments, seg_labels, signal


def save_processed_data(segments, labels, output_dir, prefix='processed'):
    df = pd.DataFrame(segments)
    df['label'] = labels
    df['sample_id'] = range(len(df))

    cols = ['sample_id'] + [c for c in df.columns if c != 'sample_id']
    df = df[cols]

    output_path = f"{output_dir}/{prefix}.csv"
    df.to_csv(output_path, index=False)
    return output_path, len(df)


def split_dataset(segments, labels, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
    np.random.seed(seed)
    n = len(segments)
    indices = np.random.permutation(n)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return {
        'train': (segments[train_idx], labels[train_idx]),
        'val': (segments[val_idx], labels[val_idx]),
        'test': (segments[test_idx], labels[test_idx])
    }


def save_split_data(splits, output_dir):
    paths = {}
    for split_name, (segs, labs) in splits.items():
        df = pd.DataFrame(segs)
        df['label'] = labs
        df['sample_id'] = range(len(df))
        cols = ['sample_id'] + [c for c in df.columns if c != 'sample_id']
        df = df[cols]
        path = f"{output_dir}/{split_name}.csv"
        df.to_csv(path, index=False)
        paths[split_name] = path
    return paths
