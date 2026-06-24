import os
import wfdb
import numpy as np
import pandas as pd

TARGET_RECORDS = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
    '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
    '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
    '222', '223', '228', '230', '231', '232', '233', '234'
]

LABEL_MAP = {
    'N': 'N', 'L': 'L', 'R': 'R', 'A': 'A', 'V': 'V',
    'a': 'A', 'e': 'A', 'j': 'R', 'S': 'N', 'E': 'V',
    'f': 'V', 'F': 'V', 'x': 'N', 'p': 'N', 'Q': 'N',
    'J': 'R', '/': 'N', 'n': 'N', 'B': 'V'
}

TARGET_LABELS = ['N', 'A', 'L', 'R', 'V']


def get_available_records(data_dir):
    records = []
    for rec in TARGET_RECORDS:
        hea_path = os.path.join(data_dir, f'{rec}.hea')
        dat_path = os.path.join(data_dir, f'{rec}.dat')
        if os.path.exists(hea_path) and os.path.exists(dat_path):
            records.append(rec)
    return records


def load_mit_bih_record(data_dir, record_name, lead_name='MLII'):
    record = wfdb.rdrecord(os.path.join(data_dir, record_name))
    annotation = wfdb.rdann(os.path.join(data_dir, record_name), 'atr')

    lead_names = record.sig_name
    if lead_name in lead_names:
        lead_idx = lead_names.index(lead_name)
    else:
        lead_idx = 0

    signal = record.p_signal[:, lead_idx]
    fs = record.fs

    r_peaks = annotation.sample
    symbols = annotation.symbol

    filtered_data = []
    for peak, sym in zip(r_peaks, symbols):
        mapped = LABEL_MAP.get(sym, None)
        if mapped in TARGET_LABELS:
            filtered_data.append({'r_peak': int(peak), 'label': mapped})

    return {
        'signal': signal,
        'fs': fs,
        'r_peaks': r_peaks,
        'symbols': symbols,
        'filtered_data': filtered_data,
        'record_name': record_name
    }


def load_all_records(data_dir, progress_callback=None):
    all_data = []
    records = get_available_records(data_dir)
    total = len(records)

    for i, rec in enumerate(records):
        if progress_callback:
            progress_callback(int((i / total) * 100), f"Loading record {rec}...")
        try:
            data = load_mit_bih_record(data_dir, rec)
            all_data.append(data)
        except Exception as e:
            print(f"Error loading {rec}: {e}")

    if progress_callback:
        progress_callback(100, "All records loaded.")

    return all_data
