import numpy as np
import pandas as pd
import tsfel
from .preprocessing_utils import load_split_data


def extract_features(data_dir, split_name, output_dir, progress_callback=None):
    csv_path = f"{data_dir}/{split_name}.csv"
    df = pd.read_csv(csv_path)
    labels = df['label'].values
    sample_ids = df['sample_id'].values
    segments = df.drop(columns=['label', 'sample_id']).values

    cfg = tsfel.get_features_by_domain()

    all_features = []
    total = len(segments)
    batch_size = 500

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = segments[start:end]
        features_list = []
        for seg in batch:
            feat = tsfel.time_series_features_extractor(cfg, seg, verbose=0)
            features_list.append(feat.values.flatten())
        all_features.extend(features_list)

        if progress_callback:
            progress_callback(int((end / total) * 100), f"Extracted {end}/{total} samples")

    feature_names = []
    sample_feat = tsfel.time_series_features_extractor(cfg, segments[0], verbose=0)
    feature_names = list(sample_feat.columns)

    feature_df = pd.DataFrame(all_features, columns=feature_names)
    feature_df['label'] = labels
    feature_df['sample_id'] = sample_ids

    cols = ['sample_id'] + [c for c in feature_df.columns if c not in ['sample_id', 'label']] + ['label']
    feature_df = feature_df[cols]

    output_path = f"{output_dir}/{split_name}_feature.csv"
    feature_df.to_csv(output_path, index=False)

    if progress_callback:
        progress_callback(100, f"Feature extraction complete: {len(feature_names)} features")

    return output_path, feature_names, feature_df


def filter_correlated_features(df, threshold=0.95):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'label' in numeric_cols:
        numeric_cols.remove('label')
    if 'sample_id' in numeric_cols:
        numeric_cols.remove('sample_id')

    corr_matrix = df[numeric_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    filtered_df = df.drop(columns=to_drop)
    return filtered_df, to_drop
