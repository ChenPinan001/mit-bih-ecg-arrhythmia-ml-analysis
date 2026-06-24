import pandas as pd
import numpy as np


def load_split_data(csv_path):
    df = pd.read_csv(csv_path)
    y = df['label'].values
    X = df.drop(columns=['label', 'sample_id']).values if 'sample_id' in df.columns else df.drop(columns=['label']).values
    return X, y
