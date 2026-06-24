import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class ECGDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.labels = df['label'].values
        self.features = df.drop(columns=['label', 'sample_id']).values.astype(np.float32)
        self.label_map = {'N': 0, 'A': 1, 'L': 2, 'R': 3, 'V': 4}
        self.label_names = ['N', 'A', 'L', 'R', 'V']
        self.encoded_labels = np.array([self.label_map.get(l, 0) for l in self.labels])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.encoded_labels[idx], dtype=torch.long)
        return x, y
