import torch
import torch.nn as nn
import torch.nn.functional as F


class ECG_CNN(nn.Module):
    def __init__(self, input_length=300, num_classes=5):
        super(ECG_CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        dummy = torch.zeros(1, 1, input_length)
        out = self.pool3(self.conv3(self.pool2(self.conv2(self.pool1(self.conv1(dummy))))))
        flat_size = out.view(1, -1).size(1)
        self.fc1 = nn.Linear(flat_size, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ECG_LSTM(nn.Module):
    def __init__(self, input_length=300, hidden_size=128, num_layers=2, num_classes=5):
        super(ECG_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(1, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class ECG_ResNet1D(nn.Module):
    def __init__(self, input_length=300, num_classes=5):
        super(ECG_ResNet1D, self).__init__()
        self.conv_in = nn.Conv1d(1, 64, kernel_size=15, padding=7)
        self.bn_in = nn.BatchNorm1d(64)

        self.layer1 = self._make_res_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_res_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_res_layer(128, 256, 2, stride=2)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def _make_res_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = [ResBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3, stride=stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ECG_Transformer(nn.Module):
    def __init__(self, input_length=300, d_model=64, nhead=4, num_layers=2, num_classes=5):
        super(ECG_Transformer, self).__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, input_length, d_model) * 0.1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(d_model, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.input_proj(x) + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        x = self.global_pool(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


MODEL_REGISTRY = {
    'CNN1D': ECG_CNN,
    'LSTM': ECG_LSTM,
    'ResNet1D': ECG_ResNet1D,
    'Transformer': ECG_Transformer,
}


def create_model(model_name, input_length=300, num_classes=5, **kwargs):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL_REGISTRY[model_name](input_length=input_length, num_classes=num_classes, **kwargs)
