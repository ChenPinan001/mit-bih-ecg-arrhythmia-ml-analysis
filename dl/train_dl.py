import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from datasets.ecg_dataset import ECGDataset
from dl.cnn_model import create_model

LABEL_NAMES = ['N', 'A', 'L', 'R', 'V']


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            focal_loss = alpha[targets] * focal_loss
        return focal_loss.mean()


def _get_class_weights(dataset):
    labels = dataset.encoded_labels
    counts = np.bincount(labels, minlength=5).astype(float)
    counts[counts == 0] = 1
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(weights)
    return torch.tensor(weights, dtype=torch.float32)


def _build_optimizer(name, model, lr):
    if name == 'Adam':
        return optim.Adam(model.parameters(), lr=lr)
    elif name == 'SGD':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif name == 'RMSprop':
        return optim.RMSprop(model.parameters(), lr=lr)
    elif name == 'AdamW':
        return optim.AdamW(model.parameters(), lr=lr)
    return optim.Adam(model.parameters(), lr=lr)


def _build_criterion(name, train_dataset):
    if 'WeightedCrossEntropy' in name:
        weights = _get_class_weights(train_dataset)
        return nn.CrossEntropyLoss(weight=weights)
    elif 'FocalLoss' in name:
        weights = _get_class_weights(train_dataset)
        return FocalLoss(alpha=weights, gamma=2.0)
    elif 'BCEWithLogitsLoss' in name:
        return nn.BCEWithLogitsLoss()
    return nn.CrossEntropyLoss()


def train_dl_model(model_name, train_csv, val_csv, output_dir, params=None, progress_callback=None):
    if params is None:
        params = {'batch_size': 256, 'learning_rate': 0.001, 'epochs': 20, 'seed': 42,
                  'optimizer': 'Adam', 'loss_function': 'CrossEntropy (多分类)'}

    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])
    device_str = params.get('device', 'GPU')
    if device_str == 'GPU' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif device_str == 'GPU':
        device = torch.device('cpu')
        if progress_callback:
            progress_callback(0, "警告: GPU不可用，自动切换到CPU")
    else:
        device = torch.device('cpu')

    train_dataset = ECGDataset(train_csv)
    val_dataset = ECGDataset(val_csv)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)

    input_length = train_dataset.features.shape[1]
    model = create_model(model_name, input_length=input_length, num_classes=5).to(device)

    optimizer_name = params.get('optimizer', 'Adam')
    loss_name = params.get('loss_function', 'CrossEntropy (多分类)')
    criterion = _build_criterion(loss_name, train_dataset)
    optimizer = _build_optimizer(optimizer_name, model, params['learning_rate'])

    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in range(params['epochs']):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        accuracy = accuracy_score(all_labels, all_preds)
        val_accuracies.append(accuracy)

        if progress_callback:
            progress_callback(
                int(((epoch + 1) / params['epochs']) * 100),
                f"Epoch {epoch+1}/{params['epochs']} - Loss: {avg_train_loss:.4f}/{avg_val_loss:.4f} - Acc: {accuracy:.4f}"
            )

    recall_val = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    precision_val = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_val = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3, 4])

    all_labels_bin = label_binarize(all_labels, classes=[0, 1, 2, 3, 4])
    fpr, tpr, _ = roc_curve(all_labels_bin.ravel(), all_probs.ravel())
    roc_auc_val = auc(fpr, tpr)

    os.makedirs(output_dir, exist_ok=True)
    safe_name = model_name.lower().replace('1d', '').replace(' ', '_')
    model_path = os.path.join(output_dir, f'{safe_name}_model.pth')
    torch.save(model.state_dict(), model_path)

    results = {
        'train_losses': train_losses, 'val_losses': val_losses, 'val_accuracies': val_accuracies,
        'accuracy': accuracy, 'recall': recall_val, 'precision': precision_val, 'f1': f1_val,
        'confusion_matrix': cm, 'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc_val,
        'classes': LABEL_NAMES, 'model_path': model_path, 'device': str(device), 'model_name': model_name,
    }
    return model, results


def load_dl_model(model_name, model_path, input_length=300, num_classes=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(model_name, input_length=input_length, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model, device


def predict_dl_model(model, device, segments):
    if segments.ndim == 1:
        segments = segments.reshape(1, -1)
    x = torch.tensor(segments, dtype=torch.float32).unsqueeze(1).to(device)
    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
    return preds.cpu().numpy(), probs.cpu().numpy()


def plot_dl_results(results, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(results['train_losses'], label='Train Loss', color='#89b4fa')
    axes[0].plot(results['val_losses'], label='Val Loss', color='#f38ba8')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curve'); axes[0].legend()

    axes[1].plot(results['val_accuracies'], label='Val Accuracy', color='#a6e3a1')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curve'); axes[1].legend()

    axes[2].plot(results['fpr'], results['tpr'], label=f'AUC = {results["roc_auc"]:.4f}')
    axes[2].plot([0, 1], [0, 1], 'k--')
    axes[2].set_xlabel('FPR'); axes[2].set_ylabel('TPR')
    axes[2].set_title('ROC Curve'); axes[2].legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/loss_curve.png", dpi=150, bbox_inches='tight')
    plt.savefig(f"{output_dir}/accuracy_curve.png", dpi=150, bbox_inches='tight')
    plt.savefig(f"{output_dir}/roc_curve.png", dpi=150, bbox_inches='tight')
    plt.close()

    cm = results['confusion_matrix']
    classes = results['classes']
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(f'{results["model_name"]} Confusion Matrix')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel('True'); ax.set_xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
