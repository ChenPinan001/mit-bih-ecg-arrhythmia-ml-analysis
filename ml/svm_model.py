import numpy as np
import pandas as pd
import pickle
import os
import warnings
from sklearn.svm import SVC
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn.svm')
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt


def load_split_data(csv_path):
    df = pd.read_csv(csv_path)
    y = df['label'].values
    X = df.drop(columns=['label', 'sample_id']).values if 'sample_id' in df.columns else df.drop(columns=['label']).values
    return X, y


def train_svm(train_path, val_path, params=None, progress_callback=None):
    X_train, y_train = load_split_data(train_path)
    X_val, y_val = load_split_data(val_path)

    X_train = np.nan_to_num(X_train, nan=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    if params is None:
        params = {
            'C': 1.0,
            'gamma': 'scale',
            'kernel': 'rbf'
        }

    model = SVC(**params, probability=True, random_state=42)

    if progress_callback:
        progress_callback(5, "准备数据...")
        progress_callback(10, "训练SVM (可能需要较长时间)...")

    model.fit(X_train_scaled, y_train)

    if progress_callback:
        progress_callback(60, "训练完成，评估中...")

    y_pred = model.predict(X_val_scaled)
    accuracy = accuracy_score(y_val, y_pred)

    if progress_callback:
        progress_callback(70, "计算指标...")

    recall = recall_score(y_val, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_val, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_val, y_pred, labels=['N', 'A', 'L', 'R', 'V'])

    classes = ['N', 'A', 'L', 'R', 'V']
    y_val_bin = label_binarize(y_val, classes=classes)

    if progress_callback:
        progress_callback(85, "计算ROC曲线...")

    y_pred_proba = model.predict_proba(X_val_scaled)
    fpr, tpr, _ = roc_curve(y_val_bin.ravel(), y_pred_proba.ravel())
    roc_auc = auc(fpr, tpr)

    if progress_callback:
        progress_callback(95, "保存模型...")

    results = {
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'classes': classes
    }

    return model, scaler, results


def save_svm_model(model, scaler, model_path, scaler_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)


def load_svm_model(model_path, scaler_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler


def predict_svm(model, scaler, X):
    X = np.nan_to_num(X, nan=0.0)
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled), model.predict_proba(X_scaled)


def plot_svm_results(results, output_dir, prefix='svm'):
    classes = results['classes']
    cm = results['confusion_matrix']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im = axes[0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[0].set_title('Confusion Matrix')
    plt.colorbar(im, ax=axes[0])
    tick_marks = np.arange(len(classes))
    axes[0].set_xticks(tick_marks)
    axes[0].set_xticklabels(classes)
    axes[0].set_yticks(tick_marks)
    axes[0].set_yticklabels(classes)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0].text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
    axes[0].set_ylabel('True')
    axes[0].set_xlabel('Predicted')

    axes[1].plot(results['fpr'], results['tpr'], label=f'ROC (AUC = {results["roc_auc"]:.4f})')
    axes[1].plot([0, 1], [0, 1], 'k--')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve')
    axes[1].legend()

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    values = [results['accuracy'], results['precision'], results['recall'], results['f1']]
    axes[2].bar(metrics, values, color=['#4CAF50', '#2196F3', '#FF9800', '#F44336'])
    axes[2].set_ylim(0, 1)
    axes[2].set_title('Performance Metrics')
    for i, v in enumerate(values):
        axes[2].text(i, v + 0.01, f'{v:.4f}', ha='center')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{prefix}_results.png", dpi=150, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title('SVM Confusion Matrix')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel('True')
    ax.set_xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{prefix}_confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
