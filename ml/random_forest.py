import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt


def load_split_data(csv_path):
    df = pd.read_csv(csv_path)
    y = df['label'].values
    X = df.drop(columns=['label', 'sample_id']).values if 'sample_id' in df.columns else df.drop(columns=['label']).values
    return X, y


def train_random_forest(train_path, val_path, params=None, progress_callback=None):
    X_train, y_train = load_split_data(train_path)
    X_val, y_val = load_split_data(val_path)

    X_train = np.nan_to_num(X_train, nan=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0)

    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'criterion': 'gini'
        }

    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)

    if progress_callback:
        progress_callback(10, "Training Random Forest...")

    model.fit(X_train, y_train)

    if progress_callback:
        progress_callback(70, "Evaluating on validation set...")

    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_val, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_val, y_pred, labels=['N', 'A', 'L', 'R', 'V'])

    classes = ['N', 'A', 'L', 'R', 'V']
    y_val_bin = label_binarize(y_val, classes=classes)
    y_pred_proba = model.predict_proba(X_val)
    fpr, tpr, _ = roc_curve(y_val_bin.ravel(), y_pred_proba.ravel())
    roc_auc = auc(fpr, tpr)

    if progress_callback:
        progress_callback(90, "Saving model...")

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

    return model, results


def save_rf_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def load_rf_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def predict_rf(model, X):
    X = np.nan_to_num(X, nan=0.0)
    return model.predict(X), model.predict_proba(X)


def plot_rf_results(results, output_dir, prefix='random_forest'):
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
    ax.set_title('Random Forest Confusion Matrix')
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
