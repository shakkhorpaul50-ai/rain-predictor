import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve
import os


def plot_confusion_matrix(cm, class_names=["No Rain", "Rain"], save_path=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    return fig


def plot_roc_curve(y_true, y_prob, save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = np.trapezoid(tpr, fpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    return fig


def plot_feature_importance(feature_importance, top_n=20, save_path=None):
    sorted_idx = np.argsort(feature_importance)[::-1][:top_n]
    sorted_vals = feature_importance[sorted_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(sorted_vals)), sorted_vals, align="center")
    ax.set_yticks(range(len(sorted_vals)))
    ax.set_yticklabels([f"Feat {i}" for i in sorted_idx])
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    return fig


def plot_precision_recall(y_true, y_prob, save_path=None):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    return fig


def plot_sample_images(images, titles=None, save_path=None, n_cols=4):
    n = len(images)
    n_cols = min(n_cols, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n > 1 else [axes]
    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(images[i], cmap="gray" if len(images[i].shape) == 2 else None)
            if titles and i < len(titles):
                ax.set_title(titles[i])
            ax.axis("off")
        else:
            ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    return fig
