import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def classification_report(
    y_true: List[int], y_pred: List[int], y_prob: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    Compute common classification metrics. If y_prob is provided, also compute aupr.
    """
    acc = metrics.accuracy_score(y_true, y_pred)
    macro_precision, macro_recall, macro_f1, _ = metrics.precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    micro_precision, micro_recall, micro_f1, _ = metrics.precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    out = {
        "accuracy": acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
    }
    if y_prob is not None:
        precision, recall, _ = metrics.precision_recall_curve(y_true, y_prob)
        aupr = metrics.auc(recall, precision)
        out["aupr"] = aupr
    return out


def save_metrics_json(metrics_dict: Dict[str, float], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)


def save_predictions_csv(
    titles: List[str],
    labels: List[int],
    preds: List[int],
    probs: List[float],
    path: str,
) -> None:
    import pandas as pd

    ensure_dir(os.path.dirname(path))
    df = pd.DataFrame(
        {
            "title": titles,
            "label": labels,
            "pred": preds,
            "prob": probs,
        }
    )
    df.to_csv(path, index=False, encoding="utf-8")


def plot_confusion_matrix(
    y_true: List[int], y_pred: List[int], path: str, labels: Tuple[str, str] = ("wrong", "correct")
) -> None:
    ensure_dir(os.path.dirname(path))
    cm = metrics.confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_pr_curve(y_true: List[int], y_prob: List[float], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_prob)
    aupr = metrics.auc(recall, precision)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, label=f"PR (AUPR={aupr:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def count_parameters(model) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable}
