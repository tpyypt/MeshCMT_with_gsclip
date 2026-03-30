from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, top_k_accuracy_score


def classification_metrics(labels, logits, topk=(1, 3)) -> Dict[str, float]:
    labels = np.asarray(labels)
    logits = np.asarray(logits)
    preds = logits.argmax(axis=1)

    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
    }
    for k in topk:
        if logits.shape[1] >= k:
            metrics[f"top_{k}_accuracy"] = float(top_k_accuracy_score(labels, logits, k=k, labels=np.arange(logits.shape[1])))
    return metrics


def per_class_accuracy(labels, logits, classnames):
    labels = np.asarray(labels)
    preds = np.asarray(logits).argmax(axis=1)
    matrix = confusion_matrix(labels, preds, labels=np.arange(len(classnames)))
    per_class = {}
    for idx, class_name in enumerate(classnames):
        denom = matrix[idx].sum()
        per_class[class_name] = 0.0 if denom == 0 else float(matrix[idx, idx] / denom)
    return per_class
