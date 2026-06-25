from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def confusion_dataframe(y_true, y_pred) -> pd.DataFrame:
    matrix = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(matrix)


def plot_confusion_matrix(y_true, y_pred, *, labels: list[str] | None = None, title: str = "Matriz de confusão"):
    matrix = confusion_matrix(y_true, y_pred)
    figure, axis = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axis)
    axis.set_xlabel("Predito")
    axis.set_ylabel("Real")
    axis.set_title(title)
    figure.tight_layout()
    return figure


def summarize_training_history(history: list[dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(history)
