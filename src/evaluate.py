"""Métricas de avaliação e comparação entre modelos."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


# ---------------------------------------------------------------------------
# Métricas de classificação binária
# ---------------------------------------------------------------------------

def classification_metrics(y_true, y_pred) -> dict[str, float]:
    """Calcula acurácia, precisão, recall e F1."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def confusion_dataframe(y_true, y_pred) -> pd.DataFrame:
    """Retorna a matriz de confusão como DataFrame."""
    matrix = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(matrix)


def plot_confusion_matrix(y_true, y_pred, *, labels: list[str] | None = None, title: str = "Matriz de confusão"):
    """Plota a matriz de confusão com heatmap."""
    matrix = confusion_matrix(y_true, y_pred)
    figure, axis = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axis)
    axis.set_xlabel("Predito")
    axis.set_ylabel("Real")
    axis.set_title(title)
    figure.tight_layout()
    return figure


def summarize_training_history(history: list[dict[str, float]]) -> pd.DataFrame:
    """Converte histórico de treino em DataFrame."""
    return pd.DataFrame(history)


# ---------------------------------------------------------------------------
# Métricas de busca semântica (Precision@k, Recall@k, MRR)
# ---------------------------------------------------------------------------

def semantic_search_metrics(
    queries: list[np.ndarray],
    relevant_ids_per_query: list[set[int]],
    corpus_embeddings: np.ndarray,
    model,
    top_k: int = 10,
) -> dict[str, float]:
    """Calcula Precision@k, Recall@k e MRR para busca semântica.

    Args:
        queries: lista de embeddings de consulta (um por query).
        relevant_ids_per_query: para cada query, conjunto de índices relevantes no corpus.
        corpus_embeddings: embeddings do corpus inteiro.
        model: não usado diretamente (compatibilidade); as queries já são embeddings.
        top_k: quantidade de resultados a considerar.

    Returns:
        Dicionário com precision@k, recall@k e mrr.
    """
    from .sbert_encoder import semantic_search

    precisions: list[float] = []
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []

    for query_emb, relevant_ids in zip(queries, relevant_ids_per_query):
        hits = semantic_search(query_emb, corpus_embeddings, top_k=top_k)
        retrieved_ids = [h["index"] for h in hits]

        # Precision@k
        relevant_retrieved = sum(1 for idx in retrieved_ids if idx in relevant_ids)
        precisions.append(relevant_retrieved / top_k)

        # Recall@k
        if len(relevant_ids) > 0:
            recalls.append(relevant_retrieved / len(relevant_ids))
        else:
            recalls.append(0.0)

        # MRR — posição do primeiro relevante
        rr = 0.0
        for rank, idx in enumerate(retrieved_ids, start=1):
            if idx in relevant_ids:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    return {
        f"precision@{top_k}": float(np.mean(precisions)),
        f"recall@{top_k}": float(np.mean(recalls)),
        "mrr": float(np.mean(reciprocal_ranks)),
    }


# ---------------------------------------------------------------------------
# Comparação entre modelos
# ---------------------------------------------------------------------------

def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """Cria tabela comparativa a partir de um dicionário {nome_modelo: métricas}.

    Exemplo:
        results = {
            "baseline": {"accuracy": 0.85, "f1": 0.80},
            "bert": {"accuracy": 0.90, "f1": 0.87},
            "sbert": {"precision@10": 0.75, "recall@10": 0.60, "mrr": 0.82},
        }
    """
    rows = []
    for model_name, metrics in results.items():
        row = {"modelo": model_name, **metrics}
        rows.append(row)
    return pd.DataFrame(rows).set_index("modelo")
