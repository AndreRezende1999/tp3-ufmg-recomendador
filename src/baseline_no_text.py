"""Baseline estrutural sem texto — regressão logística sobre features numéricas."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class BaselineResult:
    model: Pipeline
    metrics: dict[str, float]


def build_structural_features(pairs: pd.DataFrame) -> pd.DataFrame:
    """Constrói features estruturais (sem texto) a partir dos pares."""
    data = pairs.copy()
    data["completed_count"] = data["completed_codes"].map(lambda values: len(values) if isinstance(values, list) else 0)
    data["prerequisites_met_int"] = data["prerequisites_met"].astype(int)
    data["same_track_int"] = data["same_track"].astype(int)
    data["next_required_int"] = data["next_required"].astype(int)
    data["candidate_length"] = data["candidate_text"].fillna("").map(lambda text: len(str(text).split()))
    data["profile_length"] = data["perfil_texto"].fillna("").map(lambda text: len(str(text).split()))
    return data[["completed_count", "prerequisites_met_int", "same_track_int", "next_required_int", "candidate_length", "profile_length"]]


def train_baseline(train_pairs: pd.DataFrame, validation_pairs: pd.DataFrame | None = None) -> Pipeline:
    """Treina o baseline com regressão logística sobre features estruturais."""
    feature_builder = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("scaler", StandardScaler())]), ["completed_count", "prerequisites_met_int", "same_track_int", "next_required_int", "candidate_length", "profile_length"]),
        ],
        remainder="drop",
    )
    model = Pipeline(
        steps=[
            ("features", feature_builder),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    train_features = build_structural_features(train_pairs)
    model.fit(train_features, train_pairs["label"])
    return model


def evaluate_baseline(model: Pipeline, test_pairs: pd.DataFrame) -> dict[str, float]:
    """Avalia o baseline no conjunto de teste."""
    test_features = build_structural_features(test_pairs)
    predictions = model.predict(test_features)
    return {
        "accuracy": float(accuracy_score(test_pairs["label"], predictions)),
        "f1": float(f1_score(test_pairs["label"], predictions)),
    }
