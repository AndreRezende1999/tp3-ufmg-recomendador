"""Regras de filtragem de pré-requisitos e balanceamento por dificuldade."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class RecommendationCandidate:
    codigo: str
    nome: str
    score: float
    difficulty: float | None
    credits: int | None


def prerequisites_satisfied(candidate_row: pd.Series, completed_codes: Iterable[str]) -> bool:
    """Verifica se todos os pré-requisitos da candidata foram cursados."""
    prerequisites = set(candidate_row.get("pre_requisitos", []) or [])
    return prerequisites.issubset(set(completed_codes))


def filter_recommended_candidates(candidates: pd.DataFrame, completed_codes: Iterable[str]) -> pd.DataFrame:
    """Filtra candidatas cujos pré-requisitos estão satisfeitos."""
    if candidates.empty:
        return candidates.copy()
    mask = candidates.apply(lambda row: prerequisites_satisfied(row, completed_codes), axis=1)
    return candidates.loc[mask].copy()


def greedy_balance_by_difficulty(candidates: pd.DataFrame, *, difficulty_budget: float, max_courses: int | None = None) -> pd.DataFrame:
    """Seleciona disciplinas de forma gulosa respeitando orçamento de dificuldade."""
    if candidates.empty:
        return candidates.copy()

    ordered = candidates.sort_values(["predicted_probability", "difficulty"], ascending=[False, True]).copy()
    selected_rows: list[pd.Series] = []
    used_difficulty = 0.0

    for _, row in ordered.iterrows():
        difficulty = row.get("difficulty")
        numeric_difficulty = float(difficulty) if pd.notna(difficulty) else 0.0
        if max_courses is not None and len(selected_rows) >= max_courses:
            break
        if used_difficulty + numeric_difficulty > difficulty_budget:
            continue
        selected_rows.append(row)
        used_difficulty += numeric_difficulty

    if not selected_rows:
        return ordered.head(0).copy()
    return pd.DataFrame(selected_rows).reset_index(drop=True)
