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


def get_schedule_slots(row: pd.Series) -> list[tuple[str, str]]:
    """Extrai os slots de horário de uma disciplina como lista de (dia, turno)."""
    horarios = row.get("horarios")
    if not isinstance(horarios, list) or not horarios:
        return []
    return [(h.get("dia", ""), h.get("turno", "")) for h in horarios if h.get("dia") and h.get("turno")]


def has_schedule_conflict(candidate_slots: list[tuple[str, str]], selected_slots: set[tuple[str, str]]) -> bool:
    """Verifica se os slots da candidata conflitam com os já selecionados."""
    if not candidate_slots:
        return False
    return any(slot in selected_slots for slot in candidate_slots)


def filter_by_schedule(candidates: pd.DataFrame, already_selected_slots: set[tuple[str, str]] | None = None) -> pd.DataFrame:
    """Remove candidatas cujos horários conflitam com os já selecionados."""
    if candidates.empty:
        return candidates.copy()
    if already_selected_slots is None:
        already_selected_slots = set()
    if not already_selected_slots:
        return candidates.copy()

    def _no_conflict(row):
        slots = get_schedule_slots(row)
        return not slots or not has_schedule_conflict(slots, already_selected_slots)

    mask = candidates.apply(_no_conflict, axis=1)
    return candidates.loc[mask].copy()


def greedy_balance_by_difficulty(
    candidates: pd.DataFrame,
    *,
    difficulty_budget: float,
    max_courses: int | None = None,
    respect_schedule: bool = True,
) -> pd.DataFrame:
    """Seleciona disciplinas de forma gulosa respeitando orçamento de dificuldade e horários.

    Se respect_schedule=True, descarta candidatas cujos horários conflitem com as já
    selecionadas, garantindo uma grade sem choques de horário.
    """
    if candidates.empty:
        return candidates.copy()

    ordered = candidates.sort_values(["predicted_probability", "difficulty"], ascending=[False, True]).copy()
    selected_rows: list[pd.Series] = []
    used_difficulty = 0.0
    used_slots: set[tuple[str, str]] = set()

    for _, row in ordered.iterrows():
        difficulty = row.get("difficulty")
        numeric_difficulty = float(difficulty) if pd.notna(difficulty) else 0.0
        if max_courses is not None and len(selected_rows) >= max_courses:
            break
        if used_difficulty + numeric_difficulty > difficulty_budget:
            continue
        if respect_schedule:
            candidate_slots = set(get_schedule_slots(row))
            if candidate_slots and has_schedule_conflict(list(candidate_slots), used_slots):
                continue
        selected_rows.append(row)
        used_difficulty += numeric_difficulty
        used_slots.update(get_schedule_slots(row))

    if not selected_rows:
        return ordered.head(0).copy()
    return pd.DataFrame(selected_rows).reset_index(drop=True)
