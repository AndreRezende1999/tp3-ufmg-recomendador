"""Geração de perfis sintéticos e pares de treino para classificador e SBERT."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .data_curation import get_discipline_text, get_completed_profile_text


@dataclass(frozen=True)
class SyntheticProfile:
    profile_id: str
    completed_codes: tuple[str, ...]
    profile_text: str
    variant_name: str


def build_profile_text(disciplines: pd.DataFrame, completed_codes: Iterable[str], *, max_tokens_hint: int = 320) -> str:
    """Monta texto do perfil a partir das disciplinas cursadas."""
    code_set = {code for code in completed_codes if code}
    selected_rows = disciplines[disciplines["codigo"].isin(code_set)]
    parts: list[str] = []
    for _, row in selected_rows.iterrows():
        segment_parts = [str(row.get("codigo", "")).strip(), str(row.get("nome", "")).strip()]
        area = row.get("area_conhecimento")
        if pd.notna(area):
            segment_parts.append(f"Area: {area}")
        ementa = str(row.get("ementa", "")).strip()
        if ementa:
            segment_parts.append(ementa[:700])
        parts.append(" | ".join(part for part in segment_parts if part))

    profile_text = "\n\n".join(parts)
    if len(profile_text.split()) > max_tokens_hint * 2:
        return " ".join(profile_text.split()[: max_tokens_hint * 2])
    return profile_text


def generate_period_prefix_profiles(disciplines: pd.DataFrame, *, period_column: str = "periodo_sugerido") -> list[SyntheticProfile]:
    """Gera perfis acumulando disciplinas por período (prefixo crescente)."""
    profiles: list[SyntheticProfile] = []
    if period_column not in disciplines.columns:
        return profiles

    ordered = disciplines.dropna(subset=[period_column]).sort_values([period_column, "codigo"])
    if ordered.empty:
        return profiles

    max_period = int(ordered[period_column].max())
    for period in range(1, max_period):
        completed_codes = tuple(ordered.loc[ordered[period_column] <= period, "codigo"].dropna().astype(str).tolist())
        profile_text = build_profile_text(disciplines, completed_codes)
        profiles.append(
            SyntheticProfile(
                profile_id=f"prefix_p{period}",
                completed_codes=completed_codes,
                profile_text=profile_text,
                variant_name="prefix",
            )
        )
    return profiles


def generate_elective_variants(disciplines: pd.DataFrame, base_profiles: Iterable[SyntheticProfile], *, group_column: str = "grupo_optativa") -> list[SyntheticProfile]:
    """Adiciona variantes com optativas ao conjunto de perfis base."""
    if group_column not in disciplines.columns:
        return []
    variants: list[SyntheticProfile] = []
    grouped = disciplines.dropna(subset=[group_column]).groupby(group_column)
    elective_groups = {group_name: group_df["codigo"].dropna().astype(str).tolist() for group_name, group_df in grouped}

    for profile in base_profiles:
        for group_name, group_codes in elective_groups.items():
            chosen = tuple(sorted(set(profile.completed_codes).union(group_codes[: min(2, len(group_codes))])))
            profile_text = build_profile_text(disciplines, chosen)
            variants.append(
                SyntheticProfile(
                    profile_id=f"{profile.profile_id}_{group_name}",
                    completed_codes=chosen,
                    profile_text=profile_text,
                    variant_name=f"elective_{group_name}",
                )
            )
    return variants


def generate_labelled_pairs(disciplines: pd.DataFrame, profiles: Iterable[SyntheticProfile]) -> pd.DataFrame:
    """Gera pares rotulados (perfil, candidata) para o classificador binário."""
    records: list[dict] = []
    discipline_codes = disciplines["codigo"].dropna().astype(str).tolist()
    discipline_lookup = disciplines.set_index("codigo")

    for profile in profiles:
        completed_set = set(profile.completed_codes)
        completed_periods = disciplines.loc[disciplines["codigo"].isin(completed_set), "periodo_sugerido"].dropna()
        highest_completed_period = int(completed_periods.max()) if not completed_periods.empty else 0
        completed_groups: set[str] = set()

        if "grupo_optativa" in disciplines.columns:
            completed_groups = set(
                disciplines.loc[disciplines["codigo"].isin(completed_set), "grupo_optativa"]
                .dropna()
                .astype(str)
                .tolist()
            )

        for code in discipline_codes:
            if code in completed_set:
                continue
            row = discipline_lookup.loc[code]
            prerequisites = set(row.get("pre_requisitos", []) or [])
            prerequisites_met = prerequisites.issubset(completed_set)
            period = row.get("periodo_sugerido")
            same_track = pd.notna(row.get("grupo_optativa")) and str(row.get("grupo_optativa")) in completed_groups
            next_required = pd.notna(period) and int(period) <= highest_completed_period + 1
            positive = bool(prerequisites_met and (same_track or next_required))
            label = int(positive)
            records.append(
                {
                    "profile_id": profile.profile_id,
                    "variant_name": profile.variant_name,
                    "perfil_texto": profile.profile_text,
                    "completed_codes": list(profile.completed_codes),
                    "candidate_code": code,
                    "candidate_text": row.get("ementa", ""),
                    "label": label,
                    "prerequisites_met": prerequisites_met,
                    "same_track": same_track,
                    "next_required": next_required,
                }
            )
    return pd.DataFrame(records)


def split_by_profile(pairs: pd.DataFrame, *, train_size: float = 0.7, validation_size: float = 0.15, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Divide os pares por perfil (sem vazamento entre splits)."""
    unique_profiles = pairs[["profile_id"]].drop_duplicates().sample(frac=1.0, random_state=seed)
    total_profiles = len(unique_profiles)
    if total_profiles < 3:
        return {
            "train": pairs.copy().reset_index(drop=True),
            "validation": pairs.head(0).copy(),
            "test": pairs.head(0).copy(),
        }

    train_cutoff = max(1, int(total_profiles * train_size))
    validation_cutoff = max(train_cutoff + 1, int(total_profiles * (train_size + validation_size)))
    validation_cutoff = min(validation_cutoff, total_profiles - 1)

    train_profiles = set(unique_profiles.iloc[:train_cutoff]["profile_id"])
    validation_profiles = set(unique_profiles.iloc[train_cutoff:validation_cutoff]["profile_id"])
    test_profiles = set(unique_profiles.iloc[validation_cutoff:]["profile_id"])

    return {
        "train": pairs[pairs["profile_id"].isin(train_profiles)].reset_index(drop=True),
        "validation": pairs[pairs["profile_id"].isin(validation_profiles)].reset_index(drop=True),
        "test": pairs[pairs["profile_id"].isin(test_profiles)].reset_index(drop=True),
    }


# ===================================================================
# Pares de treino para Sentence-BERT (CosineSimilarityLoss)
# ===================================================================

def generate_sbert_training_pairs(
    disciplines_df: pd.DataFrame,
    profiles: Iterable[SyntheticProfile],
) -> pd.DataFrame:
    """Gera pares (text_a, text_b, label) para fine-tuning do SBERT.

    - text_a: texto do perfil (concatenação das ementas cursadas)
    - text_b: texto da disciplina candidata (nome + ementa)
    - label: 1.0 (relevante) ou 0.0 (irrelevante)
    """
    records: list[dict] = []
    discipline_codes = disciplines_df["codigo"].dropna().astype(str).tolist()
    discipline_lookup = disciplines_df.set_index("codigo")

    for profile in profiles:
        completed_set = set(profile.completed_codes)
        # Texto do perfil via ementas concatenadas
        text_a = get_completed_profile_text(disciplines_df, list(completed_set))

        completed_periods = disciplines_df.loc[
            disciplines_df["codigo"].isin(completed_set), "periodo_sugerido"
        ].dropna()
        highest_period = int(completed_periods.max()) if not completed_periods.empty else 0

        for code in discipline_codes:
            if code in completed_set:
                continue
            row = discipline_lookup.loc[code]
            prereqs = set(row.get("pre_requisitos", []) or [])
            prereqs_met = prereqs.issubset(completed_set)
            period = row.get("periodo_sugerido")
            next_required = pd.notna(period) and int(period) <= highest_period + 1
            relevant = bool(prereqs_met and next_required)

            text_b = get_discipline_text(row)
            records.append({
                "text_a": text_a,
                "text_b": text_b,
                "label": 1.0 if relevant else 0.0,
            })

    return pd.DataFrame(records)
