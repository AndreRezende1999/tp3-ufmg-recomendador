from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import PROCESSED_DIR

CODE_PATTERN = re.compile(r"\b[A-Z]{2,4}[0-9]{4}\b")


@dataclass
class DisciplineRecord:
    disciplina_id: str
    codigo: str
    nome: str
    tipo: str | None = None
    periodo_sugerido: int | None = None
    carga_horaria: int | None = None
    pre_requisitos: list[str] | None = None
    ementa: str | None = None
    area_conhecimento: str | None = None
    grupo_optativa: str | None = None
    dificuldade: float | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["pre_requisitos"] = self.pre_requisitos or []
        return data


def normalize_course_code(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = re.sub(r"\s+", "", str(value)).upper()
    return cleaned if CODE_PATTERN.fullmatch(cleaned) else cleaned


def normalize_course_name(value: str | None) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text.title()


def extract_course_codes(text: str | None) -> list[str]:
    if not text:
        return []
    return sorted(set(CODE_PATTERN.findall(text.upper())))


def reconcile_sources(primary: pd.DataFrame, secondary: pd.DataFrame) -> pd.DataFrame:
    merged = primary.copy()
    if not secondary.empty:
        secondary_indexed = secondary.set_index("codigo")
        for code, row in secondary_indexed.iterrows():
            if code in merged["codigo"].values:
                mask = merged["codigo"] == code
                for column in secondary.columns:
                    if column == "codigo":
                        continue
                    current_value = merged.loc[mask, column].iloc[0] if column in merged.columns and not merged.loc[mask, column].empty else None
                    if pd.isna(current_value) or current_value in (None, ""):
                        merged.loc[mask, column] = row.get(column)
            else:
                merged = pd.concat([merged, row.to_frame().T], ignore_index=True, sort=False)
    return merged


def load_difficulty_sheet(path_or_url: str) -> pd.DataFrame:
    if path_or_url.startswith(("http://", "https://")):
        try:
            return pd.read_html(path_or_url)[0]
        except ValueError as error:
            raise RuntimeError("Unable to parse the published spreadsheet as HTML table.") from error
    return pd.read_excel(path_or_url) if path_or_url.endswith((".xlsx", ".xls")) else pd.read_csv(path_or_url)


def attach_difficulty(disciplines: pd.DataFrame, difficulty_sheet: pd.DataFrame, *, code_column: str = "codigo", difficulty_column: str = "dificuldade") -> pd.DataFrame:
    normalized = difficulty_sheet.copy()
    normalized[code_column] = normalized[code_column].map(normalize_course_code)
    merged = disciplines.merge(normalized[[code_column, difficulty_column]], on=code_column, how="left")
    return merged


def save_processed_table(disciplines: pd.DataFrame, filename: str = "disciplinas_processadas.parquet") -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / filename
    disciplines.to_parquet(output_path, index=False)
    return output_path


def build_discipline_table(records: Iterable[DisciplineRecord]) -> pd.DataFrame:
    dataframe = pd.DataFrame([record.to_dict() for record in records])
    if not dataframe.empty:
        dataframe["codigo"] = dataframe["codigo"].map(normalize_course_code)
        dataframe["nome"] = dataframe["nome"].map(normalize_course_name)
    return dataframe
