"""Carga e tratamento dos dados curados do projeto."""
import pandas as pd
import json
from .config import DISCIPLINAS_JSON

def load_disciplines() -> pd.DataFrame:
    """Carrega os dados curados das disciplinas a partir do JSON."""
    with open(DISCIPLINAS_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    return df

def get_discipline_table() -> pd.DataFrame:
    """Retorna o DataFrame completo (neste caso, a curadoria já foi feita no builder)."""
    return load_disciplines()

def get_discipline_text(row: pd.Series) -> str:
    """Gera uma representação textual rica de uma disciplina para embedding."""
    nome = row.get("nome", "")
    area = row.get("area_conhecimento", "")
    ementa = row.get("ementa", "")
    return f"{nome}. Área: {area}. {ementa}"

def get_completed_profile_text(df: pd.DataFrame, completed_codes: list[str]) -> str:
    """Gera uma descrição textual agregada das disciplinas concluídas por um estudante."""
    text_parts = []
    for code in completed_codes:
        disc = df[df["codigo"] == code]
        if not disc.empty:
            nome = disc.iloc[0]["nome"]
            area = disc.iloc[0]["area_conhecimento"]
            text_parts.append(f"{nome} ({area})")
    return "Disciplinas cursadas com interesse: " + ", ".join(text_parts) + "."
