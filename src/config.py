"""Configurações centrais do projeto."""
from __future__ import annotations
from pathlib import Path

# Diretórios
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Arquivos de dados curados
DISCIPLINAS_JSON = DATA_DIR / "disciplinas_engsis.json"
DIFICULDADE_CSV = DATA_DIR / "dificuldade_engsis.csv"

# Modelo SBERT
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
RANDOM_STATE = 42
MAX_SEQ_LENGTH = 256
