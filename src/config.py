from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic_pairs"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "TP3_Classificador_Disciplinas_UFMG.ipynb"

UFMG_SOURCES = {
    "ppc": "https://www.ufmg.br/prograd/arquivos/cursos/projeto/PPCEngenhariaDeSistemas.pdf",
    "dee": "https://www.eng.ufmg.br/portal/graduacao/cursos/engenharia-sistemas/",
    "gees": "https://geesufmg.com/o-curso/grade-curricular",
    "curriculum": "https://www.ufmg.br/prograd/estrutura-curricular-da-graduacao/",
    "normas": "https://www.ufmg.br/prograd/arquivos/docs/normasGraduacao.pdf",
}

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
RANDOM_STATE = 42
MAX_PROFILE_TOKENS = 384
MAX_CANDIDATE_TOKENS = 192
DEFAULT_USER_AGENT = "TP3-UFMG-Recomendador/0.1 (educational project; local cache only)"
