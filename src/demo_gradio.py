"""Interface Gradio para o Recomendador de Disciplinas — TP3 FIA UFMG."""
from __future__ import annotations

import gradio as gr
import pandas as pd
import numpy as np

from src.config import MODEL_NAME
from src.data_curation import get_discipline_table, get_completed_profile_text
from src.sbert_encoder import load_sbert_model, encode_texts, encode_disciplines, semantic_search
from src.recommender_rules import (
    filter_recommended_candidates,
    greedy_balance_by_difficulty,
    get_schedule_slots,
)

_model = None
_corpus_embeddings = None
_df = None


def _ensure_loaded():
    global _model, _corpus_embeddings, _df
    if _model is None:
        _df = get_discipline_table()
        _model = load_sbert_model()
        _corpus_embeddings = encode_disciplines(_model, _df)


def recommend(interesse: str, concluidas_str: str, budget: float, max_cursos: int, top_k: int):
    _ensure_loaded()

    concluidas = [c.strip().upper() for c in concluidas_str.replace(",", " ").split() if c.strip()]
    valid_codes = set(_df["codigo"].tolist())
    concluidas = [c for c in concluidas if c in valid_codes]

    if not interesse.strip():
        return "Digite seus interesses.", "", ""

    perfil_texto = get_completed_profile_text(_df, concluidas) if concluidas else "Novo estudante."
    query = f"{perfil_texto} {interesse}"
    query_emb = encode_texts(_model, [query], batch_size=1)[0]
    hits = semantic_search(query_emb, _corpus_embeddings, top_k=top_k)

    rows = []
    for hit in hits:
        row = _df.iloc[hit["index"]].copy()
        row["predicted_probability"] = hit["score"]
        row["difficulty"] = row["dificuldade_geral"]
        rows.append(row)
    df_cand = pd.DataFrame(rows)
    df_cand = df_cand[~df_cand["codigo"].isin(concluidas)]
    df_validas = filter_recommended_candidates(df_cand, concluidas)
    df_final = greedy_balance_by_difficulty(df_validas, difficulty_budget=budget, max_courses=max_cursos, respect_schedule=True)

    # Top-10 semantic ranking (for transparency)
    top10_lines = []
    for _, row in df_cand.head(10).iterrows():
        diff = row.get("dificuldade_geral")
        diff_str = f"{diff:.1f}" if pd.notna(diff) else "—"
        top10_lines.append(f"{row['predicted_probability']:.3f}  |  {diff_str}  |  {row['codigo']} — {row['nome']}")
    top10_text = "\n".join(top10_lines)

    # Final recommendation
    if df_final.empty:
        final_text = "Nenhuma disciplina atende a todos os filtros (pré-requisitos + dificuldade + horários). Tente aumentar o orçamento de dificuldade."
        schedule_text = ""
    else:
        lines = []
        schedule_lines = []
        for _, row in df_final.iterrows():
            diff = row.get("dificuldade_geral")
            diff_str = f"{diff:.1f}" if pd.notna(diff) else "—"
            lines.append(f"**{row['codigo']}** — {row['nome']}  |  Score: {row['predicted_probability']:.3f}  |  Dificuldade: {diff_str}")
            slots = get_schedule_slots(row)
            if slots:
                for d, t in slots:
                    schedule_lines.append(f"{d} {t}  →  {row['codigo']} ({row['nome'][:40]})")
        final_text = "\n\n".join(lines)
        schedule_text = "\n".join(sorted(schedule_lines))

    diff_total = df_final["difficulty"].sum() if len(df_final) > 0 else 0
    status = (
        f"**{len(df_validas)}**/{len(df_cand)} passaram nos pré-requisitos  |  "
        f"**{len(df_final)}** recomendadas  |  "
        f"Dificuldade total: **{diff_total:.1f}**/{budget:.0f}"
    )

    return status, top10_text, final_text, schedule_text


# --- CSS customizado ---
custom_css = """
.gradio-container { max-width: 900px !important; }
h1 { text-align: center; }
"""

with gr.Blocks(css=custom_css, title="Recomendador de Disciplinas — Eng. Sistemas UFMG") as demo:
    gr.Markdown(
        "# 🎓 Recomendador de Disciplinas\n"
        "### Engenharia de Sistemas — UFMG\n"
        "Digite seus interesses em linguagem natural e selecione as disciplinas já concluídas. "
        "O sistema usa **Sentence-BERT** para busca semântica + filtros de **pré-requisitos**, "
        "**dificuldade** e **horários** para sugerir a grade ideal."
    )

    with gr.Row():
        with gr.Column(scale=2):
            interesse = gr.Textbox(
                label="Seus interesses (linguagem natural)",
                placeholder="Ex: Gosto muito de inteligência artificial, machine learning e análise de dados...",
                lines=2,
            )
            concluidas = gr.Textbox(
                label="Disciplinas já concluídas (códigos separados por espaço ou vírgula)",
                placeholder="Ex: MAT001 MAT038 QUI628 DCC203 DCC217 MAT039 FIS065",
                lines=1,
            )

        with gr.Column(scale=1):
            budget = gr.Slider(5, 40, value=22, step=1, label="Orçamento de dificuldade")
            max_cursos = gr.Slider(1, 8, value=5, step=1, label="Máximo de disciplinas")
            top_k = gr.Slider(5, 40, value=20, step=5, label="Top-K busca semântica")
            btn = gr.Button("🔍 Recomendar", variant="primary")

    status = gr.Markdown("")

    with gr.Tabs():
        with gr.TabItem("📋 Recomendação Final"):
            resultado = gr.Markdown("As disciplinas recomendadas aparecerão aqui.", label="Grade sugerida")
        with gr.TabItem("📊 Ranking Semântico (Top-10)"):
            ranking = gr.Textbox(label="Top-10 por similaridade semântica (antes dos filtros)", lines=11, interactive=False)
        with gr.TabItem("🕐 Horários"):
            horarios = gr.Textbox(label="Grade horária sugerida (sem conflitos)", lines=11, interactive=False)

    btn.click(
        fn=recommend,
        inputs=[interesse, concluidas, budget, max_cursos, top_k],
        outputs=[status, ranking, resultado, horarios],
    )

if __name__ == "__main__":
    demo.launch()
