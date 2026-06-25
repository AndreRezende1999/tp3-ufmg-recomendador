"""Interface Streamlit para o recomendador de disciplinas do TP3 FIA UFMG."""
from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from src.data_curation import get_completed_profile_text, get_discipline_table
from src.recommender_rules import (
    filter_recommended_candidates,
    get_schedule_slots,
    greedy_balance_by_difficulty,
)
from src.sbert_encoder import encode_disciplines, encode_texts, load_sbert_model, semantic_search


@st.cache_resource(show_spinner="Carregando modelo SBERT e embeddings das disciplinas...")
def load_resources() -> tuple[pd.DataFrame, object, object]:
    df = get_discipline_table()
    model = load_sbert_model()
    embeddings = encode_disciplines(model, df)
    return df, model, embeddings


def parse_completed_codes(completed_text: str, valid_codes: set[str]) -> tuple[list[str], list[str]]:
    raw_codes = [code.strip().upper() for code in re.split(r"[\s,;]+", completed_text) if code.strip()]
    completed = [code for code in raw_codes if code in valid_codes]
    invalid = [code for code in raw_codes if code not in valid_codes]
    return completed, invalid


def build_recommendation_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        slots = get_schedule_slots(row)
        schedule = ", ".join([f"{day} {time}" for day, time in slots]) if slots else "-"
        difficulty = row.get("difficulty")
        rows.append(
            {
                "Codigo": row["codigo"],
                "Disciplina": row["nome"],
                "Score semantico": round(float(row["predicted_probability"]), 3),
                "Dificuldade": round(float(difficulty), 1) if pd.notna(difficulty) else None,
                "Horarios": schedule,
            }
        )
    return pd.DataFrame(rows)


def build_schedule_text(df: pd.DataFrame) -> str:
    schedule_lines = []
    for _, row in df.iterrows():
        for day, time in get_schedule_slots(row):
            schedule_lines.append(f"{day} {time} -> {row['codigo']} - {row['nome']}")
    return "\n".join(sorted(schedule_lines))


def recommend(
    interesse: str,
    concluidas_str: str,
    budget: float,
    max_cursos: int,
    top_k: int,
) -> dict[str, object]:
    df, model, corpus_embeddings = load_resources()
    valid_codes = set(df["codigo"].tolist())
    concluidas, invalid_codes = parse_completed_codes(concluidas_str, valid_codes)

    if not interesse.strip():
        return {
            "status": "Digite seus interesses para iniciar a recomendacao.",
            "invalid_codes": invalid_codes,
            "ranking_df": pd.DataFrame(),
            "final_df": pd.DataFrame(),
            "schedule_text": "",
        }

    perfil_texto = get_completed_profile_text(df, concluidas) if concluidas else "Novo estudante."
    query = f"{perfil_texto} {interesse}"
    query_emb = encode_texts(model, [query], batch_size=1)[0]
    hits = semantic_search(query_emb, corpus_embeddings, top_k=top_k)

    candidates = []
    for hit in hits:
        row = df.iloc[hit["index"]].copy()
        row["predicted_probability"] = hit["score"]
        row["difficulty"] = row.get("dificuldade_geral", float("nan"))
        candidates.append(row)

    df_cand = pd.DataFrame(candidates)
    if df_cand.empty:
        return {
            "status": "Nenhuma disciplina encontrada na busca semantica.",
            "invalid_codes": invalid_codes,
            "ranking_df": pd.DataFrame(),
            "final_df": pd.DataFrame(),
            "schedule_text": "",
        }

    df_cand = df_cand[~df_cand["codigo"].isin(concluidas)].copy()
    df_validas = filter_recommended_candidates(df_cand, concluidas)
    df_final = greedy_balance_by_difficulty(
        df_validas,
        difficulty_budget=budget,
        max_courses=max_cursos,
        respect_schedule=True,
    )

    ranking_df = build_recommendation_table(df_cand.head(10))
    final_df = build_recommendation_table(df_final) if not df_final.empty else pd.DataFrame()
    difficulty_total = df_final["difficulty"].sum() if not df_final.empty else 0.0
    status = (
        f"{len(df_validas)}/{len(df_cand)} candidatos passaram nos pre-requisitos. "
        f"{len(df_final)} disciplinas recomendadas. "
        f"Dificuldade total: {difficulty_total:.1f}/{budget:.0f}."
    )

    return {
        "status": status,
        "invalid_codes": invalid_codes,
        "ranking_df": ranking_df,
        "final_df": final_df,
        "schedule_text": build_schedule_text(df_final),
    }


def render_flow() -> None:
    st.subheader("Fluxo simples")
    cols = st.columns(3)
    steps = [
        ("1. Perfil", "Informe interesses em texto livre e as disciplinas ja concluidas."),
        ("2. Busca", "O SBERT compara seu perfil com as ementas das disciplinas."),
        ("3. Filtros", "O app remove choques de horario, pre-requisitos pendentes e excesso de dificuldade."),
    ]
    for col, (title, description) in zip(cols, steps):
        with col:
            st.markdown(f"**{title}**")
            st.caption(description)


def main() -> None:
    st.set_page_config(page_title="Recomendador de Disciplinas UFMG", layout="wide")

    st.title("Recomendador de Disciplinas - Engenharia de Sistemas UFMG")
    st.write(
        "Interface Streamlit integrada ao classificador do notebook "
        "`TP3_Classificador_Disciplinas_UFMG.ipynb`."
    )
    render_flow()

    with st.sidebar:
        st.header("Entrada do aluno")
        with st.form("recommendation_form"):
            interesse = st.text_area(
                "Interesses",
                value="inteligencia artificial, aprendizado de maquina e analise de dados",
                height=120,
            )
            concluidas_str = st.text_input(
                "Disciplinas concluidas",
                value="MAT001 MAT038 QUI628 DCC203 DCC217 MAT039 FIS065",
                help="Use codigos separados por espaco, virgula ou ponto e virgula.",
            )
            st.header("Parametros")
            budget = st.slider("Orcamento de dificuldade", min_value=5, max_value=40, value=22, step=1)
            max_cursos = st.slider("Maximo de disciplinas", min_value=1, max_value=8, value=5, step=1)
            top_k = st.slider("Top-K da busca semantica", min_value=5, max_value=40, value=20, step=5)
            submitted = st.form_submit_button("Recomendar", type="primary")

    if not submitted:
        st.info("Ajuste as entradas na barra lateral e clique em Recomendar.")
        return

    result = recommend(
        interesse=interesse,
        concluidas_str=concluidas_str,
        budget=budget,
        max_cursos=max_cursos,
        top_k=top_k,
    )

    invalid_codes = result["invalid_codes"]
    if invalid_codes:
        st.warning("Codigos ignorados por nao existirem na base: " + ", ".join(invalid_codes))

    st.success(result["status"])

    final_df = result["final_df"]
    ranking_df = result["ranking_df"]
    schedule_text = result["schedule_text"]

    tab_final, tab_ranking, tab_schedule = st.tabs(["Recomendacao final", "Ranking semantico", "Horarios"])
    with tab_final:
        if isinstance(final_df, pd.DataFrame) and not final_df.empty:
            st.dataframe(final_df, use_container_width=True, hide_index=True)
        else:
            st.warning("Nenhuma recomendacao passou pelos filtros. Tente aumentar o orcamento ou o Top-K.")

    with tab_ranking:
        if isinstance(ranking_df, pd.DataFrame) and not ranking_df.empty:
            st.dataframe(ranking_df, use_container_width=True, hide_index=True)
        else:
            st.info("Nenhum ranking disponivel.")

    with tab_schedule:
        if schedule_text:
            st.code(schedule_text)
        else:
            st.info("Nenhum horario a exibir.")


if __name__ == "__main__":
    main()
