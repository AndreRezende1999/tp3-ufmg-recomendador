# TP3 — Recomendador de Disciplinas (Eng. Sistemas UFMG)

Sistema de recomendação de disciplinas com busca semântica (Sentence-BERT), classificador BERT fine-tuned e filtros estruturais de pré-requisitos, balanceamento de dificuldade e prevenção de conflitos de horário.

## Estrutura

```
artigo/               Artigo acadêmico (LaTeX + PDF)
data/                 Dataset (76 disciplinas com ementas, horários, dificuldade)
notebooks/            Notebook principal e versão autossuficiente para Colab
scripts/              Pipeline ETL de construção do dataset
src/                  Código-fonte do recomendador
  config.py           Configurações centrais
  data_curation.py    Carga e preparação dos dados
  sbert_encoder.py    Embeddings SBERT e busca semântica
  recommender_rules.py   Filtros de pré-requisitos, dificuldade e horários
  synthetic_profile_generator.py  Perfis sintéticos e pares de treino
  train_bert_classifier.py   Fine-tuning progressivo do BERT (3 estágios)
  baseline_no_text.py     Baseline estrutural (regressão logística)
  evaluate.py         Métricas de avaliação
  demo_streamlit.py   Interface interativa (Streamlit)
```

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
pip install -e .
```

No Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

## Streamlit

```bash
# Ative o ambiente virtual primeiro
source .venv/bin/activate      # Linux/macOS
# ou .\.venv\Scripts\Activate.ps1 no Windows

streamlit run src/demo_streamlit.py
```

Abra `http://localhost:8501` no navegador. No primeiro clique em **Recomendar**, o app carrega o modelo SBERT e gera os embeddings — isso leva alguns segundos.

A interface permite selecionar disciplinas já concluídas, descrever interesses em linguagem natural e ajustar o orçamento de dificuldade. O sistema retorna até 5 disciplinas ranqueadas com scores de similaridade, horários e alertas de conflito.

## Notebooks

```bash
jupyter notebook notebooks/TP3_Classificador_Disciplinas_UFMG.ipynb
```

Ou abra diretamente no Google Colab: `notebooks/TP3_Colab.ipynb` (autossuficiente, baixa os dados do GitHub).

## Artigo

O artigo acadêmico está em `artigo/artigo_tp3.tex` (LaTeX). Para compilar:

```bash
cd artigo
pdflatex artigo_tp3.tex
```
