# TP3 FIA — Recomendador de Disciplinas (Eng. Sistemas UFMG)

Sistema de recomendação de disciplinas para Engenharia de Sistemas da UFMG com busca semântica (Sentence-BERT), classificador BERT fine-tuned, e filtros estruturais de pré-requisitos, balanceamento de dificuldade e prevenção de conflitos de horário.

## Estrutura

- `data/disciplinas_engsis.json` — 76 disciplinas da grade 2026/1 com ementas do GEES UFMG, horários oficiais e métricas de dificuldade.
- `data/dificuldade_engsis.csv` — planilha colaborativa de avaliação de dificuldade dos alunos.
- `scripts/build_discipline_data.py` — pipeline de construção do dataset (PDF → JSON enriquecido).
- `src/config.py` — configurações centrais e caminhos.
- `src/data_curation.py` — carga e preparação dos dados unificados.
- `src/sbert_encoder.py` — codificação via SBERT e motor de busca semântica (cosseno).
- `src/train_bert_classifier.py` — fine-tuning progressivo do BERT em 3 estágios.
- `src/synthetic_profile_generator.py` — gerador de perfis de estudo e pares sintéticos de treino.
- `src/baseline_no_text.py` — baseline estrutural (regressão logística).
- `src/evaluate.py` — métricas de avaliação, matriz de confusão e comparação entre modelos.
- `src/recommender_rules.py` — filtros de pré-requisitos, dificuldade e horários.
- `src/demo_gradio.py` — interface interativa com Gradio.
- `src/demo_streamlit.py` — interface Streamlit com fluxo simples de recomendação.
- `notebooks/TP3_Classificador_Disciplinas_UFMG.ipynb` — notebook principal com 6 cenários de recomendação.
- `notebooks/TP3_Colab.ipynb` — versão auto-suficiente para Google Colab.
- `artigo/artigo_tp3.tex` — artigo acadêmico (LaTeX).

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

No Windows/PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

## Execução

```bash
# Notebook interativo
jupyter notebook notebooks/TP3_Classificador_Disciplinas_UFMG.ipynb

# Interface Gradio
python src/demo_gradio.py

# Interface Streamlit
streamlit run src/demo_streamlit.py

# Reconstruir dataset (requer PDF de oferta e CSV de horários)
python scripts/build_discipline_data.py
```

### Rodando o Streamlit no Windows

Se estiver usando PowerShell, rode os comandos abaixo a partir da pasta do projeto:

```powershell
cd C:\Users\archa\OneDrive\Documentos\GuaraTrainer\TP-FundAI\TP-IA\tp3-ufmg-recomendador
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m streamlit run src\demo_streamlit.py
```

Depois abra no navegador:

```text
http://localhost:8501
```

Importante: use o Python da pasta `.venv`. Se executar `streamlit run ...` pelo Python global, pode aparecer erro como `ModuleNotFoundError: No module named 'torch'`. No primeiro clique em **Recomendar**, o app pode demorar um pouco porque carrega o modelo SBERT e gera os embeddings.

### Rodando o Streamlit no macOS

No Terminal, rode os comandos abaixo a partir da pasta do projeto:

```bash
cd /caminho/para/tp3-ufmg-recomendador
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
python -m streamlit run src/demo_streamlit.py
```

Depois abra no navegador:

```text
http://localhost:8501
```

Se o ambiente virtual ja existir, basta ativar e rodar:

```bash
source .venv/bin/activate
python -m streamlit run src/demo_streamlit.py
```

## Arquitetura

1. **Curadoria** — 76 disciplinas com ementas do site GEES UFMG, horários da grade oficial, e dificuldade de planilha estudantil.
2. **SBERT (neuralmind/bert-base-portuguese-cased)** — embeddings de 768 dimensões para busca semântica por similaridade de cosseno.
3. **Filtros Estruturais** — pré-requisitos → balanceamento guloso de dificuldade → prevenção de conflitos de horário.
4. **BERT Classificador** — fine-tuning progressivo em 3 estágios (Feature Extraction → Partial Unfreeze → Full Fine-Tuning).
5. **Baseline** — regressão logística sobre features estruturais para comparação.
