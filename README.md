# TP3 FIA — Recomendador de Disciplinas (Eng. Sistemas UFMG)

ChatBot de recomendação de disciplinas para estudantes de Engenharia de Sistemas da UFMG, usando busca semântica com Sentence-BERT fine-tuned.

## Estrutura

- `data/disciplinas_engsis.json` — tabela curada de 80 disciplinas da grade 2026/1 (código, nome, ementa, período, pré-requisitos, dificuldade).
- `data/dificuldade_engsis.csv` — base da planilha de avaliação de dificuldade dos alunos.
- `src/config.py` — configurações centrais e caminhos.
- `src/data_curation.py` — carga e preparação dos dados unificados.
- `src/sbert_encoder.py` — codificação via SBERT e motor de busca semântica (cosseno).
- `src/train_bert_classifier.py` — pipeline para fine-tuning do SBERT e BERT-base.
- `src/synthetic_profile_generator.py` — gerador de perfis de estudo e pares sintéticos de treino.
- `src/baseline_no_text.py` — baseline estrutural sem processamento de texto.
- `src/evaluate.py` — métricas de avaliação e comparação.
- `src/recommender_rules.py` — regras rigorosas (pré-requisitos) e balanceamento de dificuldade para sugerir a grade ideal do semestre.
- `notebooks/TP3_Classificador_Disciplinas_UFMG.ipynb` — notebook principal interativo (seguindo o padrão das aulas práticas).

## Instalação e Execução

O projeto gerencia suas dependências localmente. Crie o ambiente, instale o pacote local e execute o Jupyter Notebook:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
jupyter notebook notebooks/TP3_Classificador_Disciplinas_UFMG.ipynb
```

O download dos pesos originais do `neuralmind/bert-base-portuguese-cased` ocorrerá de forma automática na primeira execução (requer internet).

## Arquitetura de Recomendação

O fluxo adota as seguintes etapas:
1. Um input natural ("Gosto muito de IA e dados") é convertido para embedding usando Sentence-BERT.
2. É calculada a similaridade semântica (Cosseno) contra os vetores pré-computados das 80 disciplinas.
3. As top N disciplinas mais similares passam por validação de **Pré-Requisitos** baseados no PPC e filtro guloso de **Dificuldade** baseado em avaliações estudantis, gerando a recomendação ideal para o estudante no momento.
