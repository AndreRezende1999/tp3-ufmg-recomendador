# Recomendador de Disciplinas — Engenharia de Sistemas UFMG

Interface Streamlit do sistema de recomendação de disciplinas com busca semântica (Sentence-BERT) e filtros estruturais de pré-requisitos, balanceamento de dificuldade e prevenção de conflitos de horário.

## Como rodar

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .\.venv\Scripts\Activate.ps1 no Windows

pip install -r requirements.txt
streamlit run app.py
```

Abra `http://localhost:8501` no navegador.

Na primeira execução, o modelo SBERT (`neuralmind/bert-base-portuguese-cased`) é baixado do HuggingFace (~500 MB) e os embeddings das 76 disciplinas são gerados. Isso leva alguns segundos e ocorre apenas uma vez.

## Uso

1. Na barra lateral, digite seus interesses em linguagem natural (ex.: "inteligência artificial e análise de dados").
2. Informe os códigos das disciplinas já concluídas, separados por espaço ou vírgula.
3. Ajuste o orçamento de dificuldade e o número máximo de disciplinas.
4. Clique em **Recomendar**.

O sistema exibe três abas: recomendação final (após filtros), ranking semântico (top-10) e grade horária.
