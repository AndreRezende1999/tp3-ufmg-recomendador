# TP3 UFMG - Recomendador de Próximas Disciplinas

Projeto para o Trabalho Prático III de Fundamentos de Inteligência Artificial (UFMG).

## Estrutura

- `src/scraping.py`: coleta e cache das fontes públicas da UFMG.
- `src/data_curation.py`: reconciliação das fontes e montagem da tabela única de disciplinas.
- `src/synthetic_profile_generator.py`: geração dos perfis sintéticos e dos pares de treino.
- `src/baseline_no_text.py`: baseline estrutural sem texto.
- `src/train_bert_classifier.py`: fine-tuning local de BERT em pares de textos.
- `src/evaluate.py`: métricas, matriz de confusão e comparação entre modelos.
- `src/recommender_rules.py`: filtro de pré-requisitos e balanceamento de dificuldade.
- `notebooks/TP3_Classificador_Disciplinas_UFMG.ipynb`: notebook final autossuficiente.

## Dados necessários

- PDFs e HTML oficiais da UFMG, baixados localmente e cacheados.
- Planilha de dificuldade das disciplinas. O link informado na conversa é um Google Sheets publicado; ele pode ser usado como fonte, mas o pipeline espera que o arquivo seja baixado e cacheado localmente antes do processamento.

## Próximos passos

1. Baixar e cachear as fontes públicas.
2. Extrair e normalizar as disciplinas.
3. Gerar perfis sintéticos e pares rotulados.
4. Treinar o baseline e o BERT localmente.
5. Integrar tudo no notebook final.
