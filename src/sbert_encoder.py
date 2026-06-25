"""Codificação de textos com Sentence-BERT e busca semântica por cosseno."""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from src.config import MODEL_NAME

def load_sbert_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """Carrega o modelo Sentence-BERT."""
    return SentenceTransformer(model_name)

def encode_texts(model: SentenceTransformer, texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Codifica uma lista de textos gerando embeddings."""
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    return embeddings

def encode_disciplines(model: SentenceTransformer, df: pd.DataFrame) -> np.ndarray:
    """Gera embeddings para o corpus completo de disciplinas."""
    from src.data_curation import get_discipline_text
    texts = df.apply(get_discipline_text, axis=1).tolist()
    return encode_texts(model, texts)

def semantic_search(query_embedding: np.ndarray, corpus_embeddings: np.ndarray, top_k: int = 10) -> list[dict]:
    """Realiza busca semântica calculando a similaridade de cosseno."""
    # sentence_transformers.util.cos_sim retorna tensores [num_queries, num_corpus]
    query_tensor = torch.tensor(query_embedding)
    corpus_tensor = torch.tensor(corpus_embeddings)
    
    hits = util.semantic_search(query_tensor, corpus_tensor, top_k=top_k)[0]
    # sentence_transformers returns 'corpus_id' instead of 'index'
    for hit in hits:
        hit['index'] = hit.pop('corpus_id')
    return hits

def save_embeddings(embeddings: np.ndarray, path: Path):
    """Salva os embeddings numpy no disco."""
    np.save(path, embeddings)

def load_embeddings(path: Path) -> np.ndarray:
    """Carrega embeddings numpy do disco."""
    return np.load(path)
