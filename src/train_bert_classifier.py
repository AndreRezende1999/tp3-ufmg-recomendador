"""Fine-tuning do BERT classificador e do Sentence-BERT."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup

from .config import MODEL_NAME


# ===================================================================
# Dataset de pares (perfil, candidata) para o classificador BERT
# ===================================================================

class PairTextDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_length_profile: int = 384, max_length_candidate: int = 192):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length_profile = max_length_profile
        self.max_length_candidate = max_length_candidate

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.dataframe.iloc[index]
        encoded = self.tokenizer(
            str(row["perfil_texto"]),
            str(row["candidate_text"]),
            truncation=True,
            padding="max_length",
            max_length=max(self.max_length_profile, self.max_length_candidate),
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(int(row["label"]), dtype=torch.long)
        return item


# ===================================================================
# Estágios de treino progressivo
# ===================================================================

@dataclass
class TrainingStage:
    name: str
    trainable_layers: str
    epochs: int
    learning_rate: float


def freeze_encoder(model: BertForSequenceClassification) -> None:
    """Congela todos os parâmetros do encoder BERT."""
    for parameter in model.bert.parameters():
        parameter.requires_grad = False


def unfreeze_last_layers(model: BertForSequenceClassification, num_layers: int = 2) -> None:
    """Descongela as últimas N camadas do encoder + classificador."""
    freeze_encoder(model)
    encoder_layers = model.bert.encoder.layer
    for layer in encoder_layers[-num_layers:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    if hasattr(model, "dropout"):
        for parameter in model.dropout.parameters():
            parameter.requires_grad = True


def unfreeze_all(model: BertForSequenceClassification) -> None:
    """Descongela todos os parâmetros."""
    for parameter in model.parameters():
        parameter.requires_grad = True


def create_model(num_labels: int = 2) -> BertForSequenceClassification:
    """Cria modelo BertForSequenceClassification a partir do BERTimbau."""
    return BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)


def create_tokenizer():
    """Cria tokenizer do BERTimbau."""
    return AutoTokenizer.from_pretrained(MODEL_NAME)


# ===================================================================
# Loop de treino e avaliação
# ===================================================================

def train_stage(model, data_loader, optimizer, scheduler, device):
    """Executa uma época de treino."""
    model.train()
    total_loss = 0.0
    for batch in data_loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += float(loss.item())
    return total_loss / max(1, len(data_loader))


def predict(model, data_loader, device):
    """Gera predições e probabilidades."""
    model.eval()
    predictions: list[int] = []
    labels: list[int] = []
    probabilities: list[float] = []
    with torch.no_grad():
        for batch in data_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            batch_labels = batch.pop("labels")
            outputs = model(**batch)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            predictions.extend(torch.argmax(probs, dim=-1).cpu().tolist())
            probabilities.extend(probs[:, 1].cpu().tolist())
            labels.extend(batch_labels.cpu().tolist())
    return labels, predictions, probabilities


def evaluate_model(model, data_loader, device):
    """Avalia o modelo no data_loader e retorna métricas."""
    labels, predictions, probabilities = predict(model, data_loader, device)
    return {
        "labels": labels,
        "predictions": predictions,
        "probabilities": probabilities,
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions)),
    }


def fit_three_stage_model(train_df: pd.DataFrame, validation_df: pd.DataFrame, *, batch_size: int = 8, seed: int = 42) -> tuple[BertForSequenceClassification, list[dict[str, float]]]:
    """Fine-tuning progressivo em 3 estágios: head → últimas camadas → full."""
    torch.manual_seed(seed)
    tokenizer = create_tokenizer()
    model = create_model(num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    training_stages = [
        TrainingStage(name="feature_extraction", trainable_layers="head_only", epochs=1, learning_rate=5e-5),
        TrainingStage(name="partial_unfreeze", trainable_layers="last_two_layers", epochs=1, learning_rate=3e-5),
        TrainingStage(name="full_finetuning", trainable_layers="all", epochs=1, learning_rate=2e-5),
    ]

    history: list[dict[str, float]] = []
    train_dataset = PairTextDataset(train_df, tokenizer)
    validation_dataset = PairTextDataset(validation_df, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)

    for stage in training_stages:
        if stage.trainable_layers == "head_only":
            freeze_encoder(model)
        elif stage.trainable_layers == "last_two_layers":
            unfreeze_last_layers(model, num_layers=2)
        else:
            unfreeze_all(model)

        optimizer = torch.optim.AdamW(filter(lambda parameter: parameter.requires_grad, model.parameters()), lr=stage.learning_rate)
        total_steps = max(1, len(train_loader) * stage.epochs)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, total_steps // 10), num_training_steps=total_steps)

        for _ in range(stage.epochs):
            loss = train_stage(model, train_loader, optimizer, scheduler, device)
            evaluation = evaluate_model(model, validation_loader, device)
            history.append({"stage": stage.name, "loss": loss, "accuracy": evaluation["accuracy"], "f1": evaluation["f1"]})

    return model, history


# ===================================================================
# Fine-tuning do Sentence-BERT com CosineSimilarityLoss
# ===================================================================

def fine_tune_sbert(
    train_pairs: pd.DataFrame,
    validation_pairs: pd.DataFrame,
    model_name: str | None = None,
    output_path: str | Path = "models/sbert-finetuned",
    epochs: int = 3,
    batch_size: int = 16,
) -> "SentenceTransformer":
    """Fine-tuna o SBERT com pares (text_a, text_b, label_score) via CosineSimilarityLoss.

    Espera DataFrames com colunas: text_a, text_b, label (float 0.0–1.0).
    """
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
    from torch.utils.data import DataLoader as STDataLoader

    name = model_name or MODEL_NAME
    model = SentenceTransformer(name)

    # Monta exemplos de treino
    train_examples = [
        InputExample(texts=[str(row["text_a"]), str(row["text_b"])],
                     label=float(row["label"]))
        for _, row in train_pairs.iterrows()
    ]
    train_dataloader = STDataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.CosineSimilarityLoss(model)

    # Avaliador de validação
    evaluator = None
    if not validation_pairs.empty:
        evaluator = EmbeddingSimilarityEvaluator(
            sentences1=validation_pairs["text_a"].astype(str).tolist(),
            sentences2=validation_pairs["text_b"].astype(str).tolist(),
            scores=validation_pairs["label"].astype(float).tolist(),
            name="validacao",
        )

    # Treino
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        output_path=str(output_path),
        show_progress_bar=True,
    )
    return model
