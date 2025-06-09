# app/usecases/embedding/usecase_train_embedding.py
import traceback
from typing import Any
from pathlib import Path

from app.services.logger import logger
from app.services.bdd.models.model_metrics import MetricsModel
from app.neural_network.nn_embedding import (
    UniversalEmbeddingModel,
    train_embedding_model,
)
from app.services.bdd.models.model_data import ModelData, ModelStatus

def train_embedding_usecase(
    model_name: str,
    vocab: dict,
    trainset: list[dict],
    inversify: Any,
    embedding_dim: int = 128,
    lstm_hidden_dim: int = 128,
    num_epochs: int = 2,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
) -> dict:
    """
    Trains a universal embedding model using provided vocabulary and trainset.
    Saves the model using the BDD service.
    :param model_name: The name for the new model
    :param vocab: The token-to-index vocabulary (dict)
    :param trainset: List of training examples (dict with 'seq1', 'seq2', 'label')
    :param inversify: Dependency injection container (for BDD and storage)
    :return: Dict with training summary/statistics
    """
    try:
        bdd = inversify.get_bdd()
        vocab_size = len(vocab)

        model = bdd.get_model(model_name, UniversalEmbeddingModel)
        if not model:
            raise Exception("Model not found")

        if model.status in (ModelStatus.TRAINING, ModelStatus.SUPER_TRAINING):
            raise Exception("Model is training")

        model.status = ModelStatus.TRAINING
        bdd.update_model(model)

        logger.info(
            f"[train_embedding_usecase] Starting training for model: {model_name}"
        )
        logger.info(
            f"[train_embedding_usecase] Vocab size: {vocab_size}, Train samples: {len(trainset)}"
        )
        
        model_path = Path("models/embedding") / f"{model_name}.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)

        _, train_stats = train_embedding_model(
            trainset=trainset,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            save_path=str(model_path)
        )

        bdd.update_model(
            ModelData(
                name=model_name,
                neural_network_type="EMBEDDING",
                status=ModelStatus.TRAINED,
                glossary=list(vocab.keys()),
                model_path=str(model_path)
            )
        )

        bdd.save_metrics(MetricsModel(
            model_name=model_name,
            metrics={
                "type": "training",
                "training_stats": train_stats
            }
        ))
        
        # For now, just return statistics to test integration
        logger.info(f"[train_embedding_usecase] Training complete.")
        return {
            "model": model_name,
            "stats": train_stats
        }

    except Exception as e:
        try:
            # Optional: mark model as FAILED in DB
            model = bdd.get_model(model_name)
            if model:
                model.status = ModelStatus.FAILED
                bdd.update_model(model)
        except Exception as e2:
            logger.error(f"[train_embedding_usecase] Failed to update model status to FAILED: {str(e2)}")

        logger.error(f"[train_embedding_usecase] Error: {str(e)}\n{traceback.format_exc()}")
        raise Exception(f"[train_embedding_usecase] {str(e)}")
