# app/usecases/embedding/usecase_train_embedding.py
import traceback
from typing import Any

from app.services.logger import logger
from app.services.bdd.models.model_metrics import MetricsModel
from app.neural_network.nn_embedding import train_embedding_model
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
        model_data = bdd.get_model(model_name)

        if not model_data:
            raise Exception(f"Model '{model_name}' not found")
        
        vocab_size = len(vocab)

        logger.info(f"[train_embedding_usecase] Starting training for model: {model_name}")
        logger.info(f"[train_embedding_usecase] Vocab size: {vocab_size}, Train samples: {len(trainset)}")

        nn_model, train_stats = train_embedding_model(
            trainset=trainset,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate
        )

        bdd.update_model(ModelData(
            name=model_name,
            neural_network_type="EMBEDDING",
            status=ModelStatus.TRAINED,
            glossary=model_data.glossary,
            dictionary=model_data.dictionary,
            indexed_dictionary=model_data.indexed_dictionary,
            nn_model=nn_model
        ))

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
        logger.error(f"[train_embedding_usecase] Error: {str(e)}\n{traceback.format_exc()}")
        raise Exception(f"[train_embedding_usecase] {str(e)}")
