# app/usecases/embedding/usecase_create_embedding.py
import traceback
from typing import Any

from app.services.logger import logger
from app.services.bdd.models.model_metrics import MetricsModel
from app.services.bdd.models.model_data import ModelData, ModelStatus
from app.usecases.embedding.usecase_train_embedding import train_embedding_usecase

def create_embedding_usecase(
    model_name: str,
    vocab: dict,
    trainset: list[dict],
    inversify: Any,
) -> dict:
    """
    Create and train a universal embedding model using provided vocabulary and trainset.
    Persists the model via the BDD service.
    """
    try:
        logger.info(f"[create_embedding_usecase] Starting for model: {model_name}")

        # Load database service
        bdd = inversify.get_bdd()

        if bdd.model_exists(model_name):
            raise Exception("Model already exists")

        # Initial save: status CREATED
        bdd.save_model(ModelData(
            name=model_name,
            neural_network_type="EMBEDDING",
            status=ModelStatus.CREATED,
            glossary=list(vocab.keys()),
        ))

        # Launch training
        training_report = train_embedding_usecase(model_name, vocab, trainset, inversify)

        return {
            "status": "model created and trained",
            "model": model_name,
            "training_stats": training_report
        }

    except Exception as e:
        logger.error(f"[create_embedding_usecase] Error: {str(e)}\n{traceback.format_exc()}")
        raise Exception(f"[create_embedding_usecase] {str(e)}")
