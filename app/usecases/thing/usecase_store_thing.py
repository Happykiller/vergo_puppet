# app/usecases/usecase_store_thing.py
import traceback

from app.inversify import Inversify
from app.services.logger import logger
from app.usecases.get_model import get_model_usecase
from app.services.bdd.models.model_thing import ThingModel
from app.usecases.embedding.usecase_encode_embedding import encode_embedding_usecase

def flatten_for_embedding(obj: dict) -> str:
    """
    Converts an object into a flat string for embedding.
    Handles nested fields like `cb`, `credential`, etc.

    :param obj: Dictionary with raw data
    :return: Concatenated string for embedding
    """
    parts = []

    for key, value in obj.items():
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, dict):
            parts.extend(str(v) for v in value.values() if isinstance(v, str))
        # Skip nulls and other types

    return " ".join(parts).strip()

def store_thing_usecase(model_name: str, collection_name: str, thing_id: str, data: dict, inversify: Inversify):
    """Store a thing vectorized with the specified embedding model."""
    try:
        if not thing_id or not isinstance(data, dict):
            raise ValueError("Missing 'id' or invalid 'data' payload.")
        
        if "label" not in data or not isinstance(data["label"], str):
            raise ValueError("Missing required field 'label'")

        text = flatten_for_embedding(data)

        model = get_model_usecase(model_name, inversify)
        if not model:
            raise ValueError(f"Model '{model_name}' not found")

        vector = encode_embedding_usecase(model_name, text, inversify)

        thing = ThingModel(
            id=thing_id,
            vector=vector,
            text=text,
            metadata=data,
            collection_name=collection_name
        )
        bdd = inversify.get_bdd()
        bdd.store_thing_embedding(thing)

        return {
            "status": "stored",
            "id": thing_id,
            "collection": collection_name,
            "text": text,
            "vector_dim": len(vector)
        }

    except Exception as e:
        logger.error(f"[store_thing_usecase] Error: {str(e)}\n{traceback.format_exc()}")
        raise Exception(f"[store_thing_usecase] {str(e)}")
