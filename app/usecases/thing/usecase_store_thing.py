# app/usecases/usecase_store_thing.py
import torch
import traceback

from app.inversify import Inversify
from app.services.logger import logger
from app.usecases.get_model import get_model_usecase
from app.services.bdd.models.model_thing import ThingModel

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

def encode_text_with_model(model, text: str) -> list[float]:
    """
    Encodes text using a trained NN model (e.g., Siamese LSTM), returns a vector.
    You need to adapt this part for your tokenization and model signature.
    """
    # Tokenization logic: replace with your real tokenizer (e.g., spacy, custom, etc.)
    tokens = text.split()  # Replace with real tokenizer if necessary
    # Suppose model has a `word2idx` glossary and an `encode` method
    indices = [model.word2idx.get(t, 1) for t in tokens]  # 1 = UNK (unknown token)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.nn_model = model.nn_model.to(device)
    model.nn_model.eval()
    with torch.no_grad():
        seq = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
        lengths = torch.tensor([len(indices)]).to(device)
        # Méthode à adapter selon le modèle (forward_once ou autre)
        embedding = model.nn_model.forward_once(seq, lengths)
        return embedding.squeeze(0).cpu().tolist()


def store_thing_usecase(item: dict, model_name: str, inversify: Inversify):
    """
    Flattens the object, encodes as an embedding using the designated model, and stores in the database.
    """
    try:
        if "id" not in item or not item.get("label"):
            raise ValueError("Missing required field: 'id' or 'label'")

        text = flatten_for_embedding(item)

        # 1. Récupération du modèle
        model = get_model_usecase(model_name, inversify)
        if not model:
            raise ValueError(f"Model '{model_name}' not found")

        # 2. Génération de l'embedding
        vector = encode_text_with_model(model, text)

        # 3. Création du ThingModel et stockage
        thing = ThingModel(
            id=item["id"],
            vector=vector,
            text=text,
            metadata=item
        )
        bdd = inversify.get_bdd()
        bdd.store_thing_embedding(thing)

        return {
            "status": "stored",
            "id": item["id"],
            "text": text,
            "vector_dim": len(vector)
        }

    except Exception as e:
        logger.error(f"Error message: {str(e)}\nStack trace:\n{traceback.format_exc()}")
        raise Exception(f"[store_thing_usecase] {str(e)}")
