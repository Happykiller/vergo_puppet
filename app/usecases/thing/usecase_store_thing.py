# app/usecases/usecase_store_thing.py
import re
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
    Encodes text into embedding using a UniversalEmbeddingModel.
    Assumes model.glossary and model.nn_model are present.
    """
    glossary = model.glossary
    if not glossary or not model.nn_model:
        raise ValueError("Model glossary or nn_model is missing")

    vocab = {token: idx for idx, token in enumerate(glossary)}
    if "<UNK>" not in vocab:
        raise ValueError("Model glossary missing <UNK> token")

    def tokenize(s: str) -> list[str]:
        return re.findall(r"\b\w+\b", s.lower())

    tokens = tokenize(text)
    indices = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
    if not indices:
        raise ValueError("Text produced no tokens after tokenization")

    seq = torch.tensor(indices, dtype=torch.long).unsqueeze(0)
    lengths = torch.tensor([len(indices)])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.nn_model = model.nn_model.to(device)
    seq = seq.to(device)
    lengths = lengths.to(device)

    model.nn_model.eval()
    with torch.no_grad():
        embedding = model.nn_model.encode(seq, lengths)
        return embedding.squeeze(0).cpu().tolist()

def store_thing_usecase(model_name: str, collection_name: str, thing_id: str, data: dict, inversify: Inversify):
    try:
        if not thing_id or not isinstance(data, dict):
            raise ValueError("Missing 'id' or invalid 'data' payload.")

        text = flatten_for_embedding(data)

        model = get_model_usecase(model_name, inversify)
        if not model:
            raise ValueError(f"Model '{model_name}' not found")

        vector = encode_text_with_model(model, text)

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
            "collection": collection_name
        }

    except Exception as e:
        logger.error(f"[store_thing_usecase] Error: {str(e)}\n{traceback.format_exc()}")
        raise Exception(f"[store_thing_usecase] {str(e)}")
