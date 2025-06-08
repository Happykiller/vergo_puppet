# app\usecases\embedding\usecase_encode_embedding.py
import torch
import traceback
from typing import Any
from pathlib import Path

from app.services.logger import logger
from app.neural_network.nn_embedding import UniversalEmbeddingModel

def encode_embedding_usecase(
    model_name: str,
    sentence: str,
    inversify: Any,
) -> list[float]:
    """
    Encodes a sentence into its embedding vector using a trained model.
    :param model_name: Name of the model to load
    :param vocab: Token-to-index mapping
    :param sentence: The sentence to encode
    :return: List of floats (embedding)
    """
    try:
        # 1. Load the model
        bdd = inversify.get_bdd()
        model = bdd.get_model(model_name, UniversalEmbeddingModel)

        if not model:
            raise Exception(f"Model '{model_name}' not found")
        
        glossary = model.glossary
        nn_model = model.nn_model
        if not nn_model:
            raise Exception("Model not completed")
        
        vocab = {token: idx for idx, token in enumerate(glossary)}
        
        # 2. Tokenize and convert to indices
        import re
        def simple_tokenize(text):
            return re.findall(r"\b\w+\b", text.lower())
        indices = [vocab.get(token, vocab["<UNK>"]) for token in simple_tokenize(sentence)]
        if not indices:
            raise ValueError("Input sentence produced no valid tokens.")

        # 3. Prepare tensors
        seq = torch.tensor(indices, dtype=torch.long).unsqueeze(0)  # batch=1
        lengths = torch.tensor([len(indices)])
        nn_model.eval()
        with torch.no_grad():
            embedding = nn_model.encode(seq, lengths)
            return embedding.squeeze(0).cpu().tolist()
    except Exception as e:
        logger.error(f"[encode_embedding_usecase] Error: {str(e)}\n{traceback.format_exc()}")
        raise Exception(f"[encode_embedding_usecase] {str(e)}")
