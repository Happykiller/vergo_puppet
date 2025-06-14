# app\usecases\embedding\usecase_encode_embedding.py
import re
import torch  # type: ignore
import traceback
import numpy as np  # type: ignore
from typing import Any
from pathlib import Path

from app.services.logger import logger
from app.neural_network.nn_embedding import UniversalEmbeddingModel

def simple_tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())

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
        
        if not model.glossary:
            raise Exception("Glossary is missing from model.")

        if not model.model_path:
            raise Exception("Model file path is missing.")
        
        vocab = {token: idx for idx, token in enumerate(model.glossary)}
        vocab_size = len(vocab)

        # Tokenize input
        indices = [vocab.get(token, vocab["<UNK>"]) for token in simple_tokenize(sentence)]
        if not indices:
            raise ValueError("Input sentence produced no valid tokens.")

        # Prepare input tensors
        seq = torch.tensor(indices, dtype=torch.long).unsqueeze(0)  # (1, seq_len)
        lengths = torch.tensor([len(indices)])

        # Reconstruct and load model
        model_path = Path(model.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        nn_model = UniversalEmbeddingModel(vocab_size=vocab_size).eval()
        nn_model.load_state_dict(torch.load(model_path, map_location="cpu"))

        # Compute embedding
        with torch.no_grad():
            embedding = nn_model.encode(seq, lengths)  # tensor
            embedding = embedding.detach().cpu().numpy()  # numpy array
            embedding = np.nan_to_num(embedding)  # Clean NaNs if any
            return embedding.squeeze(0).tolist()
    except Exception as e:
        logger.error(f"[encode_embedding_usecase] Error: {str(e)}\n{traceback.format_exc()}")
        raise Exception(f"[encode_embedding_usecase] {str(e)}")
