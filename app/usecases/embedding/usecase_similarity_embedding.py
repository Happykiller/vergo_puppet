# app/usecases/embedding/usecase_similarity_embedding.py
import numpy as np
import traceback

from app.usecases.embedding.usecase_encode_embedding import encode_embedding_usecase
from app.services.logger import logger

def cosine_similarity(vec1, vec2):
    """
    Computes cosine similarity between two vectors.
    """
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))

def similarity_embedding_usecase(
    model_name: str,
    vocab: dict,
    sentence1: str,
    sentence2: str,
    embedding_dim: int = 128,
    lstm_hidden_dim: int = 128,
) -> float:
    """
    Computes cosine similarity between two sentences using a trained embedding model.
    :param model_name: Name of the embedding model to use
    :param vocab: Token-to-index vocabulary
    :param sentence1: First sentence
    :param sentence2: Second sentence
    :param embedding_dim: Embedding vector size
    :param lstm_hidden_dim: LSTM hidden state size
    :return: Cosine similarity (float)
    """
    try:
        emb1 = encode_embedding_usecase(
            model_name=model_name,
            vocab=vocab,
            sentence=sentence1,
            embedding_dim=embedding_dim,
            lstm_hidden_dim=lstm_hidden_dim,
        )
        emb2 = encode_embedding_usecase(
            model_name=model_name,
            vocab=vocab,
            sentence=sentence2,
            embedding_dim=embedding_dim,
            lstm_hidden_dim=lstm_hidden_dim,
        )
        sim = cosine_similarity(emb1, emb2)
        logger.info(f"[similarity_embedding_usecase] similarity={sim:.4f}")
        return sim
    except Exception as e:
        logger.error(f"[similarity_embedding_usecase] Error: {str(e)}\n{traceback.format_exc()}")
        raise Exception(f"[similarity_embedding_usecase] {str(e)}")
