# app/usecases/embedding/usecase_similarity_embedding.py
import traceback
import numpy as np
from typing import Any

from app.services.logger import logger
from app.usecases.embedding.usecase_encode_embedding import encode_embedding_usecase

def cosine_similarity(vec1, vec2):
    """
    Computes the cosine similarity between two vectors and normalizes it to the [0, 1] range.
    This is the single source of truth for similarity calculation at inference time.
    
    Args:
        vec1: First vector as a NumPy array.
        vec2: Second vector as a NumPy array.

    Returns:
        The normalized similarity score (float between 0.0 and 1.0).
    """
    # Ensure they are numpy arrays
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # Raw cosine similarity [-1, 1]
    raw_similarity = np.dot(v1, v2) / (norm1 * norm2)
    
    # Normalize to [0, 1]
    normalized_similarity = (raw_similarity + 1.0) / 2.0
    
    # Clip to ensure it's strictly within [0, 1] due to potential float precision issues
    return float(np.clip(normalized_similarity, 0.0, 1.0))

def similarity_embedding_usecase(
    model_name: str,
    sentence1: str,
    sentence2: str,
    inversify: Any,
) -> float:
    """
    Computes cosine similarity between two sentences using a trained embedding model.
    :param model_name: Name of the embedding model to use
    :param vocab: Token-to-index vocabulary
    :param sentence1: First sentence
    :param sentence2: Second sentence
    :return: Cosine similarity (float)
    """
    try:
        emb1 = encode_embedding_usecase(
            model_name=model_name,
            sentence=sentence1,
            inversify=inversify,
        )
        emb2 = encode_embedding_usecase(
            model_name=model_name,
            sentence=sentence2,
            inversify=inversify,
        )
        sim = cosine_similarity(emb1, emb2)
        logger.info(f"[similarity_embedding_usecase] similarity={sim:.4f}")
        return sim
    except Exception as e:
        logger.error(f"[similarity_embedding_usecase] Error: {str(e)}\n{traceback.format_exc()}")
        raise Exception(f"[similarity_embedding_usecase] {str(e)}")
