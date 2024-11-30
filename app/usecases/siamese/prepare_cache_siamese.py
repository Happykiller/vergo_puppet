# app\usecases\siamese\prepare_cache_siamese.py
from typing import List
from app.services.logger import logger
from app.repositories.memory import get_model
from app.usecases.siamese.usecase_search_siamese import search_model_siamese

def prepare_cache(name: str, search_vectors: List[List[str]]) -> dict:
    """
    Prepares the search cache for a model by precomputing results for the given search vectors.
    :param name: Name of the model
    :param search_vectors: List of search vectors to precompute
    :return: Summary of the cache preparation
    """

    # Check if the model exists
    model = get_model(name)
    if not model:
        raise Exception("Model not found")

    logger.info(f"Preparing cache for SIAMESE model '{name}' with {len(search_vectors)} search vectors")

    # Store results for all search vectors
    results = []

    for vector in search_vectors:
        try:
            # Perform a search for the vector using the SIAMESE use case
            result = search_model_siamese(name, vector)
            results.append({"search_vector": vector, "result": result})
        except Exception as e:
            logger.error(f"Failed to prepare cache for vector {vector}: {str(e)}")
            results.append({"search_vector": vector, "error": str(e)})

    return {
        "status": "cache prepared",
        "model_name": name,
        "vectors_processed": len(search_vectors),
        "results": results,
    }
