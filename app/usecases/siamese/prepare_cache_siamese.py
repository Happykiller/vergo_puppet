# app\usecases\siamese\prepare_cache_siamese.py
from typing import List, NamedTuple

from app.inversify import Inversify
from app.services.logger import logger
from app.usecases.siamese.usecase_search_siamese import SearchSiameseUsecaseDto, search_model_siamese

class PrepareSiameseUsecaseDto(NamedTuple):
    name: str
    search_vectors: List[List[str]]
    inversify: Inversify

def prepare_cache_siamese(dto: PrepareSiameseUsecaseDto) -> dict:
    """
    Prepares the search cache for a model by precomputing results for the given search vectors.
    :param name: Name of the model
    :param search_vectors: List of search vectors to precompute
    :return: Summary of the cache preparation
    """
    # Fetch Bdd
    bdd = dto.inversify.get_bdd()

    # Check if the model exists
    model = bdd.get_model(dto.name)
    if not model:
        raise Exception("Model not found")

    logger.info(f"Preparing cache for SIAMESE model '{dto.name}' with {len(dto.search_vectors)} search vectors")

    # Store results for all search vectors
    results = []

    for vector in dto.search_vectors:
        try:
            # Perform a search for the vector using the SIAMESE use case
            result = search_model_siamese(SearchSiameseUsecaseDto(name=dto.name, search=vector, inversify=dto.inversify))
            results.append({"search_vector": vector, "result": result})
        except Exception as e:
            logger.error(f"Failed to prepare cache for vector {vector}: {str(e)}")
            results.append({"search_vector": vector, "error": str(e)})

    return {
        "status": "cache prepared",
        "model_name": dto.name,
        "vectors_processed": len(dto.search_vectors),
        "results": results,
    }
