#app\usecases\siamese\usecase_search_siamese.py
from fastapi import HTTPException # type: ignore

from app.services.logger import logger
from app.repositories.memory import get_model
from app.usecases.tokens_to_indices import tokens_to_indices
from app.machine_learning.neural_network_siamese import evaluate_similarity
from app.usecases.siamese.usecase_commons_siamese import create_indexed_glossary

def search_model_siamese(name: str, search: list):
    model = get_model(name)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    nn_model = model.get("nn_model", None)
    if not nn_model:
        raise HTTPException(status_code=400, detail="No neural network model found in the model")

    indexed_dictionary = model.get("indexed_dictionary", [])
    dictionary = model.get("dictionary", [])
    if not indexed_dictionary:
        raise HTTPException(status_code=400, detail="No vectors available in the model")
    
    glossary = model.get("glossary", [])

    logger.info(f"Type de machine learning utilisé pour la recherche SIAMESE")

    word2idx = create_indexed_glossary(glossary)
    search_indices = tokens_to_indices(search, word2idx)

    similarities = []
    for vector in dictionary:
        vector_indices = tokens_to_indices(vector, word2idx)
        similarity = evaluate_similarity(nn_model, search_indices, vector_indices)
        similarities.append((vector, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)

    accuracy = similarities[0][1]
    find = similarities[0][0]

    return {
        "search": search,
        "find": find,
        "stats": {
            "accuracy": accuracy
        }
    }
