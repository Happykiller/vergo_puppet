#app\usecases\siamese\usecase_search_siamese.py
from fastapi import HTTPException  # type: ignore

from app.services.logger import logger
from app.repositories.memory import get_model
from app.neural_network.nn_siamese import evaluate_similarity
from app.usecases.siamese.usecase_commons_siamese import create_indexed_glossary, tokens_to_indices

def search_model_siamese(name: str, search: list):
    # Retrieve the model
    model = get_model(name)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Ensure the neural network model is available
    nn_model = model.get("nn_model", None)
    if not nn_model:
        raise HTTPException(status_code=400, detail="No neural network model found in the model")

    # Retrieve the indexed dictionary and the raw dictionary
    indexed_dictionary = model.get("indexed_dictionary", [])
    dictionary = model.get("dictionary", [])
    if not indexed_dictionary:
        raise HTTPException(status_code=400, detail="No vectors available in the model")
    
    # Retrieve glossary to create a word-to-index mapping
    glossary = model.get("glossary", [])

    logger.info("Machine learning type used for SIAMESE search")

    # Convert search terms to indices
    word2idx = create_indexed_glossary(glossary)
    search_indices = tokens_to_indices(search, word2idx)

    # Calculate similarity with each vector in the dictionary
    similarities = []
    for vector in dictionary:
        vector_indices = tokens_to_indices(vector, word2idx)
        similarity = evaluate_similarity(nn_model, search_indices, vector_indices)
        similarities.append((vector, similarity))

    # Sort by similarity in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Extract the most similar vector and its accuracy score
    accuracy = similarities[0][1]
    find = similarities[0][0]

    return {
        "search": search,
        "find": find,
        "stats": {
            "accuracy": accuracy
        }
    }
