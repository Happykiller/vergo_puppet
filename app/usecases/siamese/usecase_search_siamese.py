# app\usecases\siamese\usecase_search_siamese.py
import traceback
from typing import NamedTuple

from app.inversify import Inversify
from app.services.logger import logger
from app.neural_network.nn_siamese import SiameseLSTM, evaluate_similarity
from app.usecases.siamese.usecase_commons_siamese import create_indexed_glossary, tokens_to_indices

class SearchSiameseUsecaseDto(NamedTuple):
    name: str
    search: list
    inversify: Inversify

def search_model_siamese(dto: SearchSiameseUsecaseDto):
    try:
        # Fetch Bdd
        bdd = dto.inversify.get_bdd()

        # Vérification du buffer
        search_query = str(dto.search) 
        cached_result = bdd.get_search_result(dto.name, search_query)
        if cached_result:
            logger.info(f"Returning cached result for search query: {search_query}")
            return cached_result

        # Retrieve the model
        model = bdd.get_model(dto.name, SiameseLSTM)
        if not model:
            raise Exception("Model not found")

        # Ensure the neural network model is available
        nn_model = model.nn_model
        if not nn_model:
            raise Exception("No neural network model found in the model")

        # Retrieve the indexed dictionary and the raw dictionary
        indexed_dictionary = model.indexed_dictionary
        dictionary = model.dictionary
        if not dictionary or not indexed_dictionary:
            raise Exception("No vectors available in the model")
        
        # Retrieve glossary to create a word-to-index mapping
        glossary = model.glossary
        if not glossary:
            raise Exception("No glossary available in the model")

        logger.info("Machine learning type used for SIAMESE search")

        # Convert search terms to indices
        word2idx = create_indexed_glossary(glossary)
        search_indices = tokens_to_indices(dto.search, word2idx)

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

        result = {
            "search": dto.search,
            "find": find,
            "stats": {
                "accuracy": accuracy
            }
        }

        # Save the result in the buffer
        bdd.save_search_result(dto.name, search_query, result)

        return result
    
    except Exception as e:
        logger.error(f"Error message:{str(e)}\nStack trace:\n{traceback.format_exc()}")
        raise Exception(f"[#search_model_siamese]{str(e)}")
