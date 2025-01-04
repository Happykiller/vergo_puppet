# app\usecases\gru\usecase_search_gru.py
import traceback
from typing import List, NamedTuple

from app.inversify import Inversify
from app.services.logger import logger
from app.neural_network.nn_gru import GRUClassifier, predict
from app.usecases.gru.usecase_commons_gru import process_input

class SearchGRUUsecaseDto(NamedTuple):
    name: str
    search: List[str]
    inversify: Inversify

def search_model_gru(dto: SearchGRUUsecaseDto):
    """
    Uses the GRU model to predict the category of a new sequence of tokens.
    :param name: Name of the model.
    :param search: List of tokens representing the sequence to classify.
    :return: Predicted category.
    """
    try:
        # Fetch Bdd
        bdd = dto.inversify.get_bdd()

        # Retrieve model data from memory
        model = bdd.get_model(dto.name, GRUClassifier)
        
        # Check if the model data is found
        if model is None or not model:
            # Raise a 404 error if the model is not found
            raise Exception("Model not found")
        
        # Extract the neural network model from the data
        nn_model = model.nn_model
        if nn_model is None:
            # Raise a 400 error if the model is not trained
            raise Exception("Model not trained")
        
        # Retrieve word-to-index and index-to-category mappings
        word2idx = model.word2idx
        idx2category = model.idx2category
        if word2idx is None or idx2category is None:
            # Raise a 400 error if essential model data is incomplete
            raise Exception("Model data incomplete")
        
        # Process the input sequence using the word-to-index mapping
        input = process_input(dto.search, word2idx)

        # Make a prediction using the GRU model
        predicted_idx = predict(nn_model, input)

        # Map the predicted index to the corresponding category
        predicted_category = idx2category[predicted_idx]
        
        # Return the predicted category in a dictionary
        return {"category": predicted_category}
    
    except Exception as e:
        logger.error(f"Error message:{str(e)}\nStack trace:\n{traceback.format_exc()}")
        raise Exception(f"[#search_model_simple_nn]{str(e)}")