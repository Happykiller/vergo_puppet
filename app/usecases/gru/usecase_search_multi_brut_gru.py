# app\usecases\gru\usecase_search_multi_brut_gru.py
import traceback
from typing import List, NamedTuple

from app.inversify import Inversify
from app.services.logger import logger
from app.usecases.usecase_tokenize import usecase_tokenize
from app.neural_network.nn_gru import GRUClassifier, predict
from app.usecases.gru.usecase_commons_gru import process_input
from app.apis.models.tokenize_model_data import ModelTokenizeData

class SearchMultiBrutGRUUsecaseDto(NamedTuple):
    name: str
    documents: List[ModelTokenizeData]
    inversify: Inversify

def search_multi_brut_model_gru(dto: SearchMultiBrutGRUUsecaseDto):
    try:
        result = []
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
        
        documents_tokenized = usecase_tokenize(dto.documents)

        for index, data in enumerate(documents_tokenized):
            # Process the input sequence using the word-to-index mapping
            input = process_input(data['tokens'], word2idx)

            # Make a prediction using the GRU model
            predicted_idx = predict(nn_model, input)

            # Map the predicted index to the corresponding category
            predicted_category = idx2category[predicted_idx]
        
            result.append({
                'incidentId': dto.documents[index].incidentId,
                'description': dto.documents[index].description,
                'tokens': data['tokens'],
                'predicted_category': predicted_category
            })

        return result
    
    except Exception as e:
        logger.error(f"Error message:{str(e)}\nStack trace:\n{traceback.format_exc()}")
        raise Exception(f"[#search_model_simple_nn]{str(e)}")
