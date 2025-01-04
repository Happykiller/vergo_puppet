# app\usecases\simple\usecase_search_simple.py
import traceback
from typing import NamedTuple

from app.inversify import Inversify
from app.services.logger import logger
from app.neural_network.nn_simple import SimpleNN, predict
from app.usecases.simple.usecase_commons_simple import process_input_data
from app.apis.models.simple_nn_search_model_data import SimpleNNSearchModelData

class SearchSimpleUsecaseDto(NamedTuple):
    name: str
    search: SimpleNNSearchModelData
    inversify: Inversify

def search_model_simple_nn(dto: SearchSimpleUsecaseDto):
    """
    Uses the SimpleNN model to predict the price based on input data.
    :param name: Name of the model.
    :param search: Input data as a SimpleNNSearchModelData object.
    :return: Predicted price.
    """
    try:
        # Fetch Bdd
        bdd = dto.inversify.get_bdd()

        # Retrieve the model
        model = bdd.get_model(dto.name, SimpleNN)
        
        if model is None or not model:
            raise Exception("Model not found")
        
        nn_model = model.nn_model
        if nn_model is None:
            raise Exception("Model not trained yet")
        
        # Check if essential attributes are present
        if not model.encoder or not model.scaler or not model.indices:
            raise Exception("Missing encoder, scaler, or indices in the model")
        
        # Retrieve the target normalization parameters
        categorical_indices = model.indices["categorical_indices"]
        numerical_indices = model.indices["numerical_indices"]
        targets_mean = model.targets_mean
        targets_std = model.targets_std
        if targets_mean is None or targets_std is None:
            raise Exception("Missing normalization parameters in the model")
        
        # Prepare the input data
        input_data = [
            dto.search.type,
            dto.search.surface,
            dto.search.pieces,
            dto.search.floor,
            dto.search.parking,
            dto.search.balcon,
            dto.search.ascenseur,
            dto.search.orientation,
            dto.search.transports,
            dto.search.neighborhood
        ]
        
        # Transform the input data
        input_processed = process_input_data(input_data, model.encoder, model.scaler, categorical_indices, numerical_indices)
        
        # Make the prediction
        predicted = predict(nn_model, input_processed, targets_mean, targets_std)
        
        return {"predicted_price": predicted}
    except Exception as e:
        logger.error(f"Error message:{str(e)}\nStack trace:\n{traceback.format_exc()}")
        raise Exception(f"[#search_model_simple_nn]{str(e)}")
