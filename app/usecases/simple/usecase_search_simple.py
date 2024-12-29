# app\usecases\simple\usecase_search_simple.py
import joblib
from typing import NamedTuple
from fastapi import HTTPException  # type: ignore

from app.inversify import Inversify
from app.neural_network.nn_simple import predict
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
    # Fetch Bdd
    bdd = dto.inversify.get_bdd()

    # Retrieve the model
    model = bdd.get_model(dto.name)
    
    if model is None or not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    nn_model = model.get("nn_model", None)
    if nn_model is None:
        raise HTTPException(status_code=400, detail="Model not trained yet")
    
    # Load the encoder, scaler, and indices
    encoder_filename = model.get("encoder_filename")
    scaler_filename = model.get("scaler_filename")
    indices_filename = model.get("indices_filename")
    if not encoder_filename or not scaler_filename or not indices_filename:
        raise HTTPException(status_code=400, detail="Missing encoder, scaler, or indices in the model")
    
    encoder = joblib.load(encoder_filename)
    scaler = joblib.load(scaler_filename)
    indices_info = joblib.load(indices_filename)
    categorical_indices = indices_info["categorical_indices"]
    numerical_indices = indices_info["numerical_indices"]
    
    # Retrieve the target normalization parameters
    targets_mean = model.get("targets_mean")
    targets_std = model.get("targets_std")
    if targets_mean is None or targets_std is None:
        raise HTTPException(status_code=400, detail="Missing normalization parameters in the model")
    
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
    input_processed = process_input_data(input_data, encoder, scaler, categorical_indices, numerical_indices)
    
    # Make the prediction
    predicted = predict(nn_model, input_processed, targets_mean, targets_std)
    
    return {"predicted_price": predicted}
