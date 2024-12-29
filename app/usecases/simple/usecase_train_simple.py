# app\usecases\simple\usecase_train_simple.py
import joblib
from typing import List, NamedTuple
from fastapi import HTTPException  # type: ignore

from app.inversify import Inversify
from app.services.logger import logger
from app.neural_network.nn_simple import train_model_nn
from app.usecases.simple.usecase_commons_simple import transform_data
from app.apis.models.simple_nn_training_model_data import SimpleNNTrainingModelData

class TrainSimpleUsecaseDto(NamedTuple):
    name: str
    training_data: List[SimpleNNTrainingModelData]
    inversify: Inversify

def train_model_simple_nn(dto: TrainSimpleUsecaseDto):
    """
    Train the SimpleNN model with the provided training data.
    :param name: Model name
    :param training_data: List of training data
    :return: Status of training
    """
    # Fetch Bdd
    bdd = dto.inversify.get_bdd()

    # Retrieve the model configuration
    model = bdd.get_model(dto.name)
    
    if model is None or not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if dto.training_data is None or len(dto.training_data) == 0:
        raise HTTPException(status_code=400, detail="No training data provided or training data is empty")
    
    logger.info("Machine learning type used for training: SimpleNN")
    
    # Transform the data
    features_processed, targets_standardized, encoder, scaler, targets_mean, targets_std, categorical_indices, numerical_indices = transform_data(dto.training_data)
    
    # Determine the input size for the neural network
    input_size = features_processed.shape[1]
    
    # Train the neural network
    nn_model, _ = train_model_nn(features_processed, targets_standardized, input_size)
    
    # Save the encoder, scaler, and target normalization parameters
    encoder_filename = f"{dto.name}_encoder.joblib"
    scaler_filename = f"{dto.name}_scaler.joblib"
    joblib.dump(encoder, encoder_filename)
    joblib.dump(scaler, scaler_filename)
    
    # Save categorical and numerical indices
    indices_info = {
        "categorical_indices": categorical_indices,
        "numerical_indices": numerical_indices
    }
    indices_filename = f"{dto.name}_indices.joblib"
    joblib.dump(indices_info, indices_filename)
    
    # Update the model with the trained neural network and additional metadata
    bdd.update_model(dto.name, {
        "nn_model": nn_model,
        "encoder_filename": encoder_filename,
        "scaler_filename": scaler_filename,
        "indices_filename": indices_filename,
        "targets_mean": targets_mean,
        "targets_std": targets_std
    })
    
    return {"status": "training completed", "model_name": dto.name}
