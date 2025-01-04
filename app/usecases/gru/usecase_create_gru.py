# app\usecases\gru\usecase_create_gru.py
from typing import NamedTuple

from app.inversify import Inversify
from app.services.logger import logger
from app.services.bdd.models.model_data import ModelData

class CreateGRUUsecaseDto(NamedTuple):
    name: str
    inversify: Inversify

def create_model_gru(dto: CreateGRUUsecaseDto):
    """
    Creates a new GRU model with the specified name.
    - Checks if the model already exists, and if so, raises an HTTPException.
    - Saves the model with necessary metadata if it does not exist.
    
    :param name: The name of the model to create.
    :return: A dictionary indicating the success status and model name.
    """
    # Log the model creation process with specified machine learning type
    logger.info("Machine learning type used for model creation: 'GRU'")
    
    # Fetch Bdd
    bdd = dto.inversify.get_bdd()

    # Check if a model with the given name already exists
    if bdd.model_exists(dto.name):
        # If model exists, raise an HTTP 400 error with a relevant message
        raise Exception("Model already exists")
    
    # Save the model with the provided name and data
    bdd.save_model(ModelData(name=dto.name, neural_network_type="GRU"))

    # Return success status and model name for confirmation
    return {"status": "model created", "model_name": dto.name}

