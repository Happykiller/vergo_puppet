# app\usecases\lstm\usecase_create_lstm.py
from typing import NamedTuple
from fastapi import HTTPException  # type: ignore

from app.inversify import Inversify
from app.services.logger import logger

class CreateLSTMUsecaseDto(NamedTuple):
    name: str
    inversify: Inversify

def create_lstm(dto: CreateLSTMUsecaseDto):
    # Log the type of machine learning model being created
    logger.info(f"Machine learning type used for model creation: 'LSTM'")
    
    # Fetch Bdd
    bdd = dto.inversify.get_bdd()

    # Check if the model already exists
    if bdd.model_exists(dto.name):
        # If the model exists, raise an HTTP 400 error
        raise HTTPException(status_code=400, detail="Model already exists")

    # Prepare model data with necessary attributes
    model_data = {
        "neural_network_type": "LSTM"  # Record the model type as 'LSTM'
    }
    # Save the model data
    bdd.save_model(dto.name, model_data)

    # Return a success message with the model name
    return {"status": "model created", "model_name": dto.name}
