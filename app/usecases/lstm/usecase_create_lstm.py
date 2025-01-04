# app\usecases\lstm\usecase_create_lstm.py
import traceback
from typing import NamedTuple

from app.inversify import Inversify
from app.services.logger import logger
from app.services.bdd.models.model_data import ModelData

class CreateLSTMUsecaseDto(NamedTuple):
    name: str
    inversify: Inversify

def create_lstm(dto: CreateLSTMUsecaseDto):
    try:
        # Log the type of machine learning model being created
        logger.info(f"Machine learning type used for model creation: 'LSTM'")
        
        # Fetch Bdd
        bdd = dto.inversify.get_bdd()

        # Check if the model already exists
        if bdd.model_exists(dto.name):
            # If the model exists, raise an HTTP 400 error
            raise Exception("Model already exists")

        # Save the model data
        bdd.save_model(ModelData(name=dto.name, neural_network_type="LSTM"))

        # Return a success message with the model name
        return {"status": "model created", "model_name": dto.name}
    
    except Exception as e:
        logger.error(f"Error message:{str(e)}\nStack trace:\n{traceback.format_exc()}")
        raise Exception(f"[#create_lstm]{str(e)}")
