#app\usecases\gru\usecase_create_gru.py
from app.services.logger import logger
from fastapi import HTTPException  # type: ignore
from app.repositories.memory import model_exists, save_model

def create_model_gru(name: str):
    """
    Creates a new GRU model with the specified name.
    - Checks if the model already exists, and if so, raises an HTTPException.
    - Saves the model with necessary metadata if it does not exist.
    
    :param name: The name of the model to create.
    :return: A dictionary indicating the success status and model name.
    """
    # Log the model creation process with specified machine learning type
    logger.info("Machine learning type used for model creation: 'GRU'")

    # Check if a model with the given name already exists
    if model_exists(name):
        # If model exists, raise an HTTP 400 error with a relevant message
        raise HTTPException(status_code=400, detail="Model already exists")

    # Prepare the data for saving the model with its type specified
    model_data = {
        "neural_network_type": "GRU"  # Record the model type as 'GRU'
    }
    
    # Save the model with the provided name and data
    save_model(name, model_data)

    # Return success status and model name for confirmation
    return {"status": "model created", "model_name": name}

