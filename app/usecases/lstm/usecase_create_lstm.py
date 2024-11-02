#app\usecases\lstm\usecase_create_lstm.py
from app.services.logger import logger
from fastapi import HTTPException  # type: ignore
from app.repositories.memory import model_exists, save_model

def create_lstm(name: str):
    # Log the type of machine learning model being created
    logger.info(f"Machine learning type used for model creation: 'LSTM'")

    # Check if the model already exists
    if model_exists(name):
        # If the model exists, raise an HTTP 400 error
        raise HTTPException(status_code=400, detail="Model already exists")

    # Prepare model data with necessary attributes
    model_data = {
        "neural_network_type": "LSTM"  # Record the model type as 'LSTM'
    }
    # Save the model data
    save_model(name, model_data)

    # Return a success message with the model name
    return {"status": "model created", "model_name": name}
