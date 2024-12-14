#app\usecases\lstm\usecase_train_lstm.py
import joblib
import numpy as np
import pandas as pd
from typing import List
from app.services.logger import logger
from fastapi import HTTPException  # type: ignore
from app.neural_network.nn_lstm import train_nn_lstm
from app.repositories.memory import get_model, update_model
from app.apis.models.weather_model_data import WeatherModelData
from app.usecases.lstm.usecase_commons_lstm import prepare_sequences, preprocess_data

def train_lstm(name: str, training_data: List[WeatherModelData]):
    """
    Trains the model using the provided training data.
    :param name: Name of the model.
    :param training_data: List of training data.
    """
    model = get_model(name)
    if model is None or not model:
        raise HTTPException(status_code=404, detail="Model not found")
    if training_data is None or len(training_data) == 0:
        raise HTTPException(status_code=400, detail="No training data provided or training data is empty")
    
    logger.info("Machine learning type used for training: LSTM")

    # Convert training data to DataFrame
    df = pd.DataFrame([data.dict() for data in training_data])
    
    # Data preprocessing
    df_processed, y_temp_scaled, scaler, target_scaler, coco_encoder = preprocess_data(df)
    
    # Preparing data for training
    X_train, y_train = prepare_sequences(df_processed, y_temp_scaled)

    # Check for NaNs in X_train and y_train
    if np.isnan(X_train).any():
        logger.error("X_train contains NaN values. Training aborted.")
        raise ValueError("X_train contains NaN values.")
    if np.isnan(y_train).any():
        logger.error("y_train contains NaN values. Training aborted.")
        raise ValueError("y_train contains NaN values.")

    # Save preprocessing objects
    joblib.dump(scaler, f'{name}_scaler.pkl')
    joblib.dump(target_scaler, f'{name}_target_scaler.pkl')
    joblib.dump(coco_encoder, f'{name}_coco_encoder.pkl')

    # Model training
    nn_model = train_nn_lstm(X_train, y_train)
    
    # Update the model in storage
    update_model(name, {"nn_model": nn_model})
    
    return {"status": "training completed", "model_name": name}