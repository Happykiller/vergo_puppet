# app\usecases\lstm\usecase_train_lstm.py
import traceback
import numpy as np
import pandas as pd
from typing import List, NamedTuple

from app.inversify import Inversify
from app.services.logger import logger
from app.services.bdd.models.model_data import ModelData, ModelStatus
from app.neural_network.nn_lstm import LSTMNN, train_nn_lstm
from app.apis.models.weather_model_data import WeatherModelData
from app.usecases.lstm.usecase_commons_lstm import prepare_sequences, preprocess_data

class TrainLSTMUsecaseDto(NamedTuple):
    name: str
    training_data: List[WeatherModelData]
    inversify: Inversify

def train_lstm(dto: TrainLSTMUsecaseDto):
    """
    Trains the model using the provided training data.
    :param name: Name of the model.
    :param training_data: List of training data.
    """
    try:
        # Fetch Bdd
        bdd = dto.inversify.get_bdd()

        model = bdd.get_model(dto.name, LSTMNN)
        if model is None or not model:
            raise Exception("Model not found")
        if dto.training_data is None or len(dto.training_data) == 0:
            raise Exception("No training data provided or training data is empty")
        
        logger.info("Machine learning type used for training: LSTM")

        # Update the model in storage
        model.status = ModelStatus.TRAINING
        bdd.update_model(model)

        # Convert training data to DataFrame
        df = pd.DataFrame([data.dict() for data in dto.training_data])
        
        # Data preprocessing
        df_processed, y_temp_scaled, scaler, target_scaler, encoder = preprocess_data(df)
        
        # Preparing data for training
        X_train, y_train = prepare_sequences(df_processed, y_temp_scaled)

        # Check for NaNs in X_train and y_train
        if np.isnan(X_train).any():
            raise ValueError("X_train contains NaN values.")
        if np.isnan(y_train).any():
            raise ValueError("y_train contains NaN values.")

        # Model training
        nn_model = train_nn_lstm(X_train, y_train)
        
        # Update the model in storage
        bdd.update_model(ModelData(
            name=model.name, 
            neural_network_type=model.neural_network_type,
            status=ModelStatus.TRAINED,
            scaler=scaler,
            target_scaler=target_scaler,
            encoder=encoder,
            nn_model=nn_model
        ))
        
        return {"status": "training completed", "model_name": dto.name}
    
    except Exception as e:
        logger.error(f"Error message:{str(e)}\nStack trace:\n{traceback.format_exc()}")
        raise Exception(f"[#train_lstm]{str(e)}")