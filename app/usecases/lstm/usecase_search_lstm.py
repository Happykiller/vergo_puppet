# app\usecases\lstm\usecase_search_lstm.py
import traceback
import numpy as np
import pandas as pd
from typing import NamedTuple

from app.inversify import Inversify
from app.services.logger import logger
from app.neural_network.nn_lstm import LSTMNN, predict_nn_lstm
from app.apis.models.weather_model_data import WeatherSearchModelData
from app.usecases.lstm.usecase_commons_lstm import inverse_transform_predictions, preprocess_input_data

class SearchLSTMUsecaseDto(NamedTuple):
    name: str
    search: WeatherSearchModelData
    inversify: Inversify

def search_lstm(dto: SearchLSTMUsecaseDto):
    """
    Uses the LSTM model to predict temperature based on the provided input data.
    :param name: Name of the model.
    :param input_data: Input data without the temperature.
    :return: Predicted temperature.
    """
    try:
        # Fetch Bdd
        bdd = dto.inversify.get_bdd()
        
        # Load the model
        model = bdd.get_model(dto.name, LSTMNN)
        if model is None or not model:
            raise Exception("Model not found")
        nn_model = model.nn_model
        nn_model.eval()

        # Prepare input data
        df_input = pd.DataFrame([dto.search.dict()])
        df_processed = preprocess_input_data(df_input, model.scaler, model.encoder)

        # Define the sequence length expected by the model
        sequence_length = 24  # Adjust if necessary

        # Create a sequence by repeating the input to reach the required length
        input_sequence = np.repeat(df_processed.values, sequence_length, axis=0)
        input_sequence = np.expand_dims(input_sequence, axis=0)  # Shape: (1, sequence_length, num_features)

        # Make the prediction
        prediction_normalized = predict_nn_lstm(nn_model, input_sequence)

        # Invert the normalization of the prediction
        prediction_inverse = inverse_transform_predictions(prediction_normalized, model.target_scaler)[0]

        # Convert to float for JSON serialization
        prediction_value = float(prediction_inverse)

        return {"prediction": prediction_value}
    
    except Exception as e:
        logger.error(f"Error message:{str(e)}\nStack trace:\n{traceback.format_exc()}")
        raise Exception(f"[#search_lstm]{str(e)}")
