#app\usecases\lstm\usecase_search_lstm.py
import joblib
import numpy as np
import pandas as pd
from app.repositories.memory import get_model
from fastapi import HTTPException  # type: ignore
from app.neural_network.nn_lstm import predict_nn_lstm
from app.apis.models.weather_model_data import WeatherSearchModelData
from app.usecases.lstm.usecase_commons_lstm import inverse_transform_predictions, preprocess_input_data

def search_lstm(name: str, input_data: WeatherSearchModelData):
    """
    Uses the LSTM model to predict temperature based on the provided input data.
    :param name: Name of the model.
    :param input_data: Input data without the temperature.
    :return: Predicted temperature.
    """
    # Load the model
    model_data = get_model(name)
    if model_data is None or not model_data:
        raise HTTPException(status_code=404, detail="Model not found")
    nn_model = model_data['nn_model']
    nn_model.eval()

    # Load scaler and encoder
    scaler = joblib.load(f'{name}_scaler.pkl')
    target_scaler = joblib.load(f'{name}_target_scaler.pkl')
    coco_encoder = joblib.load(f'{name}_coco_encoder.pkl')

    # Prepare input data
    df_input = pd.DataFrame([input_data.dict()])
    df_processed = preprocess_input_data(df_input, scaler, coco_encoder)

    # Define the sequence length expected by the model
    sequence_length = 24  # Adjust if necessary

    # Create a sequence by repeating the input to reach the required length
    input_sequence = np.repeat(df_processed.values, sequence_length, axis=0)
    input_sequence = np.expand_dims(input_sequence, axis=0)  # Shape: (1, sequence_length, num_features)

    # Make the prediction
    prediction_normalized = predict_nn_lstm(nn_model, input_sequence)

    # Invert the normalization of the prediction
    prediction_inverse = inverse_transform_predictions(prediction_normalized, target_scaler)[0]

    # Convert to float for JSON serialization
    prediction_value = float(prediction_inverse)

    return {"prediction": prediction_value}
