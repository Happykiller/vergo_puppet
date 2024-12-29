#app\usecases\lstm\usecase_mesure_lstm.py
import joblib
import numpy as np
import pandas as pd
from typing import List, NamedTuple
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from app.inversify import Inversify
from app.services.logger import logger
from app.neural_network.nn_lstm import predict_nn_lstm
from app.apis.models.weather_model_data import WeatherModelData
from app.usecases.lstm.usecase_commons_lstm import inverse_transform_predictions, preprocess_input_data

class MesureLSTMUsecaseDto(NamedTuple):
    name: str
    test_data: List[WeatherModelData]
    inversify: Inversify

def mesure_lstm(dto: MesureLSTMUsecaseDto):
    """
    Measures the performance of the LSTM model on the provided test data.
    :param name: Name of the model.
    :param test_data: List of test data samples.
    :return: A dictionary with performance metrics.
    """
    try:
        # Fetch Bdd
        bdd = dto.inversify.get_bdd()

        # Verify that the model exists
        model = bdd.get_model(dto.name)
        nn_model = model.get("nn_model", None)
        if nn_model is None:
            raise Exception("The model has not been trained yet")
        
        # Load scaler and encoder
        scaler = joblib.load(f'{dto.name}_scaler.pkl')
        target_scaler = joblib.load(f'{dto.name}_target_scaler.pkl')
        coco_encoder = joblib.load(f'{dto.name}_coco_encoder.pkl')
        
        # Define the sequence length used during training
        sequence_length = 24  # Adjust if necessary

        # Lists to store actual and predicted values
        y_true_list = []
        y_pred_list = []
        
        # Loop through each test sample
        for data in dto.test_data:
            # Extract the actual 'temp' value
            y_true = data.temp
            
            # Prepare input data (excluding 'temp')
            data_dict = data.dict()
            data_dict.pop('temp', None)  # Remove 'temp' from input data
            df_input = pd.DataFrame([data_dict])
            
            # Preprocess the input data
            df_processed = preprocess_input_data(df_input, scaler, coco_encoder)
            
            # Create a sequence by duplicating the input to reach the required length
            input_sequence = np.repeat(df_processed.values, sequence_length, axis=0)
            input_sequence = np.expand_dims(input_sequence, axis=0)  # Shape: (1, sequence_length, num_features)
            
            # Make the prediction
            prediction_normalized = predict_nn_lstm(nn_model, input_sequence)
            
            # Invert the normalization of the prediction
            prediction_inverse = inverse_transform_predictions(prediction_normalized, target_scaler)[0]
            
            # Add values to the lists
            y_true_list.append(y_true)
            y_pred_list.append(prediction_inverse)
        
        # Convert lists to numpy arrays
        y_true_array = np.array(y_true_list)
        y_pred_array = np.array(y_pred_list)
        
        # Calculate performance metrics
        mae = mean_absolute_error(y_true_array, y_pred_array)
        mape = mean_absolute_percentage_error(y_true_array, y_pred_array) * 100  # In percentage
        
        # Log the results
        logger.info(f"Mean Absolute Error (MAE) on the test set: {mae:.2f}")
        logger.info(f"Mean Absolute Percentage Error (MAPE) on the test set: {mape:.2f}%")
        
        # Return the metrics
        return {
            "mae": mae,
            "mape": mape,
            "test_count": len(y_pred_list)
        }
    except Exception as e:
        logger.error(f"An error occurred during measurement: {str(e)}")
        raise Exception(f"An error occurred during measurement: {str(e)}")
