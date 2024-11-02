#app\usecases\lstm\usecase_train_lstm_test.py
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from fastapi import HTTPException # type: ignore
from app.usecases.lstm.usecase_train_lstm import train_lstm
from app.apis.models.weather_model_data import WeatherModelData

@patch('app.usecases.lstm.usecase_train_lstm.joblib.dump')
@patch('app.usecases.lstm.usecase_train_lstm.prepare_sequences', return_value=(np.array([[np.nan]]), np.array([0.5])))
@patch('app.usecases.lstm.usecase_train_lstm.preprocess_data')
@patch('app.usecases.lstm.usecase_train_lstm.get_model')
def test_train_lstm_nan_in_data(mock_get_model, mock_preprocess_data, mock_prepare_sequences, mock_joblib_dump):
    # Mock the model to simulate an existing model
    mock_get_model.return_value = {"nn_model": MagicMock()}

    # Mock preprocess_data to return appropriate values
    mock_preprocess_data.return_value = (
        pd.DataFrame(),  # Mocked df_processed
        np.array([0.5]),  # Mocked y_temp_scaled
        MagicMock(name="scaler"),  # Mocked scaler
        MagicMock(name="target_scaler"),  # Mocked target_scaler
        MagicMock(name="coco_encoder")  # Mocked coco_encoder
    )

    # Training data with necessary fields
    training_data = [
        WeatherModelData(time="2023-01-01T00:00:00Z", temp=22.0, dwpt=10.0, rhum=60, prcp=0.0, snow=0.0, wdir=180, wspd=10, wpgt=15, pres=1013, tsun=0, coco=3)
    ]

    # Expect a ValueError due to NaN in X_train
    with pytest.raises(ValueError, match="X_train contains NaN values."):
        train_lstm("test_model", training_data)

    # Ensure joblib.dump is not called due to early termination on error
    mock_joblib_dump.assert_not_called()

@patch('app.usecases.lstm.usecase_train_lstm.joblib.dump')
@patch('app.usecases.lstm.usecase_train_lstm.prepare_sequences', return_value=(np.array([[[0.2]]]), np.array([0.5])))  # Make sure to return a 3D array
@patch('app.usecases.lstm.usecase_train_lstm.preprocess_data')
@patch('app.usecases.lstm.usecase_train_lstm.get_model')
def test_train_lstm_success(mock_get_model, mock_preprocess_data, mock_prepare_sequences, mock_joblib_dump):
    # Mock the model to simulate an existing model
    mock_model = {
        "nn_model": MagicMock()
    }
    mock_get_model.return_value = mock_model
    
    # Mock preprocess_data to return appropriate values
    scaler = MagicMock()
    target_scaler = MagicMock()
    coco_encoder = MagicMock()

    mock_preprocess_data.return_value = (
        pd.DataFrame(),  # Mocked df_processed
        np.array([0.5]),  # Mocked y_temp_scaled
        scaler,  # Mocked scaler
        target_scaler,  # Mocked target_scaler
        coco_encoder  # Mocked coco_encoder
    )

    # Training data with necessary fields
    training_data = [
        WeatherModelData(time="2023-01-01T00:00:00Z", temp=22.0, dwpt=10.0, rhum=60, prcp=0.0, snow=0.0, wdir=180, wspd=10, wpgt=15, pres=1013, tsun=0, coco=3)
    ]

    # Create a mock for the model storage to avoid KeyError
    mock_model_storage = {}
    
    # Mock the update_model function
    with patch('app.repositories.memory.update_model') as mock_update_model:
        mock_update_model.side_effect = lambda name, data: mock_model_storage.setdefault(name, {}).update(data)
        
        # Run the function
        result = train_lstm("test_model", training_data)

        # Verify the result
        assert result == {"status": "training completed", "model_name": "test_model"}
        
        # Check if the scalers were saved
        mock_joblib_dump.assert_called()
        assert mock_joblib_dump.call_count == 3  # scaler, target_scaler, coco_encoder should all be saved

@patch('app.usecases.lstm.usecase_train_lstm.get_model', return_value=None)
def test_train_lstm_model_not_found(mock_get_model):
    training_data = [
        WeatherModelData(time="2023-01-01T00:00:00Z", temp=22.0, dwpt=10.0, rhum=60, prcp=0.0, snow=0.0, wdir=180, wspd=10, wpgt=15, pres=1013, tsun=0, coco=3)
    ]

    # Expect an HTTPException due to model not found
    with pytest.raises(HTTPException, match="Model not found"):
        train_lstm("unknown_model", training_data)

@patch('app.usecases.lstm.usecase_train_lstm.get_model')
def test_train_lstm_no_training_data(mock_get_model):
    mock_get_model.return_value = {"nn_model": MagicMock()}

    # Expect an HTTPException due to empty training data
    with pytest.raises(HTTPException, match="No training data provided or training data is empty"):
        train_lstm("test_model", [])
