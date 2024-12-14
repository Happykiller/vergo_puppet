#app\usecases\lstm\usecase_search_lstm_test.py
import pytest
import numpy as np
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
from fastapi import HTTPException # type: ignore
from app.usecases.lstm.usecase_search_lstm import search_lstm
from app.apis.models.weather_model_data import WeatherSearchModelData

# Test for successful prediction
@patch('app.usecases.lstm.usecase_search_lstm.joblib.load')
@patch('app.usecases.lstm.usecase_search_lstm.predict_nn_lstm')
@patch('app.usecases.lstm.usecase_search_lstm.get_model')
def test_search_lstm_success(mock_get_model, mock_predict_nn_lstm, mock_joblib_load):
    # Mock model data to simulate an LSTM model
    mock_model = {
        "nn_model": MagicMock()
    }
    mock_model["nn_model"].eval = MagicMock()
    mock_get_model.return_value = mock_model

    # Mock scaler, target_scaler, and encoder
    mock_scaler = MagicMock()
    mock_target_scaler = MagicMock()
    mock_encoder = MagicMock()
    mock_joblib_load.side_effect = [mock_scaler, mock_target_scaler, mock_encoder]

    # Mock predict function to return a normalized prediction
    mock_predict_nn_lstm.return_value = np.array([[0.5]])  # Simulated normalized prediction

    # Mock inverse transformation to provide a simulated temperature prediction
    mock_target_scaler.inverse_transform.return_value = np.array([[22.5]])  # Simulated temperature prediction

    # Test data with required fields, including time
    input_data = WeatherSearchModelData(
        time=datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc),
        dwpt=10.0, rhum=60, prcp=0.0, snow=0.0, wdir=180, wspd=10, wpgt=15,
        pres=1013, tsun=0, coco=3
    )

    # Run the function
    result = search_lstm("test_model", input_data)

    # Assertions on result
    assert result["prediction"] == 22.5, "The predicted temperature should match the inverse transformed value."

# Test for model not found
@patch('app.usecases.lstm.usecase_search_lstm.get_model', return_value=None)
def test_search_lstm_model_not_found(mock_get_model):
    # Test data with required fields, including time
    input_data = WeatherSearchModelData(
        time=datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc),
        dwpt=10.0, rhum=60, prcp=0.0, snow=0.0, wdir=180, wspd=10, wpgt=15,
        pres=1013, tsun=0, coco=3
    )

    # Expect an HTTP 404 exception if the model is not found
    with pytest.raises(HTTPException) as exc_info:
        search_lstm("unknown_model", input_data)

    # Verify the exception status and message
    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Model not found"

# Test for missing scaler or encoder files
@patch('app.usecases.lstm.usecase_search_lstm.joblib.load', side_effect=FileNotFoundError("File not found"))
@patch('app.usecases.lstm.usecase_search_lstm.get_model')
def test_search_lstm_missing_files(mock_get_model, mock_joblib_load):
    # Mock model data to simulate a model with missing files
    mock_model = {
        "nn_model": MagicMock()
    }
    mock_get_model.return_value = mock_model

    # Test data with required fields, including time
    input_data = WeatherSearchModelData(
        time=datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc),
        dwpt=10.0, rhum=60, prcp=0.0, snow=0.0, wdir=180, wspd=10, wpgt=15,
        pres=1013, tsun=0, coco=3
    )

    # Expect an exception for missing files
    with pytest.raises(FileNotFoundError, match="File not found"):
        search_lstm("test_model", input_data)
