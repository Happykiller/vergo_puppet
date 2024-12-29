#app\usecases\lstm\usecase_mesure_lstm_test.py
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from app.apis.models.weather_model_data import WeatherModelData
from app.usecases.lstm.usecase_mesure_lstm import MesureLSTMUsecaseDto, mesure_lstm

# Test to verify successful measurement with valid data
@patch('app.usecases.lstm.usecase_mesure_lstm.joblib.load')
@patch('app.usecases.lstm.usecase_mesure_lstm.predict_nn_lstm')
def test_mesure_lstm_success(mock_predict_nn_lstm, mock_joblib_load, patch_inversify):
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify

    # Mock model data to simulate an LSTM model
    mock_model = {
        "nn_model": MagicMock()
    }
    mock_bdd.get_model.return_value = mock_model

    # Mock scaler, target_scaler, and encoder
    mock_scaler = MagicMock()
    mock_target_scaler = MagicMock()
    mock_encoder = MagicMock()
    mock_joblib_load.side_effect = [mock_scaler, mock_target_scaler, mock_encoder]

    # Mock prediction function to return a normalized prediction
    mock_predict_nn_lstm.return_value = np.array([[0.5]])  # Simulated normalized prediction

    # Mock inverse transformation to provide a simulated temperature prediction
    mock_target_scaler.inverse_transform.return_value = np.array([[22.5]])  # Simulated temperature prediction

    # Test data
    test_data = [
        WeatherModelData(time="2023-01-01T00:00:00Z", temp=22.0, dwpt=10.0, rhum=60, prcp=0.0, snow=0.0, wdir=180, wspd=10, wpgt=15, pres=1013, tsun=0, coco=3)
    ]

    # Run the function
    result = mesure_lstm(MesureLSTMUsecaseDto(name="test_model", test_data=test_data, inversify=mock_inversify))

    # Assertions on results
    assert result["mae"] is not None, "MAE should be calculated"
    assert result["mape"] is not None, "MAPE should be calculated"
    assert result["test_count"] == 1, "Test count should match the length of the input data"

# Test to verify missing normalization parameters
@patch('app.usecases.lstm.usecase_mesure_lstm.joblib.load')
def test_mesure_lstm_missing_normalization_parameters(mock_joblib_load, patch_inversify):
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify

    # Mock model without normalization parameters
    mock_model = {
        "nn_model": MagicMock(),
        "scaler_filename": "scaler.pkl",
        "target_scaler_filename": "target_scaler.pkl",
        "encoder_filename": "encoder.pkl"
    }
    mock_bdd.get_model.return_value = mock_model
    mock_joblib_load.side_effect = [MagicMock(), MagicMock(), MagicMock()]  # Mock scaler and encoder loading

    # Test data
    test_data = [
        WeatherModelData(time="2023-01-01T00:00:00Z", temp=22.0, dwpt=10.0, rhum=60, prcp=0.0, snow=0.0, wdir=180, wspd=10, wpgt=15, pres=1013, tsun=0, coco=3)
    ]

    # Expect an exception due to missing parameters with a specific error message
    with pytest.raises(Exception, match=r"An error occurred during measurement: Found array with 0 feature\(s\)"):
        mesure_lstm(MesureLSTMUsecaseDto(name="test_model", test_data=test_data, inversify=mock_inversify))
