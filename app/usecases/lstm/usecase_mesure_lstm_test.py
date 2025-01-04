# app\usecases\lstm\usecase_mesure_lstm_test.py
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from app.services.bdd.models.model_data import ModelData
from app.apis.models.weather_model_data import WeatherModelData
from app.usecases.lstm.usecase_mesure_lstm import MesureLSTMUsecaseDto, mesure_lstm


# Test for successful measurement
@patch('app.usecases.lstm.usecase_mesure_lstm.predict_nn_lstm')
@patch('app.usecases.lstm.usecase_mesure_lstm.preprocess_input_data')
@patch('app.usecases.lstm.usecase_mesure_lstm.inverse_transform_predictions')
def test_mesure_lstm_success(mock_inverse_transform, mock_preprocess_input, mock_predict_nn_lstm, patch_inversify):
    mock_inversify, mock_bdd = patch_inversify

    # Mock model setup
    mock_model = ModelData(
        name="test_model",
        neural_network_type="LSTMNN",
        nn_model=MagicMock(),
        scaler=MagicMock(),
        encoder=MagicMock(),
        target_scaler=MagicMock()
    )
    mock_bdd.get_model.return_value = mock_model

    # Mock preprocessing and predictions
    mock_preprocess_input.return_value = pd.DataFrame([[0.1, 0.2]], columns=["feature1", "feature2"])
    mock_predict_nn_lstm.return_value = np.array([[0.5]])
    mock_inverse_transform.return_value = np.array([22.5])

    # Test data
    test_data = [
        WeatherModelData(
            time="2023-01-01T00:00:00Z", temp=22.0, dwpt=10.0, rhum=60, prcp=0.0,
            snow=0.0, wdir=180, wspd=10, wpgt=15, pres=1013, tsun=0, coco=3
        )
    ]

    # Run the function
    result = mesure_lstm(MesureLSTMUsecaseDto(name="test_model", test_data=test_data, inversify=mock_inversify))

    # Assertions on results
    assert result["mae"] == pytest.approx(0.5, rel=1e-2), "MAE should match the difference between true and predicted"
    assert result["mape"] == pytest.approx(2.27, rel=1e-2), "MAPE should be calculated correctly as a percentage"
    assert result["test_count"] == 1, "Test count should match the input data length"


# Test when the model is not found
def test_mesure_lstm_model_not_found(patch_inversify):
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_model.return_value = None

    test_data = [
        WeatherModelData(
            time="2023-01-01T00:00:00Z", temp=22.0, dwpt=10.0, rhum=60, prcp=0.0,
            snow=0.0, wdir=180, wspd=10, wpgt=15, pres=1013, tsun=0, coco=3
        )
    ]

    with pytest.raises(Exception, match="Model not found"):
        mesure_lstm(MesureLSTMUsecaseDto(name="unknown_model", test_data=test_data, inversify=mock_inversify))


## Test for missing input data
def test_mesure_lstm_no_test_data(patch_inversify):
    mock_inversify, mock_bdd = patch_inversify

    mock_bdd.get_model.return_value = ModelData(
        name="test_model",
        neural_network_type="LSTMNN",
        nn_model=MagicMock(),
        scaler=MagicMock(),
        encoder=MagicMock(),
        target_scaler=MagicMock()
    )

    # Empty input data
    test_data = []

    # Run the function
    result = mesure_lstm(MesureLSTMUsecaseDto(name="test_model", test_data=test_data, inversify=mock_inversify))

    # Assertions
    assert result["test_count"] == 0, "Test count should be zero for empty input data"
    assert result["mae"] == pytest.approx(0.0), "MAE should be zero for no data"
    assert result["mape"] == pytest.approx(0.0), "MAPE should be zero for no data"


## Test for missing normalization parameters
def test_mesure_lstm_missing_normalization_parameters(patch_inversify):
    mock_inversify, mock_bdd = patch_inversify

    # Mock model with missing normalization parameters
    mock_model = ModelData(
        name="test_model",
        neural_network_type="LSTMNN",
        nn_model=MagicMock(),
        scaler=None,  # Missing scaler
        encoder=MagicMock(),
        target_scaler=None  # Missing target_scaler
    )
    mock_bdd.get_model.return_value = mock_model

    test_data = [
        WeatherModelData(
            time="2023-01-01T00:00:00Z", temp=22.0, dwpt=10.0, rhum=60, prcp=0.0,
            snow=0.0, wdir=180, wspd=10, wpgt=15, pres=1013, tsun=0, coco=3
        )
    ]

    with pytest.raises(Exception, match="Missing normalization parameters in the model"):
        mesure_lstm(MesureLSTMUsecaseDto(name="test_model", test_data=test_data, inversify=mock_inversify))
