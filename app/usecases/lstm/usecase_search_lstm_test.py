# app\usecases\lstm\usecase_search_lstm_test.py
from app.services.bdd.models.model_data import ModelData
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from app.apis.models.weather_model_data import WeatherSearchModelData
from app.usecases.lstm.usecase_search_lstm import SearchLSTMUsecaseDto, search_lstm


# Test for successful prediction
@patch('app.usecases.lstm.usecase_search_lstm.predict_nn_lstm')
@patch('app.usecases.lstm.usecase_search_lstm.preprocess_input_data')
@patch('app.usecases.lstm.usecase_search_lstm.inverse_transform_predictions')
def test_search_lstm_success(mock_inverse_transform, mock_preprocess_input, mock_predict_nn_lstm, patch_inversify):
    mock_inversify, mock_bdd = patch_inversify

    # Simulate model data
    mock_model = MagicMock()
    mock_model.nn_model = MagicMock()
    mock_model.nn_model.eval = MagicMock()
    mock_model.scaler = MagicMock()
    mock_model.encoder = MagicMock()
    mock_model.target_scaler = MagicMock()
    mock_bdd.get_model.return_value = mock_model

    # Mock preprocessing and predictions
    mock_preprocess_input.return_value = pd.DataFrame([[0.1, 0.2]], columns=["feature1", "feature2"])
    mock_predict_nn_lstm.return_value = np.array([[0.5]])
    mock_inverse_transform.return_value = np.array([22.5])

    # Test input data
    input_data = WeatherSearchModelData(
        time=datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc),
        dwpt=10.0, rhum=60, prcp=0.0, snow=0.0, wdir=180, wspd=10, wpgt=15,
        pres=1013, tsun=0, coco=3
    )

    # Call the function
    result = search_lstm(SearchLSTMUsecaseDto(name="test_model", search=input_data, inversify=mock_inversify))

    # Assertions
    assert result["prediction"] == 22.5, "Prediction should match the transformed value"
    mock_predict_nn_lstm.assert_called_once()

    # Validate the DataFrame passed to preprocess_input_data
    called_df = mock_preprocess_input.call_args[0][0]
    pd.testing.assert_frame_equal(
        called_df,
        pd.DataFrame([input_data.dict()]),
        check_dtype=False  # Disable dtype checking for flexibility in mocks
    )

    mock_preprocess_input.assert_called_once_with(called_df, mock_model.scaler, mock_model.encoder)
    mock_inverse_transform.assert_called_once_with(np.array([[0.5]]), mock_model.target_scaler)


# Test for model not found
def test_search_lstm_model_not_found(patch_inversify):
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_model.return_value = ModelData(
        nn_model=None,
        name="test_model",
        neural_network_type="LSTMNN"
    )

    input_data = WeatherSearchModelData(
        time=datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc),
        dwpt=10.0, rhum=60, prcp=0.0, snow=0.0, wdir=180, wspd=10, wpgt=15,
        pres=1013, tsun=0, coco=3
    )

    # Expect exception
    with pytest.raises(Exception, match="Model not found"):
        search_lstm(SearchLSTMUsecaseDto(name="unknown_model", search=input_data, inversify=mock_inversify))


# Test for missing scaler or encoder files
@patch('app.usecases.lstm.usecase_search_lstm.preprocess_input_data', side_effect=FileNotFoundError("File not found"))
def test_search_lstm_missing_files(mock_preprocess_input, patch_inversify):
    mock_inversify, mock_bdd = patch_inversify

    # Simulate model data
    mock_model = MagicMock()
    mock_model.nn_model = MagicMock()
    mock_model.scaler = MagicMock()
    mock_model.encoder = MagicMock()
    mock_model.target_scaler = MagicMock()
    mock_bdd.get_model.return_value = mock_model

    input_data = WeatherSearchModelData(
        time=datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc),
        dwpt=10.0, rhum=60, prcp=0.0, snow=0.0, wdir=180, wspd=10, wpgt=15,
        pres=1013, tsun=0, coco=3
    )

    # Expect exception
    with pytest.raises(Exception, match=r"\[#search_lstm\]File not found"):
        search_lstm(SearchLSTMUsecaseDto(name="test_model", search=input_data, inversify=mock_inversify))
