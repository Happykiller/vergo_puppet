# app\usecases\lstm\usecase_train_lstm_test.py
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from sklearn.preprocessing import StandardScaler

from app.services.bdd.models.model_data import ModelData
from app.apis.models.weather_model_data import WeatherModelData
from app.usecases.lstm.usecase_train_lstm import TrainLSTMUsecaseDto, train_lstm


# Test: Verify that an exception is raised if the model does not exist
def test_train_lstm_model_not_found(patch_inversify):
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_model.return_value = None

    training_data = [
        WeatherModelData(
            time="2023-01-01T00:00:00Z", temp=22.0, dwpt=10.0, rhum=60, prcp=0.0,
            snow=0.0, wdir=180, wspd=10, wpgt=15, pres=1013, tsun=0, coco=3
        )
    ]

    with pytest.raises(Exception, match="Model not found"):
        train_lstm(TrainLSTMUsecaseDto(name="non_existent_model", training_data=training_data, inversify=mock_inversify))


# Test: Verify that an exception is raised if no training data is provided
def test_train_lstm_no_training_data(patch_inversify):
    mock_inversify, mock_bdd = patch_inversify

    mock_bdd.get_model.return_value = MagicMock(nn_model=None)

    with pytest.raises(Exception, match="No training data provided or training data is empty"):
        train_lstm(TrainLSTMUsecaseDto(name="test_model", training_data=None, inversify=mock_inversify))


# Test: Verify that an exception is raised if the training data is empty
def test_train_lstm_empty_training_data(patch_inversify):
    mock_inversify, mock_bdd = patch_inversify

    mock_bdd.get_model.return_value = MagicMock(nn_model=None)

    with pytest.raises(Exception, match="No training data provided or training data is empty"):
        train_lstm(TrainLSTMUsecaseDto(name="test_model", training_data=[], inversify=mock_inversify))


# Test: Verify that an exception is raised if NaN values are present in X_train
@patch('app.usecases.lstm.usecase_train_lstm.prepare_sequences', return_value=(np.array([[np.nan]]), np.array([0.5])))
@patch('app.usecases.lstm.usecase_train_lstm.preprocess_data')
def test_train_lstm_nan_in_data(mock_preprocess_data, mock_prepare_sequences, patch_inversify):
    mock_inversify, mock_bdd = patch_inversify

    mock_bdd.get_model.return_value = MagicMock(nn_model=None)
    mock_preprocess_data.return_value = (
        pd.DataFrame(), np.array([0.5]), MagicMock(), MagicMock(), MagicMock()
    )

    training_data = [
        WeatherModelData(
            time="2023-01-01T00:00:00Z", temp=22.0, dwpt=10.0, rhum=60, prcp=0.0,
            snow=0.0, wdir=180, wspd=10, wpgt=15, pres=1013, tsun=0, coco=3
        )
    ]

    with pytest.raises(Exception, match=r"\[#train_model_simple_nn\]X_train contains NaN values."):
        train_lstm(TrainLSTMUsecaseDto(name="test_model", training_data=training_data, inversify=mock_inversify))


# Test: Successful training
@patch('app.usecases.lstm.usecase_train_lstm.prepare_sequences', return_value=(np.array([[[0.2]]]), np.array([[0.5]])))
@patch('app.usecases.lstm.usecase_train_lstm.preprocess_data')
def test_train_lstm_success(mock_preprocess_data, mock_prepare_sequences, patch_inversify):
    mock_inversify, mock_bdd = patch_inversify

    # Simulate the return value for an existing model
    mock_bdd.get_model.return_value = ModelData(
        nn_model=None,
        name="test_model",
        neural_network_type="LSTMNN"
    )

    # Simulate the results of preprocess_data
    scaler = StandardScaler()
    target_scaler = StandardScaler()
    encoder = MagicMock()
    mock_preprocess_data.return_value = (
        pd.DataFrame(), np.array([0.5]), scaler, target_scaler, encoder
    )

    # List to capture arguments passed to update_model
    updated_models = []

    def mock_update_model(model_data):
        # Capture the updated model
        updated_models.append(model_data)

    mock_bdd.update_model.side_effect = mock_update_model

    # Simulated training data
    training_data = [
        WeatherModelData(
            time="2023-01-01T00:00:00Z", temp=22.0, dwpt=10.0, rhum=60, prcp=0.0,
            snow=0.0, wdir=180, wspd=10, wpgt=15, pres=1013, tsun=0, coco=3
        )
    ]

    # Call the function to test
    response = train_lstm(TrainLSTMUsecaseDto(name="test_model", training_data=training_data, inversify=mock_inversify))

    # Validate the response
    assert response["status"] == "training completed"
    assert response["model_name"] == "test_model"

    # Verify that update_model was called once
    mock_bdd.update_model.assert_called_once()

    # Validate the attributes of the updated model
    updated_model = updated_models[0]  # Retrieve the captured model
    assert updated_model.name == "test_model"
    assert updated_model.neural_network_type == "LSTMNN"
    assert updated_model.nn_model is not None
    assert updated_model.scaler is scaler
    assert updated_model.target_scaler is target_scaler
    assert updated_model.encoder is encoder
