#app\usecases\lstm\usecase_train_lstm_test.py
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from sklearn.preprocessing import StandardScaler

from app.apis.models.weather_model_data import WeatherModelData
from app.usecases.lstm.usecase_train_lstm import TrainLSTMUsecaseDto, train_lstm

# Test: Verify that a 404 error is raised if the model does not exist
def test_train_lstm_model_not_found(patch_inversify):
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_model.return_value = None

    training_data = [
        WeatherModelData(time="2023-01-01T00:00:00Z", temp=22.0, dwpt=10.0, rhum=60, prcp=0.0, snow=0.0, wdir=180, wspd=10, wpgt=15, pres=1013, tsun=0, coco=3)
    ]
    
    with pytest.raises(Exception) as excinfo:
        train_lstm(TrainLSTMUsecaseDto(name="non_existent_model", training_data=training_data, inversify=mock_inversify))  # Non-existent model
    
    assert excinfo.value.status_code == 404
    assert str(excinfo.value.detail) == "Model not found"

# Test: Verify that a 400 error is raised if no training data is provided
def test_train_lstm_no_training_data(patch_inversify):
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify

    # Save an empty model
    mock_bdd.save_model("test_model", {"nn_model": None})

    with pytest.raises(Exception) as excinfo:
        train_lstm(TrainLSTMUsecaseDto(name="test_model", training_data=None, inversify=mock_inversify))
    
    assert excinfo.value.status_code == 400
    assert str(excinfo.value.detail) == "No training data provided or training data is empty"

# Test: Verify that a 400 error is raised if the training data is empty
def test_train_lstm_empty_training_data(patch_inversify):
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify

    mock_bdd.save_model("test_model", {"nn_model": None})

    with pytest.raises(Exception) as excinfo:
        train_lstm(TrainLSTMUsecaseDto(name="test_model", training_data=[], inversify=mock_inversify))
    
    assert excinfo.value.status_code == 400
    assert str(excinfo.value.detail) == "No training data provided or training data is empty"

# Test: Verify that a ValueError is raised if NaN values are present in X_train
@patch('app.usecases.lstm.usecase_train_lstm.prepare_sequences', return_value=(np.array([[np.nan]]), np.array([0.5])))
@patch('app.usecases.lstm.usecase_train_lstm.preprocess_data')
def test_train_lstm_nan_in_data(mock_preprocess_data, mock_prepare_sequences, patch_inversify):
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify

    mock_preprocess_data.return_value = (
        pd.DataFrame(), np.array([0.5]), MagicMock(), MagicMock(), MagicMock()
    )
    mock_bdd.save_model("test_model", {"nn_model": None})

    training_data = [
        WeatherModelData(time="2023-01-01T00:00:00Z", temp=22.0, dwpt=10.0, rhum=60, prcp=0.0, snow=0.0, wdir=180, wspd=10, wpgt=15, pres=1013, tsun=0, coco=3)
    ]
    
    with pytest.raises(ValueError, match="X_train contains NaN values."):
        train_lstm(TrainLSTMUsecaseDto(name="test_model", training_data=training_data, inversify=mock_inversify))

# Test: Successful training
@patch('app.usecases.lstm.usecase_train_lstm.prepare_sequences', return_value=(np.array([[[0.2]]]), np.array([[0.5]])))  # Notez que y_train est redimensionné en [[0.5]]
@patch('app.usecases.lstm.usecase_train_lstm.preprocess_data')
def test_train_lstm_success(mock_preprocess_data, mock_prepare_sequences, patch_inversify):
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_model.return_value = {
        "nn_model": MagicMock()
    }

    scaler = StandardScaler()
    target_scaler = StandardScaler()
    coco_encoder = StandardScaler()

    mock_preprocess_data.return_value = (
        pd.DataFrame(), np.array([0.5]), scaler, target_scaler, coco_encoder
    )

    mock_bdd.save_model("test_model", {"nn_model": None})

    training_data = [
        WeatherModelData(time="2023-01-01T00:00:00Z", temp=22.0, dwpt=10.0, rhum=60, prcp=0.0, snow=0.0, wdir=180, wspd=10, wpgt=15, pres=1013, tsun=0, coco=3)
    ]

    response = train_lstm(TrainLSTMUsecaseDto(name="test_model", training_data=training_data, inversify=mock_inversify))
    
    assert response["status"] == "training completed"
    assert response["model_name"] == "test_model"
    
    model = mock_bdd.get_model("test_model")
    assert "nn_model" in model