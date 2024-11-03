#app\usecases\lstm\usecase_train_lstm_test.py
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from sklearn.preprocessing import StandardScaler
from app.repositories.memory import models, save_model
from app.usecases.lstm.usecase_train_lstm import train_lstm
from app.apis.models.weather_model_data import WeatherModelData

# Reset memory before each test
def setup_function():
    models.clear()

# Test 1: Verify that a 404 error is raised if the model does not exist
def test_train_lstm_model_not_found():
    training_data = [
        WeatherModelData(time="2023-01-01T00:00:00Z", temp=22.0, dwpt=10.0, rhum=60, prcp=0.0, snow=0.0, wdir=180, wspd=10, wpgt=15, pres=1013, tsun=0, coco=3)
    ]
    
    with pytest.raises(Exception) as excinfo:
        train_lstm("non_existent_model", training_data)  # Non-existent model
    
    assert excinfo.value.status_code == 404
    assert str(excinfo.value.detail) == "Model not found"

# Test 2: Verify that a 400 error is raised if no training data is provided
def test_train_lstm_no_training_data():
    # Save an empty model
    save_model("test_model", {"nn_model": None})

    with pytest.raises(Exception) as excinfo:
        train_lstm("test_model", None)
    
    assert excinfo.value.status_code == 400
    assert str(excinfo.value.detail) == "No training data provided or training data is empty"

# Test 3: Verify that a 400 error is raised if the training data is empty
def test_train_lstm_empty_training_data():
    save_model("test_model", {"nn_model": None})

    with pytest.raises(Exception) as excinfo:
        train_lstm("test_model", [])
    
    assert excinfo.value.status_code == 400
    assert str(excinfo.value.detail) == "No training data provided or training data is empty"

# Test 4: Verify that a ValueError is raised if NaN values are present in X_train
@patch('app.usecases.lstm.usecase_train_lstm.prepare_sequences', return_value=(np.array([[np.nan]]), np.array([0.5])))
@patch('app.usecases.lstm.usecase_train_lstm.preprocess_data')
def test_train_lstm_nan_in_data(mock_preprocess_data, mock_prepare_sequences):
    mock_preprocess_data.return_value = (
        pd.DataFrame(), np.array([0.5]), MagicMock(), MagicMock(), MagicMock()
    )
    save_model("test_model", {"nn_model": None})

    training_data = [
        WeatherModelData(time="2023-01-01T00:00:00Z", temp=22.0, dwpt=10.0, rhum=60, prcp=0.0, snow=0.0, wdir=180, wspd=10, wpgt=15, pres=1013, tsun=0, coco=3)
    ]
    
    with pytest.raises(ValueError, match="X_train contains NaN values."):
        train_lstm("test_model", training_data)

# Test 5: Successful training
@patch('app.usecases.lstm.usecase_train_lstm.prepare_sequences', return_value=(np.array([[[0.2]]]), np.array([[0.5]])))  # Notez que y_train est redimensionné en [[0.5]]
@patch('app.usecases.lstm.usecase_train_lstm.preprocess_data')
def test_train_lstm_success(mock_preprocess_data, mock_prepare_sequences):
    scaler = StandardScaler()
    target_scaler = StandardScaler()
    coco_encoder = StandardScaler()

    mock_preprocess_data.return_value = (
        pd.DataFrame(), np.array([0.5]), scaler, target_scaler, coco_encoder
    )

    save_model("test_model", {"nn_model": None})

    training_data = [
        WeatherModelData(time="2023-01-01T00:00:00Z", temp=22.0, dwpt=10.0, rhum=60, prcp=0.0, snow=0.0, wdir=180, wspd=10, wpgt=15, pres=1013, tsun=0, coco=3)
    ]

    response = train_lstm("test_model", training_data)
    
    assert response["status"] == "training completed"
    assert response["model_name"] == "test_model"
    
    model = models.get("test_model")
    assert "nn_model" in model