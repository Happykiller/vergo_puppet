#app\usecases\simple\usecase_search_simple_test.py
import pytest
from unittest.mock import patch, MagicMock
from fastapi import HTTPException  # type: ignore
from app.apis.models.simple_nn_search_model_data import SimpleNNSearchModelData
from app.usecases.simple.usecase_search_simple import search_model_simple_nn

# Test successful search with a SimpleNN model
@patch('app.usecases.simple.usecase_search_simple.joblib.load')
@patch('app.usecases.simple.usecase_search_simple.predict')
@patch('app.usecases.simple.usecase_search_simple.get_model')
def test_search_model_simple_nn_success(mock_get_model, mock_predict, mock_joblib_load):
    # Mock model returned by get_model
    mock_get_model.return_value = {
        "nn_model": MagicMock(),
        "encoder_filename": "encoder.pkl",
        "scaler_filename": "scaler.pkl",
        "indices_filename": "indices.pkl",
        "targets_mean": 0.5,
        "targets_std": 0.2
    }

    # Simulate loading encoder, scaler, and indices files
    mock_joblib_load.side_effect = [
        MagicMock(),  # Mock encoder
        MagicMock(),  # Mock scaler
        {"categorical_indices": [0, 1], "numerical_indices": [2, 3]}  # Mock indices
    ]

    # Mock model prediction
    mock_predict.return_value = 350000

    # Create dummy search data
    search_data = SimpleNNSearchModelData(
        type=1,
        surface=100,
        pieces=4,
        floor=2,
        parking=1,
        balcon=0,
        ascenseur=1,
        orientation=1,
        transports=1,
        neighborhood=8
    )

    # Call the search_model_simple_nn function
    result = search_model_simple_nn("test_model", search_data)

    # Verify that the predict function was called with the correct arguments
    mock_predict.assert_called_once()

    # Check the search result
    assert result == {"predicted_price": 350000}

# Test when the model is not found
@patch('app.usecases.simple.usecase_search_simple.get_model', return_value=None)
def test_search_model_simple_nn_model_not_found(mock_get_model):
    search_data = SimpleNNSearchModelData(
        type=1,
        surface=100,
        pieces=4,
        floor=2,
        parking=1,
        balcon=0,
        ascenseur=1,
        orientation=1,
        transports=1,
        neighborhood=8
    )

    # Check that an HTTP 404 exception is raised if the model is not found
    with pytest.raises(HTTPException) as exc_info:
        search_model_simple_nn("unknown_model", search_data)
    
    # Confirm the exception is HTTPException with status 404
    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Model not found"

# Test when the model is not yet trained
@patch('app.usecases.simple.usecase_search_simple.get_model')
def test_search_model_simple_nn_model_not_trained(mock_get_model):
    # Mock a model without nn_model
    mock_get_model.return_value = {
        "nn_model": None,
        "encoder_filename": "encoder.pkl",
        "scaler_filename": "scaler.pkl",
        "indices_filename": "indices.pkl",
        "targets_mean": 0.5,
        "targets_std": 0.2
    }

    search_data = SimpleNNSearchModelData(
        type=1,
        surface=100,
        pieces=4,
        floor=2,
        parking=1,
        balcon=0,
        ascenseur=1,
        orientation=1,
        transports=1,
        neighborhood=8
    )

    # Check that an HTTP 400 exception is raised if the model is not yet trained
    with pytest.raises(HTTPException) as exc_info:
        search_model_simple_nn("test_model", search_data)

    # Confirm the exception is HTTPException with status 400
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Model not trained yet"

# Test when encoder, scaler, or indices files are missing
@patch('app.usecases.simple.usecase_search_simple.get_model')
def test_search_model_simple_nn_missing_files(mock_get_model):
    # Mock a model without encoder, scaler, or indices
    mock_get_model.return_value = {
        "nn_model": MagicMock(),
        "encoder_filename": None,
        "scaler_filename": None,
        "indices_filename": None,
        "targets_mean": 0.5,
        "targets_std": 0.2
    }

    search_data = SimpleNNSearchModelData(
        type=1,
        surface=100,
        pieces=4,
        floor=2,
        parking=1,
        balcon=0,
        ascenseur=1,
        orientation=1,
        transports=1,
        neighborhood=8
    )

    # Check that an HTTP 400 exception is raised if files are missing
    with pytest.raises(HTTPException) as exc_info:
        search_model_simple_nn("test_model", search_data)

    # Confirm the exception is HTTPException with status 400
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Missing encoder, scaler, or indices in the model"

# Test when target normalization parameters are missing
@patch('app.usecases.simple.usecase_search_simple.joblib.load')  # Simulate joblib.load
@patch('app.usecases.simple.usecase_search_simple.get_model')
def test_search_model_simple_nn_missing_normalization_parameters(mock_get_model, mock_joblib_load):
    # Mock a model without target normalization parameters
    mock_get_model.return_value = {
        "nn_model": MagicMock(),
        "encoder_filename": "encoder.pkl",
        "scaler_filename": "scaler.pkl",
        "indices_filename": "indices.pkl",
        "targets_mean": None,
        "targets_std": None
    }

    # Simulate loading of encoder, scaler, and indices with joblib.load
    mock_joblib_load.side_effect = [
        MagicMock(),  # Mock encoder
        MagicMock(),  # Mock scaler
        {"categorical_indices": [0, 1], "numerical_indices": [2, 3]}  # Mock indices
    ]

    # Create dummy search data
    search_data = SimpleNNSearchModelData(
        type=1,
        surface=100,
        pieces=4,
        floor=2,
        parking=1,
        balcon=0,
        ascenseur=1,
        orientation=1,
        transports=1,
        neighborhood=8
    )

    # Check that an HTTP 400 exception is raised if normalization parameters are missing
    with pytest.raises(HTTPException) as exc_info:
        search_model_simple_nn("test_model", search_data)

    # Confirm the exception is HTTPException with status 400
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Missing normalization parameters in the model"
