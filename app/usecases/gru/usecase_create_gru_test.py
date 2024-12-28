# app\usecases\gru\usecase_create_gru_test.py
import pytest
from fastapi import HTTPException  # type: ignore

from app.usecases.gru.usecase_create_gru import CreateGRUUsecaseDto, create_model_gru

# Test when the GRU model is created successfully
def test_create_model_gru_success(patch_inversify):
    """
    Test that the create_model_gru function successfully creates a model when it doesn't already exist.
    """
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.model_exists.return_value = False  # Simulate model does not exist
    model_name = "test_gru_model"
    
    # Call the create_model_gru function
    response = create_model_gru(CreateGRUUsecaseDto(name=model_name, inversify=mock_inversify))
    
    # Verify that save_model was called once with the correct model data
    mock_bdd.save_model.assert_called_once_with(model_name, {"neural_network_type": "GRU"})
    
    # Check the response for the expected success message
    assert response == {"status": "model created", "model_name": model_name}

# Test when the GRU model already exists
def test_create_model_gru_model_already_exists(patch_inversify):
    """
    Test that the create_model_gru function raises an HTTPException when a model with the same name already exists.
    """
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.model_exists.return_value = True  # Simulate model does not exist
    model_name = "existing_gru_model"
    
    # Check that an HTTP 400 exception is raised
    with pytest.raises(HTTPException) as exc_info:
        create_model_gru(CreateGRUUsecaseDto(name=model_name, inversify=mock_inversify))
    
    # Verify the exception's status code and error message
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Model already exists"

# Test when the GRU model is saved with correct data
def test_create_model_gru_save_called_with_correct_data(patch_inversify):
    """
    Test that the create_model_gru function calls save_model with the correct model data.
    """
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.model_exists.return_value = False  # Simulate model does not exist
    model_name = "new_gru_model"
    
    # Call the create_model_gru function
    response = create_model_gru(CreateGRUUsecaseDto(name=model_name, inversify=mock_inversify))
    
    # Define the expected data to be saved
    expected_model_data = {
        "neural_network_type": "GRU"
    }
    # Verify save_model was called with the expected arguments
    mock_bdd.save_model.assert_called_once_with(model_name, expected_model_data)
    
    # Check the response for the expected success message
    assert response == {"status": "model created", "model_name": model_name}

