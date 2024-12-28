# app\usecases\simple\usecase_create_simple_test.py
import pytest
from unittest.mock import MagicMock, patch
from fastapi import HTTPException  # type: ignore
from app.usecases.simple.usecase_create_simple import CreateSimpleUsecaseDto, create_model_simple_nn

# Test when the model is created successfully
def test_create_model_simple_nn_success(patch_inversify):
    """
    Test that the model is created successfully.
    """
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.model_exists.return_value = False  # Simulate model does not exist

    model_name = "test_model"
    
    # Call the create_model_simple_nn function
    response = create_model_simple_nn(CreateSimpleUsecaseDto(name=model_name, inversify=mock_inversify))
    
    # Verify that the save_model function was called correctly
    mock_bdd.save_model.assert_called_once_with(model_name, {"neural_network_type": "SimpleNN"})
    
    # Verify the response
    assert response == {"status": "model created", "model_name": model_name}

# Test when the model already exists
def test_create_model_simple_nn_model_already_exists(patch_inversify):
    """
    Test that the function raises an HTTP 400 exception when the model already exists.
    """
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.model_exists.return_value = True  # Simulate model already exists

    model_name = "existing_model"
    
    # Check that an HTTP 400 exception is raised
    with pytest.raises(HTTPException) as exc_info:
        create_model_simple_nn(CreateSimpleUsecaseDto(name=model_name, inversify=mock_inversify))
    
    # Verify the error message and status code
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Model already exists"

# Test when the model is saved with the correct data
def test_create_model_simple_nn_save_called_with_correct_data(patch_inversify):
    """
    Test that the model is saved with the correct data.
    """
    # Mock the Inversify instance and BDD service
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.model_exists.return_value = False  # Simulate model does not exist

    model_name = "new_model"
    
    # Call the create_model_simple_nn function
    response = create_model_simple_nn(CreateSimpleUsecaseDto(name=model_name, inversify=mock_inversify))
    
    # Verify that save_model was called with the correct arguments
    expected_model_data = {
        "neural_network_type": "SimpleNN"
    }
    mock_bdd.save_model.assert_called_once_with(model_name, expected_model_data)
    
    # Verify the response
    assert response == {"status": "model created", "model_name": model_name}
