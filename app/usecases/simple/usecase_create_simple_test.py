#app\usecases\simple\usecase_create_simple_test.py
import pytest
from unittest.mock import patch
from fastapi import HTTPException  # type: ignore
from app.usecases.simple.usecase_create_simple import create_model_simple_nn

# Test when the model is created successfully
@patch('app.usecases.simple.usecase_create_simple.save_model')
@patch('app.usecases.simple.usecase_create_simple.model_exists', return_value=False)  # Simulate that the model does not exist
def test_create_model_simple_nn_success(mock_model_exists, mock_save_model):
    model_name = "test_model"
    
    # Call the create_model_simple_nn function
    response = create_model_simple_nn(model_name)
    
    # Verify that the save_model function was called correctly
    mock_save_model.assert_called_once_with(model_name, {"neural_network_type": "SimpleNN"})
    
    # Verify the response
    assert response == {"status": "model created", "model_name": model_name}

# Test when the model already exists
@patch('app.usecases.simple.usecase_create_simple.model_exists', return_value=True)  # Simulate that the model already exists
def test_create_model_simple_nn_model_already_exists(mock_model_exists):
    model_name = "existing_model"
    
    # Check that an HTTP 400 exception is raised
    with pytest.raises(HTTPException) as exc_info:
        create_model_simple_nn(model_name)
    
    # Verify the error message and status code
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Model already exists"

# Test when the model is saved with the correct data
@patch('app.usecases.simple.usecase_create_simple.model_exists', return_value=False)  # Simulate that the model does not exist
@patch('app.usecases.simple.usecase_create_simple.save_model')  # Simulate saving the model
def test_create_model_simple_nn_save_called_with_correct_data(mock_save_model, mock_model_exists):
    model_name = "new_model"
    
    # Call the create_model_simple_nn function
    response = create_model_simple_nn(model_name)
    
    # Verify that save_model was called with the correct arguments
    expected_model_data = {
        "neural_network_type": "SimpleNN"
    }
    mock_save_model.assert_called_once_with(model_name, expected_model_data)
    
    # Verify the response
    assert response == {"status": "model created", "model_name": model_name}
