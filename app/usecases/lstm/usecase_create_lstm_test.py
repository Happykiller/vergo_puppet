#app\usecases\lstm\usecase_create_lstm_test.py
import pytest
from unittest.mock import patch
from fastapi import HTTPException  # type: ignore
from app.usecases.lstm.usecase_create_lstm import create_lstm

# Test when the LSTM model is successfully created
@patch('app.usecases.lstm.usecase_create_lstm.save_model')
@patch('app.usecases.lstm.usecase_create_lstm.model_exists', return_value=False)  # Simulate that the model does not exist
def test_create_lstm_success(mock_model_exists, mock_save_model):
    model_name = "test_model"
    
    # Call the create_lstm function
    response = create_lstm(model_name)
    
    # Verify that save_model was called once with the correct model data
    mock_save_model.assert_called_once_with(model_name, {"neural_network_type": "LSTM"})
    
    # Check the function response
    assert response == {"status": "model created", "model_name": model_name}

# Test when the model already exists
@patch('app.usecases.lstm.usecase_create_lstm.model_exists', return_value=True)  # Simulate that the model already exists
def test_create_lstm_model_already_exists(mock_model_exists):
    model_name = "existing_model"
    
    # Ensure an HTTP 400 exception is raised if the model already exists
    with pytest.raises(HTTPException) as exc_info:
        create_lstm(model_name)
    
    # Check that the exception has the expected status code and message
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Model already exists"

# Test to verify save_model is called with the correct data
@patch('app.usecases.lstm.usecase_create_lstm.model_exists', return_value=False)  # Simulate that the model does not exist
@patch('app.usecases.lstm.usecase_create_lstm.save_model')  # Simulate the model saving
def test_create_lstm_save_called_with_correct_data(mock_save_model, mock_model_exists):
    model_name = "new_model"
    
    # Call the create_lstm function
    response = create_lstm(model_name)
    
    # Check that save_model was called with the expected arguments
    expected_model_data = {
        "neural_network_type": "LSTM"
    }
    mock_save_model.assert_called_once_with(model_name, expected_model_data)
    
    # Verify the response
    assert response == {"status": "model created", "model_name": model_name}
