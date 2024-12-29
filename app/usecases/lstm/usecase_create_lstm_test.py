# app\usecases\lstm\usecase_create_lstm_test.py
import pytest
from fastapi import HTTPException  # type: ignore

from app.usecases.lstm.usecase_create_lstm import CreateLSTMUsecaseDto, create_lstm

# Test when the LSTM model is successfully created
def test_create_lstm_success(patch_inversify):
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify

    model_name = "test_model"

    # Simulate model data returned by get_model
    mock_bdd.model_exists.return_value = False
    
    # Call the create_lstm function
    response = create_lstm(CreateLSTMUsecaseDto(name=model_name, inversify=mock_inversify))
    
    # Verify that save_model was called once with the correct model data
    mock_bdd.save_model.assert_called_once_with(model_name, {"neural_network_type": "LSTM"})
    
    # Check the function response
    assert response == {"status": "model created", "model_name": model_name}

# Test when the model already exists
def test_create_lstm_model_already_exists(patch_inversify):
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify

    model_name = "existing_model"

    # Simulate model data returned by get_model
    mock_bdd.model_exists.return_value = True
    
    # Ensure an HTTP 400 exception is raised if the model already exists
    with pytest.raises(HTTPException) as exc_info:
        create_lstm(CreateLSTMUsecaseDto(name=model_name, inversify=mock_inversify))
    
    # Check that the exception has the expected status code and message
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Model already exists"

# Test to verify save_model is called with the correct data
def test_create_lstm_save_called_with_correct_data(patch_inversify):
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify

    model_name = "new_model"

    # Simulate model data returned by get_model
    mock_bdd.model_exists.return_value = False
    
    # Call the create_lstm function
    response = create_lstm(CreateLSTMUsecaseDto(name=model_name, inversify=mock_inversify))
    
    # Check that save_model was called with the expected arguments
    expected_model_data = {
        "neural_network_type": "LSTM"
    }
    mock_bdd.save_model.assert_called_once_with(model_name, expected_model_data)
    
    # Verify the response
    assert response == {"status": "model created", "model_name": model_name}
