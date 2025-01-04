# app\usecases\lstm\usecase_create_lstm_test.py
import pytest

from app.services.bdd.models.model_data import ModelData
from app.usecases.lstm.usecase_create_lstm import CreateLSTMUsecaseDto, create_lstm

# Test when the LSTM model is successfully created
def test_create_lstm_success(patch_inversify):
    """
    Test that the LSTM model is successfully created when it doesn't already exist.
    """
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify

    model_name = "test_model"

    # Simulate model data returned by model_exists
    mock_bdd.model_exists.return_value = False

    # Call the create_lstm function
    response = create_lstm(CreateLSTMUsecaseDto(name=model_name, inversify=mock_inversify))

    # Verify that save_model was called with the correct ModelData
    expected_model_data = ModelData(name=model_name, neural_network_type="LSTM")
    mock_bdd.save_model.assert_called_once()
    actual_model_data = mock_bdd.save_model.call_args[0][0]  # Get the first positional argument

    # Ensure the correct instance and attributes
    assert isinstance(actual_model_data, ModelData)
    assert actual_model_data.name == expected_model_data.name
    assert actual_model_data.neural_network_type == expected_model_data.neural_network_type

    # Verify the response
    assert response == {"status": "model created", "model_name": model_name}


# Test when the model already exists
def test_create_lstm_model_already_exists(patch_inversify):
    """
    Test that an exception is raised when a model with the same name already exists.
    """
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify

    model_name = "existing_model"

    # Simulate model data returned by model_exists
    mock_bdd.model_exists.return_value = True

    # Ensure an exception is raised if the model already exists
    with pytest.raises(Exception) as exc_info:
        create_lstm(CreateLSTMUsecaseDto(name=model_name, inversify=mock_inversify))

    # Verify the exception message
    assert str(exc_info.value) == "[#create_lstm]Model already exists"


# Test to verify save_model is called with the correct data
def test_create_lstm_save_called_with_correct_data(patch_inversify):
    """
    Test that the save_model function is called with the correct ModelData.
    """
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify

    model_name = "new_model"

    # Simulate model data returned by model_exists
    mock_bdd.model_exists.return_value = False

    # Call the create_lstm function
    response = create_lstm(CreateLSTMUsecaseDto(name=model_name, inversify=mock_inversify))

    # Verify that save_model was called with the correct ModelData
    expected_model_data = ModelData(name=model_name, neural_network_type="LSTM")
    mock_bdd.save_model.assert_called_once()
    actual_model_data = mock_bdd.save_model.call_args[0][0]  # Get the first positional argument

    # Ensure the correct instance and attributes
    assert isinstance(actual_model_data, ModelData)
    assert actual_model_data.name == expected_model_data.name
    assert actual_model_data.neural_network_type == expected_model_data.neural_network_type

    # Verify the response
    assert response == {"status": "model created", "model_name": model_name}
