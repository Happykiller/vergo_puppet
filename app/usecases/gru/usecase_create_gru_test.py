# app\usecases\gru\usecase_create_gru_test.py
import pytest

from app.services.bdd.models.model_data import ModelData
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

    # Verify that save_model was called once with the correct ModelData
    expected_model_data = ModelData(name=model_name, neural_network_type="GRU")
    mock_bdd.save_model.assert_called_once()
    actual_model_data = mock_bdd.save_model.call_args[0][0]  # Get the first positional argument

    # Ensure the correct instance and attributes
    assert isinstance(actual_model_data, ModelData)
    assert actual_model_data.name == expected_model_data.name
    assert actual_model_data.neural_network_type == expected_model_data.neural_network_type

    # Check the response for the expected success message
    assert response == {"status": "model created", "model_name": model_name}


# Test when the GRU model already exists
def test_create_model_gru_model_already_exists(patch_inversify):
    """
    Test that the create_model_gru function raises an exception when a model with the same name already exists.
    """
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.model_exists.return_value = True  # Simulate model already exists
    model_name = "existing_gru_model"

    # Check that an exception is raised
    with pytest.raises(Exception) as exc_info:
        create_model_gru(CreateGRUUsecaseDto(name=model_name, inversify=mock_inversify))

    # Verify the exception's error message
    assert str(exc_info.value) == "Model already exists"


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

    # Verify save_model was called with the correct ModelData
    expected_model_data = ModelData(name=model_name, neural_network_type="GRU")
    mock_bdd.save_model.assert_called_once()
    actual_model_data = mock_bdd.save_model.call_args[0][0]  # Get the first positional argument

    # Ensure the correct instance and attributes
    assert isinstance(actual_model_data, ModelData)
    assert actual_model_data.name == expected_model_data.name
    assert actual_model_data.neural_network_type == expected_model_data.neural_network_type

    # Check the response for the expected success message
    assert response == {"status": "model created", "model_name": model_name}
