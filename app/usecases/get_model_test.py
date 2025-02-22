# app\usecases\get_model_test.py
import pytest
from unittest.mock import MagicMock, patch
from app.usecases.get_model import get_model_usecase
from app.services.bdd.models.model_data import ModelData

@patch("app.inversify.Inversify")
def test_get_model_usecase_found(mock_inversify_class):
    """
    Test that get_model_usecase returns a valid ModelData if the model is found.
    """
    # Create a mock Inversify instance
    mock_inversify = MagicMock()
    mock_bdd = MagicMock()

    # Mock methods and their return values
    mock_bdd.get_model.return_value = ModelData(
        name="my_model",
        neural_network_type="GRU",
        nn_model=None,
        glossary=[],
        indexed_dictionary=None
    )
    mock_inversify.get_bdd.return_value = mock_bdd
    mock_inversify_class.return_value = mock_inversify

    # Call the function
    result = get_model_usecase("my_model", mock_inversify)

    # Assertions
    assert result is not None, "Expected a ModelData object, got None."
    assert result.name == "my_model", "Returned model name does not match the expected value."
    assert result.neural_network_type == "GRU", "Returned neural network type does not match."

@patch("app.inversify.Inversify")
def test_get_model_usecase_not_found(mock_inversify_class):
    """
    Test that get_model_usecase returns None if the model does not exist in the repository.
    """
    # Create a mock Inversify instance
    mock_inversify = MagicMock()
    mock_bdd = MagicMock()

    # Mock the situation where the model doesn't exist
    mock_bdd.get_model.return_value = None
    mock_inversify.get_bdd.return_value = mock_bdd
    mock_inversify_class.return_value = mock_inversify

    # Call the function
    result = get_model_usecase("unknown_model", mock_inversify)

    # Assertions
    assert result is None, "Expected None for a non-existent model, but got a value."

@patch("app.inversify.Inversify")
def test_get_model_usecase_exception(mock_inversify_class):
    """
    Test that get_model_usecase raises an exception if an error occurs in the BDD.
    """
    # Create a mock Inversify instance
    mock_inversify = MagicMock()
    mock_bdd = MagicMock()

    # Mock an exception thrown by get_model
    mock_bdd.get_model.side_effect = Exception("Database error")
    mock_inversify.get_bdd.return_value = mock_bdd
    mock_inversify_class.return_value = mock_inversify

    # Call the function and check for re-raised exception
    with pytest.raises(Exception) as exc_info:
        get_model_usecase("any_model", mock_inversify)

    assert "[get_model]" in str(exc_info.value), "Expected the exception message to contain '[get_model]'"
