# app\usecases\getall_model_test.py
import pytest
from unittest.mock import MagicMock, patch
from app.usecases.getall_model import get_all_models_usecase
from app.services.bdd.models.model_data import ModelData, ModelStatus

# Test : Verify that the function returns all available models
@patch("app.inversify.Inversify")
def test_get_all_models_with_models(mock_inversify_class):
    """
    Test that get_all_models_usecase returns a list of models when models are available.
    """
    # Create a mock Inversify instance
    mock_inversify = MagicMock()
    mock_bdd = MagicMock()

    # Mock methods and their return values
    mock_bdd.get_all_models.return_value = [
        ModelData(name="model1", neural_network_type="GRU", status=ModelStatus.TRAINED),
        ModelData(name="model2", neural_network_type="SiameseLSTM", status=ModelStatus.CREATED)
    ]
    mock_bdd.get_metrics.return_value = []
    mock_inversify.get_bdd.return_value = mock_bdd
    mock_inversify_class.return_value = mock_inversify
    
    # Call the function
    result = get_all_models_usecase(mock_inversify)
    
    # Expected output when models are present
    expected_result = [
        {"name": "model1", "neural_network_type": "GRU", "status": "trained", "history": []},
        {"name": "model2", "neural_network_type": "SiameseLSTM", "status": "created", "history": []}
    ]
    assert result == expected_result, f"Expected {expected_result} but got {result}"

# Test : Verify the response when no models are found
@patch("app.inversify.Inversify")
def test_get_all_models_no_models(mock_inversify_class):
    """
    Test that get_all_models_usecase returns a message indicating no models are found when the model list is empty.
    """
    # Create a mock Inversify instance
    mock_inversify = MagicMock()
    mock_bdd = MagicMock()
    
    # Mock methods and their return values
    mock_bdd.get_all_models.return_value = []
    mock_inversify.get_bdd.return_value = mock_bdd
    mock_inversify_class.return_value = mock_inversify
    
    # Call the function
    result = get_all_models_usecase(mock_inversify)
    
    # Expected output when no models are found
    expected_result = []
    assert result == expected_result, f"Expected {expected_result} but got {result}"
