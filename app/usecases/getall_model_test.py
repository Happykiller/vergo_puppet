import pytest
from unittest.mock import patch
from app.usecases.getall_model import get_all_models_usecase

# Test 1: Verify that the function returns all available models
@patch('app.usecases.getall_model.get_all_models')
def test_get_all_models_with_models(mock_get_all_models):
    """
    Test that get_all_models_usecase returns a list of models when models are available.
    """
    # Simulate the return value of models
    mock_get_all_models.return_value = [
        {"name": "model1", "type": "GRU"},
        {"name": "model2", "type": "Siamese"}
    ]
    
    # Call the function
    result = get_all_models_usecase()
    
    # Expected output when models are present
    expected_result = {
        "models": [
            {"name": "model1", "type": "GRU"},
            {"name": "model2", "type": "Siamese"}
        ]
    }
    assert result == expected_result, f"Expected {expected_result} but got {result}"

# Test 2: Verify the response when no models are found
@patch('app.usecases.getall_model.get_all_models')
def test_get_all_models_no_models(mock_get_all_models):
    """
    Test that get_all_models_usecase returns a message indicating no models are found when the model list is empty.
    """
    # Simulate the case where no models are available
    mock_get_all_models.return_value = []
    
    # Call the function
    result = get_all_models_usecase()
    
    # Expected output when no models are found
    expected_result = {"message": "No models found"}
    assert result == expected_result, f"Expected {expected_result} but got {result}"
