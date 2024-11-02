#app\usecases\gru\usecase_search_gru_test.py
import pytest
from unittest.mock import patch, MagicMock
from fastapi import HTTPException  # type: ignore
from app.usecases.gru.usecase_search_gru import search_model_gru

# Test for successful search
@patch('app.usecases.gru.usecase_search_gru.predict')
@patch('app.usecases.gru.usecase_search_gru.process_input')
@patch('app.usecases.gru.usecase_search_gru.get_model')
def test_search_model_gru_success(mock_get_model, mock_process_input, mock_predict):
    """
    Tests that the search_model_gru function successfully predicts a category for a given sequence.
    """
    # Simulate model data returned by get_model
    mock_get_model.return_value = {
        "nn_model": MagicMock(),
        "word2idx": {"hello": 1, "<PAD>": 0},
        "idx2category": {0: "cat1", 1: "cat2"}
    }
    
    # Mock input processing to return a list of indices
    mock_process_input.return_value = [1, 0, 1]  # Processed sequence of indices
    
    # Mock prediction to return the predicted category index
    mock_predict.return_value = 1  # Predicted category index
    
    # Call the search_model_gru function
    result = search_model_gru("test_gru_model", ["hello", "world"])
    
    # Verify the prediction result
    assert result == {"category": "cat2"}, f"Expected category 'cat2' but got {result['category']}"

# Test when the model is not found
@patch('app.usecases.gru.usecase_search_gru.get_model', return_value=None)
def test_search_model_gru_model_not_found(mock_get_model):
    """
    Tests that the search_model_gru function raises an exception when the model is not found.
    """
    # Verify that an HTTPException is raised if the model is not found
    with pytest.raises(HTTPException) as exc_info:
        search_model_gru("unknown_model", ["hello", "world"])
    
    # Check that the exception is a 404 HTTPException with the appropriate message
    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Model not found"

# Test when the model is untrained
@patch('app.usecases.gru.usecase_search_gru.get_model', return_value={"nn_model": None})
def test_search_model_gru_model_not_trained(mock_get_model):
    """
    Tests that the search_model_gru function raises an exception when the model is untrained.
    """
    # Verify that an HTTPException is raised if the model is untrained
    with pytest.raises(HTTPException) as exc_info:
        search_model_gru("test_gru_model", ["hello", "world"])
    
    # Check that the exception is a 400 HTTPException with the appropriate message
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Model not trained"

# Test when model data is incomplete
@patch('app.usecases.gru.usecase_search_gru.get_model', return_value={"nn_model": MagicMock(), "word2idx": None})
def test_search_model_gru_incomplete_model_data(mock_get_model):
    """
    Tests that the search_model_gru function raises an exception when model data is incomplete.
    """
    # Verify that an HTTPException is raised if model data is incomplete
    with pytest.raises(HTTPException) as exc_info:
        search_model_gru("test_gru_model", ["hello", "world"])
    
    # Check that the exception is a 400 HTTPException with the appropriate message
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Model data incomplete"
