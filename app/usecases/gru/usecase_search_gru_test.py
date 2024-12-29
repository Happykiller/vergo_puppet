#app\usecases\gru\usecase_search_gru_test.py
import pytest
from unittest.mock import patch, MagicMock
from fastapi import HTTPException  # type: ignore

from app.usecases.gru.usecase_search_gru import SearchGRUUsecaseDto, search_model_gru

# Test for successful search
@patch('app.usecases.gru.usecase_search_gru.predict')
@patch('app.usecases.gru.usecase_search_gru.process_input')
def test_search_model_gru_success(mock_process_input, mock_predict, patch_inversify):
    """
    Tests that the search_model_gru function successfully predicts a category for a given sequence.
    """
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify

    # Simulate model data returned by get_model
    mock_bdd.get_model.return_value = {
        "nn_model": MagicMock(),
        "word2idx": {"hello": 1, "<PAD>": 0},
        "idx2category": {0: "cat1", 1: "cat2"}
    }
    
    # Mock input processing to return a list of indices
    mock_process_input.return_value = [1, 0, 1]  # Processed sequence of indices
    
    # Mock prediction to return the predicted category index
    mock_predict.return_value = 1  # Predicted category index
    
    # Call the search_model_gru function
    result = search_model_gru(SearchGRUUsecaseDto(name="test_gru_model", search=["hello", "world"], inversify=mock_inversify))
    
    # Verify the prediction result
    assert result == {"category": "cat2"}, f"Expected category 'cat2' but got {result['category']}"

# Test when the model is not found
def test_search_model_gru_model_not_found(patch_inversify):
    """
    Tests that the search_model_gru function raises an exception when the model is not found.
    """
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_model.return_value=None

    # Verify that an HTTPException is raised if the model is not found
    with pytest.raises(HTTPException) as exc_info:
        search_model_gru(SearchGRUUsecaseDto(name="unknown_model", search=["hello", "world"], inversify=mock_inversify))
    
    # Check that the exception is a 404 HTTPException with the appropriate message
    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Model not found"

# Test when the model is untrained
def test_search_model_gru_model_not_trained(patch_inversify):
    """
    Tests that the search_model_gru function raises an exception when the model is untrained.
    """
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_model.return_value={"nn_model": None}

    # Verify that an HTTPException is raised if the model is untrained
    with pytest.raises(HTTPException) as exc_info:
        search_model_gru(SearchGRUUsecaseDto(name="test_gru_model", search=["hello", "world"], inversify=mock_inversify))
    
    # Check that the exception is a 400 HTTPException with the appropriate message
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Model not trained"

# Test when model data is incomplete
def test_search_model_gru_incomplete_model_data(patch_inversify):
    """
    Tests that the search_model_gru function raises an exception when model data is incomplete.
    """
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_model.return_value={"nn_model": MagicMock(), "word2idx": None}

    # Verify that an HTTPException is raised if model data is incomplete
    with pytest.raises(HTTPException) as exc_info:
        search_model_gru(SearchGRUUsecaseDto(name="test_gru_model", search=["hello", "world"], inversify=mock_inversify))
    
    # Check that the exception is a 400 HTTPException with the appropriate message
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Model data incomplete"
