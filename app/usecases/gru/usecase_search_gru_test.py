# app\usecases\gru\usecase_search_gru_test.py
import pytest
from unittest.mock import patch, MagicMock

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
    mock_bdd.get_model.return_value = MagicMock(
        nn_model=MagicMock(),
        word2idx={"hello": 1, "<PAD>": 0},
        idx2category={0: "cat1", 1: "cat2"}
    )
    
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
    mock_bdd.get_model.return_value = None

    # Verify that an exception is raised if the model is not found
    with pytest.raises(Exception, match="Model not found"):
        search_model_gru(SearchGRUUsecaseDto(name="unknown_model", search=["hello", "world"], inversify=mock_inversify))


# Test when the model is untrained
def test_search_model_gru_model_not_trained(patch_inversify):
    """
    Tests that the search_model_gru function raises an exception when the model is untrained.
    """
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_model.return_value = MagicMock(nn_model=None)

    # Verify that an exception is raised if the model is untrained
    with pytest.raises(Exception, match="Model not trained"):
        search_model_gru(SearchGRUUsecaseDto(name="test_gru_model", search=["hello", "world"], inversify=mock_inversify))


# Test when model data is incomplete
def test_search_model_gru_incomplete_model_data(patch_inversify):
    """
    Tests that the search_model_gru function raises an exception when model data is incomplete.
    """
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_model.return_value = MagicMock(
        nn_model=MagicMock(),
        word2idx=None,  # Missing word2idx mapping
        idx2category={0: "cat1", 1: "cat2"}
    )

    # Verify that an exception is raised if model data is incomplete
    with pytest.raises(Exception, match="Model data incomplete"):
        search_model_gru(SearchGRUUsecaseDto(name="test_gru_model", search=["hello", "world"], inversify=mock_inversify))


# Test when prediction fails
@patch('app.usecases.gru.usecase_search_gru.predict')
@patch('app.usecases.gru.usecase_search_gru.process_input')
def test_search_model_gru_prediction_error(mock_process_input, mock_predict, patch_inversify):
    """
    Tests that the search_model_gru function raises an exception if prediction fails.
    """
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify

    # Simulate model data returned by get_model
    mock_bdd.get_model.return_value = MagicMock(
        nn_model=MagicMock(),
        word2idx={"hello": 1, "<PAD>": 0},
        idx2category={0: "cat1", 1: "cat2"}
    )
    
    # Mock input processing to return a list of indices
    mock_process_input.return_value = [1, 0, 1]
    
    # Mock predict to raise an exception
    mock_predict.side_effect = Exception("Prediction error")
    
    # Verify that an exception is raised if prediction fails
    with pytest.raises(Exception, match="Prediction error"):
        search_model_gru(SearchGRUUsecaseDto(name="test_gru_model", search=["hello", "world"], inversify=mock_inversify))