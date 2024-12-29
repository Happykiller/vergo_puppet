import pytest
from unittest.mock import MagicMock, patch
from fastapi import HTTPException  # type: ignore

from app.apis.models.tokenize_model_data import ModelTokenizeData
from app.usecases.gru.usecase_search_multi_brut_gru import SearchMultiBrutGRUUsecaseDto, search_multi_brut_model_gru

# Test for successful search
@patch("app.usecases.gru.usecase_search_multi_brut_gru.predict")
@patch("app.usecases.gru.usecase_search_multi_brut_gru.process_input")
@patch("app.usecases.gru.usecase_search_multi_brut_gru.usecase_tokenize")
def test_search_multi_brut_model_gru_success(mock_usecase_tokenize, mock_process_input, mock_predict, patch_inversify):
    """
    Ensures that the `search_multi_brut_model_gru` function returns correct results
    for a list of documents.
    """
    # patch_inversify is a tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify

    # Simulate model data returned by get_model
    mock_bdd.get_model.return_value = {
        "nn_model": MagicMock(),
        "word2idx": {"hello": 1, "<PAD>": 0},
        "idx2category": {0: "cat1", 1: "cat2"}
    }

    # Mock tokenized data
    mock_usecase_tokenize.return_value = [{"tokens": [1, 0, 1]}]

    # Mock input processing and prediction
    mock_process_input.return_value = [1, 0, 1]  # Processed indices
    mock_predict.return_value = 1  # Predicted category index

    # Create a DTO for the function call
    dto = SearchMultiBrutGRUUsecaseDto(
        name="test_gru_model",
        documents=[
            ModelTokenizeData(
                incidentId="inc1",
                description="test description",
                tokens=[]
            )
        ],
        inversify=mock_inversify
    )

    # Call the function
    result = search_multi_brut_model_gru(dto)

    # Validate the results
    assert len(result) == 1
    assert result[0]["incidentId"] == "inc1"
    assert result[0]["predicted_category"] == "cat2"

# Test when model is not found
def test_search_multi_brut_model_gru_model_not_found(patch_inversify):
    """
    Ensures that an exception is raised if the model is not found.
    """
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_model.return_value = None

    dto = SearchMultiBrutGRUUsecaseDto(
        name="unknown_model",
        documents=[],
        inversify=mock_inversify
    )

    with pytest.raises(HTTPException) as exc_info:
        search_multi_brut_model_gru(dto)

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Model not found"

# Test when model is not trained
def test_search_multi_brut_model_gru_model_not_trained(patch_inversify):
    """
    Ensures that an exception is raised if the model is not trained.
    """
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_model.return_value = {"nn_model": None}

    dto = SearchMultiBrutGRUUsecaseDto(
        name="test_gru_model",
        documents=[],
        inversify=mock_inversify
    )

    with pytest.raises(HTTPException) as exc_info:
        search_multi_brut_model_gru(dto)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Model not trained"

# Test when model data is incomplete
def test_search_multi_brut_model_gru_incomplete_model_data(patch_inversify):
    """
    Ensures that an exception is raised if model data is incomplete.
    """
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_model.return_value = {"nn_model": MagicMock(), "word2idx": None}

    dto = SearchMultiBrutGRUUsecaseDto(
        name="test_gru_model",
        documents=[],
        inversify=mock_inversify
    )

    with pytest.raises(HTTPException) as exc_info:
        search_multi_brut_model_gru(dto)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Model data incomplete"
