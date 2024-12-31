# app\usecases\siamese\usecase_search_siamese_test.py
import pytest
from unittest.mock import MagicMock, patch
from fastapi import HTTPException  # type: ignore

from app.usecases.siamese.usecase_search_siamese import SearchSiameseUsecaseDto, search_model_siamese

# Test: Search in a non-existent model
def test_search_model_not_found(patch_inversify):
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_search_result.return_value = None

    mock_bdd.get_model.return_value = False

    # Attempt to search in a model that does not exist
    with pytest.raises(HTTPException) as excinfo:
        search_model_siamese(SearchSiameseUsecaseDto(name="model_not_exist", search=["cat", "dog"], inversify=mock_inversify))
    
    assert excinfo.value.status_code == 404
    assert str(excinfo.value.detail) == "Model not found"

# Test: Search with an untrained neural network
def test_search_model_no_nn_model(patch_inversify):
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_search_result.return_value = None

    # Create a model without training the neural network
    mock_bdd.get_model.return_value = {
        "name": "model4",
        "dictionary": [["cat", "dog", "bird"]],
        "indexed_dictionary": [[2, 3, 4]],
        "glossary": ["", "UNK", "cat", "dog", "bird"]
    }

    # Check that search raises an error
    with pytest.raises(HTTPException) as excinfo:
        search_model_siamese(SearchSiameseUsecaseDto(name="model4", search=["cat", "dog"], inversify=mock_inversify))
    
    assert excinfo.value.status_code == 400
    assert str(excinfo.value.detail) == "No neural network model found in the model"

# Test: Successful search with a neural network and varied data
@patch('app.usecases.siamese.usecase_search_siamese.evaluate_similarity')
def test_search_model_success(mock_evaluate_similarity, patch_inversify):
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_search_result.return_value = None
    mock_bdd.get_model.return_value = {
        "name": "model1",
        "dictionary": [["cat", "dog", "bird"]],
        "indexed_dictionary": [[2, 3, 4]],
        "glossary": ["", "UNK", "cat", "dog", "bird"],
        "nn_model": MagicMock()
    }
    mock_evaluate_similarity.return_value = 1
    mock_bdd.save_search_result.return_value = True

    # Search with a valid vector
    result = search_model_siamese(SearchSiameseUsecaseDto(name="model1", search=["cat", "dog", "bird"], inversify=mock_inversify))

    # Assertions
    assert result["search"] == ["cat", "dog", "bird"]
    assert result["find"] == ["cat", "dog", "bird"]
    assert result["stats"]["accuracy"] > 0

# Test: Successful search with an unknown word
@patch('app.usecases.siamese.usecase_search_siamese.evaluate_similarity')
def test_search_unknown_success(mock_evaluate_similarity, patch_inversify):
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_search_result.return_value = None
    mock_bdd.get_model.return_value = {
        "name": "model3",
        "dictionary": [["cat", "dog", "bird"]],
        "indexed_dictionary": [[2, 3, 4]],
        "glossary": ["", "UNK", "cat", "dog", "bird"],
        "nn_model": MagicMock()
    }
    mock_evaluate_similarity.return_value = 0.3
    mock_bdd.save_search_result.return_value = True

    # Search with a vector containing an unknown token
    result = search_model_siamese(SearchSiameseUsecaseDto(name="model3", search=["cat", "lion"], inversify=mock_inversify))

    # Assertions
    assert result["search"] == ["cat", "lion"]
    assert result["stats"]["accuracy"] > 0
