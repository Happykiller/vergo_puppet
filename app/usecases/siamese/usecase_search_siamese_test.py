# app\usecases\siamese\usecase_search_siamese_test.py
import pytest
from unittest.mock import MagicMock, patch

from app.services.bdd.models.model_data import ModelData
from app.usecases.siamese.usecase_search_siamese import SearchSiameseUsecaseDto, search_model_siamese

# Test: Search in a non-existent model
def test_search_model_not_found(patch_inversify):
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_search_result.return_value = None
    mock_bdd.get_model.return_value = None  # Simulate model not found

    with pytest.raises(Exception, match="Model not found"):
        search_model_siamese(SearchSiameseUsecaseDto(name="model_not_exist", search=["cat", "dog"], inversify=mock_inversify))

# Test: Search with an untrained neural network
def test_search_model_no_nn_model(patch_inversify):
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_search_result.return_value = None

    # Mock a model without a neural network
    mock_bdd.get_model.return_value = ModelData(
        name="model4",
        neural_network_type="SIAMESE",
        nn_model=None,  # No trained neural network
        dictionary=[["cat", "dog", "bird"]],
        indexed_dictionary=[[2, 3, 4]],
        glossary=["", "UNK", "cat", "dog", "bird"]
    )

    with pytest.raises(Exception, match="No neural network model found in the model"):
        search_model_siamese(SearchSiameseUsecaseDto(name="model4", search=["cat", "dog"], inversify=mock_inversify))

# Test: Successful search with a neural network and valid data
@patch('app.usecases.siamese.usecase_search_siamese.evaluate_similarity')
def test_search_model_success(mock_evaluate_similarity, patch_inversify):
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_search_result.return_value = None
    mock_bdd.get_model.return_value = ModelData(
        name="model1",
        neural_network_type="SIAMESE",
        nn_model=MagicMock(),  # Mock neural network
        dictionary=[["cat", "dog", "bird"]],
        indexed_dictionary=[[2, 3, 4]],
        glossary=["", "UNK", "cat", "dog", "bird"]
    )
    mock_evaluate_similarity.return_value = 1.0  # Simulate perfect similarity
    mock_bdd.save_search_result.return_value = True

    result = search_model_siamese(SearchSiameseUsecaseDto(name="model1", search=["cat", "dog", "bird"], inversify=mock_inversify))

    # Assertions
    assert result["search"] == ["cat", "dog", "bird"]
    assert result["find"] == ["cat", "dog", "bird"]
    assert result["stats"]["accuracy"] == 1.0

# Test: Successful search with an unknown word
@patch('app.usecases.siamese.usecase_search_siamese.evaluate_similarity')
def test_search_unknown_success(mock_evaluate_similarity, patch_inversify):
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_search_result.return_value = None
    mock_bdd.get_model.return_value = ModelData(
        name="model3",
        neural_network_type="SIAMESE",
        nn_model=MagicMock(),  # Mock neural network
        dictionary=[["cat", "dog", "bird"]],
        indexed_dictionary=[[2, 3, 4]],
        glossary=["", "UNK", "cat", "dog", "bird"]
    )
    mock_evaluate_similarity.return_value = 0.3  # Simulate partial similarity
    mock_bdd.save_search_result.return_value = True

    result = search_model_siamese(SearchSiameseUsecaseDto(name="model3", search=["cat", "lion"], inversify=mock_inversify))

    # Assertions
    assert result["search"] == ["cat", "lion"]
    assert result["stats"]["accuracy"] == 0.3
