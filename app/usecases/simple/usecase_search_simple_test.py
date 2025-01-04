# app\usecases\simple\usecase_search_simple_test.py
import pytest
from unittest.mock import patch, MagicMock

from app.apis.models.simple_nn_search_model_data import SimpleNNSearchModelData
from app.usecases.simple.usecase_search_simple import SearchSimpleUsecaseDto, search_model_simple_nn


# Test successful search with a SimpleNN model
@patch("app.usecases.simple.usecase_search_simple.predict")
def test_search_model_simple_nn_success(mock_predict, patch_inversify):
    """Test successful prediction with a SimpleNN model."""
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_model.return_value = MagicMock(
        nn_model=MagicMock(),
        encoder=MagicMock(),
        scaler=MagicMock(),
        indices={"categorical_indices": [0, 1], "numerical_indices": [2, 3]},
        targets_mean=0.5,
        targets_std=0.2,
    )

    # Mock prediction
    mock_predict.return_value = 350000

    search_data = SimpleNNSearchModelData(
        type=1,
        surface=100,
        pieces=4,
        floor=2,
        parking=1,
        balcon=0,
        ascenseur=1,
        orientation=1,
        transports=1,
        neighborhood=8,
    )

    # Execute function
    result = search_model_simple_nn(SearchSimpleUsecaseDto(name="test_model", search=search_data, inversify=mock_inversify))

    # Assertions
    mock_predict.assert_called_once()
    assert result == {"predicted_price": 350000}


# Test when the model is not found
def test_search_model_simple_nn_model_not_found(patch_inversify):
    """Test the case where the model is not found."""
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_model.return_value = None

    search_data = SimpleNNSearchModelData(
        type=1,
        surface=100,
        pieces=4,
        floor=2,
        parking=1,
        balcon=0,
        ascenseur=1,
        orientation=1,
        transports=1,
        neighborhood=8,
    )

    with pytest.raises(Exception) as exc_info:
        search_model_simple_nn(SearchSimpleUsecaseDto(name="unknown_model", search=search_data, inversify=mock_inversify))

    assert str(exc_info.value) == "[#search_model_simple_nn]Model not found"


# Test when the model is not trained
def test_search_model_simple_nn_model_not_trained(patch_inversify):
    """Test the case where the model is not trained yet."""
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_model.return_value = MagicMock(
        nn_model=None,
        encoder=MagicMock(),
        scaler=MagicMock(),
        indices={"categorical_indices": [0, 1], "numerical_indices": [2, 3]},
        targets_mean=0.5,
        targets_std=0.2,
    )

    search_data = SimpleNNSearchModelData(
        type=1,
        surface=100,
        pieces=4,
        floor=2,
        parking=1,
        balcon=0,
        ascenseur=1,
        orientation=1,
        transports=1,
        neighborhood=8,
    )

    with pytest.raises(Exception) as exc_info:
        search_model_simple_nn(SearchSimpleUsecaseDto(name="test_model", search=search_data, inversify=mock_inversify))

    assert str(exc_info.value) == "[#search_model_simple_nn]Model not trained yet"


# Test when encoder, scaler, or indices are missing
def test_search_model_simple_nn_missing_files(patch_inversify):
    """Test the case where encoder, scaler, or indices are missing."""
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_model.return_value = MagicMock(
        nn_model=MagicMock(),
        encoder=None,
        scaler=None,
        indices=None,
        targets_mean=0.5,
        targets_std=0.2,
    )

    search_data = SimpleNNSearchModelData(
        type=1,
        surface=100,
        pieces=4,
        floor=2,
        parking=1,
        balcon=0,
        ascenseur=1,
        orientation=1,
        transports=1,
        neighborhood=8,
    )

    with pytest.raises(Exception) as exc_info:
        search_model_simple_nn(SearchSimpleUsecaseDto(name="test_model", search=search_data, inversify=mock_inversify))

    # Validate exception message
    assert str(exc_info.value) == "[#search_model_simple_nn]Missing encoder, scaler, or indices in the model"


# Test when target normalization parameters are missing
def test_search_model_simple_nn_missing_normalization_parameters(patch_inversify):
    """Test the case where normalization parameters are missing."""
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_model.return_value = MagicMock(
        nn_model=MagicMock(),
        encoder=MagicMock(),
        scaler=MagicMock(),
        indices={"categorical_indices": [0, 1], "numerical_indices": [2, 3]},
        targets_mean=None,
        targets_std=None,
    )

    search_data = SimpleNNSearchModelData(
        type=1,
        surface=100,
        pieces=4,
        floor=2,
        parking=1,
        balcon=0,
        ascenseur=1,
        orientation=1,
        transports=1,
        neighborhood=8,
    )

    with pytest.raises(Exception) as exc_info:
        search_model_simple_nn(SearchSimpleUsecaseDto(name="test_model", search=search_data, inversify=mock_inversify))

    assert str(exc_info.value) == "[#search_model_simple_nn]Missing normalization parameters in the model"