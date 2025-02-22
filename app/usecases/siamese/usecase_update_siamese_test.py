# app\usecases\siamese\usecase_update_siamese_test.py
import pytest
from unittest.mock import MagicMock

from app.services.bdd.models.model_data import ModelData
from app.usecases.siamese.usecase_update_siamese import UpdateSiameseUsecaseDto, update_model_siamese


def test_update_model_success(patch_inversify):
    """
    Test successful update of a SIAMESE model with a new dictionary and glossary.
    """
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify

    # Mock existing model
    mock_model = ModelData(
        name="model1",
        neural_network_type="SiameseLSTM",
        dictionary=[["token1", "token2"], ["token3", "token4"]],
        glossary=["token1", "token2", "token3", "token4"],
        nn_model=MagicMock(),
    )
    mock_bdd.get_model.return_value = mock_model

    dictionary = [["new_token1", "new_token2"], ["new_token3", "new_token4"]]
    glossary = ["new_token1", "new_token2", "new_token3", "new_token4"]

    # Perform the update
    response = update_model_siamese(UpdateSiameseUsecaseDto(name="model1", dictionary=dictionary, inversify=mock_inversify))

    # Validate the response
    assert response["status"] == "model updated"
    assert response["model_name"] == "model1"

    # Validate that the search buffer was removed
    mock_bdd.clear_search_buffer.assert_called_once_with("model1")


def test_update_model_not_found(patch_inversify):
    """
    Test updating a non-existent model raises an exception.
    """
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_model.return_value = None

    with pytest.raises(Exception, match="Model not found"):
        update_model_siamese(UpdateSiameseUsecaseDto(
            name="non_existent_model",
            dictionary=[["token1"]],
            inversify=mock_inversify
        ))


def test_update_model_empty_dictionary(patch_inversify):
    """
    Test updating a model with an empty dictionary raises an exception.
    """
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify

    # Mock existing model
    mock_model = ModelData(
        name="model1",
        neural_network_type="SiameseLSTM",
        dictionary=[["token1", "token2"]],
        nn_model=MagicMock(),
    )
    mock_bdd.get_model.return_value = mock_model

    with pytest.raises(Exception, match="Dictionary cannot be empty"):
        update_model_siamese(UpdateSiameseUsecaseDto(
            name="model1",
            dictionary=[],
            inversify=mock_inversify
        ))
