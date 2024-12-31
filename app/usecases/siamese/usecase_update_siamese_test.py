# app\usecases\siamese\usecase_update_siamese_test.py
import pytest
from unittest.mock import MagicMock
from fastapi import HTTPException  # type: ignore

from app.usecases.siamese.usecase_update_siamese import UpdateSiameseUsecaseDto, update_model_siamese

def test_update_model_success(patch_inversify):
    """
    Test successful update of a SIAMESE model with a new dictionary and glossary.
    """
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_model.return_value = {
        "dictionary": [["token1", "token2"], ["token3", "token4"]],
        "glossary": ["token1", "token2", "token3", "token4", "token5"],
        "nn_model": MagicMock()
    }

    dictionary = [["new_token1", "new_token2"], ["new_token3", "new_token4"]]
    glossary = ["new_token1", "new_token2", "new_token3", "new_token4"]

    # Perform the update
    response = update_model_siamese(UpdateSiameseUsecaseDto(name="model1", dictionary=dictionary, glossary=glossary, inversify=mock_inversify))

    # Validate the response
    assert response["status"] == "model updated"
    assert response["model_name"] == "model1"

    # Validate that the model was updated
    updated_model = mock_bdd.get_model("model1")
    assert updated_model["dictionary"] == dictionary
    assert updated_model["glossary"] == ["", "UNK"] + glossary

    # Validate that the search buffer was removed
    assert "model1" not in mock_bdd.search_buffer, "Search buffer should have been removed after update"

def test_update_model_not_found(patch_inversify):
    """
    Test updating a non-existent model raises a 404 error.
    """
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_model.return_value = None
    
    with pytest.raises(HTTPException) as exc_info:
        update_model_siamese(UpdateSiameseUsecaseDto(name="non_existent_model", dictionary=[["token1"]], glossary=["token1", "token2"], inversify=mock_inversify))
    
    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Model not found"

def test_update_model_empty_dictionary(patch_inversify):
    """
    Test updating a model with an empty dictionary raises a 400 error.
    """
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_model.return_value = True

    with pytest.raises(HTTPException) as exc_info:
        update_model_siamese(UpdateSiameseUsecaseDto(name="non_existent_model", dictionary=[], glossary=["token1", "token2"], inversify=mock_inversify))
    
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Dictionary cannot be empty"

def test_update_model_empty_glossary(patch_inversify):
    """
    Test updating a model with an empty glossary raises a 400 error.
    """
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_model.return_value = True
    
    with pytest.raises(HTTPException) as exc_info:
        update_model_siamese(UpdateSiameseUsecaseDto(name="model1", dictionary=[["token1", "token2"]], glossary=[], inversify=mock_inversify))
    
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Glossary cannot be empty"