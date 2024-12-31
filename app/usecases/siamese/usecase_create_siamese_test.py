# app\usecases\siamese\usecase_create_siamese_test.py
import pytest
from fastapi import HTTPException  # type: ignore

from app.usecases.siamese.usecase_create_siamese import CreateSiameseUsecaseDto, create_model_siamese

# Test for successful model creation
def test_create_model_siamese_success(patch_inversify):
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify

    mock_bdd.model_exists.return_value = False

    # Test creating a model with a valid dictionary and glossary
    result = create_model_siamese(CreateSiameseUsecaseDto(name="model1", dictionary=[["token1", "token2"], ["token1", "token3"]], glossary=["token1", "token2", "token3"], inversify=mock_inversify))
    
    # Verify the result is as expected
    assert result == {
        "status": "model created",
        "model_name": "model1",
        "missing_tokens": []
    }

# Test for attempting to create a model that already exists
def test_create_model_siamese_already_exists(patch_inversify):
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify

    mock_bdd.model_exists.return_value = True
    
    # Attempt to create a model with the same name, which should raise an exception
    with pytest.raises(HTTPException) as excinfo:
        create_model_siamese(CreateSiameseUsecaseDto(name="model1", dictionary=[["token1", "token2"], ["token1", "token3"]], glossary=["token1", "token2", "token3"], inversify=mock_inversify))
    
    # Verify the raised exception contains the correct message
    assert str(excinfo.value.detail) == "Model already exists"

# Test for creating a model with a None dictionary
def test_create_model_dictionary_none(patch_inversify):
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify

    mock_bdd.model_exists.return_value = False
    
    # Verify the function raises an exception if the dictionary is None
    with pytest.raises(HTTPException) as excinfo:
        create_model_siamese(CreateSiameseUsecaseDto(name="model1", dictionary=None, glossary=["token1", "token2", "token3"], inversify=mock_inversify))
    
    # Verify the error message is correct
    assert excinfo.value.status_code == 400
    assert str(excinfo.value.detail) == "Dictionary cannot be None"

# Test for creating a model with a None glossary
def test_create_model_glossary_none(patch_inversify):
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify

    mock_bdd.model_exists.return_value = False

    # Verify the function raises an exception if the glossary is None
    with pytest.raises(HTTPException) as excinfo:
        create_model_siamese(CreateSiameseUsecaseDto(name="model1", dictionary=[["token1", "token2"], ["token1", "token3"]], glossary=None, inversify=mock_inversify))
    
    # Verify the error message is correct
    assert excinfo.value.status_code == 400
    assert str(excinfo.value.detail) == "Glossary cannot be None"

# Test for creating a model with an empty dictionary
def test_create_model_dictionary_empty(patch_inversify):
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify

    mock_bdd.model_exists.return_value = False

    # Verify the function raises an exception if the dictionary is empty
    with pytest.raises(HTTPException) as excinfo:
        create_model_siamese(CreateSiameseUsecaseDto(name="model1", dictionary=[], glossary=["token1", "token2", "token3"], inversify=mock_inversify))
    
    # Verify the error message is correct
    assert excinfo.value.status_code == 400
    assert str(excinfo.value.detail) == "Dictionary cannot be empty"

# Test for creating a model with an empty glossary
def test_create_model_glossary_empty(patch_inversify):
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify

    mock_bdd.model_exists.return_value = False

    # Verify the function raises an exception if the glossary is empty
    with pytest.raises(HTTPException) as excinfo:
        create_model_siamese(CreateSiameseUsecaseDto(name="model1", dictionary=[["token1", "token2"], ["token1", "token3"]], glossary=[], inversify=mock_inversify))
    
    # Verify the error message is correct
    assert excinfo.value.status_code == 400
    assert str(excinfo.value.detail) == "Glossary cannot be empty"

# Test for creating a model with unrecognized tokens
def test_create_model_with_unknown_tokens(patch_inversify):
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.model_exists.return_value = False

    # Verify the function correctly handles tokens not found in the glossary
    result = create_model_siamese(CreateSiameseUsecaseDto(name="model1", dictionary=[["token1", "tokenX"]], glossary=["token1", "token2", "token3"], inversify=mock_inversify))
    
    # Verify the model is created successfully
    assert result == {
        "status": "model created",
        "model_name": "model1",
        "missing_tokens": ["tokenX"]
    }
    
    # Verify the indexed dictionary contains None for the "tokenX"
    mock_bdd.save_model.assert_called_once_with("model1", {
        "dictionary": [["token1", "tokenX"]],
        "indexed_dictionary": [[2, 1]],
        "glossary": ["", "UNK", "token1", "token2", "token3"],
    })

# Test for creating a model with a glossary containing duplicates
def test_create_model_no_duplicates_in_glossary(patch_inversify):
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.model_exists.return_value = False

    # Verify the first occurrence of a token is used for indexing
    result = create_model_siamese(CreateSiameseUsecaseDto(name="model1", dictionary=[["token1", "token2"]], glossary=["token1", "token2", "token1", "token3"], inversify=mock_inversify))
    
    # Verify the model is created successfully
    assert result == {
        "status": "model created",
        "model_name": "model1",
        "missing_tokens": []
    }
    
    # Verify the indexed dictionary only uses the first occurrence of "token1"
    mock_bdd.save_model.assert_called_once_with("model1", {
        "dictionary": [["token1", "token2"]],
        "indexed_dictionary": [[2, 3]],
        "glossary": ["", "UNK", "token1", "token2", "token1", "token3"],
    })

# Test for creating a model with empty sublists in the dictionary
def test_create_model_empty_token_lists(patch_inversify):
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.model_exists.return_value = False

    # Verify empty sublists in the dictionary are correctly handled
    result = create_model_siamese(CreateSiameseUsecaseDto(name="model1", dictionary=[[], ["token1", "token2"], []], glossary=["token1", "token2", "token3"], inversify=mock_inversify))
    
    # Verify the model is created successfully
    assert result == {
        "status": "model created",
        "model_name": "model1",
        "missing_tokens": []
    }
    
    # Verify that the empty sublists remain empty in the indexed dictionary
    mock_bdd.save_model.assert_called_once_with("model1", {
        "dictionary": [[], ["token1", "token2"], []],
        "indexed_dictionary": [[], [2, 3], []],
        "glossary": ["", "UNK", "token1", "token2", "token3"],
    })
