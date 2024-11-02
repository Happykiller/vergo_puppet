#app\usecases\siamese\usecase_create_siamese_test.py
import pytest
from app.repositories.memory import models
from fastapi import HTTPException  # type: ignore
from app.usecases.siamese.usecase_create_siamese import create_model_siamese

def setup_function():
    """Reset the memory before each test."""
    models.clear()

# Test for successful model creation
def test_create_model_siamese_success():
    # Test creating a model with a valid dictionary and glossary
    result = create_model_siamese("model1", [["token1", "token2"], ["token1", "token3"]], ["token1", "token2", "token3"])
    
    # Verify the result is as expected
    assert result == {"status": "model created", "model_name": "model1"}
    
    # Verify the model was saved in memory
    assert "model1" in models
    
    # Verify the original dictionary is saved correctly
    assert models["model1"]["dictionary"] == [["token1", "token2"], ["token1", "token3"]]
    
    # Verify the indexed dictionary was generated correctly
    assert models["model1"]["indexed_dictionary"] == [[2, 3], [2, 4]]

# Test for attempting to create a model that already exists
def test_create_model_siamese_already_exists():
    # Create the first model
    create_model_siamese("model1", [["token1", "token2"], ["token1", "token3"]], ["token1", "token2", "token3"])
    
    # Attempt to create a model with the same name, which should raise an exception
    with pytest.raises(HTTPException) as excinfo:
        create_model_siamese("model1", [["token1", "token2"]], ["token1", "token2", "token3"])
    
    # Verify the raised exception contains the correct message
    assert str(excinfo.value.detail) == "Model already exists"

# Test for creating a model with a None dictionary
def test_create_model_dictionary_none():
    # Verify the function raises an exception if the dictionary is None
    with pytest.raises(HTTPException) as excinfo:
        create_model_siamese("model1", None, ["token1", "token2", "token3"])
    
    # Verify the error message is correct
    assert excinfo.value.status_code == 400
    assert str(excinfo.value.detail) == "Dictionary cannot be None"

# Test for creating a model with a None glossary
def test_create_model_glossary_none():
    # Verify the function raises an exception if the glossary is None
    with pytest.raises(HTTPException) as excinfo:
        create_model_siamese("model1", [["token1", "token2"]], None)
    
    # Verify the error message is correct
    assert excinfo.value.status_code == 400
    assert str(excinfo.value.detail) == "Glossary cannot be None"

# Test for creating a model with an empty dictionary
def test_create_model_dictionary_empty():
    # Verify the function raises an exception if the dictionary is empty
    with pytest.raises(HTTPException) as excinfo:
        create_model_siamese("model1", [], ["token1", "token2", "token3"])
    
    # Verify the error message is correct
    assert excinfo.value.status_code == 400
    assert str(excinfo.value.detail) == "Dictionary cannot be empty"

# Test for creating a model with an empty glossary
def test_create_model_glossary_empty():
    # Verify the function raises an exception if the glossary is empty
    with pytest.raises(HTTPException) as excinfo:
        create_model_siamese("model1", [["token1", "token2"]], [])
    
    # Verify the error message is correct
    assert excinfo.value.status_code == 400
    assert str(excinfo.value.detail) == "Glossary cannot be empty"

# Test for creating a model with unrecognized tokens
def test_create_model_with_unknown_tokens():
    # Verify the function correctly handles tokens not found in the glossary
    result = create_model_siamese("model1", [["token1", "token2", "tokenX"]], ["token1", "token2", "token3"])
    
    # Verify the model is created successfully
    assert result == {"status": "model created", "model_name": "model1"}
    
    # Verify the indexed dictionary contains None for the "tokenX"
    assert models["model1"]["indexed_dictionary"] == [[2, 3, 1]]

# Test for creating a model with a glossary containing duplicates
def test_create_model_no_duplicates_in_glossary():
    # Verify the first occurrence of a token is used for indexing
    result = create_model_siamese("model1", [["token1", "token2"]], ["token1", "token2", "token1", "token3"])
    
    # Verify the model is created successfully
    assert result == {"status": "model created", "model_name": "model1"}
    
    # Verify the indexed dictionary only uses the first occurrence of "token1"
    assert models["model1"]["indexed_dictionary"] == [[2, 3]]

# Test for creating a model with empty sublists in the dictionary
def test_create_model_empty_token_lists():
    # Verify empty sublists in the dictionary are correctly handled
    result = create_model_siamese("model1", [[], ["token1", "token2"], []], ["token1", "token2", "token3"])
    
    # Verify the model is created successfully
    assert result == {"status": "model created", "model_name": "model1"}
    
    # Verify that the empty sublists remain empty in the indexed dictionary
    assert models["model1"]["indexed_dictionary"] == [[], [2, 3], []]
