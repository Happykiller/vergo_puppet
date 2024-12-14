# app\usecases\siamese\usecase_update_siamese_test.py
import pytest
from fastapi import HTTPException  # type: ignore
from app.usecases.siamese.usecase_update_siamese import update_model_siamese
from app.repositories.memory import save_model, get_model, search_buffer

@pytest.fixture
def setup_model():
    """
    Fixture to set up an initial model in memory for testing.
    """
    save_model("model1", {
        "dictionary": [["token1", "token2"], ["token3", "token4"]],
        "glossary": ["token1", "token2", "token3", "token4", "token5"]
    })
    yield
    # Clean up after the test
    from app.repositories.memory import models
    models.clear()
    search_buffer.clear()

def test_update_model_success(setup_model):
    """
    Test successful update of a SIAMESE model with a new dictionary and glossary.
    """
    dictionary = [["new_token1", "new_token2"], ["new_token3", "new_token4"]]
    glossary = ["new_token1", "new_token2", "new_token3", "new_token4"]

    # Perform the update
    response = update_model_siamese("model1", dictionary, glossary)

    # Validate the response
    assert response["status"] == "model updated"
    assert response["model_name"] == "model1"

    # Validate that the model was updated
    updated_model = get_model("model1")
    assert updated_model["dictionary"] == dictionary
    assert updated_model["glossary"] == ["", "UNK"] + glossary

    # Validate that the search buffer was removed
    assert "model1" not in search_buffer, "Search buffer should have been removed after update"

def test_update_model_not_found():
    """
    Test updating a non-existent model raises a 404 error.
    """
    with pytest.raises(HTTPException) as exc_info:
        update_model_siamese("non_existent_model", [["token1"]], ["token1", "token2"])
    
    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Model not found"

def test_update_model_empty_dictionary(setup_model):
    """
    Test updating a model with an empty dictionary raises a 400 error.
    """
    with pytest.raises(HTTPException) as exc_info:
        update_model_siamese("model1", [], ["token1", "token2"])
    
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Dictionary cannot be empty"

def test_update_model_empty_glossary(setup_model):
    """
    Test updating a model with an empty glossary raises a 400 error.
    """
    with pytest.raises(HTTPException) as exc_info:
        update_model_siamese("model1", [["token1", "token2"]], [])
    
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Glossary cannot be empty"