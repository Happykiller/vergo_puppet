#app\usecases\siamese\usecase_search_siamese_test.py
import pytest
from fastapi import HTTPException  # type: ignore
from app.repositories.memory import models
from app.usecases.siamese.usecase_create_siamese import create_model_siamese
from app.usecases.siamese.usecase_search_siamese import search_model_siamese
from app.usecases.siamese.usecase_train_siamese import train_model_siamese

# Setup function to clear memory before each test
def setup_function():
    models.clear()

# Test 1: Successful search with a neural network and varied data
def test_search_model_success():
    # Create and train a model with varied data
    create_model_siamese(
        "model1", 
        [["cat", "dog", "bird"], ["car", "bike", "train"], ["computer", "table", "pen"]], 
        ["cat", "dog", "bird", "car", "bike", "train", "computer", "table", "pen"]
    )

    # Train the model with non-empty vectors
    train_model_siamese("model1", [
        (["cat", "dog", "bird"], ["cat", "dog", "bird"], 1), 
        (["car", "bike", "train"], ["car", "bike", "train"], 1), 
        (["computer", "table", "pen"], ["computer", "table", "pen"], 1)
    ])

    # Search with a valid vector
    result = search_model_siamese("model1", ["cat", "dog", "bird"])

    # Assertions
    assert result["search"] == ["cat", "dog", "bird"]
    assert result["find"] == ["cat", "dog", "bird"]
    assert result["stats"]["accuracy"] > 0

# Test 3: Successful search with an unknown word
def test_search_unknown_success():
    # Create and train a model with varied strings
    create_model_siamese(
        "model3", 
        [
            ["cat", "dog", "bird"], 
            ["car", "bike", "train"], 
            ["computer", "table", "pen"]
        ], 
        ["cat", "dog", "bird", "car", "bike", "train", "computer", "table", "pen"]
    )

    # Train the model with non-empty vectors
    train_model_siamese(
        "model3", 
        [
            (["cat", "dog", "bird"], ["cat", "dog", "bird"], 1),
            (["car", "bike", "train"], ["car", "bike", "train"], 1),
            (["computer", "table", "pen"], ["computer", "table", "pen"], 1)
        ]
    )

    # Search with a vector containing an unknown token
    result = search_model_siamese("model3", ["cat", "lion"])

    # Assertions
    assert result["search"] == ["cat", "lion"]
    assert result["stats"]["accuracy"] > 0

# Test 4: Search in a non-existent model
def test_search_model_not_found():
    # Attempt to search in a model that does not exist
    with pytest.raises(HTTPException) as excinfo:
        search_model_siamese("model_not_exist", ["cat", "dog"])
    
    assert excinfo.value.status_code == 404
    assert str(excinfo.value.detail) == "Model not found"

# Test 5: Search with an untrained neural network
def test_search_model_no_nn_model():
    # Create a model without training the neural network
    create_model_siamese("model4", [["cat", "dog"], ["bird", "fish"], ["car", "train"]], ["cat", "dog", "bird", "fish", "car", "train"])
    
    # Store indexed_dictionary without a neural network
    models["model4"]["indexed_dictionary"] = [[0, 1], [2, 3], [4, 5]]

    # Check that search raises an error
    with pytest.raises(HTTPException) as excinfo:
        search_model_siamese("model4", ["cat", "dog"])
    
    assert excinfo.value.status_code == 400
    assert str(excinfo.value.detail) == "No neural network model found in the model"
