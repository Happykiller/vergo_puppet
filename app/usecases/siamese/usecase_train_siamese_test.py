#app\usecases\siamese\usecase_train_siamese_test.py
import pytest
from app.repositories.memory import models, save_model
from app.usecases.siamese.usecase_train_siamese import train_model_siamese

# Reset memory before each test
def setup_function():
    models.clear()

# Test 1: Verify that a 404 error is raised if the model does not exist
def test_train_model_not_found():
    with pytest.raises(Exception) as excinfo:
        train_model_siamese("model1", [["token1", "token2", "token3"]])  # Non-existent model
    
    assert excinfo.value.status_code == 404  # Verify that the error is 404
    assert str(excinfo.value.detail) == "Model not found"

# Test 2: Verify that a 400 error is raised if no dictionary is provided
def test_train_model_no_dictionary():
    # Save an empty model
    save_model("model1", {
        "dictionary": [["token1", "token2", "token3"]],
        "indexed_dictionary": [[0,1,2]],
        "glossary": ["token1", "token2", "token3"],
        "neural_network_type": "SIAMESE"
    })

    with pytest.raises(Exception) as excinfo:
        train_model_siamese("model1", None)
    
    assert excinfo.value.status_code == 400
    assert str(excinfo.value.detail) == "No training data provided"

# Test 3: Verify that a 400 error is raised if the dictionary is empty
def test_train_model_empty_dictionary():
    # Save an empty model
    save_model("model1", {
        "dictionary": [["token1", "token2", "token3"]],
        "indexed_dictionary": [[0,1,2]],
        "glossary": ["token1", "token2", "token3"],
        "neural_network_type": "SIAMESE"
    })

    with pytest.raises(Exception) as excinfo:
        train_model_siamese("model1", [])
    
    assert excinfo.value.status_code == 400
    assert str(excinfo.value.detail) == "Training data is empty"

# Test 5: Successful training
def test_train_model_success():
    # Save a model
    save_model("model1", {
        "dictionary": [["token1", "token2", "token3"], ["token1", "token2", "token4"], ["token1", "token2", "token5"]],
        "indexed_dictionary": [[0,1,2], [0,1,3], [0,1,4]],
        "glossary": ["token1", "token2", "token3", "token4", "token5"],
        "neural_network_type": "SIAMESE"
    })

    # Call the function with valid training data
    response = train_model_siamese("model1", [
        (["token1", "token2", "token3"], ["token1", "token2", "token3"], 1), 
        (["token1", "token2", "token4"], ["token1", "token2", "token4"], 1), 
        (["token1", "token2", "token5"], ["token1", "token2", "token5"], 1)
    ])

    # Verify that the response is correct
    assert response["status"] == "training completed"
    assert response["model_name"] == "model1"
    
    # Verify that the neural network model is correctly saved
    model = models.get("model1")
    assert "nn_model" in model
