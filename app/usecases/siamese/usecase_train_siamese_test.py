# app\usecases\siamese\usecase_train_siamese_test.py
import pytest
from unittest.mock import MagicMock

from app.usecases.siamese.usecase_train_siamese import TrainSiameseUsecaseDto, train_model_siamese

# Test: Verify that a 404 error is raised if the model does not exist
def test_train_model_not_found(patch_inversify):
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify

    # Create a model without training the neural network
    mock_bdd.get_model.return_value = False

    with pytest.raises(Exception) as excinfo:
        train_model_siamese(TrainSiameseUsecaseDto(name="model1", training_data=[["token1", "token2", "token3"]], inversify=mock_inversify))  # Non-existent model
    
    assert excinfo.value.status_code == 404  # Verify that the error is 404
    assert str(excinfo.value.detail) == "Model not found"

# Test: Verify that a 400 error is raised if no dictionary is provided
def test_train_model_no_dictionary(patch_inversify):
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify

    # Save an empty model
    mock_bdd.get_model.return_value = {
        "name": "model1",
        "dictionary": [["cat", "dog", "bird"]],
        "indexed_dictionary": [[2, 3, 4]],
        "glossary": ["", "UNK", "cat", "dog", "bird"],
        "nn_model": MagicMock()
    }

    with pytest.raises(Exception) as excinfo:
        train_model_siamese(TrainSiameseUsecaseDto(name="model1", training_data=None, inversify=mock_inversify))
    
    assert excinfo.value.status_code == 400
    assert str(excinfo.value.detail) == "No training data provided"

# Test: Verify that a 400 error is raised if the dictionary is empty
def test_train_model_empty_dictionary(patch_inversify):
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify

    # Save an empty model
    mock_bdd.get_model.return_value = {
        "name": "model1",
        "dictionary": [["cat", "dog", "bird"]],
        "indexed_dictionary": [[2, 3, 4]],
        "glossary": ["", "UNK", "cat", "dog", "bird"],
        "nn_model": MagicMock()
    }

    with pytest.raises(Exception) as excinfo:
        train_model_siamese(TrainSiameseUsecaseDto(name="model1", training_data=[], inversify=mock_inversify))
    
    assert excinfo.value.status_code == 400
    assert str(excinfo.value.detail) == "Training data is empty"

# Test: Successful training
def test_train_model_success(patch_inversify):
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify

    # Save an empty model
    mock_bdd.get_model.return_value = {
        "name": "model1",
        "dictionary": [["cat", "dog", "bird"]],
        "indexed_dictionary": [[2, 3, 4]],
        "glossary": ["", "UNK", "cat", "dog", "bird"],
        "nn_model": MagicMock()
    }

    # Call the function with valid training data
    response = train_model_siamese(TrainSiameseUsecaseDto(name="model1", training_data=[
        (["token1", "token2", "token3"], ["token1", "token2", "token3"], 1), 
        (["token1", "token2", "token4"], ["token1", "token2", "token4"], 1), 
        (["token1", "token2", "token5"], ["token1", "token2", "token5"], 1)
    ], inversify=mock_inversify))

    # Verify that the response is correct
    assert response["status"] == "training completed"
    assert response["model_name"] == "model1"
