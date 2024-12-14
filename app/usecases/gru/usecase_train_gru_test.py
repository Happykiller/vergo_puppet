#app\usecases\gru\usecase_train_gru_test.py
import pytest
from fastapi import HTTPException  # type: ignore
from unittest.mock import patch, MagicMock
from app.apis.models.gru_training_model_data import GRUTrainingModelData
from app.usecases.gru.usecase_train_gru import train_model_gru

# Test to verify successful training of the GRU model
@patch('app.usecases.gru.usecase_train_gru.update_model')
@patch('app.usecases.gru.usecase_train_gru.train_gru')
@patch('app.usecases.gru.usecase_train_gru.prepare_sequences')
@patch('app.usecases.gru.usecase_train_gru.build_category_mapping')
@patch('app.usecases.gru.usecase_train_gru.build_vocab')
@patch('app.usecases.gru.usecase_train_gru.get_model')
def test_train_model_gru_success(mock_get_model, mock_build_vocab, mock_build_category_mapping, mock_prepare_sequences, mock_train_gru, mock_update_model):
    # Mock the return value for get_model to simulate model retrieval
    mock_get_model.return_value = {"nn_model": None}

    # Mock vocabulary and category mappings
    mock_build_vocab.return_value = ({"hello": 1, "<PAD>": 0}, {1: "hello", 0: "<PAD>"})
    mock_build_category_mapping.return_value = ({"cat1": 0, "cat2": 1}, {0: "cat1", 1: "cat2"})

    # Mock sequence and label preparation
    mock_prepare_sequences.return_value = (MagicMock(), MagicMock())  # sequences, labels

    # Mock the GRU model training
    mock_train_gru.return_value = (MagicMock(), {"final_loss": 0.1, "epochs_run": 5})

    # Create test training data
    training_data = [
        GRUTrainingModelData(category="cat1", tokens=["hello", "world"]),
        GRUTrainingModelData(category="cat2", tokens=["another", "sentence"])
    ]

    # Call the train_model_gru function
    result = train_model_gru("test_gru_model", training_data)

    # Verify that update_model was called to save the trained model
    mock_update_model.assert_called_once()

    # Check the return value to confirm training completion
    assert result == {
        "status": "Training complete",
        "model_name": "test_gru_model",
        "training_stats": {"final_loss": 0.1, "epochs_run": 5}
    }

# Test when the specified model cannot be found
@patch('app.usecases.gru.usecase_train_gru.get_model', return_value=None)
def test_train_model_gru_model_not_found(mock_get_model):
    # Create test training data
    training_data = [
        GRUTrainingModelData(category="cat1", tokens=["hello", "world"])
    ]

    # Expect an HTTPException with status 404 if the model is not found
    with pytest.raises(HTTPException) as exc_info:
        train_model_gru("unknown_model", training_data)

    # Verify the exception details
    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Model not found"

# Test when the training data is missing or empty
@patch('app.usecases.gru.usecase_train_gru.get_model', return_value={"nn_model": None})
def test_train_model_gru_no_training_data(mock_get_model):
    # Expect an HTTPException with status 400 if no training data is provided
    with pytest.raises(HTTPException) as exc_info:
        train_model_gru("test_gru_model", [])

    # Verify the exception details
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "No training data provided or data is empty"
