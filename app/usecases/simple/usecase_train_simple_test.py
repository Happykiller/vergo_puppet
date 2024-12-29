# app\usecases\simple\usecase_train_simple_test.py
import pytest
from unittest.mock import patch, MagicMock

from app.apis.models.simple_nn_training_model_data import SimpleNNTrainingModelData
from app.usecases.simple.usecase_train_simple import TrainSimpleUsecaseDto, train_model_simple_nn

# Test for successful training
@patch('app.usecases.simple.usecase_train_simple.joblib.dump')
@patch('app.usecases.simple.usecase_train_simple.train_model_nn')
@patch('app.usecases.simple.usecase_train_simple.transform_data')
def test_train_model_simple_nn_success(mock_transform_data, mock_train_model_nn, mock_joblib_dump, patch_inversify):
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_model.return_value = {"nn_model": None}

    # Mock the transformed data
    mock_transform_data.return_value = (
        MagicMock(),  # features_processed
        MagicMock(),  # targets_standardized
        MagicMock(),  # encoder
        MagicMock(),  # scaler
        0.5,          # targets_mean
        0.2,          # targets_std
        [0, 1],       # categorical_indices
        [2, 3]        # numerical_indices
    )

    # Mock model training
    mock_train_model_nn.return_value = (MagicMock(), [0.5, 0.2, 0.1])  # nn_model, losses

    # Create test training data
    training_data = [
        SimpleNNTrainingModelData(type=1, surface=75, pieces=3, floor=2, parking=1, balcon=0, ascenseur=1, orientation=1, transports=1, neighborhood=8, price=350000),
        SimpleNNTrainingModelData(type=2, surface=100, pieces=4, floor=1, parking=1, balcon=1, ascenseur=0, orientation=2, transports=2, neighborhood=6, price=450000)
    ]

    # Call the train_model_simple_nn function
    result = train_model_simple_nn(TrainSimpleUsecaseDto(name="test_model", training_data=training_data, inversify=mock_inversify))

    # Verify that update_model was called
    mock_bdd.update_model.assert_called_once()

    # Verify that joblib.dump was called to save encoder, scaler, and indices files
    assert mock_joblib_dump.call_count == 3, "Encoder, scaler, and indices files should be saved"
    
    # Check the function's return value
    assert result == {"status": "training completed", "model_name": "test_model"}

# Test when the model is not found
def test_train_model_simple_nn_model_not_found(patch_inversify):
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_model.return_value = None

    training_data = [
        SimpleNNTrainingModelData(type=1, surface=75, pieces=3, floor=2, parking=1, balcon=0, ascenseur=1, orientation=1, transports=1, neighborhood=8, price=350000)
    ]

    # Verify that an exception is raised if the model is not found
    with pytest.raises(Exception) as exc_info:
        train_model_simple_nn(TrainSimpleUsecaseDto(name="unknown_model", training_data=training_data, inversify=mock_inversify))
    
    # Check that the exception is an HTTPException with status 404
    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Model not found"

# Test when training data is missing
def test_train_model_simple_nn_no_training_data(patch_inversify):
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, _ = patch_inversify

    # Verify that an exception is raised if the training data is empty
    with pytest.raises(Exception) as exc_info:
        train_model_simple_nn(TrainSimpleUsecaseDto(name="unknown_model", training_data=[], inversify=mock_inversify))

    # Check that the exception is an HTTPException with status 400
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "No training data provided or training data is empty"
