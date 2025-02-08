# app\usecases\simple\usecase_train_simple_test.py
import pytest
from unittest.mock import patch, MagicMock

from app.services.bdd.models.model_data import ModelData
from app.apis.models.simple_nn_training_model_data import SimpleNNTrainingModelData
from app.usecases.simple.usecase_train_simple import TrainSimpleUsecaseDto, train_model_simple_nn


# Test for successful training
@patch("app.usecases.simple.usecase_train_simple.train_model_nn")
@patch("app.usecases.simple.usecase_train_simple.transform_data")
def test_train_model_simple_nn_success(mock_transform_data, mock_train_model_nn, patch_inversify):
    """Test successful training of the model."""
    mock_inversify, mock_bdd = patch_inversify

    # Mock model retrieval and transformed data
    mock_bdd.get_model.return_value = ModelData(
        name="test_model",
        neural_network_type="SimpleNN"
    )
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
    mock_train_model_nn.return_value = (MagicMock(), [0.5, 0.2, 0.1])  # nn_model, losses

    # Test training data
    training_data = [
        SimpleNNTrainingModelData(type=1, surface=75, pieces=3, floor=2, parking=1, balcon=0, ascenseur=1, orientation=1, transports=1, neighborhood=8, price=350000),
        SimpleNNTrainingModelData(type=2, surface=100, pieces=4, floor=1, parking=1, balcon=1, ascenseur=0, orientation=2, transports=2, neighborhood=6, price=450000)
    ]

    # Call the function
    result = train_model_simple_nn(TrainSimpleUsecaseDto(name="test_model", training_data=training_data, inversify=mock_inversify))

    # Verify update_model was called
    mock_bdd.update_model.assert_called()
    actual_model_data = mock_bdd.update_model.call_args[0][0]

    # Validate ModelData attributes
    assert isinstance(actual_model_data, ModelData)
    assert actual_model_data.name == "test_model"
    assert actual_model_data.targets_mean == 0.5
    assert actual_model_data.targets_std == 0.2
    assert actual_model_data.indices == {"categorical_indices": [0, 1], "numerical_indices": [2, 3]}

    # Check function result
    assert result == {"status": "training completed", "model_name": "test_model"}


# Test when the model is not found
def test_train_model_simple_nn_model_not_found(patch_inversify):
    """Test the case where the model is not found."""
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_model.return_value = None

    training_data = [
        SimpleNNTrainingModelData(type=1, surface=75, pieces=3, floor=2, parking=1, balcon=0, ascenseur=1, orientation=1, transports=1, neighborhood=8, price=350000)
    ]

    # Expect exception if model not found
    with pytest.raises(Exception) as exc_info:
        train_model_simple_nn(TrainSimpleUsecaseDto(name="unknown_model", training_data=training_data, inversify=mock_inversify))

    # Validate exception message
    assert str(exc_info.value) == "[#train_model_simple_nn]Model not found"


# Test when training data is missing
def test_train_model_simple_nn_no_training_data(patch_inversify):
    """Test the case where no training data is provided."""
    mock_inversify, _ = patch_inversify

    # Expect exception if training data is empty
    with pytest.raises(Exception) as exc_info:
        train_model_simple_nn(TrainSimpleUsecaseDto(name="test_model", training_data=[], inversify=mock_inversify))

    # Validate exception message
    assert str(exc_info.value) == "[#train_model_simple_nn]No training data provided or training data is empty"


# Test logging of errors during training
@patch("app.services.logger.logger.error")
def test_train_model_simple_nn_logs_error(mock_logger_error, patch_inversify):
    """Test that errors are logged during training."""
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_model.side_effect = Exception("Unexpected error")

    training_data = [
        SimpleNNTrainingModelData(type=1, surface=75, pieces=3, floor=2, parking=1, balcon=0, ascenseur=1, orientation=1, transports=1, neighborhood=8, price=350000)
    ]

    # Expect exception and check logs
    with pytest.raises(Exception):
        train_model_simple_nn(TrainSimpleUsecaseDto(name="test_model", training_data=training_data, inversify=mock_inversify))

    # Validate logger was called with the error message
    mock_logger_error.assert_called()
    assert "Unexpected error" in mock_logger_error.call_args[0][0]
