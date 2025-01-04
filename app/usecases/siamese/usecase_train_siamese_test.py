# app\usecases\siamese\usecase_train_siamese_test.py
import pytest
from unittest.mock import MagicMock, patch
from app.services.bdd.models.model_data import ModelData
from app.usecases.siamese.usecase_train_siamese import TrainSiameseUsecaseDto, train_model_siamese


# Test: Verify that an exception is raised if the model does not exist
def test_train_model_not_found(patch_inversify):
    mock_inversify, mock_bdd = patch_inversify

    # Simulate model not found
    mock_bdd.get_model.return_value = None

    with pytest.raises(Exception, match="Model not found"):
        train_model_siamese(TrainSiameseUsecaseDto(
            name="non_existent_model",
            training_data=[(["token1", "token2"], ["token3", "token4"], 0.8)],
            inversify=mock_inversify
        ))


# Test: Verify that an exception is raised if no training data is provided
def test_train_model_no_training_data(patch_inversify):
    mock_inversify, mock_bdd = patch_inversify

    # Mock model with valid structure
    mock_bdd.get_model.return_value = ModelData(
        name="model1",
        neural_network_type="SiameseLSTM",
        nn_model=MagicMock(),
        glossary=["", "UNK", "cat", "dog", "bird"],
        indexed_dictionary=[2, 3, 4]
    )

    with pytest.raises(Exception, match="No training data provided"):
        train_model_siamese(TrainSiameseUsecaseDto(
            name="model1",
            training_data=None,
            inversify=mock_inversify
        ))


# Test: Verify that an exception is raised if training data is empty
def test_train_model_empty_training_data(patch_inversify):
    mock_inversify, mock_bdd = patch_inversify

    # Mock model with valid structure
    mock_bdd.get_model.return_value = ModelData(
        name="model1",
        neural_network_type="SiameseLSTM",
        nn_model=MagicMock(),
        glossary=["", "UNK", "cat", "dog", "bird"],
        indexed_dictionary=[2, 3, 4]
    )

    with pytest.raises(Exception, match="Training data is empty"):
        train_model_siamese(TrainSiameseUsecaseDto(
            name="model1",
            training_data=[],
            inversify=mock_inversify
        ))


# Test: Verify that an exception is raised if the indexed dictionary is missing
def test_train_model_no_indexed_dictionary(patch_inversify):
    mock_inversify, mock_bdd = patch_inversify

    # Mock model without an indexed dictionary
    mock_bdd.get_model.return_value = ModelData(
        name="model1",
        neural_network_type="SiameseLSTM",
        nn_model=MagicMock(),
        glossary=["", "UNK", "cat", "dog", "bird"],
        indexed_dictionary=None
    )

    with pytest.raises(Exception, match="No vectors available in the model"):
        train_model_siamese(TrainSiameseUsecaseDto(
            name="model1",
            training_data=[(["token1", "token2"], ["token3", "token4"], 0.8)],
            inversify=mock_inversify
        ))


# Test: Successful training
@patch('app.usecases.siamese.usecase_train_siamese.train_siamese_model_nn', return_value=(MagicMock(), {"loss": 0.2}))
def test_train_model_success(mock_train_nn, patch_inversify):
    mock_inversify, mock_bdd = patch_inversify

    # Mock model with valid data
    mock_bdd.get_model.return_value = ModelData(
        name="model1",
        neural_network_type="SiameseLSTM",
        nn_model=MagicMock(),
        glossary=["", "UNK", "cat", "dog", "bird"],
        indexed_dictionary=[2, 3, 4]
    )

    # Mock update_model to validate saved model data
    def mock_update_model(model_data):
        assert model_data.name == "model1"
        assert model_data.neural_network_type == "SiameseLSTM"
        assert model_data.nn_model is not None

    mock_bdd.update_model.side_effect = mock_update_model

    # Call the function with valid training data
    response = train_model_siamese(TrainSiameseUsecaseDto(
        name="model1",
        training_data=[
            (["token1", "token2"], ["token3", "token4"], 0.8),
            (["token5"], ["token6"], 0.6)
        ],
        inversify=mock_inversify
    ))

    # Verify the response
    assert response["status"] == "training completed"
    assert response["model_name"] == "model1"
    assert "training_report" in response
    assert response["training_report"]["loss"] == 0.2
