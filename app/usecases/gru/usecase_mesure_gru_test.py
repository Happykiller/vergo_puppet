# app\usecases\gru\usecase_mesure_gru_test.py
import pytest
from unittest.mock import patch, MagicMock

from app.usecases.gru.usecase_mesure_gru import mesure_gru
from app.apis.models.gru_training_model_data import GRUTrainingModelData

# Test for successful performance measurement
@patch('app.usecases.gru.usecase_mesure_gru.logger')
@patch('app.usecases.gru.usecase_mesure_gru.predict')
@patch('app.usecases.gru.usecase_mesure_gru.process_input')
@patch('app.usecases.gru.usecase_mesure_gru.get_model')
def test_mesure_gru_success(mock_get_model, mock_process_input, mock_predict, mock_logger):
    """
    Tests that the mesure_gru function successfully calculates the model's performance metrics.
    """
    # Simulate model data returned by get_model
    mock_get_model.return_value = {
        "nn_model": MagicMock(),
        "word2idx": {"hello": 1, "<PAD>": 0},
        "idx2category": {0: "cat1", 1: "cat2"},
        "category2idx": {"cat1": 0, "cat2": 1}
    }
    
    # Mock input processing
    mock_process_input.side_effect = lambda tokens, word2idx: [word2idx.get(token, word2idx['<PAD>']) for token in tokens]
    
    # Mock prediction results
    mock_predict.side_effect = [0, 1]  # Simulate predicted category indices
    
    # Create test data
    test_data = [
        GRUTrainingModelData(category="cat1", tokens=["hello"]),
        GRUTrainingModelData(category="cat2", tokens=["hello", "world"])
    ]

    # Call the mesure_gru function
    mesure_gru("test_gru_model", test_data)
    
    # Verify that the expected info logs were called
    mock_logger.info.assert_any_call("Number of correct predictions: 2/2")
    mock_logger.info.assert_any_call("Model accuracy rate: 100.00%")

# Test when the model is not trained
@patch('app.usecases.gru.usecase_mesure_gru.get_model', return_value={"nn_model": None})
@patch('app.usecases.gru.usecase_mesure_gru.logger')
def test_mesure_gru_model_not_trained(mock_logger, mock_get_model):
    """
    Tests that the mesure_gru function raises an exception if the model is not trained.
    """
    # Create test data
    test_data = [GRUTrainingModelData(category="cat1", tokens=["hello"])]

    # Verify that an exception is raised for an untrained model
    with pytest.raises(Exception, match="Model is not trained"):
        mesure_gru("test_gru_model", test_data)
    
    # Verify that the error was logged
    mock_logger.error.assert_called_once_with("An error occurred during measurement: 400: Model is not trained")

# Test when the model data is incomplete
@patch('app.usecases.gru.usecase_mesure_gru.get_model', return_value={"nn_model": MagicMock(), "word2idx": None})
@patch('app.usecases.gru.usecase_mesure_gru.logger')
def test_mesure_gru_incomplete_model_data(mock_logger, mock_get_model):
    """
    Tests that the mesure_gru function raises an exception if model data is incomplete.
    """
    # Create test data
    test_data = [GRUTrainingModelData(category="cat1", tokens=["hello"])]

    # Verify that an exception is raised for incomplete model data
    with pytest.raises(Exception, match="Model data is incomplete"):
        mesure_gru("test_gru_model", test_data)
    
    # Verify that the error was logged
    mock_logger.error.assert_called_once_with("An error occurred during measurement: Model data is incomplete")

# Test when an unknown category is present in the test data
@patch('app.usecases.gru.usecase_mesure_gru.logger')
@patch('app.usecases.gru.usecase_mesure_gru.predict')
@patch('app.usecases.gru.usecase_mesure_gru.process_input')
@patch('app.usecases.gru.usecase_mesure_gru.get_model')
def test_mesure_gru_unknown_category_in_test_data(mock_get_model, mock_process_input, mock_predict, mock_logger):
    """
    Tests that the mesure_gru function logs a warning when an unknown category is encountered in test data.
    """
    # Simulate model data
    mock_get_model.return_value = {
        "nn_model": MagicMock(),
        "word2idx": {"hello": 1, "<PAD>": 0},
        "idx2category": {0: "cat1", 1: "cat2"},
        "category2idx": {"cat1": 0, "cat2": 1}
    }
    
    # Mock input processing and prediction
    mock_process_input.return_value = [1, 0]
    mock_predict.return_value = 0  # Predicted category 'cat1'

    # Create test data with an unknown category
    test_data = [
        GRUTrainingModelData(category="cat1", tokens=["hello"]),
        GRUTrainingModelData(category="unknown_cat", tokens=["world"])
    ]

    # Call the mesure_gru function
    mesure_gru("test_gru_model", test_data)
    
    # Verify that the warning for the unknown category was logged
    mock_logger.warning.assert_called_once_with("Unknown category in test data: 'unknown_cat'. It was not seen during training.")
