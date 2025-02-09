# app\usecases\gru\usecase_mesure_gru_test.py
import pytest
from unittest.mock import patch, MagicMock

from app.apis.models.gru_training_model_data import GRUTrainingModelData
from app.usecases.gru.usecase_mesure_gru import MesureGRUUsecaseDto, mesure_gru

# Test for successful performance measurement
@patch('app.usecases.gru.usecase_mesure_gru.logger')
@patch('app.usecases.gru.usecase_mesure_gru.predict')
@patch('app.usecases.gru.usecase_mesure_gru.process_input')
def test_mesure_gru_success(mock_process_input, mock_predict, mock_logger, patch_inversify):
    """
    Tests that the mesure_gru function successfully calculates the model's performance metrics.
    """
    mock_inversify, mock_bdd = patch_inversify

    # Mock model data
    mock_bdd.get_model.return_value = MagicMock(
        nn_model=MagicMock(),
        word2idx={"hello": 1, "<PAD>": 0},
        idx2category={0: "cat1", 1: "cat2"},
        category2idx={"cat1": 0, "cat2": 1}
    )

    # Mock input processing and predictions
    mock_process_input.side_effect = lambda tokens, word2idx: [word2idx.get(token, word2idx['<PAD>']) for token in tokens]
    mock_predict.side_effect = lambda model, x: 1  # Simulated predictions

    # Test data
    test_data = [
        GRUTrainingModelData(category="cat1", tokens=["hello"]),
        GRUTrainingModelData(category="cat2", tokens=["hello", "world"])
    ]

    # Call the function
    result = mesure_gru(MesureGRUUsecaseDto(name="test_gru_model", test_data=test_data, inversify=mock_inversify))

    # Verify results and summary
    summary = result["summary"]
    assert summary["iterations"] == 10
    assert summary["average_accuracy"] == 50.0
    assert summary["min_accuracy"] == 50.0
    assert summary["max_accuracy"] == 50.0

    detailed_results = result["history"][0]["detailed_results"]
    assert len(detailed_results) == 2
    assert detailed_results[0]["expected_category"] == "cat1"
    assert detailed_results[0]["predicted_category"] == "cat2"
    assert detailed_results[0]["is_correct"] is False


# Test when the model is not trained
def test_mesure_gru_model_not_trained(patch_inversify):
    """
    Tests that the mesure_gru function raises an exception if the model is not trained.
    """
    mock_inversify, mock_bdd = patch_inversify

    # Mock model data with an untrained model
    mock_bdd.get_model.return_value = MagicMock(nn_model=None)

    # Test data
    test_data = [GRUTrainingModelData(category="cat1", tokens=["hello"])]

    with pytest.raises(Exception, match="Model is not trained"):
        mesure_gru(MesureGRUUsecaseDto(name="test_gru_model", test_data=test_data, inversify=mock_inversify))


# Test when model data is incomplete
def test_mesure_gru_incomplete_model_data(patch_inversify):
    """
    Tests that the mesure_gru function raises an exception if model data is incomplete.
    """
    mock_inversify, mock_bdd = patch_inversify

    # Mock incomplete model data
    mock_bdd.get_model.return_value = MagicMock(
        nn_model=MagicMock(),
        word2idx=None,
        idx2category=None,
        category2idx=None
    )

    # Test data
    test_data = [GRUTrainingModelData(category="cat1", tokens=["hello"])]

    with pytest.raises(Exception, match="Model data is incomplete"):
        mesure_gru(MesureGRUUsecaseDto(name="test_gru_model", test_data=test_data, inversify=mock_inversify))


# Test when an unknown category is present in test data
@patch('app.usecases.gru.usecase_mesure_gru.logger')
@patch('app.usecases.gru.usecase_mesure_gru.predict')
@patch('app.usecases.gru.usecase_mesure_gru.process_input')
def test_mesure_gru_unknown_category_in_test_data(mock_process_input, mock_predict, mock_logger, patch_inversify):
    """
    Tests that the mesure_gru function logs a warning when an unknown category is encountered in test data.
    """
    mock_inversify, mock_bdd = patch_inversify

    # Mock model data
    mock_bdd.get_model.return_value = MagicMock(
        nn_model=MagicMock(),
        word2idx={"hello": 1, "<PAD>": 0},
        idx2category={0: "cat1", 1: "cat2"},
        category2idx={"cat1": 0, "cat2": 1}
    )

    # Mock input processing and predictions
    mock_process_input.return_value = [1, 0]
    mock_predict.return_value = 0  # Predicted category 'cat1'

    # Test data with an unknown category
    test_data = [
        GRUTrainingModelData(category="cat1", tokens=["hello"]),
        GRUTrainingModelData(category="unknown_cat", tokens=["world"])
    ]

    # Call the function
    mesure_gru(MesureGRUUsecaseDto(name="test_gru_model", test_data=test_data, inversify=mock_inversify))

    # Verify the warning log for the unknown category
    mock_logger.warning.assert_called_with("Unknown category in test data: 'unknown_cat'. Skipping sample.")


# Test when no test data is provided
def test_mesure_gru_no_test_data(patch_inversify):
    """
    Tests that the mesure_gru function handles an empty test dataset gracefully.
    """
    mock_inversify, mock_bdd = patch_inversify

    # Mock model data
    mock_bdd.get_model.return_value = MagicMock(
        nn_model=MagicMock(),
        word2idx={"hello": 1, "<PAD>": 0},
        idx2category={0: "cat1", 1: "cat2"},
        category2idx={"cat1": 0, "cat2": 1}
    )

    # Empty test data
    test_data = []

    # Call the function
    result = mesure_gru(MesureGRUUsecaseDto(name="test_gru_model", test_data=test_data, inversify=mock_inversify))

    # Verify the summary
    summary = result["summary"]
    assert summary["iterations"] == 10
    assert summary["average_accuracy"] == 0
    assert summary["min_accuracy"] == 0
    assert summary["max_accuracy"] == 0.0