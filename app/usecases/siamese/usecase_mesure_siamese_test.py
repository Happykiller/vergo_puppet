# app\usecases\siamese\usecase_mesure_siamese_test.py
import pytest
from unittest.mock import patch, MagicMock

from app.usecases.siamese.usecase_mesure_siamese import MesureSiameseUsecaseDto, mesure_siamese

# Test: Successful measurement with valid test data
@patch('app.usecases.siamese.usecase_mesure_siamese.evaluate_similarity')
@patch('app.usecases.siamese.usecase_mesure_siamese.create_indexed_glossary')
@patch('app.usecases.siamese.usecase_mesure_siamese.logger')
def test_mesure_siamese_success(mock_logger, mock_create_indexed_glossary, mock_evaluate_similarity, patch_inversify):
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify

    # Mock model data
    mock_bdd.get_model.return_value = {
        "name": "test_siamese_model",
        "nn_model": MagicMock(),
        "glossary": ["dog", "cat", "bird"]
    }
    
    # Simulate indexed glossary and similarity function
    mock_create_indexed_glossary.return_value = {"dog": 0, "cat": 1, "bird": 2}
    mock_evaluate_similarity.side_effect = [1.0, 0.7, 0.4]  # Mocked similarities
    
    # Test data
    test_data = [
        (["dog"], ["dog"], 1.0),
        (["cat"], ["bird"], 0.7),
        (["dog"], ["cat"], 0.4)
    ]
    
    # Call mesure_siamese function
    mesure_siamese(MesureSiameseUsecaseDto(name="test_siamese_model", test_data=test_data, inversify=mock_inversify))
    
    # Check that logs include the number of correct predictions and accuracy
    mock_logger.info.assert_any_call("Prediction accuracy: 100.00% (3/3)")
    mock_logger.info.assert_any_call("Average similarity precision: 100.00%")

# Test: Error if the model is not found
@patch('app.usecases.siamese.usecase_mesure_siamese.logger')
def test_mesure_siamese_model_not_found(mock_logger, patch_inversify):
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_model.return_value = None

    # Test data
    test_data = [(["dog"], ["cat"], 0.5)]
    
    # Check that an exception is raised if the model is not found
    with pytest.raises(Exception, match="Model not found"):
        mesure_siamese(MesureSiameseUsecaseDto(name="unknown_model", test_data=test_data, inversify=mock_inversify))
    
    # Verify that the error was logged
    mock_logger.error.assert_called_once_with("An error occurred during siamese testing: Model not found")

# Test: Error if the model lacks nn_model or glossary
@patch('app.usecases.siamese.usecase_mesure_siamese.logger')
def test_mesure_siamese_incomplete_model_data(mock_logger, patch_inversify):
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_model.return_value = {"glossary": ["dog", "cat", "bird"]}

    # Test data
    test_data = [(["dog"], ["cat"], 0.5)]
    
    # Check that an exception is raised if nn_model is missing
    with pytest.raises(Exception, match="Model not completed"):
        mesure_siamese(MesureSiameseUsecaseDto(name="test_siamese_model", test_data=test_data, inversify=mock_inversify))
    
    # Verify that the error was logged
    mock_logger.error.assert_called_once_with("An error occurred during siamese testing: Model not completed")

# Test: Logs prediction details
@patch('app.usecases.siamese.usecase_mesure_siamese.evaluate_similarity')
@patch('app.usecases.siamese.usecase_mesure_siamese.create_indexed_glossary')
@patch('app.usecases.siamese.usecase_mesure_siamese.logger')
def test_mesure_siamese_logs_predictions(mock_logger, mock_create_indexed_glossary, mock_evaluate_similarity, patch_inversify):
    # patch_inversify est un tuple (mock_inversify, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_model.return_value = {
        "nn_model": MagicMock(),
        "glossary": ["dog", "cat", "bird"]
    }
    
    # Simulate indexed glossary and predicted similarity
    mock_create_indexed_glossary.return_value = {"dog": 0, "cat": 1, "bird": 2}
    mock_evaluate_similarity.side_effect = [0.95]  # Mocked similarity

    # Test data
    test_data = [(["dog"], ["cat"], 1.0)]
    
    # Call mesure_siamese function
    mesure_siamese(MesureSiameseUsecaseDto(name="test_siamese_model", test_data=test_data, inversify=mock_inversify))
    
    # Check that logs for requests and predictions were called
    mock_logger.info.assert_any_call("Query: ['dog'], Image: ['cat']")
    mock_logger.info.assert_any_call("Expected similarity: 100.0%, Model similarity: 95.00%, Error: 5.00%")
