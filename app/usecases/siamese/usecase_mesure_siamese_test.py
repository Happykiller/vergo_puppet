# app\usecases\siamese\usecase_mesure_siamese_test.py
import pytest
from unittest.mock import patch, MagicMock

from app.services.bdd.models.model_data import ModelData
from app.usecases.siamese.usecase_mesure_siamese import MesureSiameseUsecaseDto, mesure_siamese

# Test: Successful measurement with valid test data
@patch('app.usecases.siamese.usecase_mesure_siamese.evaluate_similarity')
@patch('app.usecases.siamese.usecase_mesure_siamese.create_indexed_glossary')
@patch('app.usecases.siamese.usecase_mesure_siamese.logger')
def test_mesure_siamese_success(mock_logger, mock_create_indexed_glossary, mock_evaluate_similarity, patch_inversify):
    mock_inversify, mock_bdd = patch_inversify

    # Mock model data
    mock_bdd.get_model.return_value = ModelData(
        name="test_siamese_model",
        neural_network_type="SIAMESE",
        nn_model=MagicMock(),
        glossary=["dog", "cat", "bird"]
    )
    
    # Simulate indexed glossary and similarity evaluation
    mock_create_indexed_glossary.return_value = {"dog": 0, "cat": 1, "bird": 2}
    mock_evaluate_similarity.side_effect = [1.0, 0.7, 0.4]  # Mocked similarities

    # Test data
    test_data = [
        (["dog"], ["dog"], 1.0),
        (["cat"], ["bird"], 0.7),
        (["dog"], ["cat"], 0.4)
    ]
    
    # Call the function
    result = mesure_siamese(MesureSiameseUsecaseDto(name="test_siamese_model", test_data=test_data, inversify=mock_inversify))

    # Assertions on summary
    assert result["total_tests"] == 3
    assert result["correct_predictions"] == 3
    assert result["prediction_accuracy_percentage"] == 100.0
    assert result["avg_similarity_precision_percentage"] == 100.0

# Test: Error if the model is not found
def test_mesure_siamese_model_not_found(patch_inversify):
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_model.return_value = None

    test_data = [(["dog"], ["cat"], 0.5)]

    with pytest.raises(Exception, match="Model not found"):
        mesure_siamese(MesureSiameseUsecaseDto(name="unknown_model", test_data=test_data, inversify=mock_inversify))

# Test: Error if the model lacks nn_model or glossary
def test_mesure_siamese_incomplete_model_data(patch_inversify):
    mock_inversify, mock_bdd = patch_inversify

    # Mock incomplete model
    mock_bdd.get_model.return_value = ModelData(
        name="test_siamese_model",
        neural_network_type="SIAMESE",
        nn_model=None,  # Missing nn_model
        glossary=["dog", "cat", "bird"]
    )

    test_data = [(["dog"], ["cat"], 0.5)]

    with pytest.raises(Exception, match="Model not completed"):
        mesure_siamese(MesureSiameseUsecaseDto(name="test_siamese_model", test_data=test_data, inversify=mock_inversify))

# Test: Logs prediction details
@patch('app.usecases.siamese.usecase_mesure_siamese.evaluate_similarity')
@patch('app.usecases.siamese.usecase_mesure_siamese.create_indexed_glossary')
@patch('app.usecases.siamese.usecase_mesure_siamese.logger')
def test_mesure_siamese_logs_predictions(mock_logger, mock_create_indexed_glossary, mock_evaluate_similarity, patch_inversify):
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_model.return_value = ModelData(
        name="test_siamese_model",
        neural_network_type="SIAMESE",
        nn_model=MagicMock(),
        glossary=["dog", "cat", "bird"]
    )
    
   
