# app\usecases\gru\usecase_super_train_gru_test.py
import pytest
from unittest.mock import patch, MagicMock

from app.services.bdd.models.model_data import ModelData, ModelStatus
from app.apis.models.gru_training_model_data import GRUTrainingModelData
from app.usecases.gru.usecase_super_train_gru import SuperTrainGRUUsecaseDto, super_train_model_gru

# ✅ Test principal : entraînement multiple avec succès
@patch('app.usecases.gru.usecase_super_train_gru.train_model_gru')
@patch('app.usecases.gru.usecase_super_train_gru.mesure_gru')
def test_super_train_model_gru_success(mock_mesure_gru, mock_train_model_gru, patch_inversify):
    """Test successful multi-iteration training of the GRU model."""
    
    # Mock dependencies
    mock_inversify, mock_bdd = patch_inversify
    mock_model_data = MagicMock()
    mock_model_data.name = "test_gru_model"
    mock_model_data.neural_network_type = "GRUClassifier"
    mock_model_data.status = ModelStatus.TRAINED
    mock_model_data.nn_model = MagicMock()
    mock_bdd.get_model.return_value = mock_model_data

    # Mock training process
    mock_train_model_gru.return_value = {"training_stats": {"final_loss": 0.05}}

    # Mock measurement process
    mock_mesure_gru.return_value = {"summary": {"average_accuracy": 85.0}}

    # Sample training and test data
    training_data = [
        GRUTrainingModelData(category="cat1", tokens=["hello", "world"]),
        GRUTrainingModelData(category="cat2", tokens=["another", "sentence"])
    ]
    test_data = [
        GRUTrainingModelData(category="cat1", tokens=["test", "phrase"]),
        GRUTrainingModelData(category="cat2", tokens=["example", "sentence"])
    ]

    # Call the function
    result = super_train_model_gru(SuperTrainGRUUsecaseDto(
        name="test_gru_model",
        training_data=training_data,
        test_data=test_data,
        inversify=mock_inversify,
        n_iterations=3  # Small number of iterations for testing
    ))

    # Vérification que l'entraînement et la mesure ont été appelés plusieurs fois
    assert mock_train_model_gru.call_count == 3
    assert mock_mesure_gru.call_count == 3

    # Vérification que la meilleure accuracy a bien été enregistrée
    assert result["best_test_accuracy"] == 85.0
    assert result["training_report"]["training_stats"]["final_loss"] == 0.05
    assert result["measurement_report"]["summary"]["average_accuracy"] == 85.0

    # Vérification de l'état final du modèle
    mock_bdd.update_model.assert_called()

# ✅ Test si le modèle est déjà en train d’être entraîné
def test_super_train_model_gru_model_already_training(patch_inversify):
    """Test the case where the GRU model is already in training mode."""
    mock_inversify, mock_bdd = patch_inversify
    mock_model_data = MagicMock(status=ModelStatus.SUPER_TRAINING)
    mock_bdd.get_model.return_value = mock_model_data

    training_data = [GRUTrainingModelData(category="cat1", tokens=["hello", "world"])]
    test_data = [GRUTrainingModelData(category="cat1", tokens=["test", "sentence"])]

    with pytest.raises(Exception, match="Model is training"):
        super_train_model_gru(SuperTrainGRUUsecaseDto(
            name="test_gru_model",
            training_data=training_data,
            test_data=test_data,
            inversify=mock_inversify
        ))

# ✅ Test si une erreur survient pendant l’entraînement
@patch('app.usecases.gru.usecase_super_train_gru.train_model_gru')
def test_super_train_model_gru_error_during_training(mock_train_model_gru, patch_inversify):
    """Test if an error occurs during the super training process."""
    mock_inversify, mock_bdd = patch_inversify
    mock_model_data = MagicMock(status=ModelStatus.TRAINED)
    mock_bdd.get_model.return_value = mock_model_data

    # Simulation d'une erreur pendant l'entraînement
    mock_train_model_gru.side_effect = Exception("Training failure")

    training_data = [GRUTrainingModelData(category="cat1", tokens=["hello", "world"])]
    test_data = [GRUTrainingModelData(category="cat1", tokens=["test", "sentence"])]

    with pytest.raises(Exception, match="Training failure"):
        super_train_model_gru(SuperTrainGRUUsecaseDto(
            name="test_gru_model",
            training_data=training_data,
            test_data=test_data,
            inversify=mock_inversify
        ))
