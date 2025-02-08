# app/usecases/siamese/usecase_super_train_siamese_test.py
import pytest
from unittest.mock import patch, MagicMock

from app.services.bdd.models.model_data import ModelData, ModelStatus
from app.usecases.siamese.usecase_super_train_siamese import SuperTrainSiameseUsecaseDto, super_train_model_siamese

# ✅ Test principal : entraînement multiple avec succès
@patch('app.usecases.siamese.usecase_super_train_siamese.train_model_siamese')
@patch('app.usecases.siamese.usecase_super_train_siamese.mesure_siamese')
def test_super_train_model_siamese_success(mock_mesure_siamese, mock_train_model_siamese, patch_inversify):
    """Test successful multi-iteration training of the Siamese model."""
    
    # Mock dependencies
    mock_inversify, mock_bdd = patch_inversify
    mock_model_data = MagicMock()
    mock_model_data.name = "test_siamese_model"
    mock_model_data.neural_network_type = "SiameseLSTM"
    mock_model_data.status = ModelStatus.TRAINED
    mock_model_data.nn_model = MagicMock()
    mock_bdd.get_model.return_value = mock_model_data

    # Mock training process
    mock_train_model_siamese.return_value = {"training_stats": {"final_loss": 0.05}}

    # Mock measurement process
    mock_mesure_siamese.return_value = {"prediction_accuracy_percentage": 87.5}

    # Sample training and test data
    training_data = [
        (["hello", "world"], ["hi", "planet"], 0.9),
        (["another", "sentence"], ["different", "words"], 0.1)
    ]
    test_data = [
        ["test", "phrase"],
        ["example", "sentence"]
    ]

    # Call the function
    result = super_train_model_siamese(SuperTrainSiameseUsecaseDto(
        name="test_siamese_model",
        training_data=training_data,
        test_data=test_data,
        inversify=mock_inversify,
        n_iterations=3  # Small number of iterations for testing
    ))

    # Vérification que l'entraînement et la mesure ont été appelés plusieurs fois
    assert mock_train_model_siamese.call_count == 3
    assert mock_mesure_siamese.call_count == 3

    # Vérification que la meilleure accuracy a bien été enregistrée
    assert result["best_test_accuracy"] == 87.5
    assert result["training_report"]["training_stats"]["final_loss"] == 0.05
    assert result["measurement_report"]["prediction_accuracy_percentage"] == 87.5

    # Vérification de l'état final du modèle
    mock_bdd.update_model.assert_called()

# ✅ Test si le modèle est déjà en train d’être entraîné
def test_super_train_model_siamese_model_already_training(patch_inversify):
    """Test the case where the Siamese model is already in training mode."""
    mock_inversify, mock_bdd = patch_inversify
    mock_model_data = MagicMock(status=ModelStatus.SUPER_TRAINING)
    mock_bdd.get_model.return_value = mock_model_data

    training_data = [(["hello", "world"], ["hi", "planet"], 0.9)]
    test_data = [["test", "sentence"]]

    with pytest.raises(Exception, match="Model is training"):
        super_train_model_siamese(SuperTrainSiameseUsecaseDto(
            name="test_siamese_model",
            training_data=training_data,
            test_data=test_data,
            inversify=mock_inversify
        ))

# ✅ Test si une erreur survient pendant l’entraînement
@patch('app.usecases.siamese.usecase_super_train_siamese.train_model_siamese')
def test_super_train_model_siamese_error_during_training(mock_train_model_siamese, patch_inversify):
    """Test if an error occurs during the super training process."""
    mock_inversify, mock_bdd = patch_inversify
    mock_model_data = MagicMock(status=ModelStatus.TRAINED)
    mock_bdd.get_model.return_value = mock_model_data

    # Simulation d'une erreur pendant l'entraînement
    mock_train_model_siamese.side_effect = Exception("Training failure")

    training_data = [(["hello", "world"], ["hi", "planet"], 0.9)]
    test_data = [["test", "sentence"]]

    with pytest.raises(Exception, match="Training failure"):
        super_train_model_siamese(SuperTrainSiameseUsecaseDto(
            name="test_siamese_model",
            training_data=training_data,
            test_data=test_data,
            inversify=mock_inversify
        ))
