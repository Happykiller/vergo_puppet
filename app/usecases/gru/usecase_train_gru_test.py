import pytest
from fastapi import HTTPException  # type: ignore
from unittest.mock import patch, MagicMock
from app.apis.models.gru_training_model_data import GRUTrainingModelData
from app.usecases.gru.usecase_train_gru import train_model_gru

# Test du bon déroulement de l'entraînement
@patch('app.usecases.gru.usecase_train_gru.update_model')
@patch('app.usecases.gru.usecase_train_gru.train_gru')
@patch('app.usecases.gru.usecase_train_gru.prepare_sequences')
@patch('app.usecases.gru.usecase_train_gru.build_category_mapping')
@patch('app.usecases.gru.usecase_train_gru.build_vocab')
@patch('app.usecases.gru.usecase_train_gru.get_model')
def test_train_model_gru_success(mock_get_model, mock_build_vocab, mock_build_category_mapping, mock_prepare_sequences, mock_train_gru, mock_update_model):
    # Simuler le modèle renvoyé par get_model
    mock_get_model.return_value = {"nn_model": None}

    # Simuler le vocabulaire et le mapping de catégories
    mock_build_vocab.return_value = ({"hello": 1, "<PAD>": 0}, {1: "hello", 0: "<PAD>"})
    mock_build_category_mapping.return_value = ({"cat1": 0, "cat2": 1}, {0: "cat1", 1: "cat2"})

    # Simuler la préparation des séquences et des labels
    mock_prepare_sequences.return_value = (MagicMock(), MagicMock())  # sequences, labels

    # Simuler l'entraînement du modèle GRU
    mock_train_gru.return_value = MagicMock()  # nn_model

    # Créer des données d'entraînement pour les tests
    training_data = [
        GRUTrainingModelData(category="cat1", tokens=["hello", "world"]),
        GRUTrainingModelData(category="cat2", tokens=["another", "sentence"])
    ]

    # Appeler la fonction train_model_gru
    result = train_model_gru("test_gru_model", training_data)

    # Vérifier que la fonction update_model a été appelée
    mock_update_model.assert_called_once()

    # Vérifier le retour de la fonction
    assert result == {"status": "Entraînement terminé", "model_name": "test_gru_model"}

# Test lorsque le modèle est introuvable
@patch('app.usecases.gru.usecase_train_gru.get_model', return_value=None)
def test_train_model_gru_model_not_found(mock_get_model):
    # Créer des données d'entraînement pour le test
    training_data = [
        GRUTrainingModelData(category="cat1", tokens=["hello", "world"])
    ]

    # Vérifier qu'une exception est levée si le modèle est introuvable
    with pytest.raises(HTTPException) as exc_info:
        train_model_gru("unknown_model", training_data)

    # Vérifier que l'exception est bien une HTTPException avec le statut 404
    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Modèle non trouvé"

# Test lorsque les données d'entraînement sont manquantes
@patch('app.usecases.gru.usecase_train_gru.get_model', return_value={"nn_model": None})
def test_train_model_gru_no_training_data(mock_get_model):
    # Vérifier qu'une exception est levée si les données d'entraînement sont vides
    with pytest.raises(HTTPException) as exc_info:
        train_model_gru("test_gru_model", [])

    # Vérifier que l'exception est bien une HTTPException avec le statut 400
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Aucune donnée d'entraînement fournie ou données vides"
