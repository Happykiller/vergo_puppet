import pytest
from unittest.mock import patch, MagicMock
from fastapi import HTTPException  # type: ignore
from app.usecases.simple_nn.mesure_simple_nn import mesure_simple_nn
from app.apis.models.simple_nn_training_data import SimpleNNTrainingData

# Test du bon fonctionnement de mesure_simple_nn avec des données de test valides
@patch('app.usecases.simple_nn.mesure_simple_nn.predict')
@patch('app.usecases.simple_nn.mesure_simple_nn.process_input_data')
@patch('app.usecases.simple_nn.mesure_simple_nn.joblib.load')
@patch('app.usecases.simple_nn.mesure_simple_nn.get_model')
def test_mesure_simple_nn_success(mock_get_model, mock_joblib_load, mock_process_input_data, mock_predict):
    # Préparer le modèle simulé
    mock_model = {
        "nn_model": MagicMock(),
        "encoder_filename": "encoder.pkl",
        "scaler_filename": "scaler.pkl",
        "indices_filename": "indices.pkl",
        "targets_mean": 0.5,
        "targets_std": 0.2
    }
    mock_get_model.return_value = mock_model

    # Simuler le chargement de l'encodeur, scaler et indices
    mock_joblib_load.side_effect = [
        MagicMock(),  # Encoder
        MagicMock(),  # Scaler
        {"categorical_indices": [0, 4, 5], "numerical_indices": [1, 2, 3]}  # Indices
    ]

    # Simuler la transformation des données
    mock_process_input_data.return_value = [[0.5, 1.2, 0.8]]

    # Simuler la prédiction du modèle
    mock_predict.return_value = 350000

    # Préparer les données de test
    test_data = [
        SimpleNNTrainingData(
            type=1, surface=75, pieces=3, floor=2, parking=1, balcon=0, ascenseur=1, 
            orientation=1, transports=1, neighborhood=8, price=360000
        )
    ]

    # Appeler la fonction mesure_simple_nn
    mesure_simple_nn("test_model", test_data)

    # Vérifier que la prédiction a été effectuée correctement
    mock_predict.assert_called_once()

# Test de la gestion des erreurs lorsque le modèle n'est pas encore entraîné
@patch('app.usecases.simple_nn.mesure_simple_nn.get_model')
def test_mesure_simple_nn_model_not_trained(mock_get_model):
    # Modèle sans réseau de neurones entraîné
    mock_get_model.return_value = {"nn_model": None}

    # Préparer les données de test
    test_data = [
        SimpleNNTrainingData(
            type=1, surface=75, pieces=3, floor=2, parking=1, balcon=0, ascenseur=1, 
            orientation=1, transports=1, neighborhood=8, price=360000
        )
    ]

    # Vérifier qu'une exception est levée si le modèle n'est pas encore entraîné
    with pytest.raises(Exception, match="Model not trained yet"):
        mesure_simple_nn("test_model", test_data)

# Test lorsque les fichiers d'encodeur ou de scaler manquent
@patch('app.usecases.simple_nn.mesure_simple_nn.get_model')
def test_mesure_simple_nn_missing_files(mock_get_model):
    # Modèle sans encodeur/scaler/indices
    mock_get_model.return_value = {
        "nn_model": MagicMock(),
        "encoder_filename": None,
        "scaler_filename": None,
        "indices_filename": None
    }

    # Préparer les données de test
    test_data = [
        SimpleNNTrainingData(
            type=1, surface=75, pieces=3, floor=2, parking=1, balcon=0, ascenseur=1, 
            orientation=1, transports=1, neighborhood=8, price=360000
        )
    ]

    # Vérifier qu'une exception est levée si les fichiers sont manquants
    with pytest.raises(Exception, match="Missing encoder, scaler, or indices in the model"):
        mesure_simple_nn("test_model", test_data)

# Test lorsque les paramètres de normalisation des targets manquent
@patch('app.usecases.simple_nn.mesure_simple_nn.get_model')
@patch('app.usecases.simple_nn.mesure_simple_nn.joblib.load')  # Simuler le chargement de l'encodeur/scaler/indices
def test_mesure_simple_nn_missing_normalization_parameters(mock_joblib_load, mock_get_model):
    # Modèle sans paramètres de normalisation
    mock_get_model.return_value = {
        "nn_model": MagicMock(),
        "encoder_filename": "encoder.pkl",
        "scaler_filename": "scaler.pkl",
        "indices_filename": "indices.pkl",
        "targets_mean": None,
        "targets_std": None
    }

    # Simuler le chargement des fichiers d'encodeur, scaler et indices avec joblib.load
    mock_joblib_load.side_effect = [
        MagicMock(),  # Simuler l'encodeur
        MagicMock(),  # Simuler le scaler
        {"categorical_indices": [0, 4, 5], "numerical_indices": [1, 2, 3]}  # Simuler les indices
    ]

    # Préparer les données de test
    test_data = [
        SimpleNNTrainingData(
            type=1, surface=75, pieces=3, floor=2, parking=1, balcon=0, ascenseur=1, 
            orientation=1, transports=1, neighborhood=8, price=360000
        )
    ]

    # Vérifier qu'une exception est levée si les paramètres de normalisation manquent
    with pytest.raises(Exception, match="Missing normalization parameters in the model"):
        mesure_simple_nn("test_model", test_data)
