import pytest
from unittest.mock import patch, MagicMock
from fastapi import HTTPException  # type: ignore
from app.apis.models.simple_nn_search_data import SimpleNNSearchData
from app.usecases.simple_nn.search_model_simple_nn import search_model_simple_nn

# Test du bon déroulement de la recherche avec un modèle SimpleNN
@patch('app.usecases.simple_nn.search_model_simple_nn.joblib.load')
@patch('app.usecases.simple_nn.search_model_simple_nn.predict')
@patch('app.usecases.simple_nn.search_model_simple_nn.get_model')
def test_search_model_simple_nn_success(mock_get_model, mock_predict, mock_joblib_load):
    # Simuler le modèle renvoyé par get_model
    mock_get_model.return_value = {
        "nn_model": MagicMock(),
        "encoder_filename": "encoder.pkl",
        "scaler_filename": "scaler.pkl",
        "indices_filename": "indices.pkl",
        "targets_mean": 0.5,
        "targets_std": 0.2
    }

    # Simuler le chargement des fichiers encoder, scaler, et indices
    mock_joblib_load.side_effect = [
        MagicMock(),  # Simuler l'encodeur
        MagicMock(),  # Simuler le scaler
        {"categorical_indices": [0, 1], "numerical_indices": [2, 3]}  # Simuler les indices
    ]

    # Simuler la prédiction du modèle
    mock_predict.return_value = 350000

    # Créer des données de recherche fictives
    search_data = SimpleNNSearchData(
        type=1,
        surface=100,
        pieces=4,
        floor=2,
        parking=1,
        balcon=0,
        ascenseur=1,
        orientation=1,
        transports=1,
        neighborhood=8
    )

    # Appeler la fonction search_model_simple_nn
    result = search_model_simple_nn("test_model", search_data)

    # Vérifier que la fonction predict a été appelée avec les bons arguments
    mock_predict.assert_called_once()

    # Vérifier le résultat de la recherche
    assert result == {"predicted_price": 350000}

# Test lorsque le modèle est introuvable
@patch('app.usecases.simple_nn.search_model_simple_nn.get_model', return_value=None)
def test_search_model_simple_nn_model_not_found(mock_get_model):
    search_data = SimpleNNSearchData(
        type=1,
        surface=100,
        pieces=4,
        floor=2,
        parking=1,
        balcon=0,
        ascenseur=1,
        orientation=1,
        transports=1,
        neighborhood=8
    )

    # Vérifier qu'une exception HTTP 404 est levée si le modèle est introuvable
    with pytest.raises(HTTPException) as exc_info:
        search_model_simple_nn("unknown_model", search_data)
    
    # Vérifier que l'exception est bien une HTTPException avec le statut 404
    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Model not found"

# Test lorsque le modèle n'est pas encore entraîné
@patch('app.usecases.simple_nn.search_model_simple_nn.get_model')
def test_search_model_simple_nn_model_not_trained(mock_get_model):
    # Simuler un modèle sans nn_model
    mock_get_model.return_value = {
        "nn_model": None,
        "encoder_filename": "encoder.pkl",
        "scaler_filename": "scaler.pkl",
        "indices_filename": "indices.pkl",
        "targets_mean": 0.5,
        "targets_std": 0.2
    }

    search_data = SimpleNNSearchData(
        type=1,
        surface=100,
        pieces=4,
        floor=2,
        parking=1,
        balcon=0,
        ascenseur=1,
        orientation=1,
        transports=1,
        neighborhood=8
    )

    # Vérifier qu'une exception HTTP 400 est levée si le modèle n'est pas encore entraîné
    with pytest.raises(HTTPException) as exc_info:
        search_model_simple_nn("test_model", search_data)

    # Vérifier que l'exception est bien une HTTPException avec le statut 400
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Model not trained yet"

# Test lorsque les fichiers d'encodeur, scaler ou indices manquent
@patch('app.usecases.simple_nn.search_model_simple_nn.get_model')
def test_search_model_simple_nn_missing_files(mock_get_model):
    # Simuler un modèle sans encodeur, scaler ou indices
    mock_get_model.return_value = {
        "nn_model": MagicMock(),
        "encoder_filename": None,
        "scaler_filename": None,
        "indices_filename": None,
        "targets_mean": 0.5,
        "targets_std": 0.2
    }

    search_data = SimpleNNSearchData(
        type=1,
        surface=100,
        pieces=4,
        floor=2,
        parking=1,
        balcon=0,
        ascenseur=1,
        orientation=1,
        transports=1,
        neighborhood=8
    )

    # Vérifier qu'une exception HTTP 400 est levée si les fichiers manquent
    with pytest.raises(HTTPException) as exc_info:
        search_model_simple_nn("test_model", search_data)

    # Vérifier que l'exception est bien une HTTPException avec le statut 400
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Missing encoder, scaler, or indices in the model"

# Test lorsque les paramètres de normalisation des targets manquent
@patch('app.usecases.simple_nn.search_model_simple_nn.joblib.load')  # Simuler le chargement de joblib.load
@patch('app.usecases.simple_nn.search_model_simple_nn.get_model')
def test_search_model_simple_nn_missing_normalization_parameters(mock_get_model, mock_joblib_load):
    # Simuler un modèle sans paramètres de normalisation
    mock_get_model.return_value = {
        "nn_model": MagicMock(),
        "encoder_filename": "encoder.pkl",
        "scaler_filename": "scaler.pkl",
        "indices_filename": "indices.pkl",
        "targets_mean": None,
        "targets_std": None
    }

    # Simuler le chargement des fichiers d'encodeur, scaler, et indices avec joblib.load
    mock_joblib_load.side_effect = [
        MagicMock(),  # Simuler l'encodeur
        MagicMock(),  # Simuler le scaler
        {"categorical_indices": [0, 1], "numerical_indices": [2, 3]}  # Simuler les indices
    ]

    # Créer les données de recherche fictives
    search_data = SimpleNNSearchData(
        type=1,
        surface=100,
        pieces=4,
        floor=2,
        parking=1,
        balcon=0,
        ascenseur=1,
        orientation=1,
        transports=1,
        neighborhood=8
    )

    # Vérifier qu'une exception HTTP 400 est levée si les paramètres de normalisation manquent
    with pytest.raises(HTTPException) as exc_info:
        search_model_simple_nn("test_model", search_data)

    # Vérifier que l'exception est bien une HTTPException avec le statut 400
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Missing normalization parameters in the model"
