# tests/test_nn_lstm.py

import pytest
import numpy as np
import torch
from unittest.mock import patch
from app.machine_learning.nn_lstm import train_nn_lstm, predict_nn_lstm, LSTMNN

@pytest.fixture
def synthetic_data():
    """
    Fixture pour générer des données synthétiques pour les tests.
    """
    np.random.seed(0)
    torch.manual_seed(0)
    
    # Données d'entraînement synthétiques
    X_train = np.random.rand(100, 10, 5)  # 100 échantillons, 10 pas de temps, 5 caractéristiques
    y_train = np.random.rand(100, 1)       # 100 valeurs cibles
    
    # Données de test synthétiques
    X_test = np.random.rand(20, 10, 5)     # 20 échantillons de test
    
    return X_train, y_train, X_test

@patch('app.machine_learning.nn_lstm.logger')
def test_train_nn_lstm(mock_logger, synthetic_data):
    """
    Teste la fonction d'entraînement du modèle LSTM.
    """
    X_train, y_train, _ = synthetic_data
    
    # Entraînement du modèle
    model = train_nn_lstm(X_train, y_train, epochs=5, learning_rate=0.01, patience=3)
    
    # Vérification que le modèle est une instance de LSTMNN
    assert isinstance(model, LSTMNN), "Le modèle n'est pas une instance de LSTMNN"
    
    # Vérification que le modèle peut produire des sorties
    outputs = model(torch.Tensor(X_train[:1]))
    assert outputs.shape == (1, 1), f"Forme des sorties inattendue: {outputs.shape}"
    
    # Vérification que le logger a été appelé
    assert mock_logger.info.called or mock_logger.debug.called, "Le logger n'a pas été appelé"

@patch('app.machine_learning.nn_lstm.logger')
def test_predict_nn_lstm(mock_logger, synthetic_data):
    """
    Teste la fonction de prédiction du modèle LSTM.
    """
    X_train, y_train, X_test = synthetic_data
    
    # Entraînement du modèle
    model = train_nn_lstm(X_train, y_train, epochs=5, learning_rate=0.01, patience=3)
    
    # Prédictions
    predictions = predict_nn_lstm(model, X_test)
    
    # Vérification de la forme des prédictions
    assert predictions.shape == (20,), f"Forme des prédictions inattendue: {predictions.shape}"
    
    # Vérification que les prédictions sont numériques
    assert np.issubdtype(predictions.dtype, np.number), "Les prédictions ne sont pas numériques"
    
    # Vérification que le logger a été appelé pendant l'entraînement
    assert mock_logger.info.called or mock_logger.debug.called, "Le logger n'a pas été appelé pendant l'entraînement"
