import pytest
from unittest.mock import patch, MagicMock
from app.usecases.gru.usecase_mesure_gru import mesure_gru
from app.apis.models.gru_training_data import GRUTrainingData

# Test du succès de la mesure des performances
@patch('app.usecases.gru.usecase_mesure_gru.logger')
@patch('app.usecases.gru.usecase_mesure_gru.predict')
@patch('app.usecases.gru.usecase_mesure_gru.process_input')
@patch('app.usecases.gru.usecase_mesure_gru.get_model')
def test_mesure_gru_success(mock_get_model, mock_process_input, mock_predict, mock_logger):
    # Simuler les données du modèle renvoyées par get_model
    mock_get_model.return_value = {
        "nn_model": MagicMock(),
        "word2idx": {"hello": 1, "<PAD>": 0},
        "idx2category": {0: "cat1", 1: "cat2"},
        "category2idx": {"cat1": 0, "cat2": 1}
    }
    
    # Simuler le traitement de la séquence d'entrée
    mock_process_input.side_effect = lambda tokens, word2idx: [word2idx.get(token, word2idx['<PAD>']) for token in tokens]
    
    # Simuler la prédiction
    mock_predict.side_effect = [0, 1]  # Retourne les indices des catégories prédits
    
    # Créer des données de test
    test_data = [
        GRUTrainingData(category="cat1", tokens=["hello"]),
        GRUTrainingData(category="cat2", tokens=["hello", "world"])
    ]

    # Appeler la fonction mesure_gru
    mesure_gru("test_gru_model", test_data)
    
    # Vérifier que les logs d'info ont été appelés
    mock_logger.info.assert_any_call("Nombre de prédictions correctes: 2/2")
    mock_logger.info.assert_any_call("Taux de précision du modèle : 100.00%")

# Test lorsque le modèle est non entraîné
@patch('app.usecases.gru.usecase_mesure_gru.get_model', return_value={"nn_model": None})
@patch('app.usecases.gru.usecase_mesure_gru.logger')
def test_mesure_gru_model_not_trained(mock_logger, mock_get_model):
    # Créer des données de test
    test_data = [GRUTrainingData(category="cat1", tokens=["hello"])]

    # Vérifier qu'une exception est levée si le modèle n'est pas entraîné
    with pytest.raises(Exception, match="Modèle non entraîné"):
        mesure_gru("test_gru_model", test_data)
    
    # Vérifier que l'erreur a été loggée
    mock_logger.error.assert_called_once_with("Une erreur s'est produite pendant la mesure : Modèle non entraîné")

# Test lorsque les données du modèle sont incomplètes
@patch('app.usecases.gru.usecase_mesure_gru.get_model', return_value={"nn_model": MagicMock(), "word2idx": None})
@patch('app.usecases.gru.usecase_mesure_gru.logger')
def test_mesure_gru_incomplete_model_data(mock_logger, mock_get_model):
    # Créer des données de test
    test_data = [GRUTrainingData(category="cat1", tokens=["hello"])]

    # Vérifier qu'une exception est levée si les données du modèle sont incomplètes
    with pytest.raises(Exception, match="Données du modèle incomplètes"):
        mesure_gru("test_gru_model", test_data)
    
    # Vérifier que l'erreur a été loggée
    mock_logger.error.assert_called_once_with("Une erreur s'est produite pendant la mesure : Données du modèle incomplètes")

# Test lorsqu'une catégorie dans les données de test est inconnue
@patch('app.usecases.gru.usecase_mesure_gru.logger')
@patch('app.usecases.gru.usecase_mesure_gru.predict')
@patch('app.usecases.gru.usecase_mesure_gru.process_input')
@patch('app.usecases.gru.usecase_mesure_gru.get_model')
def test_mesure_gru_unknown_category_in_test_data(mock_get_model, mock_process_input, mock_predict, mock_logger):
    # Simuler les données du modèle
    mock_get_model.return_value = {
        "nn_model": MagicMock(),
        "word2idx": {"hello": 1, "<PAD>": 0},
        "idx2category": {0: "cat1", 1: "cat2"},
        "category2idx": {"cat1": 0, "cat2": 1}
    }
    
    # Simuler le traitement de la séquence et la prédiction
    mock_process_input.return_value = [1, 0]
    mock_predict.return_value = 0  # Catégorie prédite 'cat1'

    # Créer des données de test avec une catégorie inconnue
    test_data = [
        GRUTrainingData(category="cat1", tokens=["hello"]),
        GRUTrainingData(category="unknown_cat", tokens=["world"])
    ]

    # Appeler la fonction mesure_gru
    mesure_gru("test_gru_model", test_data)
    
    # Vérifier que l'avertissement pour la catégorie inconnue a été loggé
    mock_logger.warning.assert_called_once_with("Catégorie inconnue dans les données de test : 'unknown_cat'. Elle n'a pas été vue pendant l'entraînement.")
