import pytest
from unittest.mock import patch, MagicMock
from fastapi import HTTPException  # type: ignore
from app.usecases.gru.usecase_search_gru import search_model_gru

# Test du succès de la recherche
@patch('app.usecases.gru.usecase_search_gru.predict')
@patch('app.usecases.gru.usecase_search_gru.process_input')
@patch('app.usecases.gru.usecase_search_gru.get_model')
def test_search_model_gru_success(mock_get_model, mock_process_input, mock_predict):
    # Simuler les données du modèle renvoyées par get_model
    mock_get_model.return_value = {
        "nn_model": MagicMock(),
        "word2idx": {"hello": 1, "<PAD>": 0},
        "idx2category": {0: "cat1", 1: "cat2"}
    }
    
    # Simuler le traitement de la séquence d'entrée
    mock_process_input.return_value = [1, 0, 1]  # Séquence d'indices
    
    # Simuler la prédiction de la catégorie
    mock_predict.return_value = 1  # Index de la catégorie prédite
    
    # Appeler la fonction search_model_gru
    result = search_model_gru("test_gru_model", ["hello", "world"])
    
    # Vérifier le résultat de la prédiction
    assert result == {"category": "cat2"}, f"Expected category 'cat2' but got {result['category']}"

# Test lorsque le modèle est introuvable
@patch('app.usecases.gru.usecase_search_gru.get_model', return_value=None)
def test_search_model_gru_model_not_found(mock_get_model):
    # Vérifier qu'une exception est levée si le modèle est introuvable
    with pytest.raises(HTTPException) as exc_info:
        search_model_gru("unknown_model", ["hello", "world"])
    
    # Vérifier que l'exception est bien une HTTPException avec le statut 404
    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Modèle non trouvé"

# Test lorsque le modèle est non entraîné
@patch('app.usecases.gru.usecase_search_gru.get_model', return_value={"nn_model": None})
def test_search_model_gru_model_not_trained(mock_get_model):
    # Vérifier qu'une exception est levée si le modèle n'est pas entraîné
    with pytest.raises(HTTPException) as exc_info:
        search_model_gru("test_gru_model", ["hello", "world"])
    
    # Vérifier que l'exception est bien une HTTPException avec le statut 400
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Modèle non entraîné"

# Test lorsque les données du modèle sont incomplètes
@patch('app.usecases.gru.usecase_search_gru.get_model', return_value={"nn_model": MagicMock(), "word2idx": None})
def test_search_model_gru_incomplete_model_data(mock_get_model):
    # Vérifier qu'une exception est levée si les données du modèle sont incomplètes
    with pytest.raises(HTTPException) as exc_info:
        search_model_gru("test_gru_model", ["hello", "world"])
    
    # Vérifier que l'exception est bien une HTTPException avec le statut 400
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Données du modèle incomplètes"
