import pytest
from unittest.mock import patch
from fastapi import HTTPException  # type: ignore
from app.usecases.gru.usecase_create_gru import create_model_gru

# Test lorsque le modèle GRU est créé avec succès
@patch('app.usecases.gru.usecase_create_gru.save_model')
@patch('app.usecases.gru.usecase_create_gru.model_exists', return_value=False)  # Simuler que le modèle n'existe pas
def test_create_model_gru_success(mock_model_exists, mock_save_model):
    model_name = "test_gru_model"
    
    # Appeler la fonction create_model_gru
    response = create_model_gru(model_name)
    
    # Vérifier que save_model a bien été appelé
    mock_save_model.assert_called_once_with(model_name, {"neural_network_type": "GRU"})
    
    # Vérifier la réponse
    assert response == {"status": "model created", "model_name": model_name}

# Test lorsque le modèle GRU existe déjà
@patch('app.usecases.gru.usecase_create_gru.model_exists', return_value=True)  # Simuler que le modèle existe déjà
def test_create_model_gru_model_already_exists(mock_model_exists):
    model_name = "existing_gru_model"
    
    # Vérifier qu'une exception HTTP 400 est levée
    with pytest.raises(HTTPException) as exc_info:
        create_model_gru(model_name)
    
    # Vérifier le message et le code d'erreur
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Model already exists"

# Test lorsque le modèle GRU est sauvegardé avec les bonnes données
@patch('app.usecases.gru.usecase_create_gru.model_exists', return_value=False)  # Simuler que le modèle n'existe pas
@patch('app.usecases.gru.usecase_create_gru.save_model')  # Simuler l'enregistrement du modèle
def test_create_model_gru_save_called_with_correct_data(mock_save_model, mock_model_exists):
    model_name = "new_gru_model"
    
    # Appeler la fonction create_model_gru
    response = create_model_gru(model_name)
    
    # Vérifier que save_model a été appelé avec les bons arguments
    expected_model_data = {
        "neural_network_type": "GRU"
    }
    mock_save_model.assert_called_once_with(model_name, expected_model_data)
    
    # Vérifier la réponse
    assert response == {"status": "model created", "model_name": model_name}
