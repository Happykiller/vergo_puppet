import pytest
from unittest.mock import patch
from fastapi import HTTPException  # type: ignore
from app.usecases.simple_nn.create_model_simple_nn import create_model_simpleNN

# Test lorsque le modèle est créé avec succès
@patch('app.usecases.simple_nn.create_model_simple_nn.save_model')
@patch('app.usecases.simple_nn.create_model_simple_nn.model_exists', return_value=False)  # Simuler que le modèle n'existe pas
def test_create_model_simpleNN_success(mock_model_exists, mock_save_model):
    model_name = "test_model"
    
    # Appeler la fonction create_model_simpleNN
    response = create_model_simpleNN(model_name)
    
    # Vérifier que la fonction save_model a bien été appelée
    mock_save_model.assert_called_once_with(model_name, {"neural_network_type": "SimpleNN"})
    
    # Vérifier la réponse
    assert response == {"status": "model created", "model_name": model_name}

# Test lorsque le modèle existe déjà
@patch('app.usecases.simple_nn.create_model_simple_nn.model_exists', return_value=True)  # Simuler que le modèle existe déjà
def test_create_model_simpleNN_model_already_exists(mock_model_exists):
    model_name = "existing_model"
    
    # Vérifier qu'une exception HTTP 400 est levée
    with pytest.raises(HTTPException) as exc_info:
        create_model_simpleNN(model_name)
    
    # Vérifier le message et le code d'erreur
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Model already exists"

# Test lorsque le modèle est sauvegardé avec les bonnes données
@patch('app.usecases.simple_nn.create_model_simple_nn.model_exists', return_value=False)  # Simuler que le modèle n'existe pas
@patch('app.usecases.simple_nn.create_model_simple_nn.save_model')  # Simuler l'enregistrement du modèle
def test_create_model_simpleNN_save_called_with_correct_data(mock_save_model, mock_model_exists):
    model_name = "new_model"
    
    # Appeler la fonction create_model_simpleNN
    response = create_model_simpleNN(model_name)
    
    # Vérifier que save_model a été appelé avec les bons arguments
    expected_model_data = {
        "neural_network_type": "SimpleNN"
    }
    mock_save_model.assert_called_once_with(model_name, expected_model_data)
    
    # Vérifier la réponse
    assert response == {"status": "model created", "model_name": model_name}
