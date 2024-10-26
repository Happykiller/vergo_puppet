import pytest
from unittest.mock import patch
from app.usecases.getall_model import get_all_models_usecase

# Test 1 : Vérifier que la fonction retourne tous les modèles
@patch('app.usecases.getall_model.get_all_models')
def test_get_all_models_with_models(mock_get_all_models):
    # Simuler le retour de modèles
    mock_get_all_models.return_value = [
        {"name": "model1", "type": "GRU"},
        {"name": "model2", "type": "Siamese"}
    ]
    
    # Appeler la fonction
    result = get_all_models_usecase()
    
    # Vérifier le résultat
    expected_result = {
        "models": [
            {"name": "model1", "type": "GRU"},
            {"name": "model2", "type": "Siamese"}
        ]
    }
    assert result == expected_result, f"Expected {expected_result} but got {result}"

# Test 2 : Vérifier le retour lorsque aucun modèle n'est trouvé
@patch('app.usecases.getall_model.get_all_models')
def test_get_all_models_no_models(mock_get_all_models):
    # Simuler le cas où aucun modèle n'est disponible
    mock_get_all_models.return_value = []
    
    # Appeler la fonction
    result = get_all_models_usecase()
    
    # Vérifier le résultat
    expected_result = {"message": "No models found"}
    assert result == expected_result, f"Expected {expected_result} but got {result}"
