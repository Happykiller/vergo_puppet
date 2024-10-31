import pytest
from unittest.mock import patch, MagicMock
from app.usecases.siamese.usecase_mesure_siamese import mesure_siamese

# Test 1 : Succès de la mesure avec des données de test valides
@patch('app.usecases.siamese.usecase_mesure_siamese.evaluate_similarity')
@patch('app.usecases.siamese.usecase_mesure_siamese.create_indexed_glossary')
@patch('app.usecases.siamese.usecase_mesure_siamese.get_model')
@patch('app.usecases.siamese.usecase_mesure_siamese.logger')
def test_mesure_siamese_success(mock_logger, mock_get_model, mock_create_indexed_glossary, mock_evaluate_similarity):
    # Mock des données du modèle
    mock_get_model.return_value = {
        "name": "test_siamese_model",
        "nn_model": MagicMock(),
        "glossary": ["dog", "cat", "bird"]
    }
    
    # Simuler le glossaire indexé et la fonction d'évaluation
    mock_create_indexed_glossary.return_value = {"dog": 0, "cat": 1, "bird": 2}
    mock_evaluate_similarity.side_effect = [1.0, 0.7, 0.4]  # Similarités simulées
    
    # Données de test
    test_data = [
        (["dog"], ["dog"], 1.0),
        (["cat"], ["bird"], 0.7),
        (["dog"], ["cat"], 0.4)
    ]
    
    # Appeler la fonction mesure_siamese
    mesure_siamese("test_siamese_model", test_data)
    
    # Vérifier que les logs incluent le nombre de prédictions correctes et la précision
    mock_logger.info.assert_any_call("Nombre de prédictions correctes: 3/3")
    mock_logger.info.assert_any_call("Précision moyenne du modèle sur le jeu de test: 100.00%")

# Test 2 : Erreur si le modèle n'est pas trouvé
@patch('app.usecases.siamese.usecase_mesure_siamese.get_model', return_value=None)
@patch('app.usecases.siamese.usecase_mesure_siamese.logger')
def test_mesure_siamese_model_not_found(mock_logger, mock_get_model):
    # Données de test
    test_data = [(["dog"], ["cat"], 0.5)]
    
    # Vérifier qu'une exception est levée si le modèle est introuvable
    with pytest.raises(Exception, match="Model not found"):
        mesure_siamese("unknown_model", test_data)
    
    # Vérifier que l'erreur a été loggée
    mock_logger.error.assert_called_once_with("Une erreur s'est produite pendant test_siamese : Model not found")

# Test 3 : Erreur si le modèle ne contient pas de nn_model ou de glossaire
@patch('app.usecases.siamese.usecase_mesure_siamese.get_model', return_value={"glossary": ["dog", "cat", "bird"]})
@patch('app.usecases.siamese.usecase_mesure_siamese.logger')
def test_mesure_siamese_incomplete_model_data(mock_logger, mock_get_model):
    # Données de test
    test_data = [(["dog"], ["cat"], 0.5)]
    
    # Vérifier qu'une exception est levée si nn_model est manquant
    with pytest.raises(Exception, match="Model not completed"):
        mesure_siamese("test_siamese_model", test_data)
    
    # Vérifier que l'erreur a été loggée
    mock_logger.error.assert_called_once_with("Une erreur s'est produite pendant test_siamese : Model not completed")

# Test 4 : Log des détails de la prédiction
@patch('app.usecases.siamese.usecase_mesure_siamese.evaluate_similarity')
@patch('app.usecases.siamese.usecase_mesure_siamese.create_indexed_glossary')
@patch('app.usecases.siamese.usecase_mesure_siamese.get_model')
@patch('app.usecases.siamese.usecase_mesure_siamese.logger')
def test_mesure_siamese_logs_predictions(mock_logger, mock_get_model, mock_create_indexed_glossary, mock_evaluate_similarity):
    # Mock des données du modèle
    mock_get_model.return_value = {
        "nn_model": MagicMock(),
        "glossary": ["dog", "cat", "bird"]
    }
    
    # Simuler le glossaire indexé et les similarités prédites
    mock_create_indexed_glossary.return_value = {"dog": 0, "cat": 1, "bird": 2}
    mock_evaluate_similarity.side_effect = [0.95]  # Similarité simulée

    # Données de test
    test_data = [(["dog"], ["cat"], 1.0)]
    
    # Appeler la fonction mesure_siamese
    mesure_siamese("test_siamese_model", test_data)
    
    # Vérifier que les logs de requêtes et des prédictions sont appelés
    mock_logger.info.assert_any_call("Requête: ['dog'], Image: ['cat']")
    mock_logger.info.assert_any_call("Similarité attendue: 100.0%, Similarité donnée par le modèle: 95.00%, Erreur: 5.00%")
