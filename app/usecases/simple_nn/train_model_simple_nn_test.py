import pytest
from unittest.mock import patch, MagicMock
from app.apis.models.simple_nn_training_data import SimpleNNTrainingData
from app.usecases.simple_nn.train_model_simple_nn import train_model_simple_nn

# Test du bon déroulement de l'entraînement
@patch('app.usecases.simple_nn.train_model_simple_nn.joblib.dump')
@patch('app.usecases.simple_nn.train_model_simple_nn.update_model')
@patch('app.usecases.simple_nn.train_model_simple_nn.train_model_nn')
@patch('app.usecases.simple_nn.train_model_simple_nn.transform_data')
@patch('app.usecases.simple_nn.train_model_simple_nn.get_model')
def test_train_model_simple_nn_success(mock_get_model, mock_transform_data, mock_train_model_nn, mock_update_model, mock_joblib_dump):
    # Simuler le modèle renvoyé par get_model
    mock_get_model.return_value = {"nn_model": None}

    # Simuler les données transformées
    mock_transform_data.return_value = (
        MagicMock(),  # features_processed
        MagicMock(),  # targets_standardized
        MagicMock(),  # encoder
        MagicMock(),  # scaler
        0.5,          # targets_mean
        0.2,          # targets_std
        [0, 1],       # categorical_indices
        [2, 3]        # numerical_indices
    )

    # Simuler l'entraînement du modèle
    mock_train_model_nn.return_value = (MagicMock(), [0.5, 0.2, 0.1])  # nn_model, losses

    # Créer des données de test
    training_data = [
        SimpleNNTrainingData(type=1, surface=75, pieces=3, floor=2, parking=1, balcon=0, ascenseur=1, orientation=1, transports=1, neighborhood=8, price=350000),
        SimpleNNTrainingData(type=2, surface=100, pieces=4, floor=1, parking=1, balcon=1, ascenseur=0, orientation=2, transports=2, neighborhood=6, price=450000)
    ]

    # Appeler la fonction train_model_simple_nn
    result = train_model_simple_nn("test_model", training_data)

    # Vérifier que la fonction update_model a été appelée
    mock_update_model.assert_called_once()

    # Vérifier que la fonction joblib.dump a été appelée pour enregistrer les fichiers encoder et scaler
    assert mock_joblib_dump.call_count == 3, "Les fichiers encoder, scaler et indices devraient être enregistrés"
    
    # Vérifier le retour de la fonction
    assert result == {"status": "training completed", "model_name": "test_model"}

# Test lorsque le modèle est introuvable
@patch('app.usecases.simple_nn.train_model_simple_nn.get_model', return_value=None)
def test_train_model_simple_nn_model_not_found(mock_get_model):
    training_data = [
        SimpleNNTrainingData(type=1, surface=75, pieces=3, floor=2, parking=1, balcon=0, ascenseur=1, orientation=1, transports=1, neighborhood=8, price=350000)
    ]

    # Vérifier qu'une exception est levée si le modèle est introuvable
    with pytest.raises(Exception) as exc_info:
        train_model_simple_nn("unknown_model", training_data)
    
    # Vérifier que l'exception est bien une HTTPException avec le statut 404
    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Model not found"

# Test lorsque les données d'entraînement sont manquantes
@patch('app.usecases.simple_nn.train_model_simple_nn.get_model', return_value={"nn_model": None})
def test_train_model_simple_nn_no_training_data(mock_get_model):
    # Vérifier qu'une exception est levée si les données d'entraînement sont vides
    with pytest.raises(Exception) as exc_info:
        train_model_simple_nn("test_model", [])

    # Vérifier que l'exception est bien une HTTPException avec le statut 400
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "No training data provided or training data is empty"
