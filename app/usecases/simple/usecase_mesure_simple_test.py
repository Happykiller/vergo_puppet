# app\usecases\simple\usecase_mesure_simple_test.py
import pytest
from unittest.mock import patch, MagicMock

from app.usecases.simple.usecase_mesure_simple import mesure_simple_nn
from app.apis.models.simple_nn_training_model_data import SimpleNNTrainingModelData

# Test that mesure_simple_nn runs successfully with valid test data
@patch('app.usecases.simple.usecase_mesure_simple.predict')
@patch('app.usecases.simple.usecase_mesure_simple.process_input_data')
@patch('app.usecases.simple.usecase_mesure_simple.joblib.load')
def test_mesure_simple_nn_success(mock_joblib_load, mock_process_input_data, mock_predict, patch_inversify):
    # Mock a complete model setup with neural network, encoder, scaler, and indices
    # patch_inversify est un tuple (mock_inversify_instance, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_model.return_value.model = {
        "nn_model": MagicMock(),
        "encoder_filename": "encoder.pkl",
        "scaler_filename": "scaler.pkl",
        "indices_filename": "indices.pkl",
        "targets_mean": 0.5,
        "targets_std": 0.2
    }

    # Simulate loading of encoder, scaler, and indices
    mock_joblib_load.side_effect = [
        MagicMock(),  # Encoder
        MagicMock(),  # Scaler
        {"categorical_indices": [0, 4, 5], "numerical_indices": [1, 2, 3]}  # Indices info
    ]

    # Mock processed input data and model prediction
    mock_process_input_data.return_value = [[0.5, 1.2, 0.8]]
    mock_predict.return_value = 350000

    # Prepare test data
    test_data = [
        SimpleNNTrainingModelData(
            type=1, surface=75, pieces=3, floor=2, parking=1, balcon=0, ascenseur=1,  # Changed to `balcon` and `ascenseur`
            orientation=1, transports=1, neighborhood=8, price=360000
        )
    ]

    # Run mesure_simple_nn
    mesure_simple_nn("test_model", test_data, mock_inversify)

    # Ensure the prediction function was called correctly
    mock_predict.assert_called_once()

# Test handling when the model is not yet trained
def test_mesure_simple_nn_model_not_trained(patch_inversify):
    # patch_inversify est un tuple (mock_inversify_instance, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_model.return_value = {"nn_model": None}

    # Prepare test data
    test_data = [
        SimpleNNTrainingModelData(
            type=1, surface=75, pieces=3, floor=2, parking=1, balcon=0, ascenseur=1,  # Changed to `balcon` and `ascenseur`
            orientation=1, transports=1, neighborhood=8, price=360000
        )
    ]

    # Ensure an exception is raised if the model is not trained
    with pytest.raises(Exception, match="Model not trained yet"):
        mesure_simple_nn("test_model", test_data, mock_inversify)

# Test handling if encoder, scaler, or indices files are missing
def test_mesure_simple_nn_missing_files(patch_inversify):
    # patch_inversify est un tuple (mock_inversify_instance, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_model.return_value = {
        "nn_model": MagicMock(),
        "encoder_filename": None,
        "scaler_filename": None,
        "indices_filename": None
    }

    # Prepare test data
    test_data = [
        SimpleNNTrainingModelData(
            type=1, surface=75, pieces=3, floor=2, parking=1, balcon=0, ascenseur=1,  # Changed to `balcon` and `ascenseur`
            orientation=1, transports=1, neighborhood=8, price=360000
        )
    ]

    # Ensure an exception is raised if required files are missing
    with pytest.raises(Exception, match="Missing encoder, scaler, or indices in the model"):
        mesure_simple_nn("test_model", test_data, mock_inversify)

# Test handling when target normalization parameters are missing
@patch('app.usecases.simple.usecase_mesure_simple.joblib.load')  # Simulate loading encoder/scaler/indices
def test_mesure_simple_nn_missing_normalization_parameters(mock_joblib_load, patch_inversify):
    # patch_inversify est un tuple (mock_inversify_instance, mock_bdd)
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_model.return_value = {
        "nn_model": MagicMock(),
        "encoder_filename": "encoder.pkl",
        "scaler_filename": "scaler.pkl",
        "indices_filename": "indices.pkl",
        "targets_mean": None,
        "targets_std": None
    }

    # Simulate loading of encoder, scaler, and indices via joblib.load
    mock_joblib_load.side_effect = [
        MagicMock(),  # Encoder
        MagicMock(),  # Scaler
        {"categorical_indices": [0, 4, 5], "numerical_indices": [1, 2, 3]}  # Indices info
    ]

    # Prepare test data
    test_data = [
        SimpleNNTrainingModelData(
            type=1, surface=75, pieces=3, floor=2, parking=1, balcon=0, ascenseur=1,  # Changed to `balcon` and `ascenseur`
            orientation=1, transports=1, neighborhood=8, price=360000
        )
    ]

    # Ensure an exception is raised if normalization parameters are missing
    with pytest.raises(Exception, match="Missing normalization parameters in the model"):
        mesure_simple_nn("test_model", test_data, mock_inversify)
