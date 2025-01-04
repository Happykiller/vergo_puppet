# app\usecases\simple\usecase_mesure_simple_test.py
import pytest
from unittest.mock import patch, MagicMock

from app.apis.models.simple_nn_training_model_data import SimpleNNTrainingModelData
from app.usecases.simple.usecase_mesure_simple import MesureSimpleUsecaseDto, mesure_simple_nn


# Test that mesure_simple_nn returns valid results with proper test data
@patch("app.usecases.simple.usecase_mesure_simple.predict")
@patch("app.usecases.simple.usecase_mesure_simple.process_input_data")
def test_mesure_simple_nn_success(mock_process_input_data, mock_predict, patch_inversify):
    """Test successful execution of mesure_simple_nn with valid test data."""
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_model.return_value = MagicMock(
        nn_model=MagicMock(),
        encoder=MagicMock(),
        scaler=MagicMock(),
        indices={"categorical_indices": [0, 4, 5], "numerical_indices": [1, 2, 3]},
        targets_mean=0.5,
        targets_std=0.2,
    )

    # Mock processed input data and model prediction
    mock_process_input_data.return_value = [[0.5, 1.2, 0.8]]
    mock_predict.return_value = 350000

    # Prepare test data
    test_data = [
        SimpleNNTrainingModelData(
            type=1, surface=75, pieces=3, floor=2, parking=1, balcon=0, ascenseur=1,
            orientation=1, transports=1, neighborhood=8, price=360000
        )
    ]

    # Run mesure_simple_nn
    result = mesure_simple_nn(MesureSimpleUsecaseDto(name="test_model", test_data=test_data, inversify=mock_inversify))

    # Verify structure of the result
    assert "results" in result
    assert "metrics" in result

    # Verify individual test results
    assert len(result["results"]) == 1
    test_result = result["results"][0]
    assert test_result["expected_price"] == 360000
    assert test_result["predicted_price"] == 350000
    assert test_result["is_correct"] is True

    # Verify global metrics
    metrics = result["metrics"]
    assert metrics["total_tests"] == 1
    assert metrics["correct_predictions"] == 1
    assert metrics["mean_absolute_error"] == pytest.approx(10000)
    assert metrics["mean_absolute_percentage_error"] == pytest.approx(2.78, rel=1e-2)


# Test handling when the model is not trained
def test_mesure_simple_nn_model_not_trained(patch_inversify):
    """Test the case where the model is not trained yet."""
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_model.return_value = MagicMock(nn_model=None)

    test_data = [
        SimpleNNTrainingModelData(
            type=1, surface=75, pieces=3, floor=2, parking=1, balcon=0, ascenseur=1,
            orientation=1, transports=1, neighborhood=8, price=360000
        )
    ]

    with pytest.raises(Exception, match="Model not trained yet"):
        mesure_simple_nn(MesureSimpleUsecaseDto(name="test_model", test_data=test_data, inversify=mock_inversify))


# Test handling if encoder, scaler, or indices are missing
def test_mesure_simple_nn_missing_files(patch_inversify):
    """Test the case where encoder, scaler, or indices are missing."""
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_model.return_value = MagicMock(
        nn_model=MagicMock(),
        encoder=None,  # Missing encoder
        scaler=None,   # Missing scaler
        indices=None,  # Missing indices
    )

    test_data = [
        SimpleNNTrainingModelData(
            type=1, surface=75, pieces=3, floor=2, parking=1, balcon=0, ascenseur=1,
            orientation=1, transports=1, neighborhood=8, price=360000
        )
    ]

    with pytest.raises(Exception, match="Missing encoder, scaler, or indices in the model"):
        mesure_simple_nn(MesureSimpleUsecaseDto(name="test_model", test_data=test_data, inversify=mock_inversify))


# Test handling when target normalization parameters are missing
def test_mesure_simple_nn_missing_normalization_parameters(patch_inversify):
    """Test the case where normalization parameters are missing."""
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_model.return_value = MagicMock(
        nn_model=MagicMock(),
        encoder=MagicMock(),
        scaler=MagicMock(),
        indices={"categorical_indices": [0, 4, 5], "numerical_indices": [1, 2, 3]},
        targets_mean=None,  # Missing targets_mean
        targets_std=None,   # Missing targets_std
    )

    test_data = [
        SimpleNNTrainingModelData(
            type=1, surface=75, pieces=3, floor=2, parking=1, balcon=0, ascenseur=1,
            orientation=1, transports=1, neighborhood=8, price=360000
        )
    ]

    with pytest.raises(Exception, match="Missing normalization parameters in the model"):
        mesure_simple_nn(MesureSimpleUsecaseDto(name="test_model", test_data=test_data, inversify=mock_inversify))
