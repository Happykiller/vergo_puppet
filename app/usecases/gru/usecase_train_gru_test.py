# app\usecases\gru\usecase_train_gru_test.py
import pytest
from unittest.mock import patch, MagicMock

from app.apis.models.gru_training_model_data import GRUTrainingModelData
from app.usecases.gru.usecase_train_gru import TrainGRUUsecaseDto, train_model_gru

# Test to verify successful training of the GRU model
@patch('app.usecases.gru.usecase_train_gru.train_gru')
@patch('app.usecases.gru.usecase_train_gru.prepare_sequences')
@patch('app.usecases.gru.usecase_train_gru.build_category_mapping')
@patch('app.usecases.gru.usecase_train_gru.build_vocab')
def test_train_model_gru_success(mock_build_vocab, mock_build_category_mapping, mock_prepare_sequences, mock_train_gru, patch_inversify):
    """Test successful training of the GRU model."""
    # Mock dependencies
    mock_inversify, mock_bdd = patch_inversify
    mock_model_data = MagicMock()
    mock_model_data.name = "test_gru_model"
    mock_model_data.neural_network_type = "GRUClassifier"
    mock_model_data.nn_model = None
    mock_bdd.get_model.return_value = mock_model_data

    # Mock vocabulary and category mappings
    mock_build_vocab.return_value = ({"hello": 1, "<PAD>": 0}, {1: "hello", 0: "<PAD>"})
    mock_build_category_mapping.return_value = ({"cat1": 0, "cat2": 1}, {0: "cat1", 1: "cat2"})

    # Mock sequence and label preparation
    mock_prepare_sequences.return_value = (MagicMock(), MagicMock())  # sequences, labels

    # Mock GRU model training
    mock_train_gru.return_value = (MagicMock(), {"final_loss": 0.1, "epochs_run": 5})

    # Test training data
    training_data = [
        GRUTrainingModelData(category="cat1", tokens=["hello", "world"]),
        GRUTrainingModelData(category="cat2", tokens=["another", "sentence"])
    ]

    # Call the function
    result = train_model_gru(TrainGRUUsecaseDto(name="test_gru_model", training_data=training_data, inversify=mock_inversify))

    # Verify model update
    mock_bdd.update_model.assert_called()
    updated_model = mock_bdd.update_model.call_args[0][0]
    assert updated_model.name == "test_gru_model"
    assert updated_model.neural_network_type == "GRUClassifier"

    # Verify return value
    assert result == {
        "status": "Training complete",
        "model_name": "test_gru_model",
        "metrics": {
            "data_training_stats": {'num_documents': 2, 'num_categories': 2, 'dist_categories': [{'cat1': {'count': 1, 'percentage': 50.0}}, {'cat2': {'count': 1, 'percentage': 50.0}}], 'max_seq_length': 2, 'min_seq_length': 2, 'avg_seq_length': 2.0, 'vocab_size': 2},
            "training_stats": {"final_loss": 0.1, "epochs_run": 5},
        }
    }


# Test when the specified model cannot be found
def test_train_model_gru_model_not_found(patch_inversify):
    """Test the case where the specified GRU model is not found."""
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_model.return_value = None

    training_data = [
        GRUTrainingModelData(category="cat1", tokens=["hello", "world"])
    ]

    with pytest.raises(Exception, match="Model not found"):
        train_model_gru(TrainGRUUsecaseDto(name="unknown_model", training_data=training_data, inversify=mock_inversify))


# Test when training data is missing or empty
def test_train_model_gru_no_training_data(patch_inversify):
    """Test the case where no training data is provided."""
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_model.return_value = MagicMock(nn_model=None)

    with pytest.raises(Exception, match="No training data provided or data is empty"):
        train_model_gru(TrainGRUUsecaseDto(name="test_gru_model", training_data=[], inversify=mock_inversify))


# Test when there is an error during vocabulary or sequence preparation
@patch('app.usecases.gru.usecase_train_gru.build_vocab')
def test_train_model_gru_vocab_error(mock_build_vocab, patch_inversify):
    """Test error during vocabulary building."""
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_model.return_value = MagicMock(nn_model=None)
    mock_build_vocab.side_effect = Exception("Error building vocabulary")

    training_data = [
        GRUTrainingModelData(category="cat1", tokens=["hello", "world"])
    ]

    with pytest.raises(Exception, match="Error building vocabulary"):
        train_model_gru(TrainGRUUsecaseDto(name="test_gru_model", training_data=training_data, inversify=mock_inversify))


# Test when there is an error during model training
@patch('app.usecases.gru.usecase_train_gru.train_gru')
def test_train_model_gru_training_error(mock_train_gru, patch_inversify):
    """Test error during GRU model training."""
    mock_inversify, mock_bdd = patch_inversify
    mock_bdd.get_model.return_value = MagicMock(
        nn_model=None,
        name="test_gru_model",
        neural_network_type="GRUClassifier"
    )
    mock_train_gru.side_effect = Exception("Training error")

    training_data = [
        GRUTrainingModelData(category="cat1", tokens=["hello", "world"]),
        GRUTrainingModelData(category="cat2", tokens=["another", "sentence"])
    ]

    with pytest.raises(Exception, match="Training error"):
        train_model_gru(TrainGRUUsecaseDto(name="test_gru_model", training_data=training_data, inversify=mock_inversify))
