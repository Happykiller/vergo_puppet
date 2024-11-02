#app\neural_network\nn_lstm_test.py
import pytest
import numpy as np
import torch
from unittest.mock import patch
from app.neural_network.nn_lstm import train_nn_lstm, predict_nn_lstm, LSTMNN

@pytest.fixture
def synthetic_data():
    """
    Fixture to generate synthetic data for testing.
    Sets random seeds for reproducibility.
    """
    np.random.seed(0)
    torch.manual_seed(0)
    
    # Synthetic training data
    X_train = np.random.rand(100, 10, 5)  # 100 samples, 10 time steps, 5 features
    y_train = np.random.rand(100, 1)      # 100 target values
    
    # Synthetic test data
    X_test = np.random.rand(20, 10, 5)    # 20 test samples with the same time steps and features
    
    return X_train, y_train, X_test

@patch('app.neural_network.nn_lstm.logger')
def test_train_nn_lstm(mock_logger, synthetic_data):
    """
    Tests the LSTM model training function.
    Checks model instance, output shape, and logger calls.
    """
    X_train, y_train, _ = synthetic_data
    
    # Train the model
    model = train_nn_lstm(X_train, y_train, epochs=5, learning_rate=0.01, patience=3)
    
    # Verify that the model is an instance of LSTMNN
    assert isinstance(model, LSTMNN), "The model is not an instance of LSTMNN"
    
    # Verify that the model can produce outputs with the expected shape
    outputs = model(torch.Tensor(X_train[:1]))
    assert outputs.shape == (1, 1), f"Unexpected output shape: {outputs.shape}"
    
    # Check that the logger was called during training
    assert mock_logger.info.called or mock_logger.debug.called, "Logger was not called"

@patch('app.neural_network.nn_lstm.logger')
def test_predict_nn_lstm(mock_logger, synthetic_data):
    """
    Tests the LSTM model prediction function.
    Verifies output shape, data type, and logger usage.
    """
    X_train, y_train, X_test = synthetic_data
    
    # Train the model
    model = train_nn_lstm(X_train, y_train, epochs=5, learning_rate=0.01, patience=3)
    
    # Make predictions
    predictions = predict_nn_lstm(model, X_test)
    
    # Verify the shape of predictions matches the number of test samples
    assert predictions.shape == (20,), f"Unexpected prediction shape: {predictions.shape}"
    
    # Ensure predictions are numeric
    assert np.issubdtype(predictions.dtype, np.number), "Predictions are not numeric"
    
    # Verify that the logger was called during training
    assert mock_logger.info.called or mock_logger.debug.called, "Logger was not called during training"
