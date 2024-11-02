#app\usecases\simple\usecase_commons_simple_test.py
import pytest
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from app.apis.models.simple_nn_training_model_data import SimpleNNTrainingModelData
from app.usecases.simple.usecase_commons_simple import process_input_data, transform_data

# Test for the transform_data function
def test_transform_data():
    # Example data for testing, adjusted with integers as expected by SimpleNNTrainingModelData
    training_data = [
        SimpleNNTrainingModelData(type=1, surface=75, pieces=3, floor=2, parking=1, balcon=0, ascenseur=1, orientation=1, transports=1, neighborhood=8, price=350000),
        SimpleNNTrainingModelData(type=2, surface=150, pieces=5, floor=0, parking=2, balcon=1, ascenseur=0, orientation=2, transports=0, neighborhood=5, price=550000)
    ]
    
    # Run transformation
    features_processed, targets_standardized, encoder, scaler, targets_mean, targets_std, categorical_indices, numerical_indices = transform_data(training_data)

    # Verify return types
    assert isinstance(features_processed, np.ndarray), "Features should be a numpy array"
    assert isinstance(targets_standardized, np.ndarray), "Targets should be a numpy array"
    assert isinstance(encoder, OneHotEncoder), "Return should include a OneHotEncoder"
    assert isinstance(scaler, StandardScaler), "Return should include a StandardScaler"
    
    # Verify features shape
    assert features_processed.shape[0] == 2, "There should be two entries after transformation"
    assert features_processed.shape[1] > 0, "Transformed features should have columns"

    # Verify target standardization
    assert np.allclose(targets_standardized.mean(), 0), "Standardized targets should have a mean of 0"
    assert np.allclose(targets_standardized.std(), 1), "Standardized targets should have a standard deviation of 1"

# Test for the process_input_data function
def test_process_input_data():
    # Simulate the input of a new data sample with integers
    input_data = [1, 80, 4, 1, 1, 0, 1, 2, 1, 8]

    # Simulate pre-trained encoder and scaler
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    scaler = StandardScaler()
    
    # Simulated categories and training data
    training_data = [
        [1, 1, 0, 1, 1, 2, 8],
        [2, 2, 1, 0, 2, 1, 5]
    ]
    numerical_data = [
        [75, 3, 2],
        [150, 5, 0]
    ]
    
    # Train the encoder and scaler on simulated data
    encoder.fit(training_data)
    scaler.fit(numerical_data)

    # Indices of categorical and numerical variables
    categorical_indices = [0, 4, 5, 6, 7, 8, 9]
    numerical_indices = [1, 2, 3]

    # Call the process_input_data function
    input_processed = process_input_data(input_data, encoder, scaler, categorical_indices, numerical_indices)

    # Verify the output
    assert isinstance(input_processed, np.ndarray), "Processed data should be a numpy array"
    assert input_processed.shape[0] == 1, "Processed data should have one row"
    assert input_processed.shape[1] > 0, "Processed data should contain columns after transformation"

    # Verify validity of transformations
    assert np.allclose(input_processed[:, :3], scaler.transform([[80, 4, 1]])), "Numerical variables should be properly standardized"
    assert input_processed[:, 3:].shape[1] == encoder.transform([[1, 1, 0, 1, 2, 1, 8]]).shape[1], "One-Hot encoding should produce the correct number of columns"
