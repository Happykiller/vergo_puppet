# app/usecases/lstm/usecase_commons_lstm_test.py
import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from app.usecases.lstm.usecase_commons_lstm import (
    preprocess_input_data, preprocess_data, prepare_sequences, inverse_transform_predictions
)

# Test for preprocess_input_data
def test_preprocess_input_data():
    # Create sample data
    data = {
        'time': pd.to_datetime(['2023-01-01 00:00', '2023-01-01 01:00']),
        'dwpt': [2.5, 2.7],
        'rhum': [80, 85],
        'prcp': [0.0, 0.1],
        'coco': [1, 2]
    }
    df = pd.DataFrame(data)

    # Initialize scaler and encoder with mock data
    scaler = MinMaxScaler().fit(df[['dwpt', 'rhum', 'prcp']])
    coco_encoder = OneHotEncoder(sparse_output=False).fit(df[['coco']])

    # Process data
    df_processed = preprocess_input_data(df, scaler, coco_encoder)

    # Assertions
    assert 'dwpt' in df_processed.columns, "The column 'dwpt' should be present after preprocessing"
    assert 'coco_1' in df_processed.columns, "One-hot encoded coco_1 column should be present"
    assert 'coco_2' in df_processed.columns, "One-hot encoded coco_2 column should be present"
    assert not df_processed.isnull().any().any(), "No NaN values should be in the processed DataFrame"

# Test for preprocess_data
def test_preprocess_data():
    # Create sample data
    data = {
        'time': pd.to_datetime(['2023-01-01 00:00', '2023-01-01 01:00', '2023-01-01 02:00']),
        'temp': [5.0, 6.0, 7.0],
        'dwpt': [2.5, 2.7, 2.9],
        'rhum': [80, 85, 90],
        'coco': [1, 2, 3]
    }
    df = pd.DataFrame(data)

    # Process data
    df_processed, y_temp_scaled, scaler, target_scaler, coco_encoder = preprocess_data(df)

    # Assertions
    assert 'dwpt' in df_processed.columns, "The column 'dwpt' should be present after preprocessing"
    assert 'temp' not in df_processed.columns, "'temp' should not be in the input features"
    assert y_temp_scaled.shape[0] == df.shape[0], "y_temp_scaled should have the same number of rows as input data"
    assert not df_processed.isnull().any().any(), "No NaN values should be in the processed DataFrame"

# Test for prepare_sequences
def test_prepare_sequences():
    # Create sample processed data
    data = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2]])
    df_processed = pd.DataFrame(data, columns=['feature1', 'feature2', 'feature3'])
    y_temp_scaled = np.array([[0.1], [0.2], [0.3], [0.4]])

    # Sequence length
    sequence_length = 2

    # Prepare sequences
    X, y = prepare_sequences(df_processed, y_temp_scaled, sequence_length)

    # Assertions
    assert X.shape[1] == sequence_length, "X should have the correct sequence length"
    assert X.shape[0] == y.shape[0], "X and y should have the same number of samples"
    assert X.shape[2] == df_processed.shape[1], "Each sequence should have the same number of features as the original data"

# Test for inverse_transform_predictions
def test_inverse_transform_predictions():
    # Sample normalized predictions
    predictions_normalized = [0.1, 0.2, 0.3]

    # Target scaler
    y_temp = np.array([[5.0], [6.0], [7.0]])
    target_scaler = MinMaxScaler().fit(y_temp)

    # Inverse transform predictions
    predictions_inverse = inverse_transform_predictions(predictions_normalized, target_scaler)

    # Assertions
    assert len(predictions_inverse) == len(predictions_normalized), "Inverse predictions should have the same length as input"
    assert predictions_inverse[0] > 0, "Inverse transformed values should be positive"

