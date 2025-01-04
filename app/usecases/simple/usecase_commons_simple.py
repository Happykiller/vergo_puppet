# app\usecases\simple\usecase_commons_simple.py
import traceback
import numpy as np
from typing import List
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.services.logger import logger
from app.apis.models.simple_nn_training_model_data import SimpleNNTrainingModelData

def process_input_data(input_data, encoder, scaler, categorical_indices, numerical_indices):
    """
    Transforms input data by applying One-Hot encoding and standardization.
    :param input_data: List of input values
    :param encoder: Trained OneHotEncoder object
    :param scaler: Trained StandardScaler object
    :param categorical_indices: Indices of categorical variables
    :param numerical_indices: Indices of numerical variables
    :return: input_processed, the transformed data ready for the model
    """
    try:
        input_array = np.array(input_data)
        
        # Separate variables
        features_categorical = input_array[categorical_indices].reshape(1, -1)
        features_numerical = input_array[numerical_indices].astype(float).reshape(1, -1)
        
        # One-Hot encode categorical variables
        features_categorical_encoded = encoder.transform(features_categorical)
        
        # Standardize numerical variables
        features_numerical_scaled = scaler.transform(features_numerical)
        
        # Concatenate features
        input_processed = np.hstack([features_numerical_scaled, features_categorical_encoded])
        
        return input_processed
    except Exception as e:
        logger.error(f"Error message:{str(e)}\nStack trace:\n{traceback.format_exc()}")
        raise Exception(f"[#process_input_data]{str(e)}")

def transform_data(training_data: List[SimpleNNTrainingModelData]):
    """
    Transforms training data into features and targets, with One-Hot encoding for categorical variables.
    """
    try:
        # Separate features and targets
        features = []
        targets = []
        
        for data in training_data:
            features.append([
                data.type,
                data.surface,
                data.pieces,
                data.floor,
                data.parking,
                data.balcon,
                data.ascenseur,
                data.orientation,
                data.transports,
                data.neighborhood
            ])
            targets.append(data.price)
        
        features = np.array(features)
        targets = np.array(targets)
        
        # Indices of categorical and numerical variables
        categorical_indices = [0, 4, 5, 6, 7, 8, 9]  # Categorical variables
        numerical_indices = [1, 2, 3]  # Numerical variables
        
        # Separate variable types
        features_categorical = features[:, categorical_indices]
        features_numerical = features[:, numerical_indices].astype(float)
        
        # One-Hot encode categorical variables
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        features_categorical_encoded = encoder.fit_transform(features_categorical)
        
        # Standardize numerical variables
        scaler = StandardScaler()
        features_numerical_scaled = scaler.fit_transform(features_numerical)
        
        # Concatenate features
        features_processed = np.hstack([features_numerical_scaled, features_categorical_encoded])
        
        # Standardize targets
        targets_mean = targets.mean()
        targets_std = targets.std()
        targets_standardized = (targets - targets_mean) / targets_std
        
        return features_processed, targets_standardized, encoder, scaler, targets_mean, targets_std, categorical_indices, numerical_indices
    
    except Exception as e:
        logger.error(f"Error message:{str(e)}\nStack trace:\n{traceback.format_exc()}")
        raise Exception(f"[#transform_data]{str(e)}")
