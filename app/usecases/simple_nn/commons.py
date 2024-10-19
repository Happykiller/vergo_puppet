from app.apis.models.simple_nn_training_data import SimpleNNTrainingData
from fastapi import HTTPException  # type: ignore
from typing import List, Tuple
from app.services.logger import logger
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

def process_input_data(input_data, encoder, scaler, categorical_indices, numerical_indices):
    """
    Transforme les données d'entrée en appliquant l'encodage One-Hot et la standardisation.
    :param input_data: Liste des valeurs d'entrée
    :param encoder: Objet OneHotEncoder entraîné
    :param scaler: Objet StandardScaler entraîné
    :param categorical_indices: Indices des variables catégorielles
    :param numerical_indices: Indices des variables numériques
    :return: input_processed, les données transformées prêtes pour le modèle
    """
    input_array = np.array(input_data)
    
    # Séparation des variables
    features_categorical = input_array[categorical_indices].reshape(1, -1)
    features_numerical = input_array[numerical_indices].astype(float).reshape(1, -1)
    
    # Encodage One-Hot des variables catégorielles
    features_categorical_encoded = encoder.transform(features_categorical)
    
    # Standardisation des variables numériques
    features_numerical_scaled = scaler.transform(features_numerical)
    
    # Concaténation des features
    input_processed = np.hstack([features_numerical_scaled, features_categorical_encoded])
    
    return input_processed

def transform_data(training_data: List[SimpleNNTrainingData]):
    """
    Transforme les données d'entraînement en features et targets, avec encodage One-Hot pour les variables catégorielles.
    """
    # Séparer les features et les targets
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
    
    # Indices des variables catégorielles et numériques
    categorical_indices = [0, 4, 5, 6, 7, 8, 9]  # Variables catégorielles
    numerical_indices = [1, 2, 3]  # Variables numériques
    
    # Séparation des variables
    features_categorical = features[:, categorical_indices]
    features_numerical = features[:, numerical_indices].astype(float)
    
    # Encodage One-Hot des variables catégorielles
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    features_categorical_encoded = encoder.fit_transform(features_categorical)
    
    # Standardisation des variables numériques
    scaler = StandardScaler()
    features_numerical_scaled = scaler.fit_transform(features_numerical)
    
    # Concaténation des features
    features_processed = np.hstack([features_numerical_scaled, features_categorical_encoded])
    
    # Standardisation des targets
    targets_mean = targets.mean()
    targets_std = targets.std()
    targets_standardized = (targets - targets_mean) / targets_std
    
    return features_processed, targets_standardized, encoder, scaler, targets_mean, targets_std, categorical_indices, numerical_indices

