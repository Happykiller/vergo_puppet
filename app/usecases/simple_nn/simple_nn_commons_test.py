import pytest
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from app.apis.models.simple_nn_training_data import SimpleNNTrainingData
from app.usecases.simple_nn.simple_nn_commons import process_input_data, transform_data

# Test de la fonction transform_data
def test_transform_data():
    # Données d'exemple pour les tests, ajustées avec des entiers comme attendu par SimpleNNTrainingData
    training_data = [
        SimpleNNTrainingData(type=1, surface=75, pieces=3, floor=2, parking=1, balcon=0, ascenseur=1, orientation=1, transports=1, neighborhood=8, price=350000),
        SimpleNNTrainingData(type=2, surface=150, pieces=5, floor=0, parking=2, balcon=1, ascenseur=0, orientation=2, transports=0, neighborhood=5, price=550000)
    ]
    
    # Exécution de la transformation
    features_processed, targets_standardized, encoder, scaler, targets_mean, targets_std, categorical_indices, numerical_indices = transform_data(training_data)

    # Vérification des types de retour
    assert isinstance(features_processed, np.ndarray), "Les features doivent être un tableau numpy"
    assert isinstance(targets_standardized, np.ndarray), "Les targets doivent être un tableau numpy"
    assert isinstance(encoder, OneHotEncoder), "Le retour doit inclure un OneHotEncoder"
    assert isinstance(scaler, StandardScaler), "Le retour doit inclure un StandardScaler"
    
    # Vérification de la taille des features
    assert features_processed.shape[0] == 2, "Il doit y avoir deux entrées après la transformation"
    assert features_processed.shape[1] > 0, "Les features transformées doivent avoir des colonnes"

    # Vérification de la standardisation des targets
    assert np.allclose(targets_standardized.mean(), 0), "Les targets standardisées doivent avoir une moyenne de 0"
    assert np.allclose(targets_standardized.std(), 1), "Les targets standardisées doivent avoir un écart-type de 1"

# Test de la fonction process_input_data
def test_process_input_data():
    # Simuler les entrées d'une nouvelle donnée à traiter avec des entiers
    input_data = [1, 80, 4, 1, 1, 0, 1, 2, 1, 8]

    # Simulation d'encodeur et de scaler pré-entrainés
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    scaler = StandardScaler()
    
    # Catégories et données d'entraînement fictives
    training_data = [
        [1, 1, 0, 1, 1, 2, 8],
        [2, 2, 1, 0, 2, 1, 5]
    ]
    numerical_data = [
        [75, 3, 2],
        [150, 5, 0]
    ]
    
    # Entraîner l'encodeur et le scaler sur des données simulées
    encoder.fit(training_data)
    scaler.fit(numerical_data)

    # Indices des variables catégorielles et numériques
    categorical_indices = [0, 4, 5, 6, 7, 8, 9]
    numerical_indices = [1, 2, 3]

    # Appel de la fonction process_input_data
    input_processed = process_input_data(input_data, encoder, scaler, categorical_indices, numerical_indices)

    # Vérification du retour
    assert isinstance(input_processed, np.ndarray), "Les données traitées doivent être un tableau numpy"
    assert input_processed.shape[0] == 1, "Les données traitées doivent avoir une seule ligne"
    assert input_processed.shape[1] > 0, "Les données traitées doivent contenir des colonnes après transformation"

    # Vérification de la validité des transformations
    assert np.allclose(input_processed[:, :3], scaler.transform([[80, 4, 1]])), "Les variables numériques doivent être correctement standardisées"
    assert input_processed[:, 3:].shape[1] == encoder.transform([[1, 1, 0, 1, 2, 1, 8]]).shape[1], "Le One-Hot encoding doit produire le bon nombre de colonnes"
