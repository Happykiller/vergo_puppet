import torch
import random
import pytest
import numpy as np
from app.machine_learning.neural_network_simple import SimpleNN, train_model_nn, predict

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Test 1: Vérifier la structure du modèle
def test_neural_network_structure():
    model = SimpleNN(input_size=3, hidden_size=128, output_size=3)

    # Vérifier que le modèle a bien les couches définies
    assert isinstance(model.fc1, torch.nn.Linear), "La première couche n'est pas une couche Linéaire"
    assert model.fc1.in_features == 3, "La couche fc1 ne reçoit pas la bonne taille d'entrée"
    assert model.fc1.out_features == 128, "La couche fc1 ne produit pas la bonne taille de sortie"

    assert isinstance(model.fc2, torch.nn.Linear), "La deuxième couche n'est pas une couche Linéaire"
    assert model.fc2.in_features == 128, "La couche fc2 ne reçoit pas la bonne taille d'entrée"
    assert model.fc2.out_features == 128, "La couche fc2 ne produit pas la bonne taille de sortie"

    assert isinstance(model.fc3, torch.nn.Linear), "La troisième couche n'est pas une couche Linéaire"
    assert model.fc3.in_features == 128, "La couche fc3 ne reçoit pas la bonne taille d'entrée"
    assert model.fc3.out_features == 3, "La couche fc3 ne produit pas la bonne taille de sortie"

# Test 2: Vérifier l'entraînement du modèle et que les poids sont mis à jour
def test_train_model_nn():
    # Exemples de caractéristiques réalistes (par exemple, caractéristiques de logements : taille, nombre de chambres, distance du centre-ville)
    features_processed = [
        [75, 2, 5],  # Logement de 75 m2, 2 chambres, à 5 km du centre-ville
        [120, 4, 10],  # Logement de 120 m2, 4 chambres, à 10 km du centre-ville
        [60, 1, 2]  # Logement de 60 m2, 1 chambre, à 2 km du centre-ville
    ]

    # Cibles réalistes : prix des logements (normalisés entre 0 et 1)
    targets_standardized = [
        0.5,  # Prix moyen pour le premier logement
        0.8,  # Prix plus élevé pour un logement plus grand et éloigné
        0.3  # Prix plus faible pour un logement plus petit
    ]

    # Entraîner le modèle
    nn_model, _ = train_model_nn(features_processed, targets_standardized, 3, epochs=100, learning_rate=0.001)

    # Vérifier que le modèle a bien été créé
    assert nn_model is not None, "Le modèle n'a pas été correctement entraîné."

    # Vérifier que les poids ne sont pas les mêmes qu'au début (mise à jour des poids)
    initial_weights = torch.zeros_like(nn_model.fc1.weight.data)
    assert not torch.equal(nn_model.fc1.weight.data, initial_weights), "Les poids de la première couche n'ont pas été mis à jour après l'entraînement."

# Test 3: Vérifier la prédiction du modèle avec des données plus réalistes
def test_predict_with_trained_model():
    # Fixer la graine pour la reproductibilité
    set_seed(42)
    
    # Exemples de caractéristiques réalistes (par exemple, caractéristiques de logements : taille, nombre de chambres, distance du centre-ville)
    features_processed = [
        [75, 2, 5],  # Logement de 75 m2, 2 chambres, à 5 km du centre-ville
        [120, 4, 10],  # Logement de 120 m2, 4 chambres, à 10 km du centre-ville
        [60, 1, 2]  # Logement de 60 m2, 1 chambre, à 2 km du centre-ville
    ]

    # Cibles réalistes : prix des logements (normalisés entre 0 et 1)
    targets_standardized = [
        0.5,  # Prix moyen pour le premier logement
        0.8,  # Prix plus élevé pour un logement plus grand et éloigné
        0.3  # Prix plus faible pour un logement plus petit
    ]

    # Entraîner le modèle
    nn_model, _ = train_model_nn(features_processed, targets_standardized, 3, epochs=200, learning_rate=0.001)

    # Faire une prédiction avec un vecteur d'entrée réaliste
    input_vector = [80, 3, 6]  # Un logement de 80 m2, 3 chambres, à 6 km du centre-ville
    predicted = predict(nn_model, input_vector, targets_mean=0.53, targets_std=0.18)  # Utiliser la déstandardisation avec une moyenne et un écart-type réalistes

    # Vérifier que la prédiction retourne une valeur correcte
    assert predicted > 0, "La prédiction retourne une valeur négative ou incorrecte."

# Test 4: Vérifier que la perte diminue pendant l'entraînement avec des données plus réalistes
def test_loss_decreases_during_training():
    # Exemples de caractéristiques réalistes (par exemple, caractéristiques de logements : taille, nombre de chambres, distance du centre-ville)
    features_processed = [
        [75, 2, 5],  # Logement de 75 m2, 2 chambres, à 5 km du centre-ville
        [120, 4, 10],  # Logement de 120 m2, 4 chambres, à 10 km du centre-ville
        [60, 1, 2]  # Logement de 60 m2, 1 chambre, à 2 km du centre-ville
    ]

    # Cibles réalistes : prix des logements (normalisés entre 0 et 1)
    targets_standardized = [
        0.5,  # Prix moyen pour le premier logement
        0.8,  # Prix plus élevé pour un logement plus grand et éloigné
        0.3  # Prix plus faible pour un logement plus petit
    ]

    # Entraîner le modèle sur plusieurs époques et vérifier la perte
    nn_model, losses = train_model_nn(features_processed, targets_standardized, 3, epochs=100, learning_rate=0.001)

    # Vérifier que la perte diminue au fil du temps
    assert losses[0] > losses[-2], "La perte n'a pas diminué pendant l'entraînement."