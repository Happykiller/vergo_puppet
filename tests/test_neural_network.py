import pytest
import torch
from app.machine_learning.neural_network_simple import SimpleNN, train_model_nn, predict

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
    train_data = [
        ([1, 2, 3], [1, 2, 3]),
        ([4, 5, 6], [4, 5, 6]),
        ([7, 8, 9], [7, 8, 9])
    ]
    vector_size = 3

    # Entraîner le modèle
    nn_model, _ = train_model_nn(train_data, vector_size=vector_size, epochs=10, learning_rate=0.01)

    # Vérifier que le modèle a bien été créé
    assert nn_model is not None, "Le modèle n'a pas été correctement entraîné."

    # Vérifier que les poids ne sont pas les mêmes qu'au début (mise à jour des poids)
    initial_weights = torch.Tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    assert not torch.equal(nn_model.fc1.weight.data, initial_weights), "Les poids n'ont pas été mis à jour après l'entraînement."

# Test 3: Vérifier la prédiction du modèle
def test_predict_with_trained_model():
    train_data = [
        ([1, 2, 3], [1, 2, 3]),
        ([4, 5, 6], [4, 5, 6]),
        ([7, 8, 9], [7, 8, 9])
    ]
    vector_size = 3

    # Entraîner le modèle
    nn_model, _ = train_model_nn(train_data, vector_size=vector_size, epochs=2000, learning_rate=0.01)

    # Faire une prédiction avec un vecteur d'entrée
    input_vector = [1, 2, 3]
    predicted_vector = predict(nn_model, input_vector)

    # Dénormaliser le vecteur prédit pour revenir à l'échelle originale
    min_val = min(input_vector)
    max_val = max(input_vector)
    denormalized_predicted_vector = predicted_vector * (max_val - min_val) + min_val

    # Vérifier que la prédiction retourne un vecteur de la bonne taille
    assert predicted_vector.shape[0] == vector_size, "La taille du vecteur prédit est incorrecte."

    # Vérifier que la prédiction est proche du vecteur d'entrée (car le modèle est entraîné sur des exemples similaires)
    assert torch.allclose(denormalized_predicted_vector, torch.Tensor(input_vector), rtol=1, atol=1), "La prédiction n'est pas suffisamment proche du vecteur d'entrée."

# Test 4: Vérifier que la perte diminue pendant l'entraînement
def test_loss_decreases_during_training():
    train_data = [
        ([1, 2, 3], [1, 2, 3]),
        ([4, 5, 6], [4, 5, 6]),
        ([7, 8, 9], [7, 8, 9])
    ]
    vector_size = 3

    # Entraîner le modèle sur plusieurs époques et vérifier la perte
    nn_model, losses = train_model_nn(train_data, vector_size=vector_size, epochs=100, learning_rate=0.01)

    # Vérifier que la perte diminue au fil du temps
    assert losses[0] > losses[-1], "La perte n'a pas diminué pendant l'entraînement."
