import pytest
import torch
from app.machine_learning.neural_network_lstm import LSTMNN, train_lstm_model_nn, search_with_similarity
from app.usecases.indices_to_tokens import indices_to_tokens

# Test 1: Vérifier la structure du modèle LSTM
def test_lstm_neural_network_structure():
    model = LSTMNN(input_size=3, hidden_size=128, output_size=3)
    
    # Vérifier que le modèle a bien les couches définies
    assert isinstance(model.lstm, torch.nn.LSTM), "La couche LSTM n'est pas correctement définie"
    assert model.lstm.input_size == 3, "La couche LSTM ne reçoit pas la bonne taille d'entrée"
    assert model.lstm.hidden_size == 128, "La couche LSTM ne produit pas la bonne taille de sortie"

    assert isinstance(model.fc1, torch.nn.Linear), "La première couche fully connected n'est pas une couche Linéaire"
    assert model.fc1.in_features == 128, "La couche fc1 ne reçoit pas la bonne taille d'entrée"
    assert model.fc1.out_features == 128, "La couche fc1 ne produit pas la bonne taille de sortie"
    
    assert isinstance(model.fc2, torch.nn.Linear), "La deuxième couche fully connected n'est pas une couche Linéaire"
    assert model.fc2.in_features == 128, "La couche fc2 ne reçoit pas la bonne taille d'entrée"
    assert model.fc2.out_features == 3, "La couche fc2 ne produit pas la bonne taille de sortie"

# Test 2: Vérifier l'entraînement du modèle LSTM
def test_train_lstm_model_nn():
    train_data = [
        ([10, 20, 30], [10, 20, 30]),
        ([40, 50, 60], [40, 50, 60]),
        ([70, 80, 90], [70, 80, 90])
    ]
    vector_size = 3

    # Entraîner le modèle
    nn_model, _ = train_lstm_model_nn(train_data, vector_size=vector_size, epochs=10, learning_rate=0.01)

    # Vérifier que le modèle a bien été créé
    assert nn_model is not None, "Le modèle LSTM n'a pas été correctement entraîné."

# Test 3: Vérifier la fonction de recherche avec similarité cosinus
def test_search_with_similarity():
    train_data = [
        ([3.14, 2.71, 1.41], [3.14, 2.71, 1.41]),
        ([5.67, 8.32, 9.21], [5.67, 8.32, 9.21]),
        ([7.54, 3.33, 6.78], [7.54, 3.33, 6.78])
    ]
    vector_size = 3

    # Entraîner le modèle
    nn_model, losses = train_lstm_model_nn(train_data, vector_size=vector_size, epochs=2000, learning_rate=0.01)

    # Afficher la perte pour voir si elle converge
    print(f"Perte finale après 2000 epochs : {losses[-1]}")

    # Créer une recherche avec un vecteur
    search_vector = [3.14, 2.71, 1.41]
    indexed_dictionary = [[3.14, 2.71, 1.41], [5.67, 8.32, 9.21], [7.54, 3.33, 6.78]]

    # Effectuer la recherche plusieurs fois pour vérifier la stabilité
    for _ in range(10):
        result = search_with_similarity(nn_model, search_vector, indexed_dictionary)
        # Vérifier que la meilleure correspondance est correcte à chaque itération
        assert result['best_match'] == [3.14, 2.71, 1.41], "Le résultat de la recherche n'est pas stable."

# Test 4: Vérifier que la perte diminue avec Cosine Similarity
def test_loss_decreases_during_training_with_cosine_similarity():
    train_data = [
        ([10, 20, 30], [10, 20, 30]),
        ([40, 50, 60], [40, 50, 60]),
        ([70, 80, 90], [70, 80, 90])
    ]
    vector_size = 3

    # Entraîner le modèle sur plusieurs époques
    nn_model, losses = train_lstm_model_nn(train_data, vector_size=vector_size, epochs=100, learning_rate=0.01)

    # Vérifier que la perte diminue au fil du temps
    assert losses[0] > losses[-1], "La perte n'a pas diminué avec Cosine Similarity."

