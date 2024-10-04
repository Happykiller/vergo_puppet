import pytest
import torch
from app.models.neural_network_lstm import LSTMNN, train_model_nn, search_with_similarity

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
        ([1, 2, 3], [1, 2, 3]),
        ([4, 5, 6], [4, 5, 6]),
        ([7, 8, 9], [7, 8, 9])
    ]
    vector_size = 3

    # Entraîner le modèle
    nn_model, _ = train_model_nn(train_data, vector_size=vector_size, epochs=10, learning_rate=0.01)

    # Vérifier que le modèle a bien été créé
    assert nn_model is not None, "Le modèle LSTM n'a pas été correctement entraîné."

# Test 3: Vérifier la fonction de recherche avec similarité cosinus
def test_search_with_similarity():
    train_data = [
        ([1, 2, 3], [1, 2, 3]),
        ([4, 5, 6], [4, 5, 6]),
        ([7, 8, 9], [7, 8, 9])
    ]
    vector_size = 3
    glossary = ['a', 'b', 'c']  # Exemples fictifs

    # Entraîner le modèle
    nn_model, _ = train_model_nn(train_data, vector_size=vector_size, epochs=100, learning_rate=0.01)

    # Créer une recherche avec un vecteur
    search_vector = [1, 2, 3]
    indexed_dictionary = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    # Effectuer une recherche
    result = search_with_similarity(nn_model, search_vector, indexed_dictionary, glossary)

    # Vérifier que la meilleure correspondance est correcte
    assert result['best_match'] == glossary[0], "Le résultat de la recherche n'est pas correct."
    assert result['similarity_score'] > 0.9, "Le score de similarité est trop faible."

# Test 4: Vérifier que la perte diminue avec Cosine Similarity
def test_loss_decreases_during_training_with_cosine_similarity():
    train_data = [
        ([1, 2, 3], [1, 2, 3]),
        ([4, 5, 6], [4, 5, 6]),
        ([7, 8, 9], [7, 8, 9])
    ]
    vector_size = 3

    # Entraîner le modèle sur plusieurs époques
    nn_model, losses = train_model_nn(train_data, vector_size=vector_size, epochs=100, learning_rate=0.01)

    # Vérifier que la perte diminue au fil du temps
    assert losses[0] > losses[-1], "La perte n'a pas diminué avec Cosine Similarity."

