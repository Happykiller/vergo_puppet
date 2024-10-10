# test_neural_network_siamese.py

import pytest
import torch
from app.machine_learning.neural_network_siamese import (
    SiameseLSTM,
    train_siamese_model_nn,
    evaluate_similarity
)

# Test 1: Vérifier la structure du modèle Siamese LSTM
def test_siamese_lstm_structure():
    vocab_size = 100
    embedding_dim = 128
    hidden_dim = 256
    model = SiameseLSTM(vocab_size, embedding_dim, hidden_dim)
    
    # Vérifier que le modèle a bien les couches définies
    assert isinstance(model.embedding, torch.nn.Embedding), "La couche d'embedding n'est pas correctement définie"
    assert model.embedding.num_embeddings == vocab_size, "La taille du vocabulaire de l'embedding est incorrecte"
    assert model.embedding.embedding_dim == embedding_dim, "La dimension de l'embedding est incorrecte"
    
    assert isinstance(model.lstm, torch.nn.LSTM), "La couche LSTM n'est pas correctement définie"
    assert model.lstm.input_size == embedding_dim, "La couche LSTM ne reçoit pas la bonne taille d'entrée"
    assert model.lstm.hidden_size == hidden_dim, "La couche LSTM ne produit pas la bonne taille de sortie"

# Test 2: Vérifier l'entraînement du modèle Siamese LSTM
def test_train_siamese_model_nn():
    training_data = [
        (["dog", "cat", "bird"], ["dog", "cat", "bird"], 1.0),
        (["frog", "lion", "tiger"], ["frog", "lion", "tiger"], 1.0),
        (["dog", "cat", "bird"], ["frog", "lion", "tiger"], 0.5),
        (["elephant", "giraffe", "zebra"], ["elephant", "giraffe", "zebra"], 1.0),
        (["dog", "cat", "bird"], ["elephant", "giraffe", "zebra"], 0.4),
        (["frog", "lion", "tiger"], ["elephant", "giraffe", "zebra"], 0.3)
    ]

    glossary = ["<PAD>", "dog", "cat", "bird", "frog", "lion", "tiger", "elephant", "giraffe", "zebra"]
    word2idx = {word: idx for idx, word in enumerate(glossary)}
    vocab_size = len(glossary)
    
    # Entraîner le modèle
    model, losses = train_siamese_model_nn(training_data, word2idx, vocab_size, num_epochs=5)
    
    # Vérifier que le modèle a bien été créé
    assert model is not None, "Le modèle Siamese LSTM n'a pas été correctement entraîné."
    # Vérifier que la perte diminue
    assert losses[0] > losses[-1], "La perte n'a pas diminué pendant l'entraînement."

# Test 3: Vérifier la fonction d'évaluation de similarité
def test_evaluate_similarity():
    training_data = [
        (["dog", "cat", "bird"], ["dog", "cat", "bird"], 1.0),
        (["frog", "lion", "tiger"], ["frog", "lion", "tiger"], 1.0),
        (["dog", "cat", "bird"], ["frog", "lion", "tiger"], 0.5),
        (["elephant", "giraffe", "zebra"], ["elephant", "giraffe", "zebra"], 1.0),
        (["dog", "cat", "bird"], ["elephant", "giraffe", "zebra"], 0.4),
        (["frog", "lion", "tiger"], ["elephant", "giraffe", "zebra"], 0.3)
    ]

    training_glossary = ["<PAD>", "dog", "cat", "bird", "frog", "lion", "tiger", "elephant", "giraffe", "zebra"]
    training_word2idx = {word: idx for idx, word in enumerate(training_glossary)}
    vocab_size = len(training_glossary)
    
    # Entraîner le modèle
    model, _ = train_siamese_model_nn(training_data, training_word2idx, vocab_size, num_epochs=5)
    
    # Évaluer la similarité entre deux séquences identiques
    seq1 = ["man", "sit", "up"]
    seq2 = ["man", "sit", "up"]
    glossary = ["<PAD>", "man", "woman", "exercise", "sit", "up"]
    word2idx = {word: idx for idx, word in enumerate(glossary)}
    similarity = evaluate_similarity(model, seq1, seq2, word2idx)
    
    assert similarity > 0.9, "La similarité entre deux séquences identiques devrait être élevée."
    
    # Évaluer la similarité entre deux séquences différentes
    seq4 = ["woman", "exercise"]
    similarity_diff = evaluate_similarity(model, seq1, seq4, word2idx)
    
    assert similarity > similarity_diff, "La similarité devrait être plus faible pour des séquences différentes."

# Test 4: Vérifier que la perte diminue avec des labels continus
def test_loss_decreases_with_continuous_labels():
    training_data = [
        (["dog", "cat", "bird"], ["dog", "cat", "bird"], 1.0),
        (["dog", "cat", "bird"], ["frog", "cat", "bird"], 0.9),
        (["dog", "cat", "bird"], ["frog", "lion"], 0.5)
    ]
    glossary = ["<PAD>", "dog", "frog", "cat", "bird", "lion"]
    word2idx = {word: idx for idx, word in enumerate(glossary)}
    vocab_size = len(glossary)
    
    # Entraîner le modèle
    model, losses = train_siamese_model_nn(training_data, word2idx, vocab_size, num_epochs=10)
    
    # Vérifier que la perte diminue
    assert losses[0] > losses[-1], "La perte n'a pas diminué pendant l'entraînement avec labels continus."

# Test 5: Vérifier la stabilité de la recherche
def test_search_stability():
    training_data = [
        (["dog", "cat", "bird"], ["dog", "cat", "bird"], 1.0),
        (["frog", "lion", "tiger"], ["frog", "lion", "tiger"], 1.0),
        (["dog", "cat", "bird"], ["frog", "lion", "tiger"], 0.5),
        (["elephant", "giraffe", "zebra"], ["elephant", "giraffe", "zebra"], 1.0),
        (["dog", "cat", "bird"], ["elephant", "giraffe", "zebra"], 0.4),
        (["frog", "lion", "tiger"], ["elephant", "giraffe", "zebra"], 0.3)
    ]

    training_glossary = ["<PAD>", "dog", "cat", "bird", "frog", "lion", "tiger", "elephant", "giraffe", "zebra"]
    training_word2idx = {word: idx for idx, word in enumerate(training_glossary)}
    vocab_size = len(training_glossary)
    
    # Entraîner le modèle
    model, _ = train_siamese_model_nn(training_data, training_word2idx, vocab_size, num_epochs=5)
    
    # Créer une recherche avec un vecteur
    search_vector = ["man", "lifting", "weights"]
    indexed_dictionary = [
        ["man", "lifting", "weights"],
        ["woman", "lifting", "weights"],
        ["man", "sit", "up"]
    ]
    glossary = ["<PAD>", "man", "woman", "lifting", "weights"]
    word2idx = {word: idx for idx, word in enumerate(glossary)}
    
    # Effectuer la recherche plusieurs fois pour vérifier la stabilité
    for _ in range(5):
        similarities = []
        for vector in indexed_dictionary:
            similarity = evaluate_similarity(model, search_vector, vector, word2idx)
            similarities.append((vector, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        best_match = similarities[0][0]
        # Vérifier que la meilleure correspondance est correcte à chaque itération
        assert best_match == ["man", "lifting", "weights"], "Le résultat de la recherche n'est pas stable."

def test_evaluate_similarity_seq_varia():
    training_data = [
        (["dog", "cat", "bird"], ["dog", "cat", "bird"], 1.0),
        (["dog", "cat"], ["dog", "cat", "bird"], 0.6),
        (["dog", "cat", "tiger"], ["dog", "cat"], 0.6),
        (["dog", "cat"], ["dog", "bird", "cat"], 0.5),
    ]

    training_glossary = ["<PAD>", "dog", "cat", "bird"]
    training_word2idx = {word: idx for idx, word in enumerate(training_glossary)}
    vocab_size = len(training_glossary)
    
    # Entraîner le modèle
    model, _ = train_siamese_model_nn(training_data, training_word2idx, vocab_size, num_epochs=5)
    
    # Évaluer la similarité entre deux séquences identiques
    seq1 = ["man", "sit"]
    seq2 = ["man", "sit", "up"]
    glossary = ["<PAD>", "man", "sit", "up"]
    word2idx = {word: idx for idx, word in enumerate(glossary)}
    similarity = evaluate_similarity(model, seq1, seq2, word2idx)
    
    assert similarity > 0.5, "La similarité entre deux séquences identiques devrait être élevée."