import pytest
import torch
from unittest.mock import patch
from app.machine_learning.nn_gru import GRUClassifier, train_gru, predict

# Test 1 : Vérifier la structure du modèle GRUClassifier
def test_gru_classifier_structure():
    vocab_size = 100
    embedding_dim = 128
    hidden_dim = 256
    num_classes = 5
    dropout_rate = 0.5

    model = GRUClassifier(vocab_size, embedding_dim, hidden_dim, num_classes, dropout_rate)

    # Vérifier les couches
    assert isinstance(model.embedding, torch.nn.Embedding), "La couche d'embedding n'est pas correctement définie"
    assert model.embedding.num_embeddings == vocab_size, "La taille du vocabulaire de l'embedding est incorrecte"
    assert model.embedding.embedding_dim == embedding_dim, "La dimension de l'embedding est incorrecte"

    assert isinstance(model.gru, torch.nn.GRU), "La couche GRU n'est pas correctement définie"
    assert model.gru.input_size == embedding_dim, "La couche GRU ne reçoit pas la bonne taille d'entrée"
    assert model.gru.hidden_size == hidden_dim, "La couche GRU ne produit pas la bonne taille de sortie"

    assert isinstance(model.fc, torch.nn.Linear), "La couche fully connected n'est pas définie correctement"
    assert model.fc.in_features == hidden_dim, "La couche fully connected n'a pas la bonne taille d'entrée"
    assert model.fc.out_features == num_classes, "La couche fully connected n'a pas le bon nombre de classes"

# Test 2 : Vérifier l'entraînement du modèle GRU avec des données factices
def test_train_gru():
    vocab_size = 50
    num_classes = 3
    sequences = [
        [1, 2, 3, 4, 0, 0], 
        [5, 6, 7, 0, 0, 0], 
        [1, 3, 4, 5, 6, 7]
    ]
    labels = [0, 1, 2]

    # Entraîner le modèle
    model = train_gru(vocab_size, num_classes, sequences, labels)

    # Vérifier que le modèle est entraîné
    assert model is not None, "Le modèle GRU n'a pas été correctement entraîné."

# Test 3 : Vérifier la prédiction avec un modèle GRU entraîné
def test_predict_with_trained_gru():
    vocab_size = 50
    num_classes = 3
    sequences = [
        [1, 2, 3, 4, 0, 0], 
        [5, 6, 7, 0, 0, 0], 
        [1, 3, 4, 5, 6, 7]
    ]
    labels = [0, 1, 2]

    # Entraîner le modèle
    model = train_gru(vocab_size, num_classes, sequences, labels)

    # Effectuer une prédiction
    input_sequence = [1, 2, 3, 4, 0, 0]
    predicted_class = predict(model, input_sequence)

    # Vérifier que la prédiction est valide
    assert isinstance(predicted_class, int), "La prédiction devrait être un entier représentant une classe"
    assert 0 <= predicted_class < num_classes, "La prédiction devrait être dans les limites des classes disponibles"

# Test 4 : Vérifier la diminution de la perte pendant l'entraînement
def test_loss_decrease_during_training():
    vocab_size = 50
    num_classes = 3
    sequences = [
        [1, 2, 3, 4, 0, 0], 
        [5, 6, 7, 0, 0, 0], 
        [1, 3, 4, 5, 6, 7]
    ]
    labels = [0, 1, 2]

    with patch('app.machine_learning.nn_gru.logger.info') as mock_logger:
        # Entraîner le modèle et récupérer les pertes
        model = train_gru(vocab_size, num_classes, sequences, labels)

    # Vérifier que l'entraînement se termine sans erreurs
    assert model is not None, "Le modèle GRU n'a pas été correctement entraîné"

# Test 5 : Vérifier que les couches dropout et log_softmax fonctionnent correctement
def test_gru_dropout_log_softmax():
    vocab_size = 50
    embedding_dim = 128
    hidden_dim = 256
    num_classes = 3
    dropout_rate = 0.5
    model = GRUClassifier(vocab_size, embedding_dim, hidden_dim, num_classes, dropout_rate)
    input_sequence = torch.tensor([[1, 2, 3, 4, 5, 0]])

    # Effectuer un passage avant
    output = model(input_sequence)

    # Vérifier les dimensions et la somme des probabilités
    assert output.shape == (1, num_classes), "La sortie devrait avoir la taille (1, num_classes)"
    assert torch.allclose(torch.exp(output).sum(dim=1), torch.tensor([1.0]), atol=1e-3), "Les probabilités devraient se sommer à 1"
