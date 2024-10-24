from app.machine_learning.nn_gru import train_gru
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict
from app.services.logger import logger
from fastapi import HTTPException
from app.repositories.memory import get_model, update_model
from app.apis.models.gru_training_data import GRUTrainingData
from app.machine_learning.nn_gru import GRUClassifier
from collections import Counter

def train_model_gru(name: str, training_data: List[GRUTrainingData]):
    """
    Entraîne le modèle GRU avec les données d'entraînement fournies.
    :param name: Nom du modèle.
    :param training_data: Liste des données d'entraînement.
    """
    model = get_model(name)
    
    if model is None or not model:
        raise HTTPException(status_code=404, detail="Modèle non trouvé")
    
    if training_data is None or len(training_data) == 0:
        raise HTTPException(status_code=400, detail="Aucune donnée d'entraînement fournie ou données vides")
    
    logger.info("Type de machine learning utilisé pour l'entraînement : GRU")
    
    # Transformation des données
    logger.info("Transformation des données d'entraînement...")
    word2idx, idx2word = build_vocab(training_data)
    category2idx, idx2category = build_category_mapping(training_data)
    sequences, labels = prepare_sequences(training_data, word2idx, category2idx)
    
    # Paramètres du modèle (hyperparamètres ajustés)
    vocab_size = len(word2idx)
    num_classes = len(category2idx)
    embedding_dim = 128  # Augmenter la dimension des embeddings
    hidden_dim = 256     # Augmenter la dimension des états cachés du GRU
    num_epochs = 20      # Augmenter le nombre d'époques
    batch_size = 16      # Réduire la taille des lots pour des mises à jour plus fréquentes
    learning_rate = 0.0005  # Diminuer le taux d'apprentissage pour une convergence plus stable
    dropout_rate = 0.5   # Taux de Dropout pour la régularisation
    
    # Création du modèle GRU avec Dropout
    model_gru = GRUClassifier(vocab_size, embedding_dim, hidden_dim, num_classes, dropout_rate)
    
    # Entraîner le modèle
    logger.info("Entraînement du modèle...")
    train_gru(model_gru, sequences, labels, num_epochs, batch_size, learning_rate)
    
    # Enregistrer le modèle entraîné et les mappings
    model_data = {
        "neural_network_type": "GRU",
        "nn_model": model_gru.state_dict(),
        "word2idx": word2idx,
        "idx2word": idx2word,
        "category2idx": category2idx,
        "idx2category": idx2category,
        "hyperparameters": {
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "dropout_rate": dropout_rate
        }
    }
    update_model(name, model_data)
    
    return {"status": "Entraînement terminé", "model_name": name}

def build_vocab(training_data: List[GRUTrainingData]):
    """
    Construit le vocabulaire à partir des données d'entraînement.
    :param training_data: Liste des données d'entraînement.
    :return: Dictionnaires de mapping mot->indice et indice->mot.
    """
    all_tokens = [token for data in training_data for token in data.tokens]
    token_counts = Counter(all_tokens)
    # Mapping des mots vers des indices, en commençant à 1 (0 réservé pour le padding)
    word2idx = {word: idx+1 for idx, (word, _) in enumerate(token_counts.most_common())}
    word2idx['<PAD>'] = 0  # Token de padding
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word

def build_category_mapping(training_data: List[GRUTrainingData]):
    """
    Construit le mapping entre les catégories et les indices.
    :param training_data: Liste des données d'entraînement.
    :return: Dictionnaires de mapping catégorie->indice et indice->catégorie.
    """
    categories = set(data.category for data in training_data)
    category2idx = {category: idx for idx, category in enumerate(categories)}
    idx2category = {idx: category for category, idx in category2idx.items()}
    return category2idx, idx2category

def prepare_sequences(training_data: List[GRUTrainingData], word2idx: Dict[str, int], category2idx: Dict[str, int]):
    """
    Prépare les séquences et les labels pour l'entraînement.
    :param training_data: Liste des données d'entraînement.
    :param word2idx: Dictionnaire de mapping mot->indice.
    :param category2idx: Dictionnaire de mapping catégorie->indice.
    :return: Tenseurs des séquences et des labels.
    """
    sequences = []
    labels = []
    max_seq_length = max(len(data.tokens) for data in training_data)  # Longueur maximale des séquences
    for data in training_data:
        # Conversion des tokens en indices
        seq = [word2idx.get(token, word2idx['<PAD>']) for token in data.tokens]
        # Padding des séquences pour qu'elles aient toutes la même longueur
        seq += [word2idx['<PAD>']] * (max_seq_length - len(seq))
        sequences.append(seq)
        labels.append(category2idx[data.category])
    sequences = torch.tensor(sequences, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    return sequences, labels