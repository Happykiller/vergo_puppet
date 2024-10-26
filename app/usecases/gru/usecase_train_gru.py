from typing import List, Dict
from collections import Counter
from app.services.logger import logger
from app.machine_learning.nn_gru import train_gru
from fastapi import HTTPException  # type: ignore
from app.repositories.memory import get_model, update_model
from app.apis.models.gru_training_data import GRUTrainingData

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
    
    # Affichage des statistiques sur les données d'entraînement
    num_documents = len(training_data)
    categories = [data.category for data in training_data]
    category_counts = Counter(categories)
    num_categories = len(category_counts)
    
    logger.info(f"Nombre de documents : {num_documents}")
    logger.info(f"Nombre de catégories : {num_categories}")
    logger.info("Répartition des catégories :")
    for category, count in category_counts.items():
        percentage = (count / num_documents) * 100
        logger.info(f" - {category}: {count} documents ({percentage:.2f}%)")
    
    sequence_lengths = [len(data.tokens) for data in training_data]
    max_seq_length = max(sequence_lengths)
    min_seq_length = min(sequence_lengths)
    avg_seq_length = sum(sequence_lengths) / num_documents
    
    logger.info(f"Longueur maximale des séquences : {max_seq_length}")
    logger.info(f"Longueur minimale des séquences : {min_seq_length}")
    logger.info(f"Longueur moyenne des séquences : {avg_seq_length:.2f}")
    
    # Construction du vocabulaire
    word2idx, idx2word = build_vocab(training_data)
    vocab_size = len(word2idx)
    logger.info(f"Taille du vocabulaire : {vocab_size}")
    
    # Mapping des catégories
    category2idx, idx2category = build_category_mapping(training_data)
    
    # Préparation des séquences et des labels
    sequences, labels = prepare_sequences(training_data, word2idx, category2idx, max_seq_length)
    
    # Paramètres du modèle (hyperparamètres ajustés)
    num_classes = len(category2idx)

    # Entraîner le modèle
    logger.info("Entraînement du modèle...")
    model = train_gru(vocab_size, num_classes, sequences, labels)
    
    # Enregistrer le modèle entraîné, les mappings et les hyperparamètres
    model_data = {
        "neural_network_type": "GRU",
        "nn_model": model,
        "word2idx": word2idx,
        "idx2word": idx2word,
        "category2idx": category2idx,
        "idx2category": idx2category
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

def prepare_sequences(training_data: List[GRUTrainingData], word2idx: Dict[str, int], category2idx: Dict[str, int], max_seq_length: int):
    """
    Prépare les séquences et les labels pour l'entraînement.
    :param training_data: Liste des données d'entraînement.
    :param word2idx: Dictionnaire de mapping mot->indice.
    :param category2idx: Dictionnaire de mapping catégorie->indice.
    :param max_seq_length: Longueur maximale des séquences pour le padding.
    :return: Tenseurs des séquences et des labels.
    """
    sequences = []
    labels = []
    for data in training_data:
        # Conversion des tokens en indices
        seq = [word2idx.get(token, word2idx['<PAD>']) for token in data.tokens]
        # Padding des séquences pour qu'elles aient toutes la même longueur
        seq += [word2idx['<PAD>']] * (max_seq_length - len(seq))
        sequences.append(seq)
        labels.append(category2idx[data.category])
    return sequences, labels