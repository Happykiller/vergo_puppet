import torch
from typing import List
from app.repositories.memory import get_model
from fastapi import HTTPException
from app.machine_learning.nn_gru import GRUClassifier

def search_model_gru(name: str, vector: List[str]):
    """
    Utilise le modèle GRU pour prédire la catégorie d'une nouvelle séquence de tokens.
    :param name: Nom du modèle.
    :param vector: Liste de tokens représentant la séquence à classer.
    :return: Catégorie prédite.
    """
    model_data = get_model(name)
    
    if model_data is None or not model_data:
        raise HTTPException(status_code=404, detail="Modèle non trouvé")
    
    nn_model_state = model_data.get("nn_model", None)
    if nn_model_state is None:
        raise HTTPException(status_code=400, detail="Modèle non entraîné")
    
    word2idx = model_data.get("word2idx", None)
    idx2category = model_data.get("idx2category", None)
    if word2idx is None or idx2category is None:
        raise HTTPException(status_code=400, detail="Données du modèle incomplètes")
    
    # Charger le modèle avec les paramètres sauvegardés
    vocab_size = len(word2idx)
    num_classes = len(idx2category)
    embedding_dim = 100
    hidden_dim = 128

    # Récupérer les hyperparamètres sauvegardés
    hyperparameters = model_data.get("hyperparameters", None)
    if hyperparameters is None:
        raise Exception("Hyperparamètres non trouvés dans les données du modèle.")

    embedding_dim = hyperparameters['embedding_dim']
    hidden_dim = hyperparameters['hidden_dim']
    dropout_rate = hyperparameters['dropout_rate']

    # Charger le modèle avec les hyperparamètres appropriés
    model_gru = GRUClassifier(vocab_size, embedding_dim, hidden_dim, num_classes, dropout_rate)
    model_gru.load_state_dict(nn_model_state)
    model_gru.eval()
    
    # Préparer la séquence d'entrée
    input_processed = process_input(vector, word2idx)
    
    # Prédiction
    with torch.no_grad():
        outputs = model_gru(input_processed)
        predicted_idx = torch.argmax(outputs, dim=1).item()
        predicted_category = idx2category[predicted_idx]
    
    return {"category": predicted_category}

def process_input(tokens: List[str], word2idx):
    """
    Transforme la liste de tokens en tenseur d'indices avec padding.
    :param tokens: Liste de tokens de la séquence à classer.
    :param word2idx: Dictionnaire de mapping mot->indice.
    :return: Tenseur de la séquence préparée.
    """
    seq = [word2idx.get(token, word2idx['<PAD>']) for token in tokens]
    max_seq_length = len(seq)
    seq += [word2idx['<PAD>']] * (max_seq_length - len(seq))  # Padding si nécessaire
    seq_tensor = torch.tensor([seq], dtype=torch.long)
    return seq_tensor
