from app.usecases.gru.usecase_commons_gru import process_input
from typing import List
from app.repositories.memory import get_model
from fastapi import HTTPException
from app.machine_learning.nn_gru import predict

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
    
    nn_model = model_data.get("nn_model", None)
    if nn_model is None:
        raise HTTPException(status_code=400, detail="Modèle non entraîné")
    
    word2idx = model_data.get("word2idx", None)
    idx2category = model_data.get("idx2category", None)
    if word2idx is None or idx2category is None:
        raise HTTPException(status_code=400, detail="Données du modèle incomplètes")
    
    # Préparer la séquence d'entrée
    input = process_input(vector, word2idx)

    # Charger le modèle avec les hyperparamètres appropriés
    predicted_idx = predict(nn_model, input)

    predicted_category = idx2category[predicted_idx]
    
    return {"category": predicted_category}