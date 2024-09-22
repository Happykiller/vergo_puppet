from app.repositories.memory import get_model, update_model
from app.models.neural_network import train_model_nn
from app.usecases.tokens_to_indices import tokens_to_indices
from fastapi import HTTPException  # type: ignore
from typing import List, Tuple

def train_model(name: str, training_data: List[Tuple[List[str], List[str]]]):
    """
    Entraîne le modèle avec des paires (input, target).
    :param name: Nom du modèle
    :param training_data: Liste de tuples (input, target) où input et target sont des listes de tokens
    """
    model = get_model(name)

    if model is None or not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if training_data is None:
        raise HTTPException(status_code=400, detail="No training data provided")
    
    if len(training_data) == 0:
        raise HTTPException(status_code=400, detail="Training data is empty")
    
    indexed_dictionary = model.get("indexed_dictionary", [])
    if not indexed_dictionary:
        raise HTTPException(status_code=400, detail="No vectors available in the model")

    # Trouver la taille maximale des vecteurs dans le dictionnaire
    max_dictionary_size = max(len(vector) for vector in indexed_dictionary)

    # Transformer chaque paire de tokens en indices
    indexed_training_data = [
        (tokens_to_indices(input_seq, model["glossary"]), tokens_to_indices(target_seq, model["glossary"]))
        for input_seq, target_seq in training_data
    ]

    # Trouver la taille maximale des vecteurs
    max_input_size = max(len(pair[0]) for pair in indexed_training_data)
    max_target_size = max(len(pair[1]) for pair in indexed_training_data)
    max_size = max(max_input_size, max_target_size)

    if max_size != max_dictionary_size:
        raise ValueError(f"Les tailles maximum pour le dictionnaire et les données d'entrainement sont divergentes, max_size : {str(max_size)}, max_size : {str(max_dictionary_size)}")

    # Entraîner le réseau de neurones avec les vecteurs paddés
    nn_model, _ = train_model_nn(indexed_training_data, max_size)

    # Enregistrer le modèle de réseau de neurones entraîné
    update_model(name, {"nn_model": nn_model})

    return {"status": "training completed", "model_name": name}
