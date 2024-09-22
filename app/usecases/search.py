from app.commons.commons import pad_vector
from app.repositories.memory import get_model
from app.models.neural_network import predict
from app.usecases.indices_to_tokens import indices_to_tokens
from app.usecases.tokens_to_indices import tokens_to_indices
from fastapi import HTTPException # type: ignore
import torch

def search_model(name: str, search: list):
    model = get_model(name)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    nn_model = model.get("nn_model", None)
    if not nn_model:
        raise HTTPException(status_code=400, detail="No neural network model found in the model")

    indexed_dictionary = model.get("indexed_dictionary", [])
    if not indexed_dictionary:
        raise HTTPException(status_code=400, detail="No vectors available in the model")
    
    indexed_search = tokens_to_indices(search, model["glossary"])

    # Trouver la taille maximale des vecteurs dans le dictionnaire
    max_vector_size = max(len(vector) for vector in indexed_dictionary)

    # Padder le vecteur de recherche pour qu'il ait la même longueur
    padded_indexed_search = pad_vector(indexed_search, max_vector_size)

    # Utiliser le réseau de neurones pour prédire le vecteur le plus proche
    predicted_vector = predict(nn_model, padded_indexed_search)

    # Recherche du meilleur vecteur dans le dictionnaire paddé
    try:
        padded_indexed_dictionary = [pad_vector(vector, max_vector_size) for vector in indexed_dictionary]
        indexed_find = max(padded_indexed_dictionary, key=lambda v: torch.dist(torch.Tensor(v), predicted_vector).item())
    except ValueError:
        raise HTTPException(status_code=400, detail="No valid vectors to compare")

    find = indices_to_tokens(indexed_find, model["glossary"])

    accuracy = 1 / torch.dist(torch.Tensor(indexed_find), predicted_vector).item()

    return {
        "search": search,
        "indexed_search": indexed_search,
        "indexed_find": indexed_find,
        "find": find,
        "stats": {
            "accuracy": accuracy
        },
        "model_used": name
    }
