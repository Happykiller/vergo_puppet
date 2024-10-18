from app.apis.models.simple_nn_search_data import SimpleNNSearchData
from app.commons.commons import create_glossary_from_dictionary, create_indexed_glossary, pad_vector
from app.machine_learning.neural_network_siamese import evaluate_similarity
from app.repositories.memory import get_model
from app.machine_learning.neural_network_simple import predict
from app.machine_learning.neural_network_lstm import search_with_similarity
from app.services.logger import logger
from app.usecases.indices_to_tokens import indices_to_tokens
from app.usecases.tokens_to_indices import tokens_to_indices
from fastapi import HTTPException # type: ignore
import torch

def search_model_simple_nn(name: str, search: SimpleNNSearchData):
    model = get_model(name)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    nn_model = model.get("nn_model", None)
    if not nn_model:
        raise HTTPException(status_code=400, detail="No neural network model found in the model")

    transformed_data = [
        search.type,
        search.surface,
        search.pieces,
        search.floor,
        search.parking,
        search.balcon,
        search.ascenseur,
        search.orientation,
        search.transports,
        search.neighborhood
    ]

    # Récupérer le type de modèle (par défaut SimpleNN)
    neural_network_type = model.get("neural_network_type", "SimpleNN")

    logger.info(f"Type de machine learning utilisé pour la recherche {neural_network_type}")

    # Utiliser le réseau de neurones SimpleNN pour prédire le vecteur le plus proche
    predicted = predict(nn_model, transformed_data)

    return {
        "search": search,
        "find": predicted
    }
