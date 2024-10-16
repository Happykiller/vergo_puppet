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

def search_model(name: str, search: list):
    model = get_model(name)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    nn_model = model.get("nn_model", None)
    if not nn_model:
        raise HTTPException(status_code=400, detail="No neural network model found in the model")

    indexed_dictionary = model.get("indexed_dictionary", [])
    dictionary = model.get("dictionary", [])
    if not indexed_dictionary:
        raise HTTPException(status_code=400, detail="No vectors available in the model")
    
    glossary = model.get("glossary", [])
    indexed_search = tokens_to_indices(search, glossary)

    # Trouver la taille maximale des vecteurs dans le dictionnaire
    max_vector_size = max(len(vector) for vector in indexed_dictionary)

    # Padder le vecteur de recherche pour qu'il ait la même longueur
    padded_indexed_search = pad_vector(indexed_search, max_vector_size)

    # Récupérer le type de modèle (par défaut SimpleNN)
    neural_network_type = model.get("neural_network_type", "SimpleNN")

    logger.info(f"Type de machine learning utilisé pour la recherche {neural_network_type}")

    if neural_network_type == "SimpleNN":
        # Utiliser le réseau de neurones SimpleNN pour prédire le vecteur le plus proche
        predicted_vector = predict(nn_model, padded_indexed_search)

        # Recherche du meilleur vecteur dans le dictionnaire paddé
        try:
            padded_indexed_dictionary = [pad_vector(vector, max_vector_size) for vector in indexed_dictionary]
            indexed_find = min(padded_indexed_dictionary, key=lambda v: torch.dist(torch.Tensor(v), predicted_vector).item())
        except ValueError:
            raise HTTPException(status_code=400, detail="No valid vectors to compare")

        find = dictionary[padded_indexed_dictionary.index(indexed_find)]

        # Calcul de l'accuracy pour SimpleNN
        accuracy = 1 / torch.dist(torch.Tensor(indexed_find), predicted_vector).item()

    elif neural_network_type == "LSTMNN":
        # Utiliser LSTM pour la recherche basée sur la similarité cosinus
        search_result = search_with_similarity(nn_model, padded_indexed_search, indexed_dictionary)
        indexed_find = search_result['best_match']
        index_find = search_result['best_match_index']
        accuracy = search_result['similarity_score']
        find = dictionary[index_find]

    elif neural_network_type == "SIAMESE":
        word2idx = create_indexed_glossary(glossary)
        search_indices = tokens_to_indices(search, word2idx)

        similarities = []
        for vector in dictionary:
            vector_indices = tokens_to_indices(vector, word2idx)
            similarity = evaluate_similarity(nn_model, search_indices, vector_indices)
            similarities.append((vector, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)

        accuracy = similarities[0][1]
        find = similarities[0][0]
    else:
        raise HTTPException(status_code=400, detail="Invalid model type")

    return {
        "search": search,
        "find": find,
        "stats": {
            "accuracy": accuracy
        }
    }
