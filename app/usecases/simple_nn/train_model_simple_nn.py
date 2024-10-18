from app.apis.models.simple_nn_training_data import SimpleNNTrainingData
from app.commons.commons import create_glossary_from_training_data
from app.machine_learning.neural_network_siamese import train_siamese_model_nn
from app.repositories.memory import get_model, update_model
from app.machine_learning.neural_network_simple import train_model_nn
from app.machine_learning.neural_network_lstm import train_lstm_model_nn
from app.usecases.tokens_to_indices import tokens_to_indices
from fastapi import HTTPException  # type: ignore
from typing import List, Tuple
from app.services.logger import logger

# Fonction pour transformer SimpleNNTrainingData en Tuple[List[int], int]
def transform_data(training_data: List[SimpleNNTrainingData]) -> List[Tuple[List[int], int]]:
    transformed_data = []
    
    for data in training_data:
        # On transforme les attributs en une liste d'entiers sauf le prix
        input_data = [
            data.type,
            data.surface,
            data.pieces,
            data.floor,
            data.parking,
            data.balcon,
            data.ascenseur,
            data.orientation,
            data.transports,
            data.neighborhood
        ]
        
        # Le prix est la cible, sous forme de float
        target_price = float(data.price)
        
        # On ajoute le tuple (input_data, target_price) à la liste transformée
        transformed_data.append((input_data, target_price))
    
    return transformed_data

def train_model_simple_nn(name: str, training_data: List[SimpleNNTrainingData]):
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
    
    # Vérifier le type de modèle à utiliser
    neural_network_type = model.get("neural_network_type", "SimpleNN")  # Par défaut SimpleNN si non spécifié
    logger.info(f"Type de machine learning utilisé pour l'entrainement {neural_network_type}")
    
    transformed_data = transform_data(training_data)

    # Récupérer la taille des tableaux dans les tuples
    vector_size = len(transformed_data[0][0])

    # Entraîner le réseau de neurones en fonction du type de modèle
    nn_model, _ = train_model_nn(transformed_data, vector_size)

    # Enregistrer le modèle de réseau de neurones entraîné
    update_model(name, {"nn_model": nn_model})

    return {"status": "training completed", "model_name": name}
